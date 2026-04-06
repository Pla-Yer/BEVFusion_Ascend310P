"""
逐层诊断脚本：找出 ONNX 与 PyTorch 第一个产生大误差的位置

用法:
    python diagnose_layer.py \
        --config src/configs/... \
        --ckpt   work_dirs/bevfusion/epoch_20.pth \
        [--num-voxels 6000] [--seed 42]
    python diagnose_layer.py \
    --config src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    --ckpt   work_dirs/bevfusion/epoch_20.pth
"""

import argparse
import os
import os.path as osp
import sys
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(__file__))
try:
    import fix_pytorch_weights_only  # noqa
except ModuleNotFoundError:
    pass

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("pip install onnxruntime-gpu")

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model


# ════════════════════════════════════════════════════════════════════════════
#  各阶段的独立 nn.Module，方便单独导出/对比
# ════════════════════════════════════════════════════════════════════════════

class StageVoxelEncoder(nn.Module):
    """输入: voxels, num_points, coords  →  输出: voxel_features"""
    def __init__(self, model):
        super().__init__()
        self.enc = model.pts_voxel_encoder

    def forward(self, voxels, num_points, coords):
        return self.enc(voxels, num_points, coords)


class StageMiddleEncoder(nn.Module):
    """输入: voxel_features, coords  →  输出: bev_feat [B,C,H,W]"""
    def __init__(self, model):
        super().__init__()
        self.mid = model.pts_middle_encoder

    def forward(self, voxel_features, coords):
        # coords 已经是 [b,z,y,x] 格式
        coords_bzyx = coords[:, [0, 3, 1, 2]].contiguous()
        batch_size = int(coords_bzyx[-1, 0].item()) + 1
        return self.mid(voxel_features, coords_bzyx, batch_size)


class StageBackbone(nn.Module):
    """输入: bev_feat  →  输出: backbone feats (tuple)"""
    def __init__(self, model):
        super().__init__()
        self.bb = model.pts_backbone

    def forward(self, bev_feat):
        outs = self.bb(bev_feat)
        # 返回 tuple，ONNX 每个元素单独命名
        return tuple(outs)


class StageNeck(nn.Module):
    """输入: backbone feats  →  输出: neck_feat [B,C,H,W]"""
    def __init__(self, model):
        super().__init__()
        self.neck = model.pts_neck

    def forward(self, *backbone_feats):
        out = self.neck(list(backbone_feats))
        return out[0]


class StageSharedConv(nn.Module):
    """输入: neck_feat  →  输出: fusion_feat"""
    def __init__(self, model):
        super().__init__()
        self.conv = model.bbox_head.shared_conv

    def forward(self, neck_feat):
        return self.conv(neck_feat)


class StageHeatmapHead(nn.Module):
    """输入: fusion_feat  →  输出: dense_heatmap (before sigmoid)"""
    def __init__(self, model):
        super().__init__()
        self.hm = model.bbox_head.heatmap_head

    def forward(self, fusion_feat):
        return self.hm(fusion_feat)


# ════════════════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════════════════

def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def ort_run(module, inputs_np: dict, tmp_path: str):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(tmp_path, providers=providers)
    return sess.run(None, inputs_np)


def export_and_run(module, torch_inputs: tuple, input_names: list, tmp_path: str):
    """导出为临时 ONNX，再用 ORT 跑，返回 (pytorch_outs, onnx_outs)"""
    module.eval()

    with torch.no_grad():
        pt_outs = module(*torch_inputs)

    if not isinstance(pt_outs, (tuple, list)):
        pt_outs = (pt_outs,)

    output_names = [f"out_{i}" for i in range(len(pt_outs))]

    with torch.no_grad():
        torch.onnx.export(
            module,
            torch_inputs,
            tmp_path,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False,
        )

    inputs_np = {n: to_numpy(t) for n, t in zip(input_names, torch_inputs)}
    onnx_outs = ort_run(module, inputs_np, tmp_path)
    pt_outs_np = [to_numpy(o) for o in pt_outs]

    return pt_outs_np, onnx_outs


def report(stage_name, pt_outs, onnx_outs, atol=1e-2):
    sep = "─" * 64
    print(f"\n  [{stage_name}]")
    print(f"  {'输出':>4}  {'shape':<22} {'max|Δ|':>10} {'mean|Δ|':>10}  pass?")
    print(f"  {sep}")
    all_ok = True
    for i, (pt, onnx) in enumerate(zip(pt_outs, onnx_outs)):
        pt   = pt.astype(np.float32)
        onnx = onnx.astype(np.float32)
        diff = np.abs(pt - onnx)
        ok   = np.allclose(pt, onnx, atol=atol, rtol=atol)
        all_ok = all_ok and ok
        flag = "✅" if ok else "❌"
        print(f"  out{i}  {str(pt.shape):<22} {diff.max():>10.3e} {diff.mean():>10.3e}  {flag}")
    print(f"  {sep}")
    return all_ok


# ════════════════════════════════════════════════════════════════════════════
#  主诊断流程
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
    p.add_argument("--ckpt",   default="work_dirs/bevfusion/epoch_20.pth")
    p.add_argument("--num-voxels", type=int, default=6000)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--atol",       type=float, default=1e-2)
    args = p.parse_args()

    # ── 加载模型 ─────────────────────────────────────────────────────────
    print("加载模型...")
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    if "img_backbone" in cfg.model and "init_cfg" in cfg.model.get("img_backbone", {}):
        cfg.model["img_backbone"]["init_cfg"] = None
    model = init_model(cfg, args.ckpt)
    model.eval()

    # ── 生成输入 ─────────────────────────────────────────────────────────
    V, M, Cin = args.num_voxels, 32, 5
    torch.manual_seed(args.seed)
    device = "cuda"
    voxels     = torch.randn(V, M, Cin, device=device)
    num_points = torch.randint(1, M + 1, (V,), device=device)
    coords     = torch.zeros(V, 4, device=device)
    coords[:, 1] = torch.randint(0, 1,   (V,)).float()
    coords[:, 2] = torch.randint(0, 360, (V,)).float()
    coords[:, 3] = torch.randint(0, 360, (V,)).float()

    tmpdir = tempfile.mkdtemp()
    print(f"临时 ONNX 目录: {tmpdir}")
    print(f"\n{'='*64}")
    print("  逐阶段精度诊断 (atol={})".format(args.atol))
    print(f"{'='*64}")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1: VoxelEncoder
    # ══════════════════════════════════════════════════════════════════════
    stage1 = StageVoxelEncoder(model).eval().to(device)
    pt1, onnx1 = export_and_run(
        stage1,
        (voxels, num_points, coords),
        ["voxels", "num_points", "coords"],
        osp.join(tmpdir, "s1_voxel_enc.onnx")
    )
    ok1 = report("Stage1: VoxelEncoder", pt1, onnx1, args.atol)
    voxel_features_pt = torch.from_numpy(pt1[0]).to(device)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2: MiddleEncoder (Scatter to BEV)
    # ══════════════════════════════════════════════════════════════════════
    stage2 = StageMiddleEncoder(model).eval().to(device)
    pt2, onnx2 = export_and_run(
        stage2,
        (voxel_features_pt, coords),
        ["voxel_features", "coords"],
        osp.join(tmpdir, "s2_middle_enc.onnx")
    )
    ok2 = report("Stage2: MiddleEncoder (BEV scatter)", pt2, onnx2, args.atol)
    bev_feat_pt = torch.from_numpy(pt2[0]).to(device)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3: Backbone
    # ══════════════════════════════════════════════════════════════════════
    stage3 = StageBackbone(model).eval().to(device)
    pt3, onnx3 = export_and_run(
        stage3,
        (bev_feat_pt,),
        ["bev_feat"],
        osp.join(tmpdir, "s3_backbone.onnx")
    )
    ok3 = report("Stage3: Backbone", pt3, onnx3, args.atol)
    # 取第一个输出作为后续输入（通常有多尺度）
    bb_feats_pt = [torch.from_numpy(o).to(device) for o in pt3]

    # ══════════════════════════════════════════════════════════════════════
    # Stage 4: Neck (FPN)
    # ══════════════════════════════════════════════════════════════════════
    stage4 = StageNeck(model).eval().to(device)
    pt4, onnx4 = export_and_run(
        stage4,
        tuple(bb_feats_pt),
        [f"bb_feat_{i}" for i in range(len(bb_feats_pt))],
        osp.join(tmpdir, "s4_neck.onnx")
    )
    ok4 = report("Stage4: Neck (FPN)", pt4, onnx4, args.atol)
    neck_feat_pt = torch.from_numpy(pt4[0]).to(device)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 5: Shared Conv
    # ══════════════════════════════════════════════════════════════════════
    stage5 = StageSharedConv(model).eval().to(device)
    pt5, onnx5 = export_and_run(
        stage5,
        (neck_feat_pt,),
        ["neck_feat"],
        osp.join(tmpdir, "s5_shared_conv.onnx")
    )
    ok5 = report("Stage5: SharedConv", pt5, onnx5, args.atol)
    fusion_feat_pt = torch.from_numpy(pt5[0]).to(device)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 6: Heatmap Head (before sigmoid)
    # ══════════════════════════════════════════════════════════════════════
    stage6 = StageHeatmapHead(model).eval().to(device)
    pt6, onnx6 = export_and_run(
        stage6,
        (fusion_feat_pt,),
        ["fusion_feat"],
        osp.join(tmpdir, "s6_heatmap_head.onnx")
    )
    ok6 = report("Stage6: HeatmapHead (pre-sigmoid)", pt6, onnx6, args.atol)

    # ══════════════════════════════════════════════════════════════════════
    # 汇总
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    print("  诊断汇总")
    print(f"{'='*64}")
    stages = [
        ("Stage1: VoxelEncoder",        ok1),
        ("Stage2: MiddleEncoder",        ok2),
        ("Stage3: Backbone",             ok3),
        ("Stage4: Neck",                 ok4),
        ("Stage5: SharedConv",           ok5),
        ("Stage6: HeatmapHead",          ok6),
    ]
    first_fail = None
    for name, ok in stages:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name:<35}: {status}")
        if not ok and first_fail is None:
            first_fail = name

    print()
    if first_fail:
        print(f"  ⚠️  第一个失败阶段: {first_fail}")
        print(f"     → 该阶段包含 ONNX 导出不兼容的算子，请重点检查。")
        print()
        print("  常见原因及排查方向:")
        print("  Stage1 失败 → VoxelEncoder 含自定义/稀疏算子，需确认 opset 或自定义插件")
        print("  Stage2 失败 → scatter_nd / unique 等稀疏散射算子，ORT 支持不完整")
        print("  Stage3 失败 → Backbone 含 BN / GroupNorm，检查 eval() 是否生效")
        print("  Stage4 失败 → FPN 上采样插值模式与 ORT 不一致 (nearest vs bilinear)")
        print("  Stage5/6 失败 → Conv/BN 精度问题，尝试 opset_version=13 或 fp16")
    else:
        print("  ✅ 所有前向阶段精度正常，问题在 TopK/Decoder 逻辑层，请用修复版导出脚本")

    print(f"{'='*64}\n")
    print("临时 ONNX 文件保存在:", tmpdir)
    print("可用 Netron (https://netron.app) 打开各阶段 ONNX 文件进行可视化检查。")


if __name__ == "__main__":
    main()