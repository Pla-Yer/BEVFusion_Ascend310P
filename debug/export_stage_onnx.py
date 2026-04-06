"""
逐阶段定位 OM 误差来源

方法：用 PyTorch 计算每个阶段的输入，导出单阶段 ONNX，
      ATC 转换后用 msame 运行，对比输出。

阶段划分：
  Stage A: VoxelEncoder          输入: voxels, num_points, coords
  Stage B: MiddleEncoder(Scatter) 输入: voxel_features, coords
  Stage C: Backbone               输入: bev_feat
  Stage D: Neck                   输入: bb_feats
  Stage E: SharedConv+HeatmapHead 输入: neck_feat
  Stage F: Full Part2             输入: voxel_features, coords

运行方式：
  1. python export_stage_onnx.py --stage B   # 导出 Stage B
  2. atc 转换 Stage B
  3. msame 运行 Stage B，对比输出
  4. 找到第一个出错的 Stage
"""

import argparse
import os
from os import path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(__file__))
try:
    import fix_pytorch_weights_only  # noqa
except ModuleNotFoundError:
    pass

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model


# ════════════════════════════════════════════════════════════════════════════
#  单阶段模块
# ════════════════════════════════════════════════════════════════════════════
class stageA_VoxelEncoder(nn.Module):
    

    def __init__(self, model):
        super().__init__()
        self.ve = model.pts_voxel_encoder

    def forward(self, voxels, num_points, coords):
        # 将 coords也打包输出
        return self.ve(voxels, num_points, coords), coords
class StageB_Scatter(nn.Module):

    def __init__(self, model):
        super().__init__()
        orig = model.pts_middle_encoder
        self.C = orig.in_channels
        self.ny = orig.ny
        self.nx = orig.nx

    def forward(self, voxel_features, coords):

        C, ny, nx = self.C, self.ny, self.nx
        HW = ny * nx

        indices = (
            coords[:, 2].long() * nx
            + coords[:, 3].long()
        ).to(torch.int32)

        canvas = torch.zeros(
            C, HW,
            dtype=voxel_features.dtype,
            device=voxel_features.device
        )

        canvas = canvas.scatter(
            1,
            indices.unsqueeze(0).expand(C, -1),
            voxel_features.t()
        )

        return canvas.view(C, 1, ny, nx).permute(1, 0, 2, 3).contiguous()


class StageC_Backbone(nn.Module):
    """输入: bev_feat[1,C,H,W] → 输出: tuple of feats"""
    def __init__(self, model):
        super().__init__()
        self.bb = model.pts_backbone

    def forward(self, bev_feat):
        outs = self.bb(bev_feat)
        return tuple(outs)


class StageD_Neck(nn.Module):
    """输入: bb_feats → 输出: neck_feat[1,C,H,W]"""
    def __init__(self, model):
        super().__init__()
        self.neck = model.pts_neck

    def forward(self, *bb_feats):
        return self.neck(list(bb_feats))[0]


class StageE_Head(nn.Module):
    """输入: neck_feat[1,C,H,W] → 输出: dense_heatmap"""
    def __init__(self, model):
        super().__init__()
        self.shared_conv  = model.bbox_head.shared_conv
        self.heatmap_head = model.bbox_head.heatmap_head

    def forward(self, neck_feat):
        return self.heatmap_head(self.shared_conv(neck_feat))


# ════════════════════════════════════════════════════════════════════════════
#  工具
# ════════════════════════════════════════════════════════════════════════════

def make_unique_coords(num_voxels, ny, nx, device="cuda"):
    torch.manual_seed(42)
    flat_idx     = torch.randperm(ny * nx, device=device)[:num_voxels]
    coords       = torch.zeros(num_voxels, 4, device=device)
    coords[:, 2] = (flat_idx // nx).float()
    coords[:, 3] = (flat_idx %  nx).float()
    return coords[coords[:, 0].argsort()]


def export_stage(module, inputs, input_names, output_names, path, dynamic_axes=None):
    with torch.no_grad():
        torch.onnx.export(
            module, inputs, path,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False,
        )
    print(f"  导出: {path}")


def save_bin(arr, path):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr.tofile(path)
    print(f"  bin: {path}  shape={arr.shape}  dtype={arr.dtype}"
          f"  min={arr.min():.4f}  max={arr.max():.4f}")


def print_atc_cmd(onnx_path, om_path, input_shape_str, dynamic_dims_str,
                  soc="Ascend310P1"):
    print(f"""
  ATC 命令:
    atc --model="{onnx_path}" \\
        --framework=5 \\
        --output="{om_path}" \\
        --input_format=ND \\
        --input_shape="{input_shape_str}" \\
        --dynamic_dims="{dynamic_dims_str}" \\
        --soc_version={soc} \\
        --precision_mode=allow_fp32_to_fp16 \\
        --log=warning
""")


def print_msame_cmd(om_path, bin_paths, dym_dims_str, output_dir):
    bins = ",".join(bin_paths)
    print(f"""
  msame 命令:
    ./msame --model "{om_path}.om" \\
        --input "{bins}" \\
        --dymDims "{dym_dims_str}" \\
        --output {output_dir} --loop 1
""")


def compare_with_pt(pt_out, om_bin_path, shape, dtype=np.float16, label=""):
    """对比 PyTorch 输出与 OM bin 文件"""
    try:
        om_data = np.fromfile(om_bin_path, dtype=dtype).astype(np.float32)
    except Exception:
        om_data = np.fromfile(om_bin_path, dtype=np.float32)

    pt_data = pt_out.detach().cpu().numpy().astype(np.float32).flatten()
    om_data = om_data.flatten()

    min_len = min(len(pt_data), len(om_data))
    pt_data = pt_data[:min_len]
    om_data = om_data[:min_len]

    diff = np.abs(pt_data - om_data)
    print(f"  [{label}] PT: min={pt_data.min():.4f} max={pt_data.max():.4f}")
    print(f"  [{label}] OM: min={om_data.min():.4f} max={om_data.max():.4f}")
    print(f"  [{label}] 误差: max|Δ|={diff.max():.4e}  mean|Δ|={diff.mean():.4e}")

    ok = np.allclose(pt_data, om_data, atol=0.05, rtol=0.05)
    print(f"  [{label}] allclose(atol=0.05): {'✅' if ok else '❌'}")
    return ok


# ════════════════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
    p.add_argument("--ckpt",   default="work_dirs/bevfusion/epoch_20.pth")
    p.add_argument("--outdir", default="models/onnx_stages")
    p.add_argument("--bindir", default="stage_bins")
    p.add_argument("--num-voxels", type=int, default=5000)
    p.add_argument("--stage",  default="all",
                   choices=["all","A", "B", "C", "D", "E"],
                   help="只导出指定阶段（all=全部）")
    # 用于对比的 OM 输出 bin（可选）
    p.add_argument("--om-bin-B", default=None, help="Stage B OM 输出 bin 路径")
    p.add_argument("--om-bin-C", default=None, help="Stage C OM 输出 bin 路径")
    p.add_argument("--om-bin-D", default=None, help="Stage D OM 输出 bin 路径")
    p.add_argument("--om-bin-E", default=None, help="Stage E OM 输出 bin 路径")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.bindir, exist_ok=True)
    device = "cuda"

    print("[1/2] 加载模型并计算各阶段中间结果...")
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    if "img_backbone" in cfg.model and "init_cfg" in cfg.model.get("img_backbone", {}):
        cfg.model["img_backbone"]["init_cfg"] = None
    model = init_model(cfg, args.ckpt)
    model.eval()

    enc = model.pts_middle_encoder
    V   = 5000
    # torch.manual_seed(42)
    # voxels     = torch.randn(V, 32, 5, device=device)
    # num_points = torch.randint(1, 33, (V,), device=device)
    # coords     = make_unique_coords(V, enc.ny, enc.nx, device=device)
    om_sample_dir = "om_data/sample_0"
    om_voxels = np.load(os.path.join(om_sample_dir, 'voxels.npy'))
    om_coords = np.load(os.path.join(om_sample_dir, 'coords.npy'))
    om_num_points = np.load(os.path.join(om_sample_dir, 'num_points.npy'))
    # 转为 tensor
    voxels = torch.from_numpy(om_voxels).to(device)
    coords = torch.from_numpy(om_coords).to(device)
    num_points = torch.from_numpy(om_num_points).to(device)
    with torch.no_grad():
        # Stage A: voxel encoder
        sa       = stageA_VoxelEncoder(model).eval().to(device)
        vf , coords_out = sa(voxels, num_points, coords)

        # Stage B: scatter
        sb       = StageB_Scatter(model).eval().to(device)
        bev_feat = sb(vf, coords)

        # Stage C: backbone
        sc       = StageC_Backbone(model).eval().to(device)
        bb_feats = sc(bev_feat)

        # Stage D: neck
        sd       = StageD_Neck(model).eval().to(device)
        neck     = sd(*bb_feats)

        # Stage E: head
        se       = StageE_Head(model).eval().to(device)
        heatmap  = se(neck)

    print(f"  voxel_features: {vf.shape}  [{vf.min():.3f}, {vf.max():.3f}]")
    print(f"  bev_feat:       {bev_feat.shape}  [{bev_feat.min():.3f}, {bev_feat.max():.3f}]")
    print(f"  bb_feats:       {[x.shape for x in bb_feats]}")
    print(f"  neck_feat:      {neck.shape}  [{neck.min():.3f}, {neck.max():.3f}]")
    print(f"  heatmap(pre-sigmoid): {heatmap.shape}  [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # 保存各阶段输入 bin
    print("\n[2/2] 导出各阶段 ONNX 和输入 bin...")

    do_all = (args.stage == "all")
    # ── Stage A ──────────────────────────────────────────────────────────
    if do_all or args.stage == "A":
        print("\n── Stage A: VoxelEncoder ──")
        save_bin(voxels,    osp.join(args.bindir, "stageA_voxels.bin"))
        save_bin(num_points,osp.join(args.bindir, "stageA_num_points.bin"))
        save_bin(coords,    osp.join(args.bindir, "stageA_coords.bin"))
        export_stage(
            sa, (voxels, num_points, coords),
            ["voxels", "num_points", "coords"], ["voxel_features", "coords_out"],
            osp.join(args.outdir, "stageA_voxel_encoder.onnx"),
            dynamic_axes={
                "voxels": {0: "V"},
                "num_points": {0: "V"},
                "coords": {0: "V"}
            }
        )
        save_bin(vf, osp.join(args.bindir, "stageA_output_voxel_features.bin"))
        save_bin(coords_out, osp.join(args.bindir, "stageA_output_coords.bin"))
        print_atc_cmd(
            f"{args.outdir}/stageA_voxel_encoder.onnx",
            f"models/om/stageA_voxel_encoder",
            f"voxels:-1,20,5;num_points:-1;coords:-1,4",
            f"4000,4000,4000;5000,5000,5000;6000,6000,6000"
        )
        print_msame_cmd(
            "models/om/stageA_voxel_encoder",
            [f"{args.bindir}/stageA_voxels.bin",
             f"{args.bindir}/stageA_num_points.bin",
             f"{args.bindir}/stageA_coords.bin"],
            f"voxels:{V},20,5;num_points:{V};coords:{V},4",
            "om_out_A"
        )

    # ── Stage B ──────────────────────────────────────────────────────────
    if do_all or args.stage == "B":
        print("\n── Stage B: Scatter (MiddleEncoder) ──")
        save_bin(vf,     osp.join(args.bindir, "stageB_voxel_features.bin"))
        save_bin(coords, osp.join(args.bindir, "stageB_coords.bin"))
        export_stage(
            sb, (vf, coords),
            ["voxel_features", "coords"], ["bev_feat"],
            osp.join(args.outdir, "stageB_scatter.onnx"),
            dynamic_axes={"voxel_features": {0: "V"}, "coords": {0: "V"}}
        )
        save_bin(bev_feat, osp.join(args.bindir, "stageB_output_bev_feat.bin"))
        print_atc_cmd(
            f"{args.outdir}/stageB_scatter.onnx",
            f"models/om/stageB_scatter",
            f"voxel_features:-1,{vf.shape[1]};coords:-1,4",
            f"4000,4000;5000,5000;6000,6000"
        )
        print_msame_cmd(
            "models/om/stageB_scatter",
            [f"{args.bindir}/stageB_voxel_features.bin",
             f"{args.bindir}/stageB_coords.bin"],
            f"voxel_features:{V},{vf.shape[1]};coords:{V},4",
            "om_out_B"
        )
        if args.om_bin_B:
            compare_with_pt(bev_feat, args.om_bin_B, bev_feat.shape, label="StageB bev_feat")

    # ── Stage C ──────────────────────────────────────────────────────────
    if do_all or args.stage == "C":
        print("\n── Stage C: Backbone ──")
        save_bin(bev_feat, osp.join(args.bindir, "stageC_bev_feat.bin"))
        export_stage(
            sc, (bev_feat,),
            ["bev_feat"], [f"bb_out_{i}" for i in range(len(bb_feats))],
            osp.join(args.outdir, "stageC_backbone.onnx"),
        )
        for i, bf in enumerate(bb_feats):
            save_bin(bf, osp.join(args.bindir, f"stageC_output_bb_{i}.bin"))
        print_atc_cmd(
            f"{args.outdir}/stageC_backbone.onnx",
            f"models/om/stageC_backbone",
            f"bev_feat:1,{bev_feat.shape[1]},{bev_feat.shape[2]},{bev_feat.shape[3]}",
            ""  # 静态shape，无需dynamic_dims
        )
        print_msame_cmd(
            "models/om/stageC_backbone",
            [f"{args.bindir}/stageC_bev_feat.bin"],
            "",
            "om_out_C"
        )
        if args.om_bin_C:
            compare_with_pt(bb_feats[0], args.om_bin_C, bb_feats[0].shape, label="StageC bb_out_0")

    # ── Stage D ──────────────────────────────────────────────────────────
    if do_all or args.stage == "D":
        print("\n── Stage D: Neck ──")
        for i, bf in enumerate(bb_feats):
            save_bin(bf, osp.join(args.bindir, f"stageD_bb_{i}.bin"))
        bb_input_names = [f"bb_feat_{i}" for i in range(len(bb_feats))]
        export_stage(
            sd, tuple(bb_feats),
            bb_input_names, ["neck_feat"],
            osp.join(args.outdir, "stageD_neck.onnx"),
        )
        save_bin(neck, osp.join(args.bindir, "stageD_output_neck.bin"))
        shapes_str = ";".join(
            f"bb_feat_{i}:1,{bf.shape[1]},{bf.shape[2]},{bf.shape[3]}"
            for i, bf in enumerate(bb_feats)
        )
        print_atc_cmd(
            f"{args.outdir}/stageD_neck.onnx",
            f"models/om/stageD_neck",
            shapes_str, ""
        )
        if args.om_bin_D:
            compare_with_pt(neck, args.om_bin_D, neck.shape, label="StageD neck_feat")

    # ── Stage E ──────────────────────────────────────────────────────────
    if do_all or args.stage == "E":
        print("\n── Stage E: SharedConv + HeatmapHead ──")
        save_bin(neck, osp.join(args.bindir, "stageE_neck_feat.bin"))
        export_stage(
            se, (neck,),
            ["neck_feat"], ["dense_heatmap"],
            osp.join(args.outdir, "stageE_heatmap_head.onnx"),
        )
        save_bin(heatmap, osp.join(args.bindir, "stageE_output_heatmap.bin"))
        print_atc_cmd(
            f"{args.outdir}/stageE_heatmap_head.onnx",
            f"models/om/stageE_heatmap_head",
            f"neck_feat:1,{neck.shape[1]},{neck.shape[2]},{neck.shape[3]}",
            ""
        )
        print_msame_cmd(
            "models/om/stageE_heatmap_head",
            [f"{args.bindir}/stageE_neck_feat.bin"],
            "",
            "om_out_E"
        )
        if args.om_bin_E:
            compare_with_pt(heatmap, args.om_bin_E, heatmap.shape,
                            label="StageE dense_heatmap")

    print("\n操作步骤:")
    print("  1. 逐个阶段做 ATC 转换（从 B 开始）")
    print("  2. msame 运行，把输出 bin 路径通过 --om-bin-X 参数传回本脚本")
    print("  3. 找到第一个 ❌ 的阶段即为根因")
    print("  4. Stage C/D/E 使用静态 shape（无 dynamic_dims），避免 ATC tiling 问题")


if __name__ == "__main__":
    main()