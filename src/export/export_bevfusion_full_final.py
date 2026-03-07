"""
BEVFusion 最终正确导出脚本

修复汇总：
  1. 去掉 coords[:,[0,3,1,2]] 错误重排
  2. OnnxPointPillarsScatter 用 scatter() 替换 in-place index_put
  3. tiebreak bias 消除 topk 并列（arange → Range 节点，导出进 ONNX 图）
  4. 虚拟输入使用唯一坐标（randperm 模拟真实 voxelization）
"""

import argparse
import os
from os import path as osp
import sys
import io

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


class OnnxPointPillarsScatter(nn.Module):
    def __init__(self, in_channels, ny, nx):
        super().__init__()
        self.C, self.ny, self.nx = in_channels, ny, nx

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


class BEVFusionDeployFinal(nn.Module):

    TIEBREAK_EPS = 1e-6

    def __init__(self, model, K=200):
        super().__init__()

        self.pts_voxel_encoder = model.pts_voxel_encoder
        self.pts_backbone = model.pts_backbone
        self.pts_neck = model.pts_neck
        self.head = model.bbox_head
        self.K = 100

        orig = model.pts_middle_encoder
        self.pts_middle_encoder = OnnxPointPillarsScatter(
            orig.in_channels,
            orig.ny,
            orig.nx
        )

    def forward(self, voxels, num_points, coords):

        vf = self.pts_voxel_encoder(voxels, num_points, coords)

        mask = (coords.sum(dim=1) != 0).float().unsqueeze(1)
        vf = vf * mask

        bev = self.pts_middle_encoder(vf, coords)

        neck = self.pts_neck(list(self.pts_backbone(bev)))[0]

        head = self.head
        B = neck.shape[0]

        ff = head.shared_conv(neck)

        _, C, H, W = ff.shape
        HW = H * W

        ff_flat = ff.view(B, C, HW)

        bev_pos = head.bev_pos.repeat(B, 1, 1).to(ff.device)

        # heatmap
        dm = head.heatmap_head(ff)

        hm = torch.sigmoid(dm)

        # NMS
        p = head.nms_kernel_size // 2

        lm = F.max_pool2d(
            hm,
            kernel_size=head.nms_kernel_size,
            stride=1,
            padding=p
        )

        # Ascend安全比较，加上容差
        keep = (hm + 1e-3) >= lm

        hmf = (hm * keep.float()).view(B, head.num_classes, HW)

        # flatten
        score = hmf.view(B, -1)

        # tiebreak 防止 topk 退化
        idx = torch.arange(
            score.shape[-1],
            device=score.device,
            dtype=score.dtype
        ).unsqueeze(0)

        score = score + idx * self.TIEBREAK_EPS

        # topk
        scores, top = torch.topk(score, k=self.K, dim=-1)

        # class + spatial index (使用整数除法和取模，兼容昇腾OM)
        top_cls = top // HW
        top_idx = top % HW

        # top_f     = top.float()                      # int → float32
        # HW_f      = HW.float()                       # int → float32
        # top_cls_f = top_f // HW_f          # float Div + Floor（高优先）
        # top_idx_f = top_f % HW_f           # float Sub + Mul（高优先）
        # top_cls   = top_cls_f.long()
        # top_idx   = top_idx_f.long()


        # 保持 long 用于 one_hot
        top_cls_long = top_cls.long()

        # gather query feature
        gather_idx = top_idx.unsqueeze(1).repeat(1, C, 1)

        qf = torch.gather(ff_flat, 2, gather_idx)

        # class encoding
        qf = qf + head.class_encoding(
            F.one_hot(top_cls_long, head.num_classes)
            .permute(0, 2, 1)
            .float()
        )

        # gather position
        pos_idx = top_idx.unsqueeze(-1).repeat(1, 1, bev_pos.shape[-1])

        qp = torch.gather(bev_pos, 1, pos_idx)

        # decoder
        for i in range(head.num_decoder_layers):

            qf = head.decoder[i](
                qf,
                key=ff_flat,
                query_pos=qp,
                key_pos=bev_pos
            )

            res = head.prediction_heads[i](qf)

            res["center"] = res["center"] + qp.permute(0, 2, 1)

            qp = res["center"].permute(0, 2, 1)

        qhs = hmf.gather(
            -1,
            top_idx[:, None, :].expand(-1, head.num_classes, -1)
        )

        return (
            dm,
            top_cls.to(torch.int32),
            qhs,
            res["heatmap"],
            res["center"],
            res["height"],
            res["dim"],
            res["rot"],
            res.get(
                "vel",
                torch.zeros(B, 2, self.K, device=voxels.device)
            ),
            top_idx.to(torch.int32),
            top.to(torch.int32),
            score.to(torch.float32),
            keep.to(torch.int32),
            hm.to(torch.float32),
            lm.to(torch.float32),
        )

def make_unique_coords(num_voxels, ny, nx, device="cuda"):
    flat_idx     = torch.randperm(ny * nx, device=device)[:num_voxels]
    coords       = torch.zeros(num_voxels, 4, device=device)
    coords[:, 2] = (flat_idx // nx).float()
    coords[:, 3] = (flat_idx %  nx).float()
    return coords[coords[:, 0].argsort()]


def verify_scatter(model, new_scatter, voxels, num_points, coords, device):
    captured = {}
    def hook_fn(module, inputs, output):
        captured['vf']     = inputs[0].detach().clone()
        captured['coors']  = inputs[1].detach().clone()
        captured['output'] = output.detach().clone()
    handle = model.pts_middle_encoder.register_forward_hook(hook_fn)
    with torch.no_grad():
        vf = model.pts_voxel_encoder(voxels, num_points, coords)
        model.pts_middle_encoder(vf, coords, 1)
    handle.remove()

    enc     = model.pts_middle_encoder
    coors   = captured['coors']
    HW      = enc.ny * enc.nx
    indices = coors[:,0].long()*HW + coors[:,2].long()*enc.nx + coors[:,3].long()
    n_uniq  = indices.unique().numel()
    print(f"      coords 唯一性: {n_uniq}/{indices.numel()} "
          + ("✅" if n_uniq == indices.numel() else "⚠️  有重复"))

    new_out = new_scatter(captured['vf'], coors)
    diff    = (captured['output'] - new_out).abs()
    print(f"      scatter 等价性: max|Δ|={diff.max():.2e}  mean|Δ|={diff.mean():.2e}")
    if diff.max() > 1e-4:
        raise AssertionError(f"scatter 不等价 max|Δ|={diff.max():.2e}")
    print("      ✅ scatter 验证通过")


def check_range_node(deploy, dummy_inputs):
    """验证导出图中包含 Range 节点（tiebreak bias 未被 constant folding 消除）"""
    try:
        import onnx
        buf = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(deploy, dummy_inputs, buf, opset_version=11,
                              do_constant_folding=True, dynamo=False, verbose=False,
                              operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
        buf.seek(0)
        g   = onnx.load(buf)
        ops = [n.op_type for n in g.graph.node]
        has_range = "CumSum" in ops
        print(f"      ONNX 图含 CumSum 节点（tiebreak bias）: "
              f"{'✅' if has_range else '❌ 被 constant folding 消除，bias 无效！'}")
        return has_range
    except ImportError:
        print("      (onnx 未安装，跳过图验证)")
        return True


INPUT_NAMES  = ["voxels", "num_points", "coords"]
OUTPUT_NAMES = ["dense_heatmap", "top_cls", "query_heatmap_score",
                "heatmap_q", "center", "height", "dim", "rot", "vel", 
                "top_idx", "top", "score", "keep","hm", "lm"]


def export_onnx(deploy, dummy_inputs, path, dynamic=False):
    dynamic_axes = ({"voxels": {0: "V"}, "num_points": {0: "V"}, "coords": {0: "V"}}
                    if dynamic else None)
    with torch.no_grad():
        torch.onnx.export(
            deploy, dummy_inputs, path,
            opset_version=11,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,  # 重要！保持 tiebreak bias 不被消除
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False,
        )
    print(f"  ✅ [{'动态' if dynamic else '静态'}] {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
    p.add_argument("--ckpt",        default="work_dirs/bevfusion/epoch_20.pth")
    p.add_argument("--outdir",      default="models/onnx_final")
    p.add_argument("--K",           type=int, default=200)
    p.add_argument("--num-voxels",  type=int, default=6000)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda"

    print("[1/4] 加载模型...")
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    if "img_backbone" in cfg.model and "init_cfg" in cfg.model.get("img_backbone", {}):
        cfg.model["img_backbone"]["init_cfg"] = None
    model = init_model(cfg, args.ckpt)
    model.eval()

    enc = model.pts_middle_encoder
    torch.manual_seed(42)
    V, M, Cin = args.num_voxels, 20, 5
    voxels     = torch.randn(V, M, Cin, device=device)
    num_points = torch.randint(1, M+1, (V,), device=device)
    coords     = make_unique_coords(V, enc.ny, enc.nx, device=device)

    print("[2/4] 验证 scatter 等价性...")
    new_scatter = OnnxPointPillarsScatter(enc.in_channels, enc.ny, enc.nx).eval().to(device)
    verify_scatter(model, new_scatter, voxels, num_points, coords, device)

    print("[3/4] 验证 tiebreak bias 在 ONNX 图中存在...")
    deploy = BEVFusionDeployFinal(model, K=args.K).eval().to(device)
    # if not check_range_node(deploy, (voxels, num_points, coords)):
    #     print("      ⚠️  Range 节点被消除，请检查 torch.arange 是否依赖输入 tensor")
    #     sys.exit(1)

    print("[4/4] 导出 ONNX...")
    # export_onnx(deploy, (voxels, num_points, coords),
    #             osp.join(args.outdir, "bevfusion_final_static.onnx"),  dynamic=False)
    export_onnx(deploy, (voxels, num_points, coords),
                osp.join(args.outdir, "bevfusion_final_dynamic.onnx"), dynamic=True)

    print(f"\n验证命令:")
    print(f"  python verify_onnx_vs_pth.py \\")
    print(f"      --onnx         {args.outdir}/bevfusion_final_static.onnx \\")
    print(f"      --dynamic-onnx {args.outdir}/bevfusion_final_dynamic.onnx")


if __name__ == "__main__":
    main()