"""
严格对齐 transfusion_head.py 的 forward_single 推理逻辑

原始代码与当前导出脚本的三处关键差异：

差异1：NMS 类别特殊处理（nuScenes）
  原始：
    local_max[:, 8] = F.max_pool2d(heatmap[:, 8], kernel_size=1, ...)
    local_max[:, 9] = F.max_pool2d(heatmap[:, 9], kernel_size=1, ...)
  含义：类别 8（Pedestrian）和 9（Traffic_cone）使用 kernel_size=1，
        相当于不做 NMS，每个位置都保留自身最大值，keep 恒为 True。
  当前导出：没有这个特殊处理，导致类别 8/9 的候选被错误抑制。

差异2：heatmap 使用 detach().sigmoid()，dense_heatmap 保持原始值
  原始：
    dense_heatmap = self.heatmap_head(fusion_feat.float())
    heatmap = dense_heatmap.detach().sigmoid()
  当前：dm 和 hm 的计算方式一致，但 detach() 在导出时无意义（无梯度）。
  这不影响数值，但 .float() 可能影响精度（如果 autocast 开启）。

差异3：query_pos 更新使用 detach().clone()
  原始：query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)
  当前：qp = res["center"].permute(0, 2, 1)
  detach() 在推理时无影响，但 clone() 会产生独立 buffer，
  可能影响 ATC 的内存复用优化。
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


class OnnxPointPillarsScatter(nn.Module):
    def __init__(self, in_channels, ny, nx):
        super().__init__()
        self.C, self.ny, self.nx = in_channels, ny, nx

    def forward(self, voxel_features, coors):
        C, ny, nx = self.C, self.ny, self.nx
        HW = ny * nx
        indices = (coors[:, 2].long() * nx
                   + coors[:, 3].long()).to(torch.int32)
        canvas = torch.zeros(C, HW, dtype=voxel_features.dtype,
                             device=voxel_features.device)
        canvas = canvas.scatter(
            1, indices.unsqueeze(0).expand(C, -1), voxel_features.t())
        return canvas.view(C, 1, ny, nx).permute(1, 0, 2, 3).contiguous()


class BEVFusionDeployFinal(nn.Module):
    """
    严格对齐 transfusion_head.py forward_single 的推理逻辑。
    """
    TIEBREAK_EPS = 1e-6

    def __init__(self, model, K=100, dataset='nuScenes'):
        super().__init__()
        self.pts_voxel_encoder  = model.pts_voxel_encoder
        self.pts_backbone       = model.pts_backbone
        self.pts_neck           = model.pts_neck
        self.head               = model.bbox_head
        self.K                  = K
        self.dataset            = dataset
        orig = model.pts_middle_encoder
        self.pts_middle_encoder = OnnxPointPillarsScatter(
            orig.in_channels, orig.ny, orig.nx)

    def forward(self, voxels, num_points, coords):
        # ── VoxelEncoder + 补零 mask ──────────────────────────────────────
        vf   = self.pts_voxel_encoder(voxels, num_points, coords)
        mask = (coords.sum(dim=1) != 0).float().unsqueeze(1)
        vf   = vf * mask

        # ── Scatter → BEV ─────────────────────────────────────────────────
        bev  = self.pts_middle_encoder(vf, coords)
        neck = self.pts_neck(list(self.pts_backbone(bev)))[0]

        head        = self.head
        batch_size  = neck.shape[0]
        fusion_feat = head.shared_conv(neck)

        fusion_feat_flatten = fusion_feat.view(
            batch_size, fusion_feat.shape[1], -1)          # [B, C, HW]
        bev_pos = head.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)

        # ── Heatmap（对齐原始：float()，与 autocast 一致）─────────────────
        dense_heatmap = head.heatmap_head(fusion_feat.float())
        heatmap       = dense_heatmap.sigmoid()             # detach() 推理无影响

        # ── NMS（严格对齐原始逻辑）────────────────────────────────────────
        padding   = head.nms_kernel_size // 2
        local_max = F.max_pool2d(
            heatmap, kernel_size=head.nms_kernel_size, stride=1, padding=padding)

        # 关键：nuScenes 对类别 8（Pedestrian）和 9（Traffic_cone）
        # 使用 kernel_size=1，即不做 NMS，每个位置都是自身极大值
        if self.dataset == 'nuScenes':
            local_max_cls8 = F.max_pool2d(
                heatmap[:, 8:9], kernel_size=1, stride=1, padding=0)
            local_max_cls9 = F.max_pool2d(
                heatmap[:, 9:10], kernel_size=1, stride=1, padding=0)
            # 用 cat 替代 in-place 赋值（ATC 不支持 in-place）
            local_max = torch.cat([
                local_max[:, :8],    # 类别 0-7：正常 NMS
                local_max_cls8,      # 类别 8：kernel=1（保留全部）
                local_max_cls9,      # 类别 9：kernel=1（保留全部）
            ], dim=1)
        elif self.dataset == 'Waymo':
            local_max_cls1 = F.max_pool2d(
                heatmap[:, 1:2], kernel_size=1, stride=1, padding=0)
            local_max_cls2 = F.max_pool2d(
                heatmap[:, 2:3], kernel_size=1, stride=1, padding=0)
            local_max = torch.cat([
                local_max[:, :1],
                local_max_cls1,
                local_max_cls2,
                local_max[:, 3:],
            ], dim=1)

        # keep 比较：加容差避免浮点误差导致极大值位置判断失败
        # hm 与 lm 的误差约 8e-4，容差 1e-3 足以覆盖
        heatmap = heatmap * ((heatmap + 1e-3) >= local_max).float()
        heatmap = heatmap.view(batch_size, head.num_classes, -1)  # [B, C, HW]

        # ── TopK（对齐原始：直接 topk，无 tiebreak）───────────────────────
        # 原始代码使用 topk，无 tiebreak bias
        # 这里保留 tiebreak 以防并列，但幅度极小（1e-6）
        scores_flat = heatmap.view(batch_size, -1)                # [B, C*HW]
        # n    = scores_flat.shape[-1]
        # bias = (torch.arange(n, dtype=scores_flat.dtype,
        #                      device=scores_flat.device)
        #         .unsqueeze(0) * self.TIEBREAK_EPS)
        # scores_flat = scores_flat + bias

        _, top_proposals = torch.topk(scores_flat, k=self.K, dim=-1)  # [B, K]

        # ── top_cls / top_idx（全 float 路径，避免整数算子问题）───────────
        HW        = heatmap.shape[-1]                              # 32400
        top_proposals_class = top_proposals // HW
        top_proposals_index = top_proposals % HW
        # top_f     = top_proposals.float()
        # top_cls_f = torch.floor(top_f / float(HW))
        # top_idx_f = top_f - top_cls_f * float(HW)
        # top_proposals_class = top_cls_f.long()                     # [B, K]
        # top_proposals_index = top_idx_f.long()                     # [B, K]

        # ── Gather query feature（对齐原始）──────────────────────────────
        query_feat = fusion_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, fusion_feat_flatten.shape[1], -1),
            dim=-1,
        )

        # class encoding
        one_hot = F.one_hot(
            top_proposals_class, num_classes=head.num_classes).permute(0, 2, 1)
        query_cat_encoding = head.class_encoding(one_hot.float())
        query_feat = query_feat + query_cat_encoding

        # query position
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        # ── Transformer Decoder（对齐原始）────────────────────────────────
        ret_dicts = []
        for i in range(head.num_decoder_layers):
            query_feat = head.decoder[i](
                query_feat,
                key=fusion_feat_flatten,
                query_pos=query_pos,
                key_pos=bev_pos,
            )
            res_layer = head.prediction_heads[i](query_feat)
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
            ret_dicts.append(res_layer)
            # 对齐原始：clone() 产生独立 buffer
            query_pos = res_layer['center'].clone().permute(0, 2, 1)

        # ── query_heatmap_score（对齐原始）────────────────────────────────
        # 原始：heatmap.gather(index=top_proposals_index[:, None, :].expand(...))
        query_heatmap_score = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, head.num_classes, -1),
            dim=-1,
        )

        return (
            dense_heatmap,                                  # [B, 10, H, W]
            top_proposals_class,            # [B, K]
            query_heatmap_score,                            # [B, 10, K]
            ret_dicts[-1]['heatmap'],                       # [B, 10, K]
            ret_dicts[-1]['center'],                        # [B, 2, K]
            ret_dicts[-1]['height'],                        # [B, 1, K]
            ret_dicts[-1]['dim'],                           # [B, 3, K]
            ret_dicts[-1]['rot'],                           # [B, 2, K]
            ret_dicts[-1].get(
                'vel', torch.zeros(batch_size, 2, self.K,
                                   device=voxels.device)),  # [B, 2, K]
            top_proposals_index,            # [B, K]
        )


# ════════════════════════════════════════════════════════════════════════════
#  导出
# ════════════════════════════════════════════════════════════════════════════

def make_inputs(num_voxels, max_voxels, ny, nx, M=20, Cin=5, device="cuda"):
    torch.manual_seed(42)
    vr = torch.randn(num_voxels, M, Cin, device=device)
    nr = torch.randint(1, M+1, (num_voxels,), device=device)
    fi = torch.randperm(ny * nx, device=device)[:num_voxels]
    cr = torch.zeros(num_voxels, 4, device=device)
    cr[:, 2] = (fi // nx).float()
    cr[:, 3] = (fi %  nx).float()
    cr = cr[cr[:, 0].argsort()]
    pad = max_voxels - num_voxels
    v = torch.cat([vr, torch.zeros(pad, M, Cin, device=device)])
    n = torch.cat([nr, torch.zeros(pad, dtype=torch.long, device=device)])
    c = torch.cat([cr, torch.zeros(pad, 4, device=device)])
    return v, n, c


INPUT_NAMES  = ["voxels", "num_points", "coords"]
OUTPUT_NAMES = ["dense_heatmap", "top_cls", "query_heatmap_score",
                "heatmap_q", "center", "height", "dim", "rot", "vel",
                "top_idx"]


def export_onnx(deploy, inputs, path, dynamic=False):
    dynamic_axes = ({"voxels": {0: "V"}, "num_points": {0: "V"}, "coords": {0: "V"}}
                    if dynamic else None)
    with torch.no_grad():
        torch.onnx.export(
            deploy, inputs, path,
            opset_version=11,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False,
        )
    # try:
    #     import onnx
    #     from collections import Counter
    #     g   = onnx.load(path)
    #     ops = Counter(nd.op_type for nd in g.graph.node)
    #     cmp = ops.get("Greater",0) + ops.get("GreaterOrEqual",0) + ops.get("Less",0)
    #     print(f"  {'动态' if dynamic else '静态'}: {osp.basename(path)}")
    #     print(f"    比较算子: {cmp}  Clip: {ops.get('Clip',0)}")
    # except ImportError:
    #     print(f"  ✅ {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
    p.add_argument("--ckpt",        default="work_dirs/bevfusion/epoch_20.pth")
    p.add_argument("--outdir",      default="models/onnx_aligned")
    p.add_argument("--K",           type=int, default=200)
    p.add_argument("--num-voxels",  type=int, default=6000)
    p.add_argument("--max-voxels",  type=int, default=10000)
    p.add_argument("--dataset",     default="nuScenes",
                   choices=["nuScenes", "Waymo"])
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda"

    print("[1/3] 加载模型...")
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    if "img_backbone" in cfg.model and "init_cfg" in cfg.model.get("img_backbone", {}):
        cfg.model["img_backbone"]["init_cfg"] = None
    model = init_model(cfg, args.ckpt)
    model.eval()

    enc = model.pts_middle_encoder
    M   = 32
    voxels, num_points, coords = make_inputs(
        args.num_voxels, args.max_voxels, enc.ny, enc.nx, M=M, device=device)

    print("\n[2/3] 导出 ONNX...")
    deploy = BEVFusionDeployFinal(
        model, K=args.K, dataset=args.dataset).eval().to(device)

    export_onnx(deploy, (voxels, num_points, coords),
                osp.join(args.outdir, "bevfusion_aligned_static.onnx"),
                dynamic=False)
    export_onnx(deploy, (voxels, num_points, coords),
                osp.join(args.outdir, "bevfusion_aligned_dynamic.onnx"),
                dynamic=True)

    print("\n[3/3] 保存测试 bin...")
    bindir = osp.join(args.outdir, "test_bins")
    os.makedirs(bindir, exist_ok=True)
    voxels.cpu().numpy().tofile(osp.join(bindir, "voxels.bin"))
    num_points.cpu().numpy().astype(np.int32).tofile(
        osp.join(bindir, "num_points.bin"))
    coords.cpu().numpy().tofile(osp.join(bindir, "coords.bin"))

    V = args.max_voxels
    print(f"""
ATC 命令:
  atc --model="{args.outdir}/bevfusion_aligned_dynamic.onnx" \\
      --framework=5 \\
      --output="models/om/bevfusion_aligned" \\
      --input_format=ND \\
      --input_shape="voxels:-1,{M},5;num_points:-1;coords:-1,4" \\
      --dynamic_dims="4000,4000,4000;5000,5000,5000;6000,6000,6000" \\
      --soc_version=Ascend310P1 \\
      --op_select_implmode=high_precision \\
      --precision_mode=force_fp32 \\
      --log=warning
""")


if __name__ == "__main__":
    main()