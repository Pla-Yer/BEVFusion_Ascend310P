"""
BEVFusion LiDAR + Camera 融合导出脚本

导出两个独立的ONNX模型：
1. LiDAR特征提取网络：VoxelEncoder -> Scatter -> Backbone -> Neck
   输入: voxels, num_points, coords
   输出: lidar_bev_feat [B, 256, 180, 180]

2. Fusion + Head网络：ConvFuser -> Backbone -> Neck -> TransFusionHead
   输入: fused_feat [B, 256, 180, 180]
   输出: 检测结果

这样设计的原因：
- 雷达分支和相机分支可以并行执行
- 相机BEV特征在CPU上计算后，与雷达BEV特征融合
- 便于调试和优化
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
#  Part 1: LiDAR特征提取网络
# ════════════════════════════════════════════════════════════════════════════

class OnnxPointPillarsScatter(nn.Module):
    """ONNX兼容的PointPillarsScatter"""
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


class LiDARFeatureExtractor(nn.Module):
    """
    LiDAR特征提取网络
    
    输入:
        - voxels: [V, M, C] 体素特征 (M=32, C=5)
        - num_points: [V] 每个体素的点数
        - coords: [V, 4] 体素坐标 (batch_id, z, y, x)
    
    输出:
        - lidar_bev_feat: [B, 256, 180, 180] LiDAR BEV特征
    """
    def __init__(self, model):
        super().__init__()
        self.pts_voxel_encoder = model.pts_voxel_encoder
        self.pts_backbone = model.pts_backbone
        self.pts_neck = model.pts_neck
        
        # 替换为ONNX兼容的Scatter
        orig = model.pts_middle_encoder
        self.pts_middle_encoder = OnnxPointPillarsScatter(
            orig.in_channels, orig.ny, orig.nx)
    
    def forward(self, voxels, num_points, coords):
        # VoxelEncoder
        vf = self.pts_voxel_encoder(voxels, num_points, coords)
        
        # 补零mask
        mask = (coords.sum(dim=1) != 0).float().unsqueeze(1)
        vf = vf * mask
        
        # Scatter -> BEV
        bev = self.pts_middle_encoder(vf, coords)
        
        
        return bev  # [B, 256, 180, 180]


# ════════════════════════════════════════════════════════════════════════════
#  Part 2: Fusion + Head网络
# ════════════════════════════════════════════════════════════════════════════


class FusionHeadNetwork(nn.Module):
    """
    Fusion + Head网络
    
    输入:
        - fused_feat: [B, 256, 180, 180] 融合后的BEV特征
    
    输出:
        - dense_heatmap: [B, 10, 90, 90] 原始热力图
        - top_cls: [B, K] 候选类别
        - query_heatmap_score: [B, 10, K] 热力图分数
        - heatmap_q: [B, 10, K] decoder热力图
        - center: [B, 2, K] 中心坐标
        - height: [B, 1, K] 高度
        - dim: [B, 3, K] 尺寸
        - rot: [B, 2, K] 旋转
        - vel: [B, 2, K] 速度
        - top_idx: [B, K] 候选索引
    """
    TIEBREAK_EPS = 1e-6

    def __init__(self, model, K=200, dataset='nuScenes'):
        super().__init__()
        
        # Fusion layer (ConvFuser)
        self.fusion_layer = model.fusion_layer
        
        # Backbone (SECOND)
        self.pts_backbone = model.pts_backbone
        
        # Neck (SECONDFPN)
        self.pts_neck = model.pts_neck
        
        # Detection Head (TransFusionHead)
        self.head = model.bbox_head
        
        self.K = K
        self.dataset = dataset

    def forward(self, pts_bev, img_bev):
        """
        Args:
            pts_bev: [B, 256, 180, 180] LiDAR BEV特征
            img_bev: [B, 80, 180, 180] 图像 BEV特征

        Returns:
            检测结果
        """
        fused_feat = self.fusion_layer([img_bev, pts_bev])
       
        head = self.head
        batch_size = fused_feat.shape[0]
        
        # Backbone
        backbone_feats = self.pts_backbone(fused_feat)
        # Neck
        neck = self.pts_neck(list(backbone_feats))[0]
        
        # Shared conv
        fusion_feat = head.shared_conv(neck)
        
        fusion_feat_flatten = fusion_feat.view(
            batch_size, fusion_feat.shape[1], -1)  # [B, C, HW]
        bev_pos = head.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)
        
        # Heatmap
        dense_heatmap = head.heatmap_head(fusion_feat.float())
        heatmap = dense_heatmap.sigmoid()
        
        # NMS
        padding = head.nms_kernel_size // 2
        local_max = F.max_pool2d(
            heatmap, kernel_size=head.nms_kernel_size, stride=1, padding=padding)
        
        # nuScenes特殊处理：类别8和9不做NMS
        if self.dataset == 'nuScenes':
            local_max_cls8 = F.max_pool2d(
                heatmap[:, 8:9], kernel_size=1, stride=1, padding=0)
            local_max_cls9 = F.max_pool2d(
                heatmap[:, 9:10], kernel_size=1, stride=1, padding=0)
            local_max = torch.cat([
                local_max[:, :8],
                local_max_cls8,
                local_max_cls9,
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
        
        # Keep mask
        heatmap = heatmap * ((heatmap + 1e-3) >= local_max).float()
        heatmap = heatmap.view(batch_size, head.num_classes, -1)  # [B, C, HW]
        
        # TopK
        scores_flat = heatmap.view(batch_size, -1)  # [B, C*HW]
        _, top_proposals = torch.topk(scores_flat, k=self.K, dim=-1)  # [B, K]
        
        # top_cls / top_idx
        HW = heatmap.shape[-1]
        top_proposals_class = top_proposals // HW
        top_proposals_index = top_proposals % HW
        
        # Gather query feature
        query_feat = fusion_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, fusion_feat_flatten.shape[1], -1),
            dim=-1,
        )
        
        # Class encoding
        one_hot = F.one_hot(
            top_proposals_class, num_classes=head.num_classes).permute(0, 2, 1)
        query_cat_encoding = head.class_encoding(one_hot.float())
        query_feat = query_feat + query_cat_encoding
        
        # Query position
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        
        # Transformer Decoder
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
            query_pos = res_layer['center'].clone().permute(0, 2, 1)
        
        # query_heatmap_score
        query_heatmap_score = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, head.num_classes, -1),
            dim=-1,
        )
        
        return (
            dense_heatmap,  # [B, 10, 90, 90]
            top_proposals_class,  # [B, K]
            query_heatmap_score,  # [B, 10, K]
            ret_dicts[-1]['heatmap'],  # [B, 10, K]
            ret_dicts[-1]['center'],  # [B, 2, K]
            ret_dicts[-1]['height'],  # [B, 1, K]
            ret_dicts[-1]['dim'],  # [B, 3, K]
            ret_dicts[-1]['rot'],  # [B, 2, K]
            ret_dicts[-1].get(
                'vel', torch.zeros(batch_size, 2, self.K,
                                   device=fused_feat.device)),  # [B, 2, K]
            top_proposals_index,  # [B, K]
        )


# ════════════════════════════════════════════════════════════════════════════
#  导出工具函数
# ════════════════════════════════════════════════════════════════════════════

def make_lidar_inputs(num_voxels, max_voxels, ny, nx, M=32, Cin=5, device="cuda"):
    """生成LiDAR输入数据"""
    torch.manual_seed(42)
    vr = torch.randn(num_voxels, M, Cin, device=device)
    nr = torch.randint(1, M+1, (num_voxels,), device=device)
    fi = torch.randperm(ny * nx, device=device)[:num_voxels]
    cr = torch.zeros(num_voxels, 4, device=device)
    cr[:, 2] = (fi // nx).float()
    cr[:, 3] = (fi % nx).float()
    cr = cr[cr[:, 0].argsort()]
    pad = max_voxels - num_voxels
    v = torch.cat([vr, torch.zeros(pad, M, Cin, device=device)])
    n = torch.cat([nr, torch.zeros(pad, dtype=torch.long, device=device)])
    c = torch.cat([cr, torch.zeros(pad, 4, device=device)])
    return v, n, c


def make_fusion_inputs(batch_size=1, device="cuda"):
    """生成Fusion输入数据"""
    torch.manual_seed(42)
    pts_bev = torch.randn(batch_size, 256, 360, 360, device=device)
    img_bev = torch.randn(batch_size, 80, 360, 360, device=device)
    return pts_bev, img_bev


def export_lidar_onnx(deploy, inputs, path, dynamic=False):
    """导出LiDAR特征提取ONNX"""
    input_names = ["voxels", "num_points", "coords"]
    output_names = ["lidar_bev_feat"]
    
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "voxels": {0: "V"},
            "num_points": {0: "V"},
            "coords": {0: "V"},
            "lidar_bev_feat": {0: "B"}
        }
    
    with torch.no_grad():
        torch.onnx.export(
            deploy, inputs, path,
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


def export_fusion_head_onnx(deploy, inputs, path, dynamic=False):
    """导出Fusion+Head ONNX"""
    input_names = ["pts_bev", "img_bev"]
    output_names = [
        "dense_heatmap", "top_cls", "query_heatmap_score",
        "heatmap_q", "center", "height", "dim", "rot", "vel", "top_idx"
    ]
    
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "fused_feat": {0: "B"},
            "dense_heatmap": {0: "B"},
            "top_cls": {0: "B"},
            "query_heatmap_score": {0: "B"},
            "heatmap_q": {0: "B"},
            "center": {0: "B"},
            "height": {0: "B"},
            "dim": {0: "B"},
            "rot": {0: "B"},
            "vel": {0: "B"},
            "top_idx": {0: "B"},
        }
    
    with torch.no_grad():
        torch.onnx.export(
            deploy, inputs, path,
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


# ════════════════════════════════════════════════════════════════════════════
#  主函数
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/configs/bevfusion_lidar-cam_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d_resnet50.py")
    p.add_argument("--ckpt", default="work_dirs/bevfusion_lidar-cam_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d_resnet50/epoch_6.pth")
    p.add_argument("--outdir", default="models/onnx_lidar_fusion_head")
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--num-voxels", type=int, default=6000)
    p.add_argument("--max-voxels", type=int, default=10000)
    p.add_argument("--dataset", default="nuScenes",
                   choices=["nuScenes", "Waymo"])
    p.add_argument("--stage", type=str, default="all",
                   choices=["lidar", "fusion_head", "all"],
                   help="导出哪个阶段")
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
    M = 32

    # ═══════════════════════════════════════════════════════════════════════
    # 导出 Part 1: LiDAR特征提取网络
    # ═══════════════════════════════════════════════════════════════════════
    if args.stage in ["lidar", "all"]:
        print("\n[2/4] 导出 LiDAR特征提取网络...")
        
        # 生成输入数据
        voxels, num_points, coords = make_lidar_inputs(
            args.num_voxels, args.max_voxels, enc.ny, enc.nx, M=M, device=device)
        
        # 创建部署模型
        lidar_deploy = LiDARFeatureExtractor(model).eval().to(device)
        
        # 导出静态shape ONNX
        # export_lidar_onnx(
        #     lidar_deploy, (voxels, num_points, coords),
        #     osp.join(args.outdir, "lidar_feature_extractor_static.onnx"),
        #     dynamic=False
        # )
        
        # 导出动态shape ONNX
        export_lidar_onnx(
            lidar_deploy, (voxels, num_points, coords),
            osp.join(args.outdir, "lidar_feature_extractor_dynamic.onnx"),
            dynamic=True
        )
        
        # 保存测试数据
        bindir = osp.join(args.outdir, "test_bins_lidar")
        os.makedirs(bindir, exist_ok=True)
        voxels.cpu().numpy().tofile(osp.join(bindir, "voxels.bin"))
        num_points.cpu().numpy().astype(np.int32).tofile(
            osp.join(bindir, "num_points.bin"))
        coords.cpu().numpy().tofile(osp.join(bindir, "coords.bin"))
        
        print(f"  ✓ LiDAR特征提取网络已导出")
        print(f"  ✓ 测试数据已保存到 {bindir}")
    else:
        print("\n[2/4] 跳过 LiDAR特征提取网络导出")

    # ═══════════════════════════════════════════════════════════════════════
    # 导出 Part 2: Fusion + Head网络
    # ═══════════════════════════════════════════════════════════════════════
    if args.stage in ["fusion_head", "all"]:
        print("\n[3/4] 导出 Fusion+Head网络...")
        
        # 生成输入数据
        pts_bev, img_bev = make_fusion_inputs(batch_size=1, device=device)
        
        # 创建部署模型
        fusion_head_deploy = FusionHeadNetwork(
            model, K=args.K, dataset=args.dataset).eval().to(device)
        
        # 导出静态shape ONNX
        export_fusion_head_onnx(
            fusion_head_deploy, (pts_bev, img_bev),
            osp.join(args.outdir, "fusion_head_static.onnx"),
            dynamic=False
        )
        
        # # 导出动态shape ONNX
        # export_fusion_head_onnx(
        #     fusion_head_deploy, (fused_feat,),
        #     osp.join(args.outdir, "fusion_head_dynamic.onnx"),
        #     dynamic=True
        # )
        
        # 保存测试数据
        bindir = osp.join(args.outdir, "test_bins_fusion_head")
        os.makedirs(bindir, exist_ok=True)
        pts_bev.cpu().numpy().tofile(osp.join(bindir, "pts_bev.bin"))
        img_bev.cpu().numpy().tofile(osp.join(bindir, "img_bev.bin"))

        print(f"  ✓ Fusion+Head网络已导出")
        print(f"  ✓ 测试数据已保存到 {bindir}")
    else:
        print("\n[3/4] 跳过 Fusion+Head网络导出")

    # ═══════════════════════════════════════════════════════════════════════
    # 打印ATC命令
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[4/4] ATC转换命令:")
    
    if args.stage in ["lidar", "all"]:
        print(f"""
Part 1: LiDAR特征提取网络
────────────────────────────────────────────────────────────────────────────
atc --model="{args.outdir}/lidar_feature_extractor_dynamic.onnx" \\
    --framework=5 \\
    --output="models/om/lidar_feature_extractor" \\
    --input_format=ND \\
    --input_shape="voxels:-1,{M},5;num_points:-1;coords:-1,4" \\
    --dynamic_dims="6000,6000,6000;8000,8000,8000;10000,10000,10000" \\
    --soc_version=Ascend310P1 \\
    --op_select_implmode=high_precision \\
    --precision_mode=force_fp32 \\
    --log=warning
""")
    
    if args.stage in ["fusion_head", "all"]:
        print(f"""
Part 2: Fusion+Head网络
────────────────────────────────────────────────────────────────────────────
atc --model="{args.outdir}/fusion_head_dynamic.onnx" \\
    --framework=5 \\
    --output="models/om/fusion_head" \\
    --input_format=NCHW \\
    --input_shape="pts_bev:1,256,360,360;img_bev:1,80,360,360" \\
    --soc_version=Ascend310P1 \\
    --op_select_implmode=high_precision \\
    --precision_mode=force_fp32 \\
    --log=warning
""")

    print("""
═══════════════════════════════════════════════════════════════════════════
导出完成！
═══════════════════════════════════════════════════════════════════════════

使用流程:
1. 运行LiDAR特征提取网络，得到 lidar_bev_feat [B, 256, 180, 180]
2. 运行相机分支（Backbone -> Neck -> DepthNet），得到 cam_feats
3. CPU计算：深度图生成 -> 几何特征计算 -> BEV Pool，得到 camera_bev_feat [B, 80, 360, 360]
4. 下采样 camera_bev_feat 到 [B, 80, 180, 180]
5. Concat: [lidar_bev_feat, camera_bev_feat] -> [B, 336, 180, 180]
6. Conv: [B, 336, 180, 180] -> [B, 256, 180, 180] (fused_feat)
7. 运行Fusion+Head网络，得到检测结果

注意：
- 步骤1和步骤2可以并行执行
- 步骤3-6在CPU上执行
- 步骤7在NPU上执行
""")


if __name__ == "__main__":
    main()
