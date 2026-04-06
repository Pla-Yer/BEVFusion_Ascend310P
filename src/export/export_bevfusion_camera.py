"""
BEVFusion 图像分支导出脚本

参考雷达分支的导出方式，导出图像分支的三个部分：
1. Backbone (ResNet50)
2. Neck (GeneralizedLSSFPN)
3. ViewTransform (DepthLSSTransform)

导出策略：
- 将可以在CPU上计算的部分（几何计算、深度图生成）不导出到ONNX
- 提供分阶段导出和完整导出两种方式

阶段划分：
- Stage 1: img_backbone (ResNet50)
- Stage 2: img_neck (GeneralizedLSSFPN)
- Stage 3: view_transform_depthnet (dtransform + depthnet)
- Stage 4: view_transform_bevpool (bev_pool_scatter)
- Full: 完整图像分支（Backbone + Neck + ViewTransform核心部分）
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
#  Stage 1: Backbone (ResNet50)
# ════════════════════════════════════════════════════════════════════════════
class Stage1_Backbone(nn.Module):
    """
    输入:  imgs_bn = [B*N, 3, H, W]  (N=6视角时，B=1则 BN=6)
    输出:  img_feat_bn = [B*N, 256, 32, 88]  (对齐 view_transform.in_channels=256)
    """
    def __init__(self, model):
        super().__init__()
        self.backbone = model.img_backbone
        self.neck = model.img_neck

    def forward(self, imgs_bn):
        feats = self.backbone(imgs_bn)   # tuple/list 多尺度   
        outs = self.neck(feats)          # 一般是 list
        # GeneralizedLSSFPN 在 BEVFusion 里通常输出一个 level：[BN,256,32,88]
        img_feat = outs[0]
        
        return img_feat

# ════════════════════════════════════════════════════════════════════════════
#  Stage 2: Neck (GeneralizedLSSFPN)
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
#  Stage 3: ViewTransform - DepthNet部分
# ════════════════════════════════════════════════════════════════════════════
class Stage3_DepthNet(nn.Module):
    """ViewTransform的DepthNet部分
    
    输入:
        - img_feat: [B*N, C, fH, fW]  (C=256, fH=32, fW=88)
        - depth: [B*N, 1, iH, iW]  (iH=256, iW=704)
    输出:
        - cam_feats: [B, N, C, D, fH, fW]  (D=118 depth bins)
    
    说明:
        - dtransform: 处理深度图 [B*N, 1, iH, iW] -> [B*N, 64, fH, fW]
        - depthnet: 预测深度和特征 [B*N, 256+64, fH, fW] -> [B*N, D+C, fH, fW]
        - softmax + outer product: 生成cam_feats
    """
    def __init__(self, model):
        super().__init__()
        vt = model.view_transform
        self.dtransform = vt.dtransform
        self.depthnet = vt.depthnet
        self.D = vt.D  # depth bins
        self.C = vt.C  # out_channels
    
    def forward(self, img_feat, depth):
        B, N, C, fH, fW = img_feat.shape
        
        # 处理深度图
        d = depth.view(B * N, *depth.shape[2:])  # [B*N, 1, iH, iW]
        d = self.dtransform(d)  # [B*N, 64, fH, fW]
        
        # 拼接特征
        x = img_feat.view(B * N, C, fH, fW)
        x = torch.cat([d, x], dim=1)  # [B*N, 320, fH, fW]
        
        # DepthNet预测
        x = self.depthnet(x)  # [B*N, D+C, fH, fW]
        
        # 提取深度和特征
        depth_logits = x[:, :self.D]  # [B*N, D, fH, fW]
        feat = x[:, self.D:(self.D + self.C)]  # [B*N, C, fH, fW]
        
        # Softmax + outer product
        depth_prob = depth_logits.softmax(dim=1)  # [B*N, D, fH, fW]
        cam_feats = depth_prob.unsqueeze(1) * feat.unsqueeze(2)  # [B*N, C, D, fH, fW]
        
        # Reshape to [B, N, C, D, fH, fW]
        cam_feats = cam_feats.view(B, N, self.C, self.D, fH, fW)
        
        return cam_feats


# ════════════════════════════════════════════════════════════════════════════
#  Stage 4: ViewTransform - BEV Pool部分
# ════════════════════════════════════════════════════════════════════════════
class Stage4_BEVPool(nn.Module):
    """ViewTransform的BEV Pool部分
    
    输入:
        - cam_feats: [B, N, C, D, fH, fW]
        - geom_feats: [B, N, D, fH, fW, 3]  (lidar坐标系下的几何坐标)
    输出:
        - bev_feat: [B, C*nx2, nx0, nx1]
    
    说明:
        - geom_feats需要在CPU上计算（通过get_geometry）
        - bev_pool_scatter将特征投影到BEV空间
    """
    def __init__(self, model):
        super().__init__()
        vt = model.view_transform
        # 将参数转换为常量tensor，避免device问题
        self.register_buffer('bx', vt.bx.detach().clone())
        self.register_buffer('dx', vt.dx.detach().clone())
        self.register_buffer('nx', vt.nx.detach().clone())
        self.downsample = vt.downsample
    
    def forward(self, cam_feats, geom_feats):
        # bev_pool_scatter
        # cam_feats: [B, N, C, D, H, W]
        # geom_feats: [B, N, D, H, W, 3]
        B, N, C, D, H, W = cam_feats.shape
        
        # Permute cam_feats to [B, N, D, H, W, C]
        cam_feats = cam_feats.permute(0, 1, 3, 4, 5, 2).contiguous()
        
        # Flatten
        feats = cam_feats.reshape(-1, C)  # [Nprime, C]
        coords = geom_feats.reshape(-1, 3)  # [Nprime, 3]
        
        # Grid index
        idx = ((coords - (self.bx - self.dx / 2.0)) / self.dx).to(torch.int64)
        x_id = idx[:, 0]
        y_id = idx[:, 1]
        z_id = idx[:, 2]
        
        # Batch id
        Nprime = B * N * D * H * W
        pts_per_batch = Nprime // B
        b_id = torch.arange(B, device=feats.device, dtype=torch.int64).repeat_interleave(pts_per_batch)
        
        # Valid mask
        nx0 = int(self.nx[0].item())
        nx1 = int(self.nx[1].item())
        nx2 = int(self.nx[2].item())
        valid = (x_id >= 0) & (x_id < nx0) & (y_id >= 0) & (y_id < nx1) & (z_id >= 0) & (z_id < nx2)
        
        # Mask invalid points
        feats = feats * valid.to(feats.dtype).unsqueeze(1)
        
        # Clamp to avoid out-of-bounds
        x_id = x_id.clamp(0, nx0 - 1)
        y_id = y_id.clamp(0, nx1 - 1)
        z_id = z_id.clamp(0, nx2 - 1)
        
        # Linear index
        stride_x = int(nx1)
        stride_z = int(nx0 * nx1)
        stride_b = int(nx2 * nx0 * nx1)
        lin = b_id * stride_b + z_id * stride_z + x_id * stride_x + y_id
        
        # Scatter reduce sum
        out_cells = B * nx2 * nx0 * nx1
        out = torch.zeros((out_cells, C), device=feats.device, dtype=feats.dtype)
        # 使用scatter_add代替index_add，更兼容ONNX
        # 注意：scatter_add在PyTorch 1.12+中已弃用，但ONNX支持
        idx_expanded = lin.unsqueeze(1).expand(-1, C)
        out = out.scatter_add(0, idx_expanded, feats)
        
        # Reshape
        out = out.view(B, nx2, nx0, nx1, C).permute(0, 4, 1, 2, 3).contiguous()
        bev_feat = out.reshape(B, C * nx2, nx0, nx1)
        
        # Downsample (Identity, skip)
        # bev_feat = self.downsample(bev_feat)
        
        return bev_feat


# ════════════════════════════════════════════════════════════════════════════
#  Full: 完整图像分支
# ════════════════════════════════════════════════════════════════════════════
class CameraBranchFull(nn.Module):
    """完整图像分支（Backbone + Neck + ViewTransform核心部分）
    
    输入:
        - img: [B, N, 3, H, W]
        - depth: [B, N, 1, iH, iW]  (从点云生成的深度图，CPU计算)
        - geom_feats: [B, N, D, fH, fW, 3]  (几何坐标，CPU计算)
    输出:
        - bev_feat: [B, C*nx2, nx0, nx1]
    
    说明:
        - depth和geom_feats需要在CPU上预计算
        - 这样可以避免在ONNX中包含复杂的几何计算
    """
    def __init__(self, model):
        super().__init__()
        self.backbone = model.img_backbone
        self.neck = model.img_neck
        vt = model.view_transform
        self.dtransform = vt.dtransform
        self.depthnet = vt.depthnet
        self.D = vt.D
        self.C = vt.C
        # 使用register_buffer避免device问题
        self.register_buffer('bx', vt.bx.detach().clone())
        self.register_buffer('dx', vt.dx.detach().clone())
        self.register_buffer('nx', vt.nx.detach().clone())
        # downsample是Identity或Sequential，需要特殊处理
        if isinstance(vt.downsample, nn.Identity):
            self.downsample = nn.Identity()
        else:
            self.downsample = vt.downsample
    
    def forward(self, img, depth, geom_feats):
        B, N, C, H, W = img.shape
        
        # Backbone
        img_flat = img.view(B * N, C, H, W)
        bb_feats = self.backbone(img_flat)  # tuple of 3
        
        # Neck
        neck_feats = self.neck(list(bb_feats))  # tuple of 2
        img_feat = neck_feats[0]  # [B*N, 256, fH, fW]
        
        # DepthNet
        fH, fW = img_feat.shape[2:]
        d = depth.view(B * N, *depth.shape[2:])
        d = self.dtransform(d)
        x = torch.cat([d, img_feat], dim=1)
        x = self.depthnet(x)
        
        depth_logits = x[:, :self.D]
        feat = x[:, self.D:(self.D + self.C)]
        depth_prob = depth_logits.softmax(dim=1)
        cam_feats = depth_prob.unsqueeze(1) * feat.unsqueeze(2)
        cam_feats = cam_feats.view(B, N, self.C, self.D, fH, fW)
        
        # BEV Pool
        bev_feat = self.bev_pool_scatter(cam_feats, geom_feats)
        bev_feat = self.downsample(bev_feat)
        
        return bev_feat
    
    def bev_pool_scatter(self, cam_feats, geom_feats):
        # cam_feats: [B, N, C, D, H, W]
        # geom_feats: [B, N, D, H, W, 3]
        B, N, C, D, H, W = cam_feats.shape
        
        # Permute cam_feats to [B, N, D, H, W, C]
        cam_feats = cam_feats.permute(0, 1, 3, 4, 5, 2).contiguous()
        
        feats = cam_feats.reshape(-1, C)
        coords = geom_feats.reshape(-1, 3)
        
        idx = ((coords - (self.bx - self.dx / 2.0)) / self.dx).to(torch.int64)
        x_id, y_id, z_id = idx[:, 0], idx[:, 1], idx[:, 2]
        
        Nprime = B * N * D * H * W
        pts_per_batch = Nprime // B
        b_id = torch.arange(B, device=feats.device, dtype=torch.int64).repeat_interleave(pts_per_batch)
        
        nx0, nx1, nx2 = int(self.nx[0].item()), int(self.nx[1].item()), int(self.nx[2].item())
        valid = (x_id >= 0) & (x_id < nx0) & (y_id >= 0) & (y_id < nx1) & (z_id >= 0) & (z_id < nx2)
        
        feats = feats * valid.to(feats.dtype).unsqueeze(1)
        x_id, y_id, z_id = x_id.clamp(0, nx0-1), y_id.clamp(0, nx1-1), z_id.clamp(0, nx2-1)
        
        stride_x, stride_z, stride_b = nx1, nx0*nx1, nx2*nx0*nx1
        lin = b_id * stride_b + z_id * stride_z + x_id * stride_x + y_id
        
        out = torch.zeros((B * nx2 * nx0 * nx1, C), device=feats.device, dtype=feats.dtype)
        idx2 = lin.view(-1, 1).expand(-1, C)
        out = out.scatter_reduce(0, idx2, feats, reduce="sum", include_self=True)
        
        out = out.view(B, nx2, nx0, nx1, C).permute(0, 4, 1, 2, 3).contiguous()
        return out.reshape(B, C * nx2, nx0, nx1)


# ════════════════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════════════════
def export_onnx(module, inputs, input_names, output_names, path, dynamic_axes=None):
    """导出ONNX模型"""
    with torch.no_grad():
        torch.onnx.export(
            module, inputs, path,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False,
        )
    print(f"  导出: {path}")


def save_bin(arr, path):
    """保存bin文件"""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr.tofile(path)
    print(f"  bin: {path}  shape={arr.shape}  dtype={arr.dtype}")


def make_camera_inputs(B=1, N=6, H=256, W=704, device="cuda"):
    """生成相机输入数据"""
    torch.manual_seed(42)
    img = torch.randn(B*N, 3, H, W, device=device)
    return img


def make_depth_inputs(B=1, N=6, iH=256, iW=704, device="cuda"):
    """生成深度图输入（模拟）"""
    torch.manual_seed(42)
    depth = torch.rand(B, N, 1, iH, iW, device=device) * 60.0  # [0, 60] meters
    return depth


def make_geom_feats(B=1, N=6, D=118, fH=32, fW=88, device="cuda"):
    """生成几何特征（模拟，实际应在CPU计算）"""
    torch.manual_seed(42)
    # 模拟lidar坐标系下的点 [x, y, z]
    # x: [-54, 54], y: [-54, 54], z: [-10, 10]
    geom = torch.zeros(B, N, D, fH, fW, 3, device=device)
    geom[..., 0] = torch.rand(B, N, D, fH, fW, device=device) * 108.0 - 54.0  # x
    geom[..., 1] = torch.rand(B, N, D, fH, fW, device=device) * 108.0 - 54.0  # y
    geom[..., 2] = torch.rand(B, N, D, fH, fW, device=device) * 20.0 - 10.0   # z
    return geom


def print_atc_cmd(onnx_path, om_path, input_shape_str, dynamic_dims_str=None, soc="Ascend310P1"):
    """打印ATC命令"""
    cmd = f"""
ATC 命令:
  atc --model="{onnx_path}" \\
      --framework=5 \\
      --output="{om_path}" \\
      --input_format=ND \\
      --input_shape="{input_shape_str}" \\
      --soc_version={soc} \\
      --precision_mode=allow_fp32_to_fp16 \\
      --log=warning
"""
    if dynamic_dims_str:
        cmd = f"""
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
"""
    print(cmd)


# ════════════════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="BEVFusion图像分支导出")
    p.add_argument("--config", default="src/configs/bevfusion_lidar-cam_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d_resnet50.py")
    p.add_argument("--ckpt", default="work_dirs/bevfusion_lidar-cam_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d_resnet50/epoch_6.pth")
    p.add_argument("--outdir", default="models/onnx_camera")
    p.add_argument("--bindir", default="camera_bins")
    p.add_argument("--stage", default="all", choices=["all", "1", "2", "3", "4", "full"])
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument("--num-cameras", type=int, default=6, help="Number of cameras")
    args = p.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.bindir, exist_ok=True)
    device = "cuda"
    
    print("[1/3] 加载模型...")
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    # if "img_backbone" in cfg.model and "init_cfg" in cfg.model.get("img_backbone", {}):
    #     cfg.model["img_backbone"]["init_cfg"] = None
    model = init_model(cfg, args.ckpt)
    model.eval()
    
    B, N = args.batch_size, args.num_cameras
    H, W = 256, 704
    fH, fW = 32, 88
    D = model.view_transform.D
    
    print(f"\n模型参数:")
    print(f"  Batch size: {B}")
    print(f"  Num cameras: {N}")
    print(f"  Image size: {H}x{W}")
    print(f"  Feature size: {fH}x{fW}")
    print(f"  Depth bins: {D}")
    
    # 生成输入数据
    print("\n[2/3] 生成输入数据...")
    img = make_camera_inputs(B, N, H, W, device)
    depth = make_depth_inputs(B, N, H, W, device)
    geom_feats = make_geom_feats(B, N, D, fH, fW, device)
    
    # 计算各阶段输出
    print("\n[3/3] 导出ONNX模型...")
    do_all = (args.stage == "all")
    
    # ── Stage 1: Backbone ───────────────────────────────────────────────
    if do_all or args.stage == "1":
        print("\n── Stage 1: Backbone (ResNet50) ──")
        s1 = Stage1_Backbone(model).eval().to(device)
        with torch.no_grad():
            bb_feats = s1(img)
        
        print(f"  输入: img {img.shape}")
        print(f"  输出: {[f.shape for f in bb_feats]}")
        
        # 保存输入输出bin
        save_bin(img, osp.join(args.bindir, "stage1_img.bin"))
        for i, f in enumerate(bb_feats):
            save_bin(f, osp.join(args.bindir, f"stage1_feat{i}.bin"))
        
        # 导出ONNX
        export_onnx(
            s1, (img,),
            ["img"], ["img_feat_bn"],
            osp.join(args.outdir, "stage1_backbone.onnx"),
        )
        
        # ATC命令
        print_atc_cmd(
            f"{args.outdir}/stage1_backbone.onnx",
            f"models/om/stage1_backbone",
            f"img:{B*N},3,{H},{W}"
        )
    
    
    # ── Stage 3: DepthNet ───────────────────────────────────────────────
    if do_all or args.stage == "3":
        print("\n── Stage 3: ViewTransform - DepthNet ──")
        s1 = Stage1_Backbone(model).eval().to(device)
        s3 = Stage3_DepthNet(model).eval().to(device)
        
        with torch.no_grad():
            bb_feats = s1(img)
            img_feat = bb_feats.view(B, N, -1, fH, fW)  # [B, N, 256, fH, fW]
            
            cam_feats = s3(img_feat, depth)
        
        print(f"  输入: img_feat {img_feat.shape}, depth {depth.shape}")
        print(f"  输出: cam_feats {cam_feats.shape}")
        
        # 保存输入输出bin
        save_bin(img_feat, osp.join(args.bindir, "stage3_img_feat.bin"))
        save_bin(depth, osp.join(args.bindir, "stage3_depth.bin"))
        save_bin(cam_feats, osp.join(args.bindir, "stage3_cam_feats.bin"))
        
        # 导出ONNX
        export_onnx(
            s3, (img_feat, depth),
            ["img_feat", "depth"], ["cam_feats"],
            osp.join(args.outdir, "stage3_depthnet.onnx"),
            dynamic_axes={"img_feat": {0: "B"}, "depth": {0: "B"}, "cam_feats": {0: "B"}}
        )
        
        # ATC命令
        print_atc_cmd(
            f"{args.outdir}/stage3_depthnet.onnx",
            f"models/om/stage3_depthnet",
            f"img_feat:{B},{N},256,{fH},{fW};depth:{B},{N},1,{H},{W}"
        )
    
    # ── Stage 4: BEV Pool ───────────────────────────────────────────────
    if do_all or args.stage == "4":
        print("\n── Stage 4: ViewTransform - BEV Pool ──")
        print("  注意: Stage 4涉及scatter操作，ONNX导出可能有兼容性问题")
        print("  建议: 在CPU上实现bev_pool_scatter，或使用自定义CUDA算子")
        print("  跳过Stage 4的ONNX导出...")
        # s1 = Stage1_Backbone(model).eval().to(device)
        # s2 = Stage2_Neck(model).eval().to(device)
        # s3 = Stage3_DepthNet(model).eval().to(device)
        # s4 = Stage4_BEVPool(model).eval().to(device)
        #
        # with torch.no_grad():
        #     bb_feats = s1(img)
        #     neck_feats = s2(*bb_feats)
        #     img_feat = neck_feats[0].view(B, N, -1, fH, fW)
        #     cam_feats = s3(img_feat, depth)
        #     bev_feat = s4(cam_feats, geom_feats)
        #
        # print(f"  输入: cam_feats {cam_feats.shape}, geom_feats {geom_feats.shape}")
        # print(f"  输出: bev_feat {bev_feat.shape}")
        #
        # # 保存输入输出bin
        # save_bin(cam_feats, osp.join(args.bindir, "stage4_cam_feats.bin"))
        # save_bin(geom_feats, osp.join(args.bindir, "stage4_geom_feats.bin"))
        # save_bin(bev_feat, osp.join(args.bindir, "stage4_bev_feat.bin"))
        #
        # # 导出ONNX
        # export_onnx(
        #     s4, (cam_feats, geom_feats),
        #     ["cam_feats", "geom_feats"], ["bev_feat"],
        #     osp.join(args.outdir, "stage4_bevpool.onnx"),
        #     dynamic_axes={"cam_feats": {0: "B"}, "geom_feats": {0: "B"}, "bev_feat": {0: "B"}}
        # )
        #
        # # ATC命令
        # print_atc_cmd(
        #     f"{args.outdir}/stage4_bevpool.onnx",
        #     f"models/om/stage4_bevpool",
        #     f"cam_feats:{B},{N},{model.view_transform.C},{D},{fH},{fW};geom_feats:{B},{N},{D},{fH},{fW},3"
        # )
    
    # ── Full: 完整图像分支 ──────────────────────────────────────────────
    if do_all or args.stage == "full":
        print("\n── Full: 完整图像分支 ──")
        print("  注意: 完整分支包含BEV Pool，ONNX导出可能有兼容性问题")
        print("  建议: 使用Stage 1-3的分阶段导出，Stage 4在CPU上实现")
        print("  跳过Full的ONNX导出...")
        # full = CameraBranchFull(model).eval().to(device)
        #
        # with torch.no_grad():
        #     bev_feat = full(img, depth, geom_feats)
        #
        # print(f"  输入: img {img.shape}, depth {depth.shape}, geom_feats {geom_feats.shape}")
        # print(f"  输出: bev_feat {bev_feat.shape}")
        #
        # # 保存输入输出bin
        # save_bin(img, osp.join(args.bindir, "full_img.bin"))
        # save_bin(depth, osp.join(args.bindir, "full_depth.bin"))
        # save_bin(geom_feats, osp.join(args.bindir, "full_geom_feats.bin"))
        # save_bin(bev_feat, osp.join(args.bindir, "full_bev_feat.bin"))
        #
        # # 导出ONNX
        # export_onnx(
        #     full, (img, depth, geom_feats),
        #     ["img", "depth", "geom_feats"], ["bev_feat"],
        #     osp.join(args.outdir, "camera_branch_full.onnx"),
        #     dynamic_axes={"img": {0: "B"}, "depth": {0: "B"}, "geom_feats": {0: "B"}, "bev_feat": {0: "B"}}
        # )
        #
        # # ATC命令
        # print_atc_cmd(
        #     f"{args.outdir}/camera_branch_full.onnx",
        #     f"models/om/camera_branch_full",
        #     f"img:{B},{N},3,{H},{W};depth:{B},{N},1,{H},{W};geom_feats:{B},{N},{D},{fH},{fW},3"
        # )
    
    print("\n" + "="*80)
    print("导出完成！")
    print("="*80)
    print("\n部署建议:")
    print("1. depth和geom_feats应在CPU上预计算:")
    print("   - depth: 从点云投影到图像平面（参考depth_lss.py的BaseDepthTransform.forward）")
    print("   - geom_feats: 通过get_geometry计算（涉及矩阵逆运算）")
    print("2. 推荐使用分阶段导出，便于调试和优化:")
    print("   - Stage 1+2: Backbone + Neck (纯卷积，稳定)")
    print("   - Stage 3: DepthNet (需要depth输入)")
    print("   - Stage 4: BEV Pool (需要geom_feats输入)")
    print("3. 完整导出适合端到端验证，但需要提前准备好depth和geom_feats")


if __name__ == "__main__":
    main()
