"""
BEVFusion 全NPU统一导出脚本
================================================================================
新流程设计：
  预计算 (CPU): depth_gen + geo_gen
  OM模型 (NPU): LiDAR → Camera → BEVPool → FusionHead

与旧方案对比：
  旧: lidar(NPU) + camera_backbone(NPU) + [depth+geo+bevpool(CPU)] + fusion_head(NPU)
  新: [depth+geo 预计算(CPU)] + 全流程一个OM(NPU)

ONNX输入:
  - voxels      : [max_V, 32, 5]     LiDAR体素特征
  - num_points  : [max_V]             每体素点数
  - coords      : [max_V, 4]          体素坐标 (b,z,y,x)
  - imgs        : [B, N, 3, H, W]    相机图像
  - depth       : [B, N, 1, H, W]    预计算深度图
  - geom_feats  : [B, N, D, fH, fW, 3] 预计算几何坐标

ONNX输出 (TransFusionHead):
  - dense_heatmap, top_cls, query_heatmap_score,
  - heatmap_q, center, height, dim, rot, vel, top_idx
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


# ═══════════════════════════════════════════════════════════════════════════
#  ONNX兼容的 PointPillarsScatter
# ═══════════════════════════════════════════════════════════════════════════
class OnnxPointPillarsScatter(nn.Module):
    """替换原始PointPillarsScatter，使用ONNX兼容的scatter操作。"""

    def __init__(self, in_channels, ny, nx):
        super().__init__()
        self.C, self.ny, self.nx = in_channels, ny, nx

    def forward(self, voxel_features, coors):
        C, ny, nx = self.C, self.ny, self.nx
        HW = ny * nx
        # coors: [V, 4] -> column 2=y, column 3=x
        indices = (coors[:, 2].long() * nx + coors[:, 3].long()).to(torch.int32)
        canvas = torch.zeros(C, HW, dtype=voxel_features.dtype, device=voxel_features.device)
        canvas = canvas.scatter(
            1,
            indices.unsqueeze(0).expand(C, -1),
            voxel_features.t()
        )
        return canvas.view(C, 1, ny, nx).permute(1, 0, 2, 3).contiguous()


# ═══════════════════════════════════════════════════════════════════════════
#  全流程统一模型
# ═══════════════════════════════════════════════════════════════════════════
class BEVFusionFullNPU(nn.Module):
    """
    BEVFusion全NPU推理模型（统一ONNX）

    整合流程：
      LiDAR分支: VoxelEncoder → OnnxScatter → [直接输入到Fusion]
      相机分支:  Backbone → Neck → DepthNet
      BEVPool:  cam_feats + pool_lookup/pool_mask(预计算LUT) → camera_bev
      FusionHead: [camera_bev, lidar_scatter] → ConvFuser → Backbone → Neck → TransFusionHead

    注意：
      - depth 由外部预计算传入；geom_feats 在 __init__ 时一次性转换为 LUT
      - BEVPool 使用 Gather + Mul + ReduceSum，完全消除 ScatterElements
      - ONNX 输入为 pool_lookup [out_cells, max_pts] int64
                    pool_mask   [out_cells, max_pts] float32
      - LiDAR分支: Scatter输出直接进入ConvFuser（backbone/neck在fusion后）
    """

    # nuScenes: cls 8,9 不做NMS；其他类做3x3 max_pool NMS
    NMS_DATASET = 'nuScenes'

    def __init__(self, model, K=200, dataset='nuScenes',
                 B=1, N=6, fH=32, fW=88, max_pts: int = 16):
        super().__init__()
        self.K = K
        self.dataset = dataset

        # ── LiDAR分支 ──────────────────────────────────────────────────────
        self.pts_voxel_encoder = model.pts_voxel_encoder
        orig_scatter = model.pts_middle_encoder
        self.pts_middle_encoder = OnnxPointPillarsScatter(
            orig_scatter.in_channels, orig_scatter.ny, orig_scatter.nx)

        # ── 相机分支 ──────────────────────────────────────────────────────
        self.img_backbone = model.img_backbone
        self.img_neck = model.img_neck
        vt = model.view_transform
        self.dtransform = vt.dtransform
        self.depthnet = vt.depthnet
        self.D    = int(vt.D)   # depth bins,  e.g. 118  — Python int, not tensor
        self.C_cam = int(vt.C)  # cam channels, e.g. 80

        # ── BEVPool: 所有标量提前固化为 Python int / float ─────────────────
        # register_buffer 只用来保证权重随 .to(device) 正确迁移；
        # forward 内不得再调用 .item() / 做 tensor 整除。
        self.register_buffer('_bx', vt.bx.detach().clone().float())  # [3]
        self.register_buffer('_dx', vt.dx.detach().clone().float())  # [3]

        # Python int 版，用于 arange / zeros / view 等需要 Python 标量的地方
        self._nx0 = int(round(vt.nx[0].item()))  # x grid count, e.g. 360
        self._nx1 = int(round(vt.nx[1].item()))  # y grid count, e.g. 360
        self._nx2 = int(round(vt.nx[2].item()))  # z grid count, e.g. 1

        # BEV pool 输入形状（trace 期间的常量）
        self._B    = int(B)
        self._N    = int(N)
        self._fH   = int(fH)
        self._fW   = int(fW)
        # pts_per_batch = B*N*D*fH*fW // B = N*D*fH*fW，纯 Python int
        self._pts_per_batch = int(N) * self.D * int(fH) * int(fW)
        # out_cells = B * nx2 * nx0 * nx1
        self._out_cells = int(B) * self._nx2 * self._nx0 * self._nx1

        # max_pts: 每个 voxel 最多容纳几个相机点（LUT 槽位数）
        # 典型 BEV 配置下实测 max_pts ≤ 8，16 足够安全
        self._max_pts = int(max_pts)

        # ── Fusion + Head ─────────────────────────────────────────────────
        self.fusion_layer = model.fusion_layer   # ConvFuser
        self.pts_backbone = model.pts_backbone   # SECOND
        self.pts_neck = model.pts_neck           # SECONDFPN
        self.head = model.bbox_head              # TransFusionHead

    # ────────────────────────────────────────────────────────────────────
    #  BEV Pool —— Gather + Mul + ReduceSum（彻底消除 ScatterElements）
    # ────────────────────────────────────────────────────────────────────
    def _bev_pool(self, cam_feats: torch.Tensor,
                  pool_lookup: torch.Tensor,
                  pool_mask: torch.Tensor) -> torch.Tensor:
        """
        BEV Pool via Gather + Mul + ReduceSum（ONNX opset-11，NPU 向量化友好）。

        Args:
            cam_feats  : [B, N, C, D, fH, fW]
            pool_lookup: [out_cells, max_pts]  int64，点的全局行下标（预计算，固定）
            pool_mask  : [out_cells, max_pts]  float32，有效槽位标记（预计算，固定）
        Returns:
            bev_feat   : [B, C*nz, nx0, nx1]

        ONNX 算子：Gather → Mul → ReduceSum（全部向量化，无写冲突）
        """
        B         = self._B
        C         = self.C_cam
        nx0       = self._nx0
        nx1       = self._nx1
        nx2       = self._nx2
        out_cells = self._out_cells   # Python int: B*nx2*nx0*nx1
        max_pts   = self._max_pts     # Python int，LUT 槽位数

        # ── [B,N,C,D,fH,fW] → [Nprime, C] ──────────────────────────────
        feats = cam_feats.permute(0, 1, 3, 4, 5, 2).contiguous().reshape(-1, C)
        # feats.shape = [Nprime, C]

        # ── Gather：pool_lookup 展平后一次性取出所有需要的行 ────────────
        # pool_lookup 作为 ONNX 输入已与模型在同一 device
        flat_idx = pool_lookup.reshape(-1)                  # [out_cells * max_pts]
        gathered = feats[flat_idx]                          # ONNX: Gather
        gathered = gathered.reshape(out_cells, max_pts, C)  # [out_cells, max_pts, C]

        # ── Mul：屏蔽填充槽位（mask=0 → 贡献归零） ──────────────────────
        # pool_mask: [out_cells, max_pts] → unsqueeze → [out_cells, max_pts, 1]
        mask = pool_mask.to(feats.dtype).unsqueeze(-1)
        gathered = gathered * mask                          # ONNX: Mul

        # ── ReduceSum over max_pts 维 ──────────────────────────────────
        out = gathered.sum(dim=1)                           # [out_cells, C]

        # ── Reshape → [B, C*nx2, nx0, nx1] ─────────────────────────────
        out = out.view(B, nx2, nx0, nx1, C).permute(0, 4, 1, 2, 3).contiguous()
        return out.reshape(B, C * nx2, nx0, nx1)

    # ────────────────────────────────────────────────────────────────────
    #  LUT 预构建（__init__ 或 export 前调用一次，不进入 ONNX 图）
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_lut(geom_feats: torch.Tensor,
                   bx: torch.Tensor, dx: torch.Tensor,
                   B: int, N: int, D: int, fH: int, fW: int,
                   nx0: int, nx1: int, nx2: int,
                   max_pts: int):
        """
        CPU 端从 geom_feats 预构建 BEV pool Gather 查找表（向量化 numpy，无 Python 逐点循环）。

        仅在 __init__ / export 前调用一次，结果作为 ONNX 静态输入（pool_lookup / pool_mask）。

        Args:
            geom_feats : [B, N, D, fH, fW, 3]  lidar 坐标（CPU tensor 或 CUDA tensor 均可）
            bx, dx     : [3] BEV 网格原点/步长
            B,N,D,fH,fW: Python int，形状参数
            nx0,nx1,nx2: Python int，BEV 网格大小
            max_pts    : int，每个 voxel LUT 槽位数上限

        Returns:
            pool_lookup : LongTensor  [out_cells, max_pts]  CPU，点的全局行下标
            pool_mask   : FloatTensor [out_cells, max_pts]  CPU，有效槽位标记
        """
        import numpy as np

        ppb       = N * D * fH * fW          # 每 batch 的点数
        out_cells = B * nx2 * nx0 * nx1      # BEV 格子总数

        # ── 计算每个点的 voxel 线性索引（numpy 向量化） ──────────────────
        coords = geom_feats.reshape(-1, 3).cpu().float().numpy()   # [Nprime, 3]
        bx_np  = bx.cpu().float().numpy()
        dx_np  = dx.cpu().float().numpy()

        idx  = ((coords - (bx_np - dx_np * 0.5)) / dx_np).astype(np.int64)
        x_id, y_id, z_id = idx[:, 0], idx[:, 1], idx[:, 2]

        valid = (
            (x_id >= 0) & (x_id < nx0) &
            (y_id >= 0) & (y_id < nx1) &
            (z_id >= 0) & (z_id < nx2)
        )

        b_id = np.repeat(np.arange(B, dtype=np.int64), ppb)
        x_c  = np.clip(x_id, 0, nx0 - 1)
        y_c  = np.clip(y_id, 0, nx1 - 1)
        z_c  = np.clip(z_id, 0, nx2 - 1)

        lin = b_id * (nx2 * nx0 * nx1) + z_c * (nx0 * nx1) + x_c * nx1 + y_c

        # ── 向量化槽位分配 ───────────────────────────────────────────────
        pool_lookup = np.zeros((out_cells, max_pts), dtype=np.int64)
        pool_mask   = np.zeros((out_cells, max_pts), dtype=np.float32)

        valid_pt_idx  = np.where(valid)[0]          # [M]，有效点的全局下标
        valid_vox     = lin[valid_pt_idx]            # [M]，对应 voxel 线性索引

        # 按 voxel 排序，方便分组
        order         = np.argsort(valid_vox, kind='stable')
        sorted_pt     = valid_pt_idx[order]
        sorted_vox    = valid_vox[order]

        # 每个点在其所属 voxel 内的槽位编号（segment-cumsum 技巧）
        is_new_vox    = np.concatenate([[True], sorted_vox[1:] != sorted_vox[:-1]])
        # 累加 segment 起点编号，再用全局位置减去起点 = 组内偏移
        seg_start_pos = np.where(is_new_vox, np.arange(len(sorted_pt), dtype=np.int64), 0)
        # 将各起点传播到组内所有成员（maximum.accumulate = 向前传播最大值）
        seg_start_pos = np.maximum.accumulate(seg_start_pos)
        slot_arr      = np.arange(len(sorted_pt), dtype=np.int64) - seg_start_pos  # 组内偏移

        # 截断超过 max_pts 的点（极少数 voxel 点密度异常情况）
        keep      = slot_arr < max_pts
        final_pt  = sorted_pt[keep]
        final_vox = sorted_vox[keep]
        final_slot = slot_arr[keep]

        pool_lookup[final_vox, final_slot] = final_pt
        pool_mask  [final_vox, final_slot] = 1.0

        return (torch.from_numpy(pool_lookup),
                torch.from_numpy(pool_mask))

    # ────────────────────────────────────────────────────────────────────
    #  Forward
    # ────────────────────────────────────────────────────────────────────
    def forward(self, voxels, num_points, coords, imgs, depth,
                pool_lookup, pool_mask):
        """
        Args:
            voxels    : [max_V, 32, 5]          体素特征（已pad）
            num_points: [max_V]                   每体素点数
            coords    : [max_V, 4]               体素坐标 (b,z,y,x)，float
            imgs      : [B, N, 3, H, W]          相机图像
            depth     : [B, N, 1, H, W]          预计算深度图
            pool_lookup: [out_cells, max_pts]  int64  BEV pool Gather 查找表（预计算）
            pool_mask  : [out_cells, max_pts]  float32 有效槽位掩码（预计算）

        Returns:
            tuple: (dense_heatmap, top_cls, query_heatmap_score,
                    heatmap_q, center, height, dim, rot, vel, top_idx)
        """
        B, N, C_img, H, W = imgs.shape

        # ── 1. LiDAR 分支：体素编码 + Scatter ────────────────────────────
        vf = self.pts_voxel_encoder(voxels, num_points, coords)
        mask = (coords.sum(dim=1) != 0).float().unsqueeze(1)
        vf = vf * mask
        lidar_bev = self.pts_middle_encoder(vf, coords)  # [1, 256, ny, nx]

        # ── 2. 相机分支：Backbone → Neck → DepthNet ───────────────────────
        # 使用 self._B/_N 等 Python int 常量，不从 shape 解包
        B_int  = self._B
        N_int  = self._N
        fH_int = self._fH
        fW_int = self._fW
        D_int  = self.D
        C_int  = self.C_cam

        imgs_flat = imgs.view(B_int * N_int, C_img, H, W)

        # Backbone (ResNet50)
        bb_feats = self.img_backbone(imgs_flat)  # tuple of 3 scales

        # Neck (GeneralizedLSSFPN)
        neck_feats = self.img_neck(list(bb_feats))
        img_feat = neck_feats[0]                          # [B*N, 256, fH, fW]

        # dtransform: 深度辅助特征
        d = depth.view(B_int * N_int, *depth.shape[2:])  # [B*N, 1, H, W]
        d = self.dtransform(d)                            # [B*N, 64, fH, fW]

        # depthnet: 深度概率 + 视觉特征
        x = torch.cat([d, img_feat], dim=1)              # [B*N, 320, fH, fW]
        x = self.depthnet(x)                              # [B*N, D+C_cam, fH, fW]

        depth_logits = x[:, :D_int]                       # [B*N, D, fH, fW]
        feat         = x[:, D_int: D_int + C_int]         # [B*N, C_cam, fH, fW]
        depth_prob   = depth_logits.softmax(dim=1)

        # outer product → [B*N, C_cam, D, fH, fW]
        cam_feats = depth_prob.unsqueeze(1) * feat.unsqueeze(2)
        # reshape 全用 Python int，不触发 symbolic tensor
        cam_feats = cam_feats.view(B_int, N_int, C_int, D_int, fH_int, fW_int)

        # ── 3. BEV Pool（pool_lookup/pool_mask 预计算传入，无 ScatterElements） ─
        camera_bev = self._bev_pool(cam_feats, pool_lookup, pool_mask)  # [B, C_cam*nz, nx0, nx1]

        # ── 4. Fusion: ConvFuser([camera_bev, lidar_bev]) ─────────────────
        fused_feat = self.fusion_layer([camera_bev, lidar_bev])  # [B, 256, ny, nx]

        # ── 5. Backbone + Neck（处理融合后特征）──────────────────────────
        backbone_feats = self.pts_backbone(fused_feat)
        neck_out = self.pts_neck(list(backbone_feats))[0]  # [B, C_neck, H', W']

        # ── 6. TransFusionHead ────────────────────────────────────────────
        head = self.head
        batch_size = neck_out.shape[0]

        fusion_feat = head.shared_conv(neck_out)
        fusion_feat_flatten = fusion_feat.view(batch_size, fusion_feat.shape[1], -1)
        bev_pos = head.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)

        # Heatmap
        dense_heatmap = head.heatmap_head(fusion_feat.float())
        heatmap = dense_heatmap.sigmoid()

        # NMS（3x3 max pool）
        padding = head.nms_kernel_size // 2
        local_max = F.max_pool2d(
            heatmap, kernel_size=head.nms_kernel_size, stride=1, padding=padding)

        if self.dataset == 'nuScenes':
            # cls 8,9 不做NMS（行人、交通锥）
            local_max_cls8 = F.max_pool2d(heatmap[:, 8:9], kernel_size=1, stride=1, padding=0)
            local_max_cls9 = F.max_pool2d(heatmap[:, 9:10], kernel_size=1, stride=1, padding=0)
            local_max = torch.cat([local_max[:, :8], local_max_cls8, local_max_cls9], dim=1)
        elif self.dataset == 'Waymo':
            local_max_cls1 = F.max_pool2d(heatmap[:, 1:2], kernel_size=1, stride=1, padding=0)
            local_max_cls2 = F.max_pool2d(heatmap[:, 2:3], kernel_size=1, stride=1, padding=0)
            local_max = torch.cat(
                [local_max[:, :1], local_max_cls1, local_max_cls2, local_max[:, 3:]], dim=1)

        heatmap = heatmap * ((heatmap + 1e-3) >= local_max).float()
        heatmap = heatmap.view(batch_size, head.num_classes, -1)  # [B, C, HW]

        # TopK 候选
        scores_flat = heatmap.view(batch_size, -1)
        _, top_proposals = torch.topk(scores_flat, k=self.K, dim=-1)
        HW = heatmap.shape[-1]
        top_proposals_class = top_proposals // HW
        top_proposals_index = top_proposals % HW

        # 抽取 query 特征
        query_feat = fusion_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, fusion_feat_flatten.shape[1], -1),
            dim=-1)

        # 类别编码
        one_hot = F.one_hot(top_proposals_class, num_classes=head.num_classes).permute(0, 2, 1)
        query_cat_encoding = head.class_encoding(one_hot.float())
        query_feat = query_feat + query_cat_encoding

        # Query 位置
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1)

        # Transformer Decoder
        ret_dicts = []
        for i in range(head.num_decoder_layers):
            query_feat = head.decoder[i](
                query_feat, key=fusion_feat_flatten,
                query_pos=query_pos, key_pos=bev_pos)
            res = head.prediction_heads[i](query_feat)
            res['center'] = res['center'] + query_pos.permute(0, 2, 1)
            ret_dicts.append(res)
            query_pos = res['center'].clone().permute(0, 2, 1)

        # query_heatmap_score
        query_heatmap_score = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, head.num_classes, -1),
            dim=-1)

        last = ret_dicts[-1]
        vel = last.get('vel', torch.zeros(batch_size, 2, self.K, device=fused_feat.device))

        return (
            dense_heatmap,          # [B, num_cls, H', W']
            top_proposals_class,    # [B, K]
            query_heatmap_score,    # [B, num_cls, K]
            last['heatmap'],        # [B, num_cls, K]
            last['center'],         # [B, 2, K]
            last['height'],         # [B, 1, K]
            last['dim'],            # [B, 3, K]
            last['rot'],            # [B, 2, K]
            vel,                    # [B, 2, K]
            top_proposals_index,    # [B, K]
        )


# ═══════════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════════

def make_dummy_inputs(model, max_voxels=10000, B=1, N=6,
                      H=256, W=704, fH=32, fW=88, device='cuda',
                      max_pts: int = 16):
    """
    生成哑元输入，用于 ONNX trace。

    与旧版区别：geom_feats 在函数内部生成并立即转换为 pool_lookup / pool_mask（CPU），
    trace 输入中不再包含 geom_feats，改为传入 LUT 张量。
    """
    torch.manual_seed(0)
    D = model.view_transform.D

    voxels = torch.randn(max_voxels, 32, 5, device=device)
    num_points = torch.randint(1, 33, (max_voxels,), device=device).float()
    # coords: (b,z,y,x) — 模拟真实坐标分布
    enc = model.pts_middle_encoder
    ny, nx = enc.ny, enc.nx
    fi = torch.randperm(ny * nx, device=device)[:max_voxels]
    coords = torch.zeros(max_voxels, 4, device=device)
    coords[:, 2] = (fi // nx).float()
    coords[:, 3] = (fi % nx).float()

    imgs = torch.randn(B, N, 3, H, W, device=device)

    # 预计算 depth（外部提供，这里用随机值模拟）
    depth = torch.rand(B, N, 1, H, W, device=device) * 59.0 + 1.0

    # ── 内部生成 geom_feats，然后预构建 LUT ──────────────────────────────
    # 实际部署时应使用真实标定数据；这里用均匀随机坐标模拟（覆盖 BEV 范围）
    vt  = model.view_transform
    bx  = vt.bx.detach().cpu().float()
    dx  = vt.dx.detach().cpu().float()
    nx0 = int(round(vt.nx[0].item()))
    nx1 = int(round(vt.nx[1].item()))
    nx2 = int(round(vt.nx[2].item()))

    geom_feats = torch.zeros(B, N, D, fH, fW, 3)
    geom_feats[..., 0] = torch.rand(B, N, D, fH, fW) * 108.0 - 54.0
    geom_feats[..., 1] = torch.rand(B, N, D, fH, fW) * 108.0 - 54.0
    geom_feats[..., 2] = torch.rand(B, N, D, fH, fW) * 20.0 - 10.0

    print('  构建 BEV pool LUT（基于哑元 geom_feats）...')
    pool_lookup, pool_mask = BEVFusionFullNPU._build_lut(
        geom_feats, bx, dx, B, N, int(D), fH, fW, nx0, nx1, nx2, max_pts)
    pool_lookup = pool_lookup.to(device)
    pool_mask   = pool_mask.to(device)

    print(f'  pool_lookup: {pool_lookup.shape} dtype={pool_lookup.dtype}')
    print(f'  pool_mask  : {pool_mask.shape}  dtype={pool_mask.dtype}')

    return voxels, num_points, coords, imgs, depth, pool_lookup, pool_mask


def save_bin(arr, path):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr.tofile(path)
    print(f"  bin: {path}  shape={arr.shape}  dtype={arr.dtype}")


def print_atc_cmd(onnx_path, om_path, max_voxels, B=1, N=6,
                  H=256, W=704, fH=32, fW=88, D=118,
                  nx0=360, nx1=360, nx2=1, max_pts=16,
                  soc='Ascend310P1'):
    """打印 ATC 编译命令。"""
    out_cells = B * nx2 * nx0 * nx1
    input_shape = (
        f"voxels:{max_voxels},32,5;"
        f"num_points:{max_voxels};"
        f"coords:{max_voxels},4;"
        f"imgs:{B},{N},3,{H},{W};"
        f"depth:{B},{N},1,{H},{W};"
        f"pool_lookup:{out_cells},{max_pts};"
        f"pool_mask:{out_cells},{max_pts}"
    )
    print(f"""
ATC 编译命令 (静态 shape):
─────────────────────────────────────────────────────────────────────────
atc --model="{onnx_path}" \\
    --framework=5 \\
    --output="{om_path}" \\
    --input_format=ND \\
    --input_shape="{input_shape}" \\
    --soc_version={soc} \\
    --op_select_implmode=high_precision \\
    --precision_mode=allow_fp32_to_fp16 \\
    --log=warning
─────────────────────────────────────────────────────────────────────────

动态体素数版本（推荐部署时使用）:
─────────────────────────────────────────────────────────────────────────
atc --model="{onnx_path}" \\
    --framework=5 \\
    --output="{om_path}_dynamic" \\
    --input_format=ND \\
    --input_shape="voxels:-1,32,5;num_points:-1;coords:-1,4;imgs:{B},{N},3,{H},{W};depth:{B},{N},1,{H},{W};pool_lookup:{out_cells},{max_pts};pool_mask:{out_cells},{max_pts}" \\
    --dynamic_dims="6000,6000,6000;8000,8000,8000;10000,10000,10000" \\
    --soc_version={soc} \\
    --op_select_implmode=high_precision \\
    --precision_mode=allow_fp32_to_fp16 \\
    --log=warning
""")


# ═══════════════════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='BEVFusion 全NPU统一ONNX导出')
    p.add_argument('--config',
                   default='src/configs/bevfusion_lidar-cam_voxel03_second_secfpn_'
                           '8xb4-cyclic-20e_nus-3d_resnet50.py')
    p.add_argument('--ckpt',
                   default='work_dirs/bevfusion_lidar-cam_voxel03_second_secfpn_'
                           '8xb4-cyclic-20e_nus-3d_resnet50/epoch_6.pth')
    p.add_argument('--outdir', default='models/onnx_full_npu')
    p.add_argument('--bindir', default='bins_full_npu')
    p.add_argument('--max-voxels', type=int, default=10000)
    p.add_argument('--K', type=int, default=200)
    p.add_argument('--dataset', default='nuScenes', choices=['nuScenes', 'Waymo'])
    p.add_argument('--save-bins', action='store_true', help='保存测试bin文件')
    p.add_argument('--soc', default='Ascend310P1')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = 'cuda'

    # ── 1. 加载模型 ───────────────────────────────────────────────────────
    print('[1/4] 加载 BEVFusion 模型...')
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    # 禁用 pretrain 加载（已有 checkpoint）
    if 'img_backbone' in cfg.model:
        cfg.model['img_backbone']['init_cfg'] = None
    model = init_model(cfg, args.ckpt)
    model.eval()

    D = model.view_transform.D
    enc = model.pts_middle_encoder
    H, W = 256, 704
    fH, fW = 32, 88
    B, N = 1, 6

    print(f'\n模型关键参数:')
    print(f'  Depth bins D = {D}')
    print(f'  Scatter grid: ny={enc.ny}, nx={enc.nx}')
    print(f'  Camera: {N} views, image {H}x{W}, feat {fH}x{fW}')
    print(f'  Max voxels: {args.max_voxels}')

    # ── 2. 构建部署模型 ────────────────────────────────────────────────────
    print('\n[2/4] 构建 BEVFusionFullNPU 部署模型...')
    MAX_PTS = 16   # LUT 槽位上限；实测 BEV 配置 max_pts ≤ 8，16 有裕量
    deploy_model = BEVFusionFullNPU(
        model, K=args.K, dataset=args.dataset,
        B=B, N=N, fH=fH, fW=fW, max_pts=MAX_PTS)
    deploy_model.eval().to(device)

    # 计算 BEV 网格参数（用于 ATC 命令及输出打印）
    vt   = model.view_transform
    nx0  = int(round(vt.nx[0].item()))
    nx1  = int(round(vt.nx[1].item()))
    nx2  = int(round(vt.nx[2].item()))

    # ── 3. 生成哑元输入并验证前向 ──────────────────────────────────────────
    print('\n[3/4] 生成测试输入并运行前向验证...')
    dummy = make_dummy_inputs(
        model, max_voxels=args.max_voxels,
        B=B, N=N, H=H, W=W, fH=fH, fW=fW,
        device=device, max_pts=MAX_PTS)
    voxels, num_points, coords, imgs, depth, pool_lookup, pool_mask = dummy

    print(f'  voxels     : {voxels.shape}')
    print(f'  num_points : {num_points.shape}')
    print(f'  coords     : {coords.shape}')
    print(f'  imgs       : {imgs.shape}')
    print(f'  depth      : {depth.shape}')
    print(f'  pool_lookup: {pool_lookup.shape}  dtype={pool_lookup.dtype}')
    print(f'  pool_mask  : {pool_mask.shape}   dtype={pool_mask.dtype}')

    with torch.no_grad():
        try:
            outputs = deploy_model(*dummy)
            print('\n  前向验证成功！输出:')
            output_names = [
                'dense_heatmap', 'top_cls', 'query_heatmap_score',
                'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel', 'top_idx'
            ]
            for name, out in zip(output_names, outputs):
                print(f'    {name}: {out.shape}  dtype={out.dtype}')
        except Exception as e:
            print(f'  [WARNING] 前向验证失败: {e}')
            print('  继续尝试 ONNX 导出...')
            import traceback
            traceback.print_exc()

    # 可选：保存 bin 测试数据
    if args.save_bins:
        os.makedirs(args.bindir, exist_ok=True)
        print(f'\n  保存测试 bin 到 {args.bindir}/')
        save_bin(voxels,      osp.join(args.bindir, 'voxels.bin'))
        save_bin(num_points,  osp.join(args.bindir, 'num_points.bin'))
        save_bin(coords,      osp.join(args.bindir, 'coords.bin'))
        save_bin(imgs,        osp.join(args.bindir, 'imgs.bin'))
        save_bin(depth,       osp.join(args.bindir, 'depth.bin'))
        save_bin(pool_lookup, osp.join(args.bindir, 'pool_lookup.bin'))
        save_bin(pool_mask,   osp.join(args.bindir, 'pool_mask.bin'))

    # ── 4. 导出 ONNX ──────────────────────────────────────────────────────
    print('\n[4/4] 导出 ONNX...')
    onnx_path = osp.join(args.outdir, 'bevfusion_full_npu.onnx')
    om_path   = 'models/om/bevfusion_full_npu'

    input_names = [
        'voxels', 'num_points', 'coords', 'imgs', 'depth',
        'pool_lookup', 'pool_mask',
    ]
    output_names = [
        'dense_heatmap', 'top_cls', 'query_heatmap_score',
        'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel', 'top_idx'
    ]

    # 动态轴：仅体素维度动态；pool_lookup / pool_mask 是固定形状
    dynamic_axes = {
        'voxels':     {0: 'max_V'},
        'num_points': {0: 'max_V'},
        'coords':     {0: 'max_V'},
    }

    with torch.no_grad():
        torch.onnx.export(
            deploy_model,
            dummy,
            onnx_path,
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
    print(f'  ✓ ONNX 已保存: {onnx_path}')

    # ── 打印 ATC 命令 ──────────────────────────────────────────────────────
    print_atc_cmd(onnx_path, om_path,
                  max_voxels=args.max_voxels,
                  B=B, N=N, H=H, W=W, fH=fH, fW=fW, D=D,
                  nx0=nx0, nx1=nx1, nx2=nx2, max_pts=MAX_PTS,
                  soc=args.soc)

    print('=' * 72)
    print('导出完成！')
    print('=' * 72)
    print(f"""
新流程说明:
  推理时外部预计算 (每帧一次，CPU):
    depth = DepthGeometryCalculator.generate_depth_map(points, metas)

  BEV pool LUT (固定标定时只需计算一次，CPU):
    pool_lookup, pool_mask = BEVFusionFullNPU._build_lut(
        geom_feats, bx, dx, B, N, D, fH, fW, nx0, nx1, nx2, max_pts={MAX_PTS})

  OM模型输入 (7个张量 → NPU):
    voxels, num_points, coords, imgs, depth, pool_lookup, pool_mask

  OM模型输出 (10个张量):
    dense_heatmap, top_cls, query_heatmap_score, heatmap_q,
    center, height, dim, rot, vel, top_idx

  ONNX BEV pool 算子: Gather + Mul + ReduceSum  (ScatterElements 已消除)
""")


if __name__ == '__main__':
    main()