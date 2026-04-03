"""
BEVFusion 全NPU评估脚本
================================================================================
相比旧版 bevfusion_camera_lidar_evaluator.py 的核心变化:

  旧版推理流程 (每帧):
    load_points → voxelize → [lidar NPU]
    load_images → [camera_backbone NPU] → [depth_gen CPU] → [depthnet NPU]
    → [geo_gen CPU] → [bevpool CPU] → [fusion_head NPU]

  新版推理流程 (每帧):
    load_points → voxelize
    load_images
    ★ precompute_depth_and_geometry()  ← CPU轻量预计算（depth + geom_feats）
    → [bevfusion_full_npu.om]          ← 单次NPU调用，完成全流程

新增方法:
    precompute_depth_and_geometry(points, metas, B, N)
        → depth [B,N,1,H,W], geom_feats [B,N,D,fH,fW,3]

时延统计新增两个桶:
    precompute_depth    : 深度图预计算耗时
    precompute_geometry : 几何坐标预计算耗时
"""

import time
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bevfusion_full_npu_net_source import BEVFusionFullNPUNet, init_acl

from bevfusion.ops.voxel import Voxelization
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.utils.splits import create_splits_scenes

import cv2  # 用于图像读取与预处理

# ─── 类别常量 ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

NMS_THRESHOLDS = {
    'car': 0.0, 'truck': 0.0, 'construction_vehicle': 0.0,
    'bus': 0.0, 'trailer': 0.0, 'barrier': 0.0,
    'motorcycle': 0.0, 'bicycle': 0.0,
    'pedestrian': 0.175, 'traffic_cone': 0.175,
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════════════
#  图像预处理
# ═══════════════════════════════════════════════════════════════════════════
#
#  ascend 环境中 numpy 1.x/2.x ABI 混装，numpy.core.multiarray 部分符号
#  （integer/inexact）缺失，导致 numpy ufunc 出错时错误格式化器本身崩溃：
#    RuntimeError: Unable to configure default ndarray.__repr__
#
#  根治方案：把所有浮点运算从 numpy 移到 PyTorch，完全绕开 numpy ufunc。
#  流水线：cv2.imread(uint8) → torch.from_numpy → .float()
#          → (x-mean)/std(torch算术) → .numpy() → float32 ndarray
# ─────────────────────────────────────────────────────────────────────────

# 模块级 torch tensor，只初始化一次（CPU），归一化用
_MEAN_T = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
_STD_T  = torch.tensor([58.395,  57.12,  57.375], dtype=torch.float32)


def load_and_preprocess_image(img_path: str,
                               target_size=(256, 704)) -> np.ndarray:
    """
    读取单张图像，返回 [3, H, W] float32 归一化数组（BGR，与训练一致）。
    全程用 PyTorch 做浮点运算，规避 ascend 环境 numpy ufunc 路径的 bug。
    """
    img_bgr = cv2.imread(img_path)          # [H_orig, W_orig, 3]  uint8
    if img_bgr is None:
        raise FileNotFoundError(f'Image not found: {img_path}')

    # Resize 仍用 cv2（输出 uint8 ndarray，不涉及 numpy 浮点运算）
    img_bgr = cv2.resize(img_bgr,
                         (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LINEAR)    # [H, W, 3]  uint8

    # ── 从这里开始完全用 PyTorch，不碰 numpy 算术 ────────────────────────
    # 确保使用的是numpy.ndarray类型以避免命名空间冲突
    img_bgr = np.asarray(img_bgr)           # [H, W, 3]  uint8
    t = torch.from_numpy(img_bgr)           # [H, W, 3]  uint8 tensor，零拷贝
    t = t.float()                           # [H, W, 3]  float32
    t = (t - _MEAN_T) / _STD_T             # 归一化，[3] 自动广播最后一维
    t = t.permute(2, 0, 1)                  # [3, H, W]
    return t.contiguous().numpy()           # [3, H, W]  float32


# ═══════════════════════════════════════════════════════════════════════════
#  深度图 + 几何坐标预计算器
# ═══════════════════════════════════════════════════════════════════════════

class DepthGeometryPrecomputer:
    """
    将深度图生成与几何坐标计算提前到推理外部（CPU端，每帧一次）。

    对应原 DepthGeometryCalculator，但拆分为两个独立接口方便分别计时。
    """

    def __init__(self,
                 image_size=(256, 704),
                 feature_size=(32, 88),
                 xbound=(-54.0, 54.0, 0.3),
                 ybound=(-54.0, 54.0, 0.3),
                 zbound=(-10.0, 10.0, 20.0),
                 dbound=(1.0, 60.0, 0.5)):
        self.image_size  = image_size   # (H, W)
        self.feature_size = feature_size  # (fH, fW)
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        # BEV 网格参数
        self.dx = torch.tensor([xbound[2], ybound[2], zbound[2]])
        self.bx = torch.tensor([
            xbound[0] + xbound[2] / 2,
            ybound[0] + ybound[2] / 2,
            zbound[0] + zbound[2] / 2,
        ])
        self.nx = torch.LongTensor([
            round((xbound[1] - xbound[0]) / xbound[2]),
            round((ybound[1] - ybound[0]) / ybound[2]),
            round((zbound[1] - zbound[0]) / zbound[2]),
        ])

        # 创建深度视锥
        self.frustum = self._create_frustum()  # [D, fH, fW, 3]
        self.D = self.frustum.shape[0]

    def _create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size
        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D = ds.shape[0]
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        return torch.stack((xs, ys, ds), -1)  # [D, fH, fW, 3]

    # ────────────────────────────────────────────────────────────────────
    def compute_depth_map(self,
                          points: torch.Tensor,
                          img_aug_matrix: torch.Tensor,
                          lidar_aug_matrix: torch.Tensor,
                          lidar2image: torch.Tensor,
                          B: int, N: int) -> torch.Tensor:
        """
        从点云生成多视角深度图。

        Args:
            points          : [M, 3+] 点云（lidar系）
            img_aug_matrix  : [B, N, 4, 4]
            lidar_aug_matrix: [B, 4, 4]
            lidar2image     : [B, N, 4, 4]
            B, N            : batch size & camera count

        Returns:
            depth : [B, N, 1, H, W]  float32
        """
        iH, iW = self.image_size
        depth = torch.zeros(B, N, 1, iH, iW, dtype=torch.float32)

        for b in range(B):
            cur_pts = points[:, :3].clone()                    # [M, 3]
            cur_l2i = lidar2image[b]                           # [N, 4, 4]
            cur_laug = lidar_aug_matrix[b]                     # [4, 4]
            cur_iaug = img_aug_matrix[b]                       # [N, 4, 4]

            # 反LiDAR增强：坐标还原
            cur_pts -= cur_laug[:3, 3]
            cur_pts = torch.inverse(cur_laug[:3, :3]).matmul(cur_pts.t())  # [3, M]

            # lidar → image (N cameras)
            cur_pts_n = cur_l2i[:, :3, :3].matmul(cur_pts)    # [N, 3, M]
            cur_pts_n += cur_l2i[:, :3, 3:4]                  # broadcast [N,3,1]

            dist = cur_pts_n[:, 2, :]                          # [N, M]
            cur_pts_n[:, 2, :] = cur_pts_n[:, 2, :].clamp(1e-5, 1e5)
            cur_pts_n[:, :2, :] /= cur_pts_n[:, 2:3, :]       # pixel coords

            # 图像增强变换
            cur_pts_n = cur_iaug[:, :3, :3].matmul(cur_pts_n)  # [N, 3, M]
            cur_pts_n += cur_iaug[:, :3, 3:4]
            pix = cur_pts_n[:, :2, :].permute(0, 2, 1)         # [N, M, 2]
            pix = pix[..., [1, 0]]                              # swap x,y → row,col

            on_img = (
                (pix[..., 0] >= 0) & (pix[..., 0] < iH) &
                (pix[..., 1] >= 0) & (pix[..., 1] < iW)
            )

            for c in range(N):
                valid_pix  = pix[c, on_img[c]].long()          # [K, 2]
                valid_dist = dist[c, on_img[c]]                 # [K]
                if len(valid_pix) > 0:
                    depth[b, c, 0,
                          valid_pix[:, 0],
                          valid_pix[:, 1]] = valid_dist

        return depth  # [B, N, 1, H, W]

    # ────────────────────────────────────────────────────────────────────
    def compute_geometry(self,
                         cam2lidar_rots: torch.Tensor,
                         cam2lidar_trans: torch.Tensor,
                         intrins: torch.Tensor,
                         post_rots: torch.Tensor,
                         post_trans: torch.Tensor,
                         extra_rots: torch.Tensor = None,
                         extra_trans: torch.Tensor = None) -> torch.Tensor:
        """
        计算每个视锥点在 lidar 坐标系下的 3D 坐标。

        Args:
            cam2lidar_rots  : [B, N, 3, 3]
            cam2lidar_trans : [B, N, 3]
            intrins         : [B, N, 3, 3]  相机内参
            post_rots       : [B, N, 3, 3]  图像增强旋转
            post_trans      : [B, N, 3]     图像增强平移
            extra_rots      : [B, 3, 3]     lidar增强旋转（可选）
            extra_trans     : [B, 3]        lidar增强平移（可选）

        Returns:
            geom_feats: [B, N, D, fH, fW, 3]
        """
        B, N, _ = cam2lidar_trans.shape

        # 反图像增强：将视锥坐标变换到原始相机坐标
        # frustum: [D, fH, fW, 3] → 广播到 [B, N, D, fH, fW, 3]
        pts = self.frustum.unsqueeze(0).unsqueeze(0) \
                  .expand(B, N, -1, -1, -1, -1).clone()        # [B,N,D,fH,fW,3]

        pts = pts - post_trans.view(B, N, 1, 1, 1, 3)
        pts = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3) \
                  .matmul(pts.unsqueeze(-1)).squeeze(-1)         # [B,N,D,fH,fW,3]

        # cam → lidar：将齐次像素坐标乘以深度，然后乘以旋转矩阵
        pts = torch.cat([
            pts[..., :2] * pts[..., 2:3],
            pts[..., 2:3]
        ], dim=-1)

        combine = cam2lidar_rots.matmul(torch.inverse(intrins))  # [B,N,3,3]
        pts = combine.view(B, N, 1, 1, 1, 3, 3) \
                  .matmul(pts.unsqueeze(-1)).squeeze(-1)          # [B,N,D,fH,fW,3]
        pts += cam2lidar_trans.view(B, N, 1, 1, 1, 3)

        # 可选：应用 lidar 增强
        if extra_rots is not None:
            pts = extra_rots.view(B, 1, 1, 1, 1, 3, 3) \
                      .matmul(pts.unsqueeze(-1)).squeeze(-1)
        if extra_trans is not None:
            pts += extra_trans.view(B, 1, 1, 1, 1, 3)

        return pts  # [B, N, D, fH, fW, 3]


# ═══════════════════════════════════════════════════════════════════════════
#  主评估器
# ═══════════════════════════════════════════════════════════════════════════

class BEVFusionFullNPUEvaluator:
    """
    BEVFusion 全NPU评估器（NuScenes验证集）。

    相比旧版的核心差异：
    1. 推理前调用 precompute_depth_and_geometry() 预计算 depth + geom_feats
    2. net.forward() 直接接收 6 个张量，内部只做一次 NPU 调用
    3. 时延统计分离出 precompute_depth / precompute_geometry 两个桶
    """

    def __init__(self,
                 dataroot: str = 'data/nuscenes-mini',
                 version: str = 'v1.0-mini'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        # BEVFusion 参数（与配置文件对齐）
        self.pc_range    = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size  = [0.30, 0.30, 8.0]
        self.grid_size   = [360, 360, 1]
        self.out_size_factor = 2

        self.image_size   = (256, 704)
        self.feature_size = (32, 88)

        # 相机列表（与训练时顺序一致）
        self.cam_keys = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        # 体素化算子
        self.voxelizer = Voxelization(
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=32,
            max_voxels=10000,
            deterministic=True,
        )

        # 深度/几何预计算器
        self.precomputer = DepthGeometryPrecomputer(
            image_size=self.image_size,
            feature_size=self.feature_size,
            xbound=(-54.0, 54.0, 0.3),
            ybound=(-54.0, 54.0, 0.3),
            zbound=(-10.0, 10.0, 20.0),
            dbound=(1.0, 60.0, 0.5),
        )

        self.results = {}

        # 时延统计（相比旧版新增 precompute_depth / precompute_geometry）
        self.timing_stats = {
            'sweep_loading':        [],
            'voxelization':         [],
            'image_loading':        [],
            'precompute_depth':     [],   # ← 新增
            'precompute_geometry':  [],   # ← 新增
            'npu_inference':        [],   # 原 lidar+camera+fusion_head 之和
            'decode':               [],
            'nms':                  [],
            'transform':            [],
            'total_per_sample':     [],
        }

    # ────────────────────────────────────────────────────────────────────
    #  点云加载
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_tf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3]  = t
        return T

    def load_points_with_sweeps(self, lidar_data, sweeps_num=9,
                                remove_close_radius=1.0):
        """
        加载当前帧 + 历史 sweeps 点云，转到当前帧 lidar 坐标系。
        返回: [M, 5] float32 (x,y,z,intensity,time_lag)
        """
        nusc = self.nusc
        cs_ref   = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose_ref = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        R_g_from_l = (Quaternion(pose_ref['rotation']).rotation_matrix
                      @ Quaternion(cs_ref['rotation']).rotation_matrix)
        t_g_from_l = (Quaternion(pose_ref['rotation']).rotation_matrix
                      @ np.array(cs_ref['translation'], dtype=np.float64)
                      + np.array(pose_ref['translation'], dtype=np.float64))
        T_ref     = self._make_tf(R_g_from_l, t_g_from_l)
        T_ref_inv = np.linalg.inv(T_ref)
        ref_ts    = lidar_data['timestamp']

        # 当前帧
        pts = np.fromfile(os.path.join(nusc.dataroot, lidar_data['filename']),
                          dtype=np.float32).reshape(-1, 5)[:, :5]
        pts[:, 4] = 0.0
        if remove_close_radius > 0:
            mask = np.linalg.norm(pts[:, :2], axis=1) >= remove_close_radius
            pts = pts[mask]
        all_pts = [pts]

        cur_token = lidar_data['prev']
        loaded = 0
        while cur_token and loaded < sweeps_num:
            sw = nusc.get('sample_data', cur_token)
            cs_sw   = nusc.get('calibrated_sensor', sw['calibrated_sensor_token'])
            pose_sw = nusc.get('ego_pose', sw['ego_pose_token'])
            R_sw = (Quaternion(pose_sw['rotation']).rotation_matrix
                    @ Quaternion(cs_sw['rotation']).rotation_matrix)
            t_sw = (Quaternion(pose_sw['rotation']).rotation_matrix
                    @ np.array(cs_sw['translation'], dtype=np.float64)
                    + np.array(pose_sw['translation'], dtype=np.float64))
            T_sw  = self._make_tf(R_sw, t_sw)
            T_rel = T_ref_inv @ T_sw

            sw_pts = np.fromfile(os.path.join(nusc.dataroot, sw['filename']),
                                 dtype=np.float32).reshape(-1, 5)[:, :5]
            if remove_close_radius > 0:
                mask = np.linalg.norm(sw_pts[:, :2], axis=1) >= remove_close_radius
                sw_pts = sw_pts[mask]

            xyz1 = np.hstack([sw_pts[:, :3],
                               np.ones((sw_pts.shape[0], 1), dtype=np.float32)])
            xyz_ref = (T_rel @ xyz1.T).T[:, :3].astype(np.float32)
            time_lag = (ref_ts - sw['timestamp']) * 1e-6
            sw_out = np.hstack([xyz_ref, sw_pts[:, 3:4],
                                np.full((sw_pts.shape[0], 1), time_lag, np.float32)])
            all_pts.append(sw_out)
            loaded += 1
            cur_token = sw['prev']

        return np.concatenate(all_pts, axis=0).astype(np.float32)

    # ────────────────────────────────────────────────────────────────────
    #  图像与元数据加载
    # ────────────────────────────────────────────────────────────────────

    def load_images_and_metas(self, sample_token: str, lidar_data: dict):
        """
        加载多视角图像及标定元数据。

        Returns:
            imgs  : [1, N, 3, H, W]  float32
            metas : dict（tensor）
        """
        nusc = self.nusc
        sample = nusc.get('sample', sample_token)
        N = len(self.cam_keys)

        imgs_list          = []
        lidar2image_list   = []
        cam2img_list       = []
        cam2lidar_list     = []
        img_aug_matrix_list = []

        cs_lidar = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        R_l2ego  = Quaternion(cs_lidar['rotation']).rotation_matrix
        t_l2ego  = np.array(cs_lidar['translation'], dtype=np.float64)

        for cam_key in self.cam_keys:
            cam_data = nusc.get('sample_data', sample['data'][cam_key])

            # 加载图像
            img_path = os.path.join(nusc.dataroot, cam_data['filename'])
            img = load_and_preprocess_image(img_path, self.image_size)
            imgs_list.append(img)

            # 标定
            cs_cam  = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            pose_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])
            pose_lidar = nusc.get('ego_pose', lidar_data['ego_pose_token'])

            # 相机内参（4x4）
            K = np.eye(4, dtype=np.float32)
            K[:3, :3] = np.array(cs_cam['camera_intrinsic'], dtype=np.float32)
            cam2img_list.append(K)

            # cam → ego → global（相机时刻）
            R_c2ego  = Quaternion(cs_cam['rotation']).rotation_matrix
            t_c2ego  = np.array(cs_cam['translation'], dtype=np.float64)
            R_ego2g_cam  = Quaternion(pose_cam['rotation']).rotation_matrix
            t_ego2g_cam  = np.array(pose_cam['translation'], dtype=np.float64)

            # lidar → ego → global（lidar时刻）
            R_ego2g_lid  = Quaternion(pose_lidar['rotation']).rotation_matrix
            t_ego2g_lid  = np.array(pose_lidar['translation'], dtype=np.float64)

            R_l2g = R_ego2g_lid @ R_l2ego
            t_l2g = R_ego2g_lid @ t_l2ego + t_ego2g_lid

            R_c2g = R_ego2g_cam @ R_c2ego
            t_c2g = R_ego2g_cam @ t_c2ego + t_ego2g_cam

            # lidar → camera
            R_l2c = np.linalg.inv(R_c2g) @ R_l2g
            t_l2c = np.linalg.inv(R_c2g) @ (t_l2g - t_c2g)

            l2c = np.eye(4, dtype=np.float32)
            l2c[:3, :3] = R_l2c;  l2c[:3, 3] = t_l2c
            lidar2image_list.append(K @ l2c)

            # camera → lidar
            c2l = np.eye(4, dtype=np.float32)
            c2l[:3, :3] = R_l2c.T;  c2l[:3, 3] = -R_l2c.T @ t_l2c
            cam2lidar_list.append(c2l)

            # 图像增强矩阵（推理时为单位矩阵）
            img_aug_matrix_list.append(np.eye(4, dtype=np.float32))

        B = 1
        imgs = np.stack(imgs_list, axis=0).reshape(B, N, 3, *self.image_size)
        lidar2image  = torch.from_numpy(np.stack(lidar2image_list).reshape(B, N, 4, 4))
        cam2img      = torch.from_numpy(np.stack(cam2img_list).reshape(B, N, 4, 4))
        cam2lidar    = torch.from_numpy(np.stack(cam2lidar_list).reshape(B, N, 4, 4))
        img_aug      = torch.from_numpy(np.stack(img_aug_matrix_list).reshape(B, N, 4, 4))
        lidar_aug    = torch.from_numpy(np.eye(4, dtype=np.float32).reshape(B, 4, 4))

        metas = {
            'lidar2image':    lidar2image,
            'cam2img':        cam2img,
            'cam2lidar':      cam2lidar,
            'img_aug_matrix': img_aug,
            'lidar_aug_matrix': lidar_aug,
        }
        return imgs, metas

    # ────────────────────────────────────────────────────────────────────
    #  ★ 核心新增：深度 + 几何 预计算
    # ────────────────────────────────────────────────────────────────────

    def precompute_depth_and_geometry(self,
                                      points: np.ndarray,
                                      metas: dict,
                                      B: int = 1,
                                      N: int = 6):
        """
        在 NPU 推理之前，CPU 端预计算：
          1. depth     [B, N, 1, H, W]      深度图（由点云投影）
          2. geom_feats [B, N, D, fH, fW, 3] 视锥几何坐标（lidar系）

        这两个量与当前帧的标定矩阵绑定，但**不含**任何可学习网络参数，
        因此可以安全地在推理前CPU完成，作为OM模型的静态输入传入。

        Args:
            points : [M, 5] float32 点云（lidar系）
            metas  : load_images_and_metas 返回的 dict
            B, N   : batch size & camera count

        Returns:
            depth      : np.ndarray [B, N, 1, H, W]
            geom_feats : np.ndarray [B, N, D, fH, fW, 3]
        """
        pts_tensor = torch.from_numpy(points[:, :3])

        # ── 1. 深度图预计算 ────────────────────────────────────────────────
        t0 = time.time()
        depth_t = self.precomputer.compute_depth_map(
            pts_tensor,
            metas['img_aug_matrix'],
            metas['lidar_aug_matrix'],
            metas['lidar2image'],
            B, N,
        )  # [B, N, 1, H, W]
        print(f"DepthMap: {time.time()-t0:.3f}s")
        self.timing_stats['precompute_depth'].append(time.time() - t0)

        # ── 2. 几何坐标预计算 ──────────────────────────────────────────────
        t0 = time.time()
        geom_t = self.precomputer.compute_geometry(
            metas['cam2lidar'][..., :3, :3],   # cam2lidar_rots
            metas['cam2lidar'][..., :3, 3],    # cam2lidar_trans
            metas['cam2img'][..., :3, :3],     # intrins
            metas['img_aug_matrix'][..., :3, :3],  # post_rots
            metas['img_aug_matrix'][..., :3, 3],   # post_trans
            metas['lidar_aug_matrix'][..., :3, :3],  # extra_rots
            metas['lidar_aug_matrix'][..., :3, 3],   # extra_trans
        )  # [B, N, D, fH, fW, 3]
        self.timing_stats['precompute_geometry'].append(time.time() - t0)

        return (np.ascontiguousarray(depth_t.numpy().astype(np.float32)),
                np.ascontiguousarray(geom_t.numpy().astype(np.float32)))

    # ────────────────────────────────────────────────────────────────────
    #  检测结果解码
    # ────────────────────────────────────────────────────────────────────

    def decode_outputs(self, outputs: list, score_threshold=0.01):
        """将 TransFusionHead 输出解码为检测框列表。"""
        t0 = time.time()
        try:
            K = 200
            num_cls = len(CLASS_NAMES)
            top_cls      = outputs[1].reshape(K).astype(np.int64)
            qhs          = outputs[2].reshape(num_cls, K).astype(np.float32)
            heatmap_q    = outputs[3].reshape(num_cls, K).astype(np.float32)
            center       = outputs[4].reshape(2, K).astype(np.float32)
            height       = outputs[5].reshape(K).astype(np.float32)
            dim_out      = outputs[6].reshape(3, K).astype(np.float32)
            rot          = outputs[7].reshape(2, K).astype(np.float32)
            vel          = outputs[8].reshape(2, K).astype(np.float32) if len(outputs) > 8 else np.zeros((2, K))

            hq_sig = 1.0 / (1.0 + np.exp(-np.clip(heatmap_q, -88, 88)))
            one_hot = np.zeros((num_cls, K), np.float32)
            for i in range(K):
                c = int(top_cls[i])
                if 0 <= c < num_cls:
                    one_hot[c, i] = 1.0

            scores = (hq_sig * qhs * one_hot).max(axis=0)
            labels = (hq_sig * qhs * one_hot).argmax(axis=0)

            mask = scores > score_threshold
            if not mask.any():
                self.timing_stats['decode'].append(time.time() - t0)
                return []

            vx, vy = self.voxel_size[0], self.voxel_size[1]
            px, py = self.pc_range[0],    self.pc_range[1]
            osf    = self.out_size_factor

            x   = center[0, mask] * osf * vx + px
            y   = center[1, mask] * osf * vy + py
            z   = height[mask]
            dim_exp = np.exp(np.clip(dim_out[:, mask], -10, 10))
            w, l, h = dim_exp[0], dim_exp[1], dim_exp[2]
            yaw = np.arctan2(rot[0, mask], rot[1, mask]) + np.pi / 2

            boxes = []
            for i in range(mask.sum()):
                cid = int(labels[mask][i])
                boxes.append({
                    'pos':        [float(x[i]), float(y[i]), float(z[i])],
                    'dim':        [float(w[i]), float(l[i]), float(h[i])],
                    'yaw':        float(yaw[i]),
                    'vel':        [float(vel[0, mask][i]), float(vel[1, mask][i])],
                    'score':      float(scores[mask][i]),
                    'class_id':   cid,
                    'class_name': CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else 'car',
                })
            print(f'  Decoded {len(boxes)} boxes (threshold={score_threshold})')
        except Exception as e:
            print(f'  [decode_outputs] Error: {e}')
            import traceback; traceback.print_exc()
            boxes = []

        self.timing_stats['decode'].append(time.time() - t0)
        return boxes

    # ────────────────────────────────────────────────────────────────────
    #  Circle NMS
    # ────────────────────────────────────────────────────────────────────

    def nms_3d(self, boxes: list, default_threshold=1.5) -> list:
        if not boxes:
            return []
        t0 = time.time()
        boxes = sorted(boxes, key=lambda x: x['score'], reverse=True)
        keep = []
        while boxes:
            cur = boxes.pop(0)
            keep.append(cur)
            thr = NMS_THRESHOLDS.get(cur['class_name'], default_threshold)
            boxes = [b for b in boxes
                     if np.hypot(cur['pos'][0] - b['pos'][0],
                                 cur['pos'][1] - b['pos'][1]) > thr]
        self.timing_stats['nms'].append(time.time() - t0)
        return keep

    # ────────────────────────────────────────────────────────────────────
    #  坐标变换 (lidar → global)
    # ────────────────────────────────────────────────────────────────────

    def transform_to_global(self, boxes: list, lidar_data: dict) -> list:
        t0 = time.time()
        cs   = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        R_l2e = Quaternion(cs['rotation']).rotation_matrix
        R_e2g = Quaternion(pose['rotation']).rotation_matrix

        out = []
        for box in boxes:
            pos = R_e2g @ (R_l2e @ np.array(box['pos'], dtype=np.float64)
                           + np.array(cs['translation'], dtype=np.float64)) \
                  + np.array(pose['translation'], dtype=np.float64)
            yaw_q   = Quaternion(axis=[0, 0, 1], radians=box['yaw'])
            total_q = Quaternion(pose['rotation']) * Quaternion(cs['rotation']) * yaw_q
            vel     = R_e2g[:2, :2] @ (R_l2e[:2, :2] @ np.array(box['vel'], dtype=np.float64))
            cname   = box['class_name'] if box['class_name'] in CLASS_NAMES else 'car'
            out.append({
                'sample_token':    lidar_data['sample_token'],
                'translation':     [float(p) for p in pos],
                'size':            [float(s) for s in box['dim']],
                'rotation':        [float(r) for r in total_q.elements],
                'velocity':        [float(v) for v in vel],
                'detection_name':  cname,
                'detection_score': float(box['score']),
                'attribute_name':  '',
            })
        self.timing_stats['transform'].append(time.time() - t0)
        return out

    # ────────────────────────────────────────────────────────────────────
    #  主评估循环
    # ────────────────────────────────────────────────────────────────────

    def evaluate(self, net: BEVFusionFullNPUNet,
                 output_path: str = 'results_full_npu.json'):
        """
        在 NuScenes 验证集上运行评估。

        Args:
            net         : BEVFusionFullNPUNet 实例
            output_path : 结果 JSON 保存路径
        """
        print('\n' + '=' * 62)
        print('BEVFusion Full-NPU  Evaluation Start')
        print('=' * 62)

        split_name = 'mini_val' if 'mini' in self.nusc.version else 'val'
        val_scenes  = set(create_splits_scenes()[split_name])
        val_samples = [s for s in self.nusc.sample
                       if self.nusc.get('scene', s['scene_token'])['name'] in val_scenes]
        print(f'Validation samples: {len(val_samples)}')

        N = len(self.cam_keys)
        B = 1

        for idx, sample in enumerate(val_samples):
            print(f'\n[{idx+1}/{len(val_samples)}] {sample["token"]}')
            t_start = time.time()

            lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

            # ── Step 1: 加载点云（含 sweeps） ─────────────────────────────
            t0 = time.time()
            points = self.load_points_with_sweeps(lidar_data, sweeps_num=9)
            self.timing_stats['sweep_loading'].append(time.time() - t0)
            print(f'  Points: {points.shape[0]}')

            # ── Step 2: 体素化 ─────────────────────────────────────────────
            t0 = time.time()
            pts_t = torch.from_numpy(points).float()
            voxels_t, coords_t, npts_t = self.voxelizer(pts_t)
            voxels    = voxels_t.numpy().astype(np.float32)
            npts      = npts_t.numpy().astype(np.float32)
            zero_col  = np.zeros((coords_t.shape[0], 1), dtype=np.float32)
            coords    = np.hstack([zero_col, coords_t.numpy()])[:, [0, 3, 1, 2]]  # b,z,y,x
            voxels    = np.ascontiguousarray(voxels)
            npts      = np.ascontiguousarray(npts)
            coords    = np.ascontiguousarray(coords.astype(np.float32))
            self.timing_stats['voxelization'].append(time.time() - t0)
            print(f'  Voxels: {voxels.shape[0]}')

            # ── Step 3: 加载图像与元数据 ──────────────────────────────────
            t0 = time.time()
            imgs, metas = self.load_images_and_metas(sample['token'], lidar_data)
            self.timing_stats['image_loading'].append(time.time() - t0)

            # ── Step 4: ★ 预计算 depth + geometry ────────────────────────
            #   这两步取代了旧版 "CPU Parts" 中间的 depth_gen + geo_gen，
            #   提前到推理前完成，作为 OM 输入直接传入 NPU。
            depth_np, geom_np = self.precompute_depth_and_geometry(
                points, metas, B=B, N=N)
            print(f'  depth: {depth_np.shape}  geom: {geom_np.shape}')

            # ── Step 5: NPU 推理（全流程单次调用） ───────────────────────
            t0 = time.time()
            outputs = net.forward(
                voxels, npts, coords,
                imgs,
                depth_np, geom_np,
            )
            self.timing_stats['npu_inference'].append(time.time() - t0)

            # ── Step 6: 解码 ───────────────────────────────────────────────
            boxes = self.decode_outputs(outputs, score_threshold=0.01)

            # ── Step 7: NMS ───────────────────────────────────────────────
            boxes = self.nms_3d(boxes, default_threshold=0.5)

            # ── Step 8: 坐标变换 ───────────────────────────────────────────
            boxes = self.transform_to_global(boxes, lidar_data)
            boxes.sort(key=lambda x: x['detection_score'], reverse=True)
            self.results[sample['token']] = boxes

            t_total = time.time() - t_start
            self.timing_stats['total_per_sample'].append(t_total)
            print(f'  Total={t_total*1000:.1f}ms  Boxes={len(boxes)}')

        # ── 保存结果 ──────────────────────────────────────────────────────
        submission = {
            'meta': {
                'use_camera': True, 'use_lidar': True,
                'use_radar': False, 'use_map': False,
                'use_external_track': False,
            },
            'results': self.results,
        }
        with open(output_path, 'w') as f:
            json.dump(submission, f, cls=NpEncoder, indent=2)
        print(f'\nResults saved: {output_path}')

        self._print_timing_summary()

        # ── NuScenes 官方评估 ─────────────────────────────────────────────
        print('\nRunning NuScenes official evaluation...')
        try:
            os.makedirs('eval_output', exist_ok=True)
            nusc_eval = NuScenesEval(
                self.nusc,
                config=config_factory('detection_cvpr_2019'),
                result_path=output_path,
                eval_set=split_name,
                output_dir='eval_output',
                verbose=True,
            )
            nusc_eval.main(plot_examples=0, render_curves=True)
        except Exception as e:
            print(f'NuScenes eval failed: {e}')
            import traceback; traceback.print_exc()

        print('\nNet timing:')
        net.print_timing_summary()

    # ────────────────────────────────────────────────────────────────────
    def _print_timing_summary(self):
        print('\n' + '=' * 72)
        print('TIMING STATISTICS SUMMARY')
        print('=' * 72)
        fmt = '{:<28} {:>8.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}'
        print(f'{"Task":<28} {"Total(s)":>8} {"Mean(ms)":>10} {"Std(ms)":>10}'
              f' {"Min(ms)":>10} {"Max(ms)":>10}')
        print('-' * 72)
        for name, times in self.timing_stats.items():
            if not times:
                continue
            a = np.array(times)
            print(fmt.format(name, a.sum(),
                             a.mean()*1000, a.std()*1000,
                             a.min()*1000, a.max()*1000))
        print('=' * 72)


# ═══════════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # 1. 初始化 ACL
    print('Initializing ACL...')
    ctx = init_acl(device_id=0)

    # 2. 初始化统一 NPU 网络（单个 OM）
    print('\nInitializing BEVFusionFullNPUNet...')
    net = BEVFusionFullNPUNet(
        model_path='models/om/bevfusion_full_npu_source_dynamic.om',
        gears=[6000, 8000, 10000],
        D=118,
        feature_size=[32, 88],
        image_size=[256, 704],
        num_cameras=6,
    )

    # 3. 初始化评估器
    print('\nInitializing evaluator...')
    evaluator = BEVFusionFullNPUEvaluator(
        dataroot='data/nuscenes-mini',
        version='v1.0-mini',
    )

    # 4. 运行评估
    evaluator.evaluate(net, output_path='bevfusion_full_npu_results.json')
    print('\nDone.')


if __name__ == '__main__':
    main()