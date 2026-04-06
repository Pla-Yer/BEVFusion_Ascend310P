"""
BEVFusion 并行推理评估脚本
================================================================================
推理流程（每帧）：
  1. 加载点云 + sweeps
  2. 体素化
  3. 并行加载 6 张图像（ThreadPoolExecutor）
  4. 深度图预计算（纯 numpy 向量化，每帧）
  5. LUT 预热（仅首帧；后续帧标定固定，直接复用）
  6. NPU 并行推理：
       Stream 1 → lidar_branch (async)  ─┐
       Stream 2 → camera_branch (async) ─┤→ Barrier → fusion_head (sync)
  7. 解码 + NMS + 坐标变换

相比旧版串行方案的优化：
  - image_loading  : 并行加载，~300ms → ~50ms
  - precompute_depth: 向量化 numpy，~165ms → ~10ms
  - precompute_geometry: 缓存，仅首帧 ~6ms，后续 0ms
  - NPU inference  : 并行两路，~182ms → ~152ms

总预期每帧耗时：~300ms（旧版 ~790ms）
"""

import time
import os
import sys
import json
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bevfusion_parallel_net import BEVFusionParallelNet, init_acl

from bevfusion.ops.voxel import Voxelization
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.utils.splits import create_splits_scenes

import cv2

# ── 类别常量 ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
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


# ══════════════════════════════════════════════════════════════════════════
#  图像预处理（全 PyTorch，规避 ascend 环境 numpy ufunc bug）
# ══════════════════════════════════════════════════════════════════════════
_MEAN_T = torch.tensor([123.675, 116.28,  103.53],  dtype=torch.float32)
_STD_T  = torch.tensor([58.395,  57.12,   57.375],  dtype=torch.float32)


def load_and_preprocess_image(img_path: str,
                               target_size=(256, 704)) -> np.ndarray:
    """读取单张图像，返回 [3, H, W] float32（BGR，与训练一致）。"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f'Image not found: {img_path}')
    img_bgr = cv2.resize(img_bgr, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(np.asarray(img_bgr)).float()
    t = (t - _MEAN_T) / _STD_T
    return t.permute(2, 0, 1).contiguous().numpy()


# ══════════════════════════════════════════════════════════════════════════
#  深度/几何预计算器
# ══════════════════════════════════════════════════════════════════════════
class DepthGeometryPrecomputer:
    """
    优化版：
      compute_depth_map_fast —— 纯 numpy 向量化，无 per-camera Python 循环
      compute_geometry       —— 保持原 torch 逻辑（标定固定时只调用一次）
    """

    def __init__(self,
                 image_size=(256, 704),
                 feature_size=(32, 88),
                 xbound=(-54.0, 54.0, 0.3),
                 ybound=(-54.0, 54.0, 0.3),
                 zbound=(-10.0, 10.0, 20.0),
                 dbound=(1.0, 60.0, 0.5)):
        self.image_size   = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

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

        self.frustum = self._create_frustum()
        self.D = self.frustum.shape[0]

    def _create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size
        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D  = ds.shape[0]
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        return torch.stack((xs, ys, ds), -1)

    # ── 优化版深度图（主流程使用） ────────────────────────────────────────
    def compute_depth_map_fast(self,
                               points_xyz: np.ndarray,
                               lidar2image_np: np.ndarray) -> np.ndarray:
        """
        纯 numpy 向量化深度图生成（一次完成所有相机，无 Python 循环）。

        优化点：
          1. lidar2image[:, :3, :] @ pts.T 一次投影所有相机
          2. np.where(valid) + fancy-index scatter，无 per-camera 循环
          3. 降序排列：远点先写，近点后写（近点优先保留）
          4. 推理时 lidar_aug / img_aug 为单位矩阵，直接省略

        Args:
            points_xyz    : [M, 3+] float32，lidar 系点云
            lidar2image_np: [N, 4, 4] float32，投影矩阵（K @ l2c，含内参）

        Returns:
            depth : [1, N, 1, H, W] float32
        """
        iH, iW = self.image_size
        N = lidar2image_np.shape[0]
        M = points_xyz.shape[0]

        pts_hom = np.concatenate(
            [points_xyz[:, :3].astype(np.float32),
             np.ones((M, 1), dtype=np.float32)], axis=1)      # [M, 4]

        pts_cam = lidar2image_np[:, :3, :] @ pts_hom.T        # [N, 3, M]

        depth_z = pts_cam[:, 2, :].copy()
        z_safe  = np.maximum(pts_cam[:, 2, :], 1e-5)
        pix_u   = pts_cam[:, 0, :] / z_safe                   # 列
        pix_v   = pts_cam[:, 1, :] / z_safe                   # 行

        valid = ((depth_z > 0) &
                 (pix_u >= 0) & (pix_u < iW) &
                 (pix_v >= 0) & (pix_v < iH))

        col_i = np.clip(pix_u.astype(np.int32), 0, iW - 1)
        row_i = np.clip(pix_v.astype(np.int32), 0, iH - 1)

        cam_idx, pt_idx = np.where(valid)
        r = row_i[cam_idx, pt_idx]
        c = col_i[cam_idx, pt_idx]
        d = depth_z[cam_idx, pt_idx]

        # 降序：远点先写，近点后写（近点覆盖远点）
        order  = np.argsort(-d)
        linear = cam_idx[order] * (iH * iW) + r[order] * iW + c[order]

        depth_out = np.zeros(N * iH * iW, dtype=np.float32)
        depth_out[linear] = d[order]
        return depth_out.reshape(1, N, 1, iH, iW)

    def compute_geometry(self,
                         cam2lidar_rots: torch.Tensor,
                         cam2lidar_trans: torch.Tensor,
                         intrins: torch.Tensor,
                         post_rots: torch.Tensor,
                         post_trans: torch.Tensor,
                         extra_rots: torch.Tensor = None,
                         extra_trans: torch.Tensor = None) -> torch.Tensor:
        """计算视锥点在 lidar 坐标系下的 3D 坐标 [B, N, D, fH, fW, 3]。"""
        B, N, _ = cam2lidar_trans.shape
        pts = (self.frustum.unsqueeze(0).unsqueeze(0)
               .expand(B, N, -1, -1, -1, -1).clone())

        pts = pts - post_trans.view(B, N, 1, 1, 1, 3)
        pts = (torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)
               .matmul(pts.unsqueeze(-1)).squeeze(-1))

        pts = torch.cat([pts[..., :2] * pts[..., 2:3], pts[..., 2:3]], dim=-1)

        combine = cam2lidar_rots.matmul(torch.inverse(intrins))
        pts = (combine.view(B, N, 1, 1, 1, 3, 3)
               .matmul(pts.unsqueeze(-1)).squeeze(-1))
        pts += cam2lidar_trans.view(B, N, 1, 1, 1, 3)

        if extra_rots is not None:
            pts = (extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                   .matmul(pts.unsqueeze(-1)).squeeze(-1))
        if extra_trans is not None:
            pts += extra_trans.view(B, 1, 1, 1, 1, 3)

        return pts


# ══════════════════════════════════════════════════════════════════════════
#  BEVFusion 并行推理评估器
# ══════════════════════════════════════════════════════════════════════════
class BEVFusionParallelEvaluator:
    """
    BEVFusion 并行推理评估器（NuScenes 验证集）。

    与旧版的差异：
      1. net 换为 BEVFusionParallelNet（两路 ACL Stream 并行）
      2. 图像并行加载（ThreadPoolExecutor）
      3. 传感器标定预缓存（__init__ 一次性提取，消除逐帧 nusc.get()）
      4. 深度图向量化 numpy（compute_depth_map_fast）
      5. geom_feats/LUT 仅首帧计算，后续复用
    """

    def __init__(self,
                 dataroot: str = 'data/nuscenes-mini',
                 version:  str = 'v1.0-mini'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        self.pc_range         = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size       = [0.30, 0.30, 8.0]
        self.grid_size        = [360, 360, 1]
        self.out_size_factor  = 2
        self.image_size       = (256, 704)
        self.feature_size     = (32, 88)

        self.cam_keys = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK',  'CAM_BACK_LEFT',   'CAM_BACK_RIGHT',
        ]

        self.voxelizer = Voxelization(
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=32,
            max_voxels=10000,
            deterministic=True,
        )

        self.precomputer = DepthGeometryPrecomputer(
            image_size=self.image_size,
            feature_size=self.feature_size,
            xbound=(-54.0, 54.0, 0.3),
            ybound=(-54.0, 54.0, 0.3),
            zbound=(-10.0, 10.0, 20.0),
            dbound=(1.0, 60.0, 0.5),
        )

        self.results = {}

        self.timing_stats = {
            'sweep_loading':       [],
            'voxelization':        [],
            'image_loading':       [],
            'precompute_depth':    [],
            'precompute_geometry': [],
            'npu_inference':       [],
            'decode':              [],
            'nms':                 [],
            'transform':           [],
            'total_per_sample':    [],
        }

        # 预缓存固定传感器标定（只做一次）
        self._calib_cache = {}
        self._preload_sensor_calib()

    # ── 传感器标定预缓存 ────────────────────────────────────────────────────
    def _preload_sensor_calib(self):
        """
        将固定不变的 calibrated_sensor 参数缓存为 numpy 数组。
        NuScenes 同一数据集内传感器配置固定，无需每帧重复 nusc.get()。
        """
        nusc    = self.nusc
        sample0 = nusc.sample[0]
        ld      = nusc.get('sample_data', sample0['data']['LIDAR_TOP'])
        cs_lid  = nusc.get('calibrated_sensor', ld['calibrated_sensor_token'])

        self._calib_cache = {
            'R_l2ego': Quaternion(cs_lid['rotation']).rotation_matrix,
            't_l2ego': np.array(cs_lid['translation'], dtype=np.float64),
            'cams':    {},
        }

        for cam_key in self.cam_keys:
            cd     = nusc.get('sample_data', sample0['data'][cam_key])
            cs_cam = nusc.get('calibrated_sensor', cd['calibrated_sensor_token'])
            K4     = np.eye(4, dtype=np.float32)
            K4[:3, :3] = np.array(cs_cam['camera_intrinsic'], dtype=np.float32)
            self._calib_cache['cams'][cam_key] = {
                'R_c2ego': Quaternion(cs_cam['rotation']).rotation_matrix,
                't_c2ego': np.array(cs_cam['translation'], dtype=np.float64),
                'K4':      K4,
            }

        print(f'[Evaluator] Sensor calib preloaded for {len(self.cam_keys)} cameras.')

    # ── 工具函数 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _make_tf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3]  = t
        return T

    # ── 点云加载 ──────────────────────────────────────────────────────────────
    def load_points_with_sweeps(self, lidar_data, sweeps_num=9,
                                remove_close_radius=1.0):
        """加载当前帧 + 历史 sweeps，转到当前帧 lidar 坐标系。"""
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

        pts = np.fromfile(os.path.join(nusc.dataroot, lidar_data['filename']),
                          dtype=np.float32).reshape(-1, 5)[:, :5]
        pts[:, 4] = 0.0
        if remove_close_radius > 0:
            pts = pts[np.linalg.norm(pts[:, :2], axis=1) >= remove_close_radius]
        all_pts = [pts]

        cur_token = lidar_data['prev']
        loaded = 0
        while cur_token and loaded < sweeps_num:
            sw      = nusc.get('sample_data', cur_token)
            cs_sw   = nusc.get('calibrated_sensor', sw['calibrated_sensor_token'])
            pose_sw = nusc.get('ego_pose', sw['ego_pose_token'])
            R_sw    = (Quaternion(pose_sw['rotation']).rotation_matrix
                       @ Quaternion(cs_sw['rotation']).rotation_matrix)
            t_sw    = (Quaternion(pose_sw['rotation']).rotation_matrix
                       @ np.array(cs_sw['translation'], dtype=np.float64)
                       + np.array(pose_sw['translation'], dtype=np.float64))
            T_rel   = T_ref_inv @ self._make_tf(R_sw, t_sw)

            sw_pts = np.fromfile(os.path.join(nusc.dataroot, sw['filename']),
                                 dtype=np.float32).reshape(-1, 5)[:, :5]
            if remove_close_radius > 0:
                sw_pts = sw_pts[np.linalg.norm(sw_pts[:, :2], axis=1) >= remove_close_radius]

            xyz1    = np.hstack([sw_pts[:, :3], np.ones((sw_pts.shape[0], 1), np.float32)])
            xyz_ref = (T_rel @ xyz1.T).T[:, :3].astype(np.float32)
            time_lag = (ref_ts - sw['timestamp']) * 1e-6
            all_pts.append(np.hstack([
                xyz_ref, sw_pts[:, 3:4],
                np.full((sw_pts.shape[0], 1), time_lag, np.float32)
            ]))
            loaded    += 1
            cur_token  = sw['prev']

        return np.concatenate(all_pts, axis=0).astype(np.float32)

    # ── 图像 + 元数据加载（并行 I/O + 缓存标定） ─────────────────────────────
    def load_images_and_metas(self, sample_token: str, lidar_data: dict):
        """
        并行加载 6 张图像 + 计算标定矩阵。

        优化：
          1. ThreadPoolExecutor 并行 cv2.imread（串行→并行，~300ms→~50ms）
          2. 传感器固定标定从 _calib_cache 读取，跳过重复 nusc.get()
          3. 返回 lidar2image_np [N,4,4]（供 compute_depth_map_fast 直接使用）
        """
        nusc   = self.nusc
        sample = nusc.get('sample', sample_token)
        N      = len(self.cam_keys)
        cache  = self._calib_cache

        cam_data_list = [
            nusc.get('sample_data', sample['data'][ck]) for ck in self.cam_keys
        ]
        img_paths = [
            os.path.join(nusc.dataroot, cd['filename']) for cd in cam_data_list
        ]

        # 并行加载图像
        with ThreadPoolExecutor(max_workers=N) as exe:
            imgs_list = list(exe.map(
                lambda p: load_and_preprocess_image(p, self.image_size),
                img_paths))

        # lidar ego pose（每帧不同，必须实时 lookup）
        pose_lidar   = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        R_ego2g_lid  = Quaternion(pose_lidar['rotation']).rotation_matrix
        t_ego2g_lid  = np.array(pose_lidar['translation'], dtype=np.float64)

        R_l2ego = cache['R_l2ego']
        t_l2ego = cache['t_l2ego']
        R_l2g   = R_ego2g_lid @ R_l2ego
        t_l2g   = R_ego2g_lid @ t_l2ego + t_ego2g_lid

        lidar2image_list = []
        cam2img_list     = []
        cam2lidar_list   = []

        for i, cam_key in enumerate(self.cam_keys):
            cd          = cam_data_list[i]
            pose_cam    = nusc.get('ego_pose', cd['ego_pose_token'])
            R_ego2g_cam = Quaternion(pose_cam['rotation']).rotation_matrix
            t_ego2g_cam = np.array(pose_cam['translation'], dtype=np.float64)

            cc    = cache['cams'][cam_key]
            R_c2g = R_ego2g_cam @ cc['R_c2ego']
            t_c2g = R_ego2g_cam @ cc['t_c2ego'] + t_ego2g_cam

            R_c2g_inv = np.linalg.inv(R_c2g)
            R_l2c     = R_c2g_inv @ R_l2g
            t_l2c     = R_c2g_inv @ (t_l2g - t_c2g)

            K4 = cc['K4']
            l2c = np.eye(4, dtype=np.float32)
            l2c[:3, :3] = R_l2c;  l2c[:3, 3] = t_l2c
            lidar2image_list.append((K4 @ l2c).astype(np.float32))

            c2l = np.eye(4, dtype=np.float32)
            c2l[:3, :3] = R_l2c.T;  c2l[:3, 3] = -R_l2c.T @ t_l2c
            cam2lidar_list.append(c2l)
            cam2img_list.append(K4.astype(np.float32))

        B = 1
        imgs              = np.stack(imgs_list).reshape(B, N, 3, *self.image_size)
        lidar2image_np    = np.stack(lidar2image_list).astype(np.float32)  # [N,4,4]

        metas = {
            'lidar2image':      torch.from_numpy(lidar2image_np.reshape(B, N, 4, 4)),
            'cam2img':          torch.from_numpy(np.stack(cam2img_list).reshape(B, N, 4, 4)),
            'cam2lidar':        torch.from_numpy(np.stack(cam2lidar_list).reshape(B, N, 4, 4)),
            'img_aug_matrix':   torch.from_numpy(np.tile(np.eye(4, dtype=np.float32), (B, N, 1, 1))),
            'lidar_aug_matrix': torch.from_numpy(np.eye(4, dtype=np.float32).reshape(B, 4, 4)),
        }

        return imgs, metas, lidar2image_np   # 第3个返回值供 compute_depth_map_fast

    # ── 仅计算深度图（每帧调用） ──────────────────────────────────────────────
    def _compute_depth_only(self,
                             points: np.ndarray,
                             lidar2image_np: np.ndarray) -> np.ndarray:
        """每帧调用，使用向量化 numpy 版本（~10ms）。"""
        return self.precomputer.compute_depth_map_fast(
            points[:, :3], lidar2image_np)

    # ── 首帧：深度 + 几何（用于 LUT 构建） ───────────────────────────────────
    def _compute_geometry_for_lut(self,
                                   metas: dict,
                                   B: int = 1,
                                   N: int = 6) -> np.ndarray:
        """仅首帧调用，计算 geom_feats 供 net.precompute_lut() 使用。"""
        geom_t = self.precomputer.compute_geometry(
            metas['cam2lidar'][..., :3, :3],
            metas['cam2lidar'][..., :3, 3],
            metas['cam2img'][..., :3, :3],
            metas['img_aug_matrix'][..., :3, :3],
            metas['img_aug_matrix'][..., :3, 3],
            metas['lidar_aug_matrix'][..., :3, :3],
            metas['lidar_aug_matrix'][..., :3, 3],
        )
        return np.ascontiguousarray(geom_t.numpy().astype(np.float32))

    # ── 检测结果解码 ──────────────────────────────────────────────────────────
    def decode_outputs(self, outputs: list, score_threshold=0.01) -> list:
        t0 = time.time()
        try:
            K       = 200
            num_cls = len(CLASS_NAMES)
            top_cls  = outputs[1].reshape(K).astype(np.int64)
            qhs      = outputs[2].reshape(num_cls, K).astype(np.float32)
            heatmap_q = outputs[3].reshape(num_cls, K).astype(np.float32)
            center   = outputs[4].reshape(2, K).astype(np.float32)
            height   = outputs[5].reshape(K).astype(np.float32)
            dim_out  = outputs[6].reshape(3, K).astype(np.float32)
            rot      = outputs[7].reshape(2, K).astype(np.float32)
            vel      = (outputs[8].reshape(2, K).astype(np.float32)
                        if len(outputs) > 8 else np.zeros((2, K)))

            hq_sig  = 1.0 / (1.0 + np.exp(-np.clip(heatmap_q, -88, 88)))
            one_hot = np.zeros((num_cls, K), np.float32)
            for i in range(K):
                c = int(top_cls[i])
                if 0 <= c < num_cls:
                    one_hot[c, i] = 1.0

            scores = (hq_sig * qhs * one_hot).max(axis=0)
            labels = (hq_sig * qhs * one_hot).argmax(axis=0)
            mask   = scores > score_threshold

            if not mask.any():
                self.timing_stats['decode'].append(time.time() - t0)
                return []

            vx, vy = self.voxel_size[0], self.voxel_size[1]
            px, py = self.pc_range[0],   self.pc_range[1]
            osf    = self.out_size_factor

            x   = center[0, mask] * osf * vx + px
            y   = center[1, mask] * osf * vy + py
            z   = height[mask]
            dexp = np.exp(np.clip(dim_out[:, mask], -10, 10))
            w, l, h = dexp[0], dexp[1], dexp[2]
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
                    'class_name': CLASS_NAMES[cid] if 0 <= cid < num_cls else 'car',
                })
            print(f'  Decoded {len(boxes)} boxes (thr={score_threshold})')
        except Exception as e:
            print(f'  [decode] Error: {e}')
            import traceback; traceback.print_exc()
            boxes = []

        self.timing_stats['decode'].append(time.time() - t0)
        return boxes

    # ── Circle NMS ───────────────────────────────────────────────────────────
    def nms_3d(self, boxes: list, default_threshold=1.5) -> list:
        if not boxes:
            return []
        t0 = time.time()
        boxes = sorted(boxes, key=lambda x: x['score'], reverse=True)
        keep  = []
        while boxes:
            cur = boxes.pop(0)
            keep.append(cur)
            thr = NMS_THRESHOLDS.get(cur['class_name'], default_threshold)
            boxes = [b for b in boxes
                     if np.hypot(cur['pos'][0] - b['pos'][0],
                                 cur['pos'][1] - b['pos'][1]) > thr]
        self.timing_stats['nms'].append(time.time() - t0)
        return keep

    # ── 坐标变换（lidar → global） ────────────────────────────────────────────
    def transform_to_global(self, boxes: list, lidar_data: dict) -> list:
        t0   = time.time()
        cs   = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        R_l2e = Quaternion(cs['rotation']).rotation_matrix
        R_e2g = Quaternion(pose['rotation']).rotation_matrix

        out = []
        for box in boxes:
            pos     = (R_e2g @ (R_l2e @ np.array(box['pos'], np.float64)
                                + np.array(cs['translation'], np.float64))
                       + np.array(pose['translation'], np.float64))
            yaw_q   = Quaternion(axis=[0, 0, 1], radians=box['yaw'])
            total_q = Quaternion(pose['rotation']) * Quaternion(cs['rotation']) * yaw_q
            vel     = R_e2g[:2, :2] @ (R_l2e[:2, :2] @ np.array(box['vel'], np.float64))
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

    # ── 主评估循环 ────────────────────────────────────────────────────────────
    def evaluate(self, net: BEVFusionParallelNet,
                 output_path: str = 'results_parallel.json'):
        """在 NuScenes 验证集上运行评估。"""
        print('\n' + '=' * 64)
        print('BEVFusion Parallel  Evaluation Start')
        print('=' * 64)

        split_name  = 'mini_val' if 'mini' in self.nusc.version else 'val'
        val_scenes  = set(create_splits_scenes()[split_name])
        val_samples = [s for s in self.nusc.sample
                       if self.nusc.get('scene', s['scene_token'])['name']
                       in val_scenes]
        print(f'Validation samples: {len(val_samples)}')

        N   = len(self.cam_keys)
        B   = 1
        lut_warmed = False

        for idx, sample in enumerate(val_samples):
            print(f'\n[{idx+1}/{len(val_samples)}] {sample["token"]}')
            t_start = time.time()

            lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

            # Step 1: 点云加载
            t0 = time.time()
            points = self.load_points_with_sweeps(lidar_data, sweeps_num=9)
            self.timing_stats['sweep_loading'].append(time.time() - t0)
            print(f'  Points: {points.shape[0]}')

            # Step 2: 体素化
            t0 = time.time()
            pts_t = torch.from_numpy(points).float()
            voxels_t, coords_t, npts_t = self.voxelizer(pts_t)
            voxels = voxels_t.numpy().astype(np.float32)
            npts   = npts_t.numpy().astype(np.float32)
            zero_col = np.zeros((coords_t.shape[0], 1), dtype=np.float32)
            coords   = np.hstack([zero_col, coords_t.numpy()])[:, [0, 3, 1, 2]]
            voxels   = np.ascontiguousarray(voxels)
            npts     = np.ascontiguousarray(npts)
            coords   = np.ascontiguousarray(coords.astype(np.float32))
            self.timing_stats['voxelization'].append(time.time() - t0)
            print(f'  Voxels: {voxels.shape[0]}')

            # Step 3: 并行图像加载 + 标定矩阵
            t0 = time.time()
            imgs, metas, lidar2image_np = self.load_images_and_metas(
                sample['token'], lidar_data)
            self.timing_stats['image_loading'].append(time.time() - t0)

            # Step 4a: 深度图预计算（每帧，~10ms）
            t0 = time.time()
            depth_np = self._compute_depth_only(points, lidar2image_np)
            self.timing_stats['precompute_depth'].append(time.time() - t0)
            print(f'  depth: {depth_np.shape}')

            # Step 4b: LUT 预热（仅首帧，标定固定后续复用）
            if not lut_warmed:
                t0 = time.time()
                geom_np = self._compute_geometry_for_lut(metas, B=B, N=N)
                self.timing_stats['precompute_geometry'].append(time.time() - t0)
                net.precompute_lut(geom_np)
                lut_warmed = True
                print(f'  geom: {geom_np.shape}  [LUT precomputed, reused hereafter]')
            else:
                self.timing_stats['precompute_geometry'].append(0.0)

            # Step 5: 并行 NPU 推理
            # Stream1→lidar_branch (async) 与 Stream2→camera_branch (async) 同时执行
            t0 = time.time()
            outputs = net.forward(
                voxels, npts, coords,
                imgs,
                depth_np,
                # geom_feats 省略：net 内部使用缓存 LUT
            )
            self.timing_stats['npu_inference'].append(time.time() - t0)

            # Step 6: 解码
            boxes = self.decode_outputs(outputs, score_threshold=0.01)

            # Step 7: NMS
            boxes = self.nms_3d(boxes, default_threshold=0.5)

            # Step 8: 坐标变换
            boxes = self.transform_to_global(boxes, lidar_data)
            boxes.sort(key=lambda x: x['detection_score'], reverse=True)
            self.results[sample['token']] = boxes

            t_total = time.time() - t_start
            self.timing_stats['total_per_sample'].append(t_total)
            print(f'  Total={t_total*1000:.1f}ms  Boxes={len(boxes)}')

        # 保存结果
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

        # NuScenes 官方评估
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


# ══════════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════════
def main():
    # 1. 初始化 ACL
    print('Initializing ACL...')
    ctx = init_acl(device_id=0)

    # 2. 初始化并行推理网络
    print('\nInitializing BEVFusionParallelNet...')
    net = BEVFusionParallelNet(
        lidar_om  = 'models/om_parallel/lidar_branch.om',
        camera_om = 'models/om_parallel/camera_branch.om',
        fusion_om = 'models/om_parallel/fusion_head.om',
        gears     = [6000, 8000, 10000],
        # BEV 网格参数（与训练配置、export 脚本保持一致）
        xbound=(-54.0, 54.0, 0.3),
        ybound=(-54.0, 54.0, 0.3),
        zbound=(-10.0, 10.0, 20.0),
        max_pts=16,
        D=118,
        feature_size=(32, 88),
        image_size=(256, 704),
        num_cameras=6,
    )

    # 3. 初始化评估器
    print('\nInitializing evaluator...')
    evaluator = BEVFusionParallelEvaluator(
        dataroot='data/nuscenes-mini',
        version='v1.0-mini',
    )

    # 4. 运行评估
    evaluator.evaluate(net, output_path='bevfusion_parallel_results.json')
    print('\nDone.')


if __name__ == '__main__':
    main()
