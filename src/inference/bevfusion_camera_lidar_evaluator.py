
"""
BEVFusion Camera+Lidar OM Model Inference and Evaluation Script
This script performs complete inference and evaluation on NuScenes dataset.

Architecture:
    1. Lidar Branch (NPU): Voxelization -> VoxelEncoder -> Scatter -> Backbone -> Neck
    2. Camera Branch (NPU): Backbone -> Neck -> DepthNet
    3. CPU Parts: Depth generation, Geometry calculation, BEV Pool
    4. Fusion + Detection Head (NPU): Fusion -> Backbone -> Neck -> Head

Requirements:
    - acl (Ascend Computing Language)
    - numpy
    - pyquaternion
    - nuscenes-devkit
    - torch (for CPU parts)

Usage:
    python bevfusion_camera_lidar_evaluator.py
"""

import time
import numpy as np
import torch
import acl
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bevfusion_camera_lidar_net import BEVFusionCameraLidarNet, init_acl, check_ret
from bevfusion.ops.voxel import Voxelization

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.utils import splits
import cv2

_MEAN_T = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
_STD_T  = torch.tensor([58.395,  57.12,  57.375], dtype=torch.float32)
# =============================================================================
# Class Definitions
# =============================================================================
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Recommended NMS thresholds per class (meters)
NMS_THRESHOLDS = {
    "car": 0,
    "truck": 0,
    "construction_vehicle": 0,
    "bus": 0,
    "trailer": 0,
    "barrier": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "pedestrian": 0.175,
    "traffic_cone": 0.175
}


class NpEncoder(json.JSONEncoder):
    """JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)


class BEVFusionCameraLidarEvaluator:
    """BEVFusion Camera+Lidar evaluator for NuScenes dataset."""

    def __init__(self, dataroot='data/nuscenes-mini', version='v1.0-mini'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        
        # BEVFusion parameters
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.30, 0.30, 8.0]
        self.grid_size = [360, 360, 1]
        self.out_size_factor = 2
        
        # Camera parameters
        self.image_size = [256, 704]
        self.feature_size = [32, 88]
        self.xbound = [-54.0, 54.0, 0.3]
        self.ybound = [-54.0, 54.0, 0.3]
        self.zbound = [-10.0, 10.0, 20.0]
        self.dbound = [1.0, 60.0, 0.5]
        
        self.results = {}

        # Initialize Voxelization operator
        self.voxelizer = Voxelization(
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=32,
            max_voxels=10000,
            deterministic=True
        )

        # Timing statistics
        self.timing_stats = {
            'sweep_loading': [],
            'voxelization': [],
            'lidar_inference': [],
            'image_loading': [],
            'cpu_parts': [],
            'fusion_inference': [],
            'decode': [],
            'nms': [],
            'transform': [],
            'total_per_sample': []
        }

    # =========================================================================
    # Sweep Point Cloud Aggregation
    # =========================================================================

    @staticmethod
    def _make_tf(rotation_matrix, translation):
        """Construct 4x4 homogeneous transformation matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation
        return T

    def load_points_with_sweeps(self, lidar_data, sweeps_num=9, remove_close_radius=1.0):
        """
        Load current keyframe and aggregate historical sweeps.
        (Same as lidar-only version)
        """
        nusc = self.nusc

        # Current keyframe lidar2global transform
        cs_ref = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose_ref = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        R_ego_from_lidar_ref = Quaternion(cs_ref['rotation']).rotation_matrix
        t_ego_from_lidar_ref = np.array(cs_ref['translation'])
        R_global_from_ego_ref = Quaternion(pose_ref['rotation']).rotation_matrix
        t_global_from_ego_ref = np.array(pose_ref['translation'])

        T_ref = self._make_tf(
            R_global_from_ego_ref @ R_ego_from_lidar_ref,
            R_global_from_ego_ref @ t_ego_from_lidar_ref + t_global_from_ego_ref
        )
        T_ref_inv = np.linalg.inv(T_ref)

        ref_timestamp = lidar_data['timestamp']

        # Load current keyframe
        pcl_path = os.path.join(nusc.dataroot, lidar_data['filename'])
        pts_current = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)[:, :5]
        pts_current[:, 4] = 0.0

        if remove_close_radius > 0:
            dist_xy = np.linalg.norm(pts_current[:, :2], axis=1)
            pts_current = pts_current[dist_xy >= remove_close_radius]

        all_points = [pts_current]
        print(f"  [Sweep] keyframe: {pts_current.shape[0]} pts")

        # Load historical sweeps
        cur_sd_token = lidar_data['prev']
        loaded_sweeps = 0

        while cur_sd_token != '' and loaded_sweeps < sweeps_num:
            sweep_sd = nusc.get('sample_data', cur_sd_token)

            cs_sweep = nusc.get('calibrated_sensor', sweep_sd['calibrated_sensor_token'])
            pose_sweep = nusc.get('ego_pose', sweep_sd['ego_pose_token'])

            R_ego_from_lidar_sw = Quaternion(cs_sweep['rotation']).rotation_matrix
            t_ego_from_lidar_sw = np.array(cs_sweep['translation'])
            R_global_from_ego_sw = Quaternion(pose_sweep['rotation']).rotation_matrix
            t_global_from_ego_sw = np.array(pose_sweep['translation'])

            T_sweep = self._make_tf(
                R_global_from_ego_sw @ R_ego_from_lidar_sw,
                R_global_from_ego_sw @ t_ego_from_lidar_sw + t_global_from_ego_sw
            )

            T_rel = T_ref_inv @ T_sweep
            time_lag = (ref_timestamp - sweep_sd['timestamp']) * 1e-6

            sweep_path = os.path.join(nusc.dataroot, sweep_sd['filename'])
            pts_sweep = np.fromfile(sweep_path, dtype=np.float32).reshape(-1, 5)[:, :5]

            if remove_close_radius > 0:
                dist_xy = np.linalg.norm(pts_sweep[:, :2], axis=1)
                pts_sweep = pts_sweep[dist_xy >= remove_close_radius]

            pts_xyz1 = np.hstack([pts_sweep[:, :3], np.ones((pts_sweep.shape[0], 1), dtype=np.float32)])
            pts_xyz_ref = (T_rel @ pts_xyz1.T).T[:, :3].astype(np.float32)

            pts_sweep_ref = np.hstack([
                pts_xyz_ref,
                pts_sweep[:, 3:4],
                np.full((pts_sweep.shape[0], 1), time_lag, dtype=np.float32)
            ])

            all_points.append(pts_sweep_ref)
            loaded_sweeps += 1
            print(f"  [Sweep] sweep -{loaded_sweeps}: {pts_sweep_ref.shape[0]} pts, time_lag={time_lag:.4f}s")

            cur_sd_token = sweep_sd['prev']

        if loaded_sweeps < sweeps_num:
            print(f"  [Sweep] Warning: only found {loaded_sweeps} sweeps (expected {sweeps_num})")

        points = np.concatenate(all_points, axis=0).astype(np.float32)
        print(f"  [Sweep] Aggregated: {len(all_points)} frames, total {points.shape[0]} pts")
        return points
    def load_and_preprocess_image(self, img_path: str,
                               target_size=(256, 704)) -> np.ndarray:
        """
        读取单张图像，返回 [3, H, W] float32 归一化数组（BGR，与训练一致）。
        全程用 PyTorch 做浮点运算，规避 ascend 环境 numpy ufunc 路径的 bug。
        """
        # 确保img_path是字符串类型
        img_path = str(img_path)
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
    def load_images_and_metas(self, sample_token, lidar_data):
        """
        Load multi-view images and metadata for camera branch.
        
        Returns:
            imgs: [B, N, 3, H, W] images
            metas: dict containing transformation matrices
        """
        nusc = self.nusc
        sample = nusc.get('sample', sample_token)
        
        # Get camera sample_data tokens
        cam_keys = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        imgs = []
        lidar2image_list = []
        cam2img_list = []
        cam2lidar_list = []
        img_aug_matrix_list = []
        
        for cam_key in cam_keys:
            cam_data = nusc.get('sample_data', sample['data'][cam_key])
            
            # Load image
            img_path = os.path.join(nusc.dataroot, cam_data['filename'])
            # Note: In real deployment, you need to load and preprocess image
            # Here we use placeholder
            img = self.load_and_preprocess_image(img_path)
            imgs.append(img)
            
            # Get calibration
            cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
            
            # Camera intrinsics
            cam_intrinsic = np.eye(4, dtype=np.float32)
            cam_intrinsic[:3, :3] = np.array(cs_record['camera_intrinsic'])
            cam2img_list.append(cam_intrinsic)
            
            # Camera to ego
            R_cam2ego = Quaternion(cs_record['rotation']).rotation_matrix
            t_cam2ego = np.array(cs_record['translation'])
            
            # Ego to global
            R_ego2global = Quaternion(pose_record['rotation']).rotation_matrix
            t_ego2global = np.array(pose_record['translation'])
            
            # Lidar to ego
            cs_lidar = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            R_lidar2ego = Quaternion(cs_lidar['rotation']).rotation_matrix
            t_lidar2ego = np.array(cs_lidar['translation'])
            
            # Lidar to camera
            R_lidar2global = R_ego2global @ R_lidar2ego
            t_lidar2global = R_ego2global @ t_lidar2ego + t_ego2global
            
            R_cam2global = R_ego2global @ R_cam2ego
            t_cam2global = R_ego2global @ t_cam2ego + t_ego2global
            
            R_lidar2cam = np.linalg.inv(R_cam2global) @ R_lidar2global
            t_lidar2cam = np.linalg.inv(R_cam2global) @ (t_lidar2global - t_cam2global)
            
            # Lidar to image
            lidar2cam = np.eye(4, dtype=np.float32)
            lidar2cam[:3, :3] = R_lidar2cam
            lidar2cam[:3, 3] = t_lidar2cam
            lidar2image = cam_intrinsic @ lidar2cam
            lidar2image_list.append(lidar2image)
            
            # Camera to lidar
            cam2lidar = np.eye(4, dtype=np.float32)
            cam2lidar[:3, :3] = R_lidar2cam.T
            cam2lidar[:3, 3] = -R_lidar2cam.T @ t_lidar2cam
            cam2lidar_list.append(cam2lidar)
            
            # Image augmentation (identity for inference)
            img_aug = np.eye(4, dtype=np.float32)
            img_aug_matrix_list.append(img_aug)
        
        # Stack to batch format
        B = 1
        N = len(cam_keys)
        
        imgs = np.stack(imgs, axis=0).reshape(1, N, 3, *self.image_size)
        lidar2image = np.stack(lidar2image_list, axis=0).reshape(1, N, 4, 4)
        cam2img = np.stack(cam2img_list, axis=0).reshape(1, N, 4, 4)
        cam2lidar = np.stack(cam2lidar_list, axis=0).reshape(1, N, 4, 4)
        img_aug_matrix = np.stack(img_aug_matrix_list, axis=0).reshape(1, N, 4, 4)
        lidar_aug_matrix = np.eye(4, dtype=np.float32).reshape(1, 4, 4)

        # convert to tensor for CPU processing
        
        lidar_aug_matrix = torch.from_numpy(lidar_aug_matrix).to(torch.float32)
        cam2img = torch.from_numpy(cam2img).to(torch.float32)
        cam2lidar = torch.from_numpy(cam2lidar).to(torch.float32)
        lidar2image = torch.from_numpy(lidar2image).to(torch.float32)
        img_aug_matrix = torch.from_numpy(img_aug_matrix).to(torch.float32)
        

        
        metas = {
            'lidar2image': lidar2image,
            'cam2img': cam2img,
            'cam2lidar': cam2lidar,
            'img_aug_matrix': img_aug_matrix,
            'lidar_aug_matrix': lidar_aug_matrix,
            'points': []  # Will be filled with point cloud
        }
        
        return imgs, metas

    def decode_outputs(self, outputs, score_threshold=0.01):
        """
        Decode BEVFusion outputs to bounding boxes.
        (Same as lidar-only version)
        """
        start_time = time.time()

        try:
            B = 1
            num_classes = len(CLASS_NAMES)

            dense_heatmap_raw = outputs[0]
            top_cls_raw = outputs[1]
            qhs_raw = outputs[2]
            heatmap_q_raw = outputs[3]
            center_raw = outputs[4]
            height_raw = outputs[5]
            dim_raw = outputs[6]
            rot_raw = outputs[7]
            vel_raw = outputs[8] if len(outputs) > 8 else None

            K = 200

            print(f"Output shapes: dense_heatmap={dense_heatmap_raw.shape}, "
                  f"top_cls={top_cls_raw.shape}, "
                  f"query_heatmap_score={qhs_raw.shape}, "
                  f"heatmap_q={heatmap_q_raw.shape}, "
                  f"center={center_raw.shape}")

            top_cls = top_cls_raw.reshape(K).astype(np.int64)
            query_heatmap_score = qhs_raw.reshape(num_classes, K).astype(np.float32)
            heatmap_q = heatmap_q_raw.reshape(num_classes, K).astype(np.float32)
            center = center_raw.reshape(2, K).astype(np.float32)
            height = height_raw.reshape(K).astype(np.float32)
            dim = dim_raw.reshape(3, K).astype(np.float32)
            rot = rot_raw.reshape(2, K).astype(np.float32)
            if vel_raw is not None and vel_raw.size > 0:
                vel = vel_raw.reshape(2, K).astype(np.float32)
            else:
                vel = np.zeros((2, K), dtype=np.float32)

            # Calculate final scores
            heatmap_q_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(heatmap_q, -88, 88)))

            one_hot = np.zeros((num_classes, K), dtype=np.float32)
            for i in range(K):
                cls_id = int(top_cls[i])
                if 0 <= cls_id < num_classes:
                    one_hot[cls_id, i] = 1.0

            batch_score = heatmap_q_sigmoid * query_heatmap_score * one_hot

            final_scores = batch_score.max(axis=0)
            final_labels = batch_score.argmax(axis=0)

            print(f"Score range: [{final_scores.min():.4f}, {final_scores.max():.4f}], "
                  f"mean={final_scores.mean():.4f}")

            # Filter by threshold
            mask = final_scores > score_threshold
            num_valid = int(mask.sum())
            print(f"Valid detections (score > {score_threshold}): {num_valid}/{K}")

            if num_valid == 0:
                elapsed = time.time() - start_time
                self.timing_stats['decode'].append(elapsed)
                return []

            scores = final_scores[mask]
            labels = final_labels[mask]
            c_filt = center[:, mask]
            h_filt = height[mask]
            d_filt = dim[:, mask]
            r_filt = rot[:, mask]
            v_filt = vel[:, mask]

            # Decode bounding boxes
            out_size_factor = self.out_size_factor
            vx, vy = self.voxel_size[0], self.voxel_size[1]
            px, py = self.pc_range[0], self.pc_range[1]

            x = c_filt[0] * out_size_factor * vx + px
            y = c_filt[1] * out_size_factor * vy + py

            dim_exp = np.exp(np.clip(d_filt, -10, 10))
            w = dim_exp[0]
            l = dim_exp[1]
            h = dim_exp[2]

            z = h_filt

            yaw = np.arctan2(r_filt[0], r_filt[1]) + np.pi / 2

            # Assemble results
            final_boxes = []
            for i in range(num_valid):
                class_id = int(labels[i])
                class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else 'car'

                final_boxes.append({
                    'pos': [float(x[i]), float(y[i]), float(z[i])],
                    'dim': [float(w[i]), float(l[i]), float(h[i])],
                    'yaw': float(yaw[i]),
                    'vel': [float(v_filt[0, i]), float(v_filt[1, i])],
                    'score': float(scores[i]),
                    'class_id': class_id,
                    'class_name': class_name,
                })

            elapsed = time.time() - start_time
            self.timing_stats['decode'].append(elapsed)
            print(f"Decoded {len(final_boxes)} boxes")
            return final_boxes

        except Exception as e:
            print(f"Error in decode_outputs: {e}")
            import traceback
            traceback.print_exc()
            elapsed = time.time() - start_time
            self.timing_stats['decode'].append(elapsed)
            return []

    def nms_3d(self, boxes, dist_threshold=1.5):
        """Simple distance-based Circle NMS."""
        if len(boxes) == 0:
            return []

        start_time = time.time()
        boxes = sorted(boxes, key=lambda x: x['score'], reverse=True)
        keep = []

        while boxes:
            curr = boxes.pop(0)
            keep.append(curr)
            
            # Apply class-specific NMS threshold
            class_name = curr['class_name']
            threshold = NMS_THRESHOLDS.get(class_name, dist_threshold)
            
            boxes = [
                b for b in boxes
                if np.hypot(curr['pos'][0] - b['pos'][0],
                            curr['pos'][1] - b['pos'][1]) > threshold
            ]

        self.timing_stats['nms'].append(time.time() - start_time)
        return keep

    def transform_to_global(self, boxes, lidar_data):
        """Transform boxes from LiDAR to global coordinate system."""
        start_time = time.time()

        cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        converted_boxes = []
        for box in boxes:
            # Lidar → Ego
            rot_matrix = Quaternion(cs_record['rotation']).rotation_matrix
            pos = np.dot(rot_matrix, np.array(box['pos'])) + np.array(cs_record['translation'])

            # Ego → Global
            pos = (np.dot(Quaternion(pose_record['rotation']).rotation_matrix, pos)
                   + np.array(pose_record['translation']))

            # Rotation quaternion
            yaw_quat = Quaternion(axis=[0, 0, 1], radians=box['yaw'])
            total_quat = (Quaternion(pose_record['rotation'])
                          * Quaternion(cs_record['rotation'])
                          * yaw_quat)

            # Velocity transform
            vel = np.array(box['vel'])
            vel_global = np.dot(
                Quaternion(pose_record['rotation']).rotation_matrix[:2, :2],
                np.dot(rot_matrix[:2, :2], vel)
            )

            class_name = box['class_name']
            if class_name not in CLASS_NAMES:
                class_name = 'car'

            converted_boxes.append({
                "sample_token": lidar_data['sample_token'],
                "translation": [float(p) for p in pos],
                "size": [float(s) for s in box['dim']],
                "rotation": [float(r) for r in total_quat.elements],
                "velocity": [float(v) for v in vel_global],
                "detection_name": class_name,
                "detection_score": float(box['score']),
                "attribute_name": "",
            })

        self.timing_stats['transform'].append(time.time() - start_time)
        return converted_boxes

    def print_timing_summary(self):
        """Print timing statistics summary."""
        print("\n" + "="*60)
        print("TIMING STATISTICS SUMMARY")
        print("="*60)

        num_samples = len(self.timing_stats['total_per_sample'])
        stats = {}
        for task_name, times in self.timing_stats.items():
            if times:
                stats[task_name] = {
                    'total': sum(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                }

        print(f"\nTotal samples: {num_samples}")
        print(f"\n{'Task':<25} {'Total(s)':<10} {'Mean(ms)':<10} {'Std(ms)':<10} {'Min(ms)':<10} {'Max(ms)':<10}")
        print("-" * 75)

        for task_name, s in stats.items():
            print(f"{task_name:<25} {s['total']:<10.2f} {s['mean']*1000:<10.2f} "
                  f"{s['std']*1000:<10.2f} {s['min']*1000:<10.2f} {s['max']*1000:<10.2f}")

        print("="*60)

    def evaluate(self, net, output_path='results.json'):
        """
        Run evaluation on NuScenes validation set.
        
        Args:
            net: BEVFusionCameraLidarNet instance
            output_path: Path to save results
        """
        print("\n" + "="*60)
        print("Starting BEVFusion Camera+Lidar Evaluation")
        print("="*60)

        # Get validation samples for the correct split
        from nuscenes.utils.splits import create_splits_scenes
        split_name = 'mini_val' if 'mini' in self.nusc.version else 'val'
        val_scenes = set(create_splits_scenes()[split_name])
        val_samples = [
            samp for samp in self.nusc.sample
            if self.nusc.get('scene', samp['scene_token'])['name'] in val_scenes
        ]
        print(f"Total samples: {len(val_samples)}")

        for idx, sample in enumerate(val_samples):
            print(f"\n[{idx+1}/{len(val_samples)}] Processing sample: {sample['token']}")
            start_time = time.time()

            # Get LIDAR_TOP data
            lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

            # 1. Load point cloud with sweeps
            t0 = time.time()
            points = self.load_points_with_sweeps(lidar_data, sweeps_num=9)
            self.timing_stats['sweep_loading'].append(time.time() - t0)

            # 2. Voxelization
            t0 = time.time()
            points_tensor  = torch.from_numpy(points).float()
            voxels, coords, num_points_per_voxel = self.voxelizer(points_tensor)

            voxels     = voxels.numpy()     if isinstance(voxels,     torch.Tensor) else voxels
            coords     = coords.numpy()     if isinstance(coords,     torch.Tensor) else coords
            num_points = num_points_per_voxel.numpy() if isinstance(num_points_per_voxel, torch.Tensor) else num_points_per_voxel
            # Prepend batch dim and reorder (b,y,x,z) → (b,z,y,x)
            zero_col   = np.zeros((coords.shape[0], 1), dtype=coords.dtype)
            coords     = np.hstack([zero_col, coords])
            coords     = coords[:, [0, 3, 1, 2]]

            voxels     = np.ascontiguousarray(voxels.astype(np.float32))
            coords     = np.ascontiguousarray(coords.astype(np.float32))  
            num_points = np.ascontiguousarray(num_points.astype(np.int64))

            self.timing_stats['voxelization'].append(time.time() - t0)
            print(f"  Voxelization: {voxels.shape[0]} voxels")

            # 3. Load images and metadata
            t0 = time.time()
            imgs, metas = self.load_images_and_metas(sample['token'], lidar_data)
            metas['points'] = [torch.from_numpy(points[:, :3])]  # For depth generation
            self.timing_stats['image_loading'].append(time.time() - t0)   # FIX: correct bucket name

            # 4. Run inference
            t0 = time.time()
            outputs = net.forward(voxels, num_points, coords, imgs, metas)
            inference_time = time.time() - t0
            self.timing_stats['fusion_inference'].append(inference_time)
            print(f"  Inference time: {inference_time*1000:.2f}ms")

            # 5. Decode outputs
            boxes = self.decode_outputs(outputs, score_threshold=0.01)

            # 6. NMS
            boxes = self.nms_3d(boxes, dist_threshold=0.5)

            # 7. Transform to global
            boxes = self.transform_to_global(boxes, lidar_data)

            # Store results
            boxes.sort(key=lambda x: x['detection_score'], reverse=True)
            self.results[sample['token']] = boxes

            total_time = time.time() - start_time
            self.timing_stats['total_per_sample'].append(total_time)
            print(f"  Total time: {total_time*1000:.2f}ms, boxes: {len(boxes)}")

        # Print timing summary
        self.print_timing_summary()
        
        # Save results in NuScenes submission format
        submission = {
            "meta": {
                "use_camera":         True,
                "use_lidar":          True,
                "use_radar":          False,
                "use_map":            False,
                "use_external_track": False,
            },
            "results": self.results,
        }
        with open(output_path, 'w') as f:
            json.dump(submission, f, cls=NpEncoder, indent=2)
        print(f"\nResults saved to: {output_path}")

        # Run NuScenes evaluation
        print("\nRunning NuScenes evaluation...")
        try:
            output_dir = "eval_output"
            os.makedirs(output_dir, exist_ok=True)
            eval_cfg  = config_factory('detection_cvpr_2019')
            nusc_eval = NuScenesEval(
                self.nusc,
                config=eval_cfg,
                result_path=output_path,
                eval_set=split_name,
                output_dir=output_dir,
                verbose=True,
            )
            nusc_eval.main(plot_examples=0, render_curves=True)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

        # Print network timing summary
        print("\nNetwork timing summary:")
        net.print_timing_summary()


def main():
    """Main function."""
    # Initialize ACL
    print("Initializing ACL...")
    ctx = init_acl(0)

    # Initialize network
    print("\nInitializing BEVFusion Camera+Lidar network...")
    net = BEVFusionCameraLidarNet(
        lidar_model_path="models/om/lidar_feature_extractor.om",
        camera_backbone_path="models/om/stage1_backbone.om",
        camera_depthnet_path="models/om/stage3_depthnet.om",
        fusion_head_path="models/om/fusion_head.om",
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        image_size=[256, 704],
        feature_size=[32, 88],
        voxel_size=[0.3, 0.3, 8.0],
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        max_voxels=10000,
        gears=[6000, 8000, 10000]
    )

    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = BEVFusionCameraLidarEvaluator(
        dataroot='data/nuscenes-mini',
        version='v1.0-mini'
    )

    # Run evaluation
    evaluator.evaluate(net, output_path='bevfusion_camera_lidar_results.json')

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
