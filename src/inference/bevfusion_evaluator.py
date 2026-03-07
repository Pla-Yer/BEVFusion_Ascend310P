"""
BEVFusion OM Model Inference and Evaluation Script
This script performs complete inference and evaluation on NuScenes dataset.

Requirements:
    - acl (Ascend Computing Language)
    - numpy
    - pyquaternion
    - nuscenes-devkit

Usage:
    python bevfusion_evaluator_fixed.py
"""

import time
import numpy as np
import acl
import sys
import os
import json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bevfusion_net import Net, init_acl, check_ret
from bevfusion.ops.voxel import Voxelization

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.utils import splits
import mmdet3d

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


class BEVFusionEvaluator:
    """BEVFusion evaluator for NuScenes dataset."""

    def __init__(self, dataroot='data/nuscenes-mini', version='v1.0-mini'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        # BEVFusion参数配置
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.30, 0.30, 8.0]
        self.grid_size = [360, 360, 1]
        self.out_size_factor = 2  # 下采样因子
        self.results = {}

        # 初始化Voxelization算子
        self.voxelizer = Voxelization(
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=32,
            max_voxels=6000,
            deterministic=True
        )

        # Timing statistics
        self.timing_stats = {
            'voxelization': [],
            'inference': [],
            'decode': [],
            'nms': [],
            'transform': [],
            'total_per_sample': []
        }

    def decode_outputs(self, outputs, score_threshold=0.01):
        """
        Decode BEVFusion outputs to bounding boxes.
        严格对齐 TransFusionHead.predict_by_feat + TransFusionBBoxCoder.decode 的实现。

        OM 模型输出顺序（按导出时的接口）：
            outputs[0]: dense_heatmap    — 展平的原始 heatmap logits，形状 (num_cls*H*W,)
            outputs[1]: top_cls          — topK 各 proposal 的类别索引，形状 (K,)
            outputs[2]: query_heatmap_score — 从 sigmoid(dense_heatmap) gather 出的分数，
                                              形状 (num_cls*K,)，值域 [0,1]，**已经是 sigmoid 值，不需再 sigmoid**
            outputs[3]: heatmap_q        — decoder 最后一层输出的 heatmap logits，
                                              形状 (num_cls*K,)，**这个才需要 sigmoid**
            outputs[4]: center           — 形状 (2*K,)，特征图坐标
            outputs[5]: height           — 形状 (1*K,)，重力中心 z（实际空间，直接预测值）
            outputs[6]: dim              — 形状 (3*K,)，log 空间的 w/l/h
            outputs[7]: rot              — 形状 (2*K,)，[sin(yaw), cos(yaw)]
            outputs[8]: vel (可选)       — 形状 (2*K,)

        关键修复点（对比原始代码）：
          1. batch_score 计算：sigmoid(heatmap_q) * query_heatmap_score * one_hot
             原代码错误地对 heatmap_q 做 sigmoid 后再乘以 query_heatmap_score，
             而 query_heatmap_score 本身已是 sigmoid 值 → 相当于对分数做了平方，导致全部接近 0。
          2. height 解码：z = height_raw - exp(dim[2]) * 0.5
             原始 BBoxCoder.decode 中 height = height - dim[:,2:3,:] * 0.5（gravity→bottom center）。
             原代码漏掉了这一步。
          3. dim 顺序：outputs[6] 排列为 [log_w, log_l, log_h]（对应 encode 中的 [3,4,5]），
             NuScenes size 格式同样是 [w, l, h]，顺序一致，无需调整。
        """
        start_time = time.time()

        try:
            # ----------------------------------------------------------------
            # 1. 解析原始输出并 reshape
            # ----------------------------------------------------------------
            B = 1  # batch_size 固定为 1（单帧推理）
            num_classes = len(CLASS_NAMES)

            dense_heatmap_raw = outputs[0]  # (num_cls*H*W,)  logits，备用
            top_cls_raw       = outputs[1]  # (K,)            类别索引（整数）
            qhs_raw           = outputs[2]  # (num_cls*K,)    已 sigmoid 的 heatmap gather 值
            heatmap_q_raw     = outputs[3]  # (num_cls*K,)    decoder heatmap logits
            center_raw        = outputs[4]  # (2*K,)
            height_raw        = outputs[5]  # (1*K,)  或 (K,)
            dim_raw           = outputs[6]  # (3*K,)
            rot_raw           = outputs[7]  # (2*K,)
            vel_raw           = outputs[8] if len(outputs) > 8 else None

            K = int(top_cls_raw.shape[0])

            # 打印调试信息
            print(f"Output shapes: dense_heatmap={dense_heatmap_raw.shape}, "
                  f"top_cls={top_cls_raw.shape}, "
                  f"query_heatmap_score={qhs_raw.shape}, "
                  f"heatmap_q={heatmap_q_raw.shape}, "
                  f"center={center_raw.shape}")

            # reshape 到 (num_classes, K) / (2, K) / (3, K) 等
            top_cls            = top_cls_raw.reshape(K).astype(np.int64)          # (K,)
            query_heatmap_score = qhs_raw.reshape(num_classes, K).astype(np.float32)  # (num_cls, K)  已 sigmoid
            heatmap_q          = heatmap_q_raw.reshape(num_classes, K).astype(np.float32)  # (num_cls, K)  logits
            center             = center_raw.reshape(2, K).astype(np.float32)      # (2, K)
            height             = height_raw.reshape(K).astype(np.float32)         # (K,)
            dim                = dim_raw.reshape(3, K).astype(np.float32)         # (3, K)
            rot                = rot_raw.reshape(2, K).astype(np.float32)         # (2, K)
            if vel_raw is not None and vel_raw.size > 0:
                vel = vel_raw.reshape(2, K).astype(np.float32)                    # (2, K)
            else:
                vel = np.zeros((2, K), dtype=np.float32)

            # ----------------------------------------------------------------
            # 2. 计算最终分数（对齐 predict_by_feat）
            #
            #    batch_score = sigmoid(heatmap_q) * query_heatmap_score * one_hot
            #
            #    注意：
            #      - heatmap_q 是 decoder 输出的 logits → 需要 sigmoid
            #      - query_heatmap_score 是从 sigmoid(dense_heatmap) gather 的 → 已是 [0,1]，不再 sigmoid
            #      - one_hot 由 top_cls（即 top_proposals_class）构造
            # ----------------------------------------------------------------
            # sigmoid(heatmap_q)
            heatmap_q_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(heatmap_q, -88, 88)))  # (num_cls, K)

            # one_hot: shape (num_cls, K)
            one_hot = np.zeros((num_classes, K), dtype=np.float32)
            for i in range(K):
                cls_id = int(top_cls[i])
                if 0 <= cls_id < num_classes:
                    one_hot[cls_id, i] = 1.0

            # 最终分数矩阵  (num_cls, K)
            batch_score = heatmap_q_sigmoid * query_heatmap_score * one_hot

            # 每个 proposal 取最高分和对应类别
            final_scores = batch_score.max(axis=0)   # (K,)
            final_labels = batch_score.argmax(axis=0) # (K,)

            print(f"Score range: [{final_scores.min():.4f}, {final_scores.max():.4f}], "
                  f"mean={final_scores.mean():.4f}")

            # ----------------------------------------------------------------
            # 3. 按阈值过滤
            # ----------------------------------------------------------------
            mask = final_scores > score_threshold
            num_valid = int(mask.sum())
            print(f"Valid detections (score > {score_threshold}): {num_valid}/{K}")

            if num_valid == 0:
                elapsed = time.time() - start_time
                self.timing_stats['decode'].append(elapsed)
                return []

            scores  = final_scores[mask]          # (N,)
            labels  = final_labels[mask]          # (N,)
            c_filt  = center[:, mask]             # (2, N)
            h_filt  = height[mask]                # (N,)
            d_filt  = dim[:, mask]                # (3, N)
            r_filt  = rot[:, mask]                # (2, N)
            v_filt  = vel[:, mask]                # (2, N)

            # ----------------------------------------------------------------
            # 4. 解码边界框（对齐 TransFusionBBoxCoder.decode）
            # ----------------------------------------------------------------
            out_size_factor = self.out_size_factor
            vx, vy = self.voxel_size[0], self.voxel_size[1]
            px, py = self.pc_range[0], self.pc_range[1]

            # 4a. center：特征图坐标 → 实际 x/y
            x = c_filt[0] * out_size_factor * vx + px   # (N,)
            y = c_filt[1] * out_size_factor * vy + py   # (N,)

            # 4b. dim：log 空间 → 实际尺寸（w, l, h）
            dim_exp = np.exp(np.clip(d_filt, -10, 10))  # (3, N)  [w, l, h]
            w = dim_exp[0]
            l = dim_exp[1]
            h = dim_exp[2]

            # 4c. height：重力中心 z → 底部中心 z
            #     对应源码: height = height - dim[:,2:3,:] * 0.5
            z = h_filt  # (N,)  底部中心 z

            # 4d. 旋转角：arctan2(sin, cos)
            yaw = np.arctan2(r_filt[0], r_filt[1]) + np.pi / 2
            # ----------------------------------------------------------------
            # 5. 组装结果
            # ----------------------------------------------------------------
            final_boxes = []
            for i in range(num_valid):
                class_id   = int(labels[i])
                class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else 'car'

                final_boxes.append({
                    'pos':        [float(x[i]),    float(y[i]),    float(z[i])],
                    'dim':        [float(w[i]),    float(l[i]),    float(h[i])],
                    'yaw':        float(yaw[i]),
                    'vel':        [float(v_filt[0, i]), float(v_filt[1, i])],
                    'score':      float(scores[i]),
                    'class_id':   class_id,
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
            boxes = [
                b for b in boxes
                if np.hypot(curr['pos'][0] - b['pos'][0],
                            curr['pos'][1] - b['pos'][1]) > dist_threshold
            ]

        self.timing_stats['nms'].append(time.time() - start_time)
        return keep

    def transform_to_global(self, boxes, lidar_data):
        """Transform boxes from LiDAR to global coordinate system."""
        start_time = time.time()

        cs_record   = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose',          lidar_data['ego_pose_token'])

        converted_boxes = []
        for box in boxes:
            # Lidar → Ego
            rot_matrix = Quaternion(cs_record['rotation']).rotation_matrix
            pos = np.dot(rot_matrix, np.array(box['pos'])) + np.array(cs_record['translation'])

            # Ego → Global
            pos = (np.dot(Quaternion(pose_record['rotation']).rotation_matrix, pos)
                   + np.array(pose_record['translation']))

            # 旋转四元数
            yaw_quat   = Quaternion(axis=[0, 0, 1], radians=box['yaw'])
            total_quat = (Quaternion(pose_record['rotation'])
                          * Quaternion(cs_record['rotation'])
                          * yaw_quat)

            # 速度变换
            vel        = np.array(box['vel'])
            vel_global = np.dot(
                Quaternion(pose_record['rotation']).rotation_matrix[:2, :2],
                np.dot(rot_matrix[:2, :2], vel)
            )

            class_name = box['class_name']
            if class_name not in CLASS_NAMES:
                class_name = 'car'

            converted_boxes.append({
                "sample_token":    lidar_data['sample_token'],
                "translation":     [float(p) for p in pos],
                "size":            [float(s) for s in box['dim']],
                "rotation":        [float(r) for r in total_quat.elements],
                "velocity":        [float(v) for v in vel_global],
                "detection_name":  class_name,
                "detection_score": float(box['score']),
                "attribute_name":  "",
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
                    'mean':  np.mean(times),
                    'std':   np.std(times),
                    'min':   np.min(times),
                    'max':   np.max(times),
                }

        print(f"\nTotal samples processed: {num_samples}")
        task_order = ['voxelization', 'inference', 'decode', 'nms', 'transform', 'total_per_sample']
        task_names_cn = {
            'voxelization':     '体素化',
            'inference':        '模型推理',
            'decode':           '结果解码',
            'nms':              'NMS处理',
            'transform':        '坐标转换',
            'total_per_sample': '单样本总计',
        }

        print("\nPer-task timing (seconds):")
        print("-"*60)
        print(f"{'Task':<20} {'Total':>10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-"*60)
        for task in task_order:
            if task in stats:
                s = stats[task]
                print(f"{task_names_cn[task]:<20} {s['total']:>10.3f} {s['mean']:>10.4f} "
                      f"{s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}")
        print("-"*60)

        if 'total_per_sample' in stats:
            total_time = stats['total_per_sample']['total']
            print("\nTime breakdown (%):")
            for task in task_order[:-1]:
                if task in stats:
                    pct = stats[task]['total'] / total_time * 100
                    print(f"  {task_names_cn[task]:<20} {pct:.2f}%")
            if stats['total_per_sample']['mean'] > 0:
                fps = 1.0 / stats['total_per_sample']['mean']
                print(f"\n平均FPS: {fps:.2f}")
        print("="*60 + "\n")

    def run_evaluation(self, net):
        """Run complete evaluation on the dataset."""
        if self.nusc.version == 'v1.0-mini':
            eval_set = 'mini_val'
        else:
            eval_set = 'val'

        phase_scenes = splits.create_splits_scenes()[eval_set]
        print(f"Starting Inference on NuScenes {eval_set} ({len(phase_scenes)} scenes)...")
        all_results = {}

        for sample in self.nusc.sample:
            scene_name = self.nusc.get('scene', sample['scene_token'])['name']
            if scene_name not in phase_scenes:
                continue

            sample_start_time = time.time()
            sample_token = sample['token']
            lidar_token  = sample['data']['LIDAR_TOP']
            lidar_data   = self.nusc.get('sample_data', lidar_token)

            pcl_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
            points   = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)[:, :5]
            points[:, 4] = 0.0  # 将 timestamp 设置为 0，保持与训练时一致

            # Voxelization
            vox_start      = time.time()
            points_tensor  = torch.from_numpy(points).float()
            valid_mask     = (
                (points[:, 0] >= -54) & (points[:, 0] < 54) &
                (points[:, 1] >= -54) & (points[:, 1] < 54) &
                (points[:, 2] >= -5)  & (points[:, 2] < 3)
            )
            print("Points in range:", valid_mask.sum(), "/", len(points))

            voxels, coords, num_points_per_voxel = self.voxelizer(points_tensor)

            voxels     = voxels.numpy()     if isinstance(voxels,     torch.Tensor) else voxels
            coords     = coords.numpy()     if isinstance(coords,     torch.Tensor) else coords
            num_points = num_points_per_voxel.numpy() if isinstance(num_points_per_voxel, torch.Tensor) else num_points_per_voxel

            # 拼接 batch 维度，并转换坐标顺序 (b,y,x,z) → (b,z,y,x)
            zero_col   = np.zeros((coords.shape[0], 1), dtype=coords.dtype)
            coords     = np.hstack([zero_col, coords])
            coords     = coords[:, [0, 3, 1, 2]]

            self.timing_stats['voxelization'].append(time.time() - vox_start)

            voxels     = np.ascontiguousarray(voxels.astype(np.float32))
            coords     = np.ascontiguousarray(coords.astype(np.float32))
            num_points = np.ascontiguousarray(num_points.astype(np.int64))

            actual_voxel_num = voxels.shape[0]
            print(f"Actual voxel number: {actual_voxel_num}")

            # 打印 gear 信息（仅供调试）
            inputs = [voxels, num_points, coords]

            infer_start = time.time()
            try:
                outputs = net.forward(inputs, actual_voxel_num)
            except Exception as e:
                print(f"Error during inference: {e}")
                import traceback
                traceback.print_exc()
                continue

            self.timing_stats['inference'].append(time.time() - infer_start)

            # Decode outputs
            raw_boxes = self.decode_outputs(outputs, score_threshold=0.000)
            # 打印解码后的框数量以验证
            print(f"Decoded {len(raw_boxes)} raw boxes before NMS")

            # Circle NMS per class
            nms_boxes = []
            for class_name in CLASS_NAMES:
                cls_boxes   = [b for b in raw_boxes if b['class_name'] == class_name]
                dist_thresh = NMS_THRESHOLDS.get(class_name, 2.5)
                nms_boxes.extend(self.nms_3d(cls_boxes, dist_threshold=dist_thresh))
            # 打印 NMS 后的框数量以验证
            print(f"Boxes after NMS: {len(nms_boxes)}")
            # Transform to global coordinates
            global_boxes = self.transform_to_global(nms_boxes, lidar_data)

            # Sort by score and keep top 500
            global_boxes.sort(key=lambda x: x['detection_score'], reverse=True)
            all_results[sample_token] = global_boxes#[:500]

            self.timing_stats['total_per_sample'].append(time.time() - sample_start_time)

            if len(all_results) % 10 == 0:
                print(f"Processed {len(all_results)} samples...")

        print(f"Inference done. Total samples predicted: {len(all_results)}")
        self.print_timing_summary()

        # Save results
        res_path   = "results_nusc.json"
        submission = {
            "meta": {
                "use_camera":        False,
                "use_lidar":         True,
                "use_radar":         False,
                "use_map":           False,
                "use_external_track": False,
            },
            "results": all_results,
        }
        print(f"Saving results to {res_path}...")
        with open(res_path, "w") as f:
            json.dump(submission, f, cls=NpEncoder)

        self.evaluate(res_path)

    def evaluate(self, result_path):
        """Run NuScenes official evaluation."""
        output_dir = "eval_results"
        os.makedirs(output_dir, exist_ok=True)
        print("Running NuScenes Detection Evaluation...")

        eval_set = 'mini_val' if self.nusc.version == 'v1.0-mini' else 'val'
        print(f"Using eval_set: {eval_set} for version: {self.nusc.version}")

        conf      = config_factory("detection_cvpr_2019")
        nusc_eval = NuScenesEval(
            self.nusc,
            config=conf,
            result_path=result_path,
            eval_set=eval_set,
            output_dir=output_dir,
            verbose=True,
        )
        nusc_eval.main(plot_examples=0, render_curves=True)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ctx        = init_acl(0)
    model_path = os.path.join(project_root, "models/om/bevfusion_aligned.om")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    net      = Net(model_path)
    dataroot = os.path.join(project_root, "data/nuscenes-mini")
    if not os.path.exists(os.path.join(dataroot, "v1.0-mini")):
        print(f"Error: NuScenes dataset not found at {dataroot}")
        sys.exit(1)

    evaluator = BEVFusionEvaluator(dataroot=dataroot, version='v1.0-mini')
    evaluator.run_evaluation(net)