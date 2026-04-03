"""
BEVFusion 并行 NPU 评估脚本
================================================================================
在原 full-NPU evaluator 的基础上，增加两类优化：
  1. NPU 侧：配合 bevfusion_parallel_npu_net.py，LiDAR / Camera 两个 OM 双 stream 并发。
  2. CPU 侧：点云加载 vs 图像加载并行；体素化 vs 深度图预计算并行；首帧 geometry 单独计算，
     避免重复调用 precompute_depth_and_geometry 导致深度图重复计算。
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
import torch

from bevfusion_parallel_npu_net import BEVFusionParallelNPUNet, init_acl
from bevfusion_full_npu_evaluator import BEVFusionFullNPUEvaluator


class BEVFusionParallelNPUEvaluator(BEVFusionFullNPUEvaluator):
    """基于原 evaluator 的轻量派生版，只重写调度逻辑。"""

    def _voxelize_points(self, points: np.ndarray):
        pts_t = torch.from_numpy(points).float()
        voxels_t, coords_t, npts_t = self.voxelizer(pts_t)
        voxels = voxels_t.numpy().astype(np.float32)
        npts = npts_t.numpy().astype(np.float32)
        zero_col = np.zeros((coords_t.shape[0], 1), dtype=np.float32)
        coords = np.hstack([zero_col, coords_t.numpy()])[:, [0, 3, 1, 2]]
        return (
            np.ascontiguousarray(voxels),
            np.ascontiguousarray(npts),
            np.ascontiguousarray(coords.astype(np.float32)),
        )

    def precompute_geometry_only(self, metas: dict, B: int = 1, N: int = 6):
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

    def _run_parallel_prep(
        self,
        sample_token: str,
        lidar_data: dict,
        warmup_geometry: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
        """
        两阶段并行：
          Stage-A: 点云加载  || 图像加载+标定装配
          Stage-B: 体素化    || 深度图预计算 || (首帧) geometry 计算
        """
        # ── Stage-A ──────────────────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=2) as exe:
            fut_points = exe.submit(self.load_points_with_sweeps, lidar_data, 9)
            fut_imgs = exe.submit(self.load_images_and_metas, sample_token, lidar_data)

            t0 = time.time()
            points = fut_points.result()
            sweep_elapsed = time.time() - t0

            t0 = time.time()
            imgs, metas, lidar2image_np = fut_imgs.result()
            img_elapsed = time.time() - t0

        # 这里记录“各自任务本身”的时间更合理，所以重新单独测一遍包装时间。
        # 为了不重复跑函数，使用 wall clock 近似会偏小；因此改为显式包裹 task。
        # 实际统计在 evaluate() 中通过 task wrapper 写入，这里只负责返回数据。
        _ = sweep_elapsed
        _ = img_elapsed

        # ── Stage-B ──────────────────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=3 if warmup_geometry else 2) as exe:
            fut_vox = exe.submit(self._voxelize_points, points)
            fut_depth = exe.submit(self._compute_depth_only, points, lidar2image_np)
            fut_geom = exe.submit(self.precompute_geometry_only, metas, 1, len(self.cam_keys)) \
                if warmup_geometry else None

            voxels, npts, coords = fut_vox.result()
            depth_np = fut_depth.result()
            geom_np = fut_geom.result() if fut_geom is not None else None

        return points, voxels, npts, coords, imgs, depth_np, metas, geom_np

    def evaluate(self, net: BEVFusionParallelNPUNet,
                 output_path: str = 'results_parallel_npu.json'):
        print('\n' + '=' * 62)
        print('BEVFusion Parallel-NPU Evaluation Start')
        print('=' * 62)

        split_name = 'mini_val' if 'mini' in self.nusc.version else 'val'
        val_scenes = set(self.nusc.get('scene', s['scene_token'])['name'] for s in self.nusc.sample)
        from nuscenes.utils.splits import create_splits_scenes
        valid_names = set(create_splits_scenes()[split_name])
        val_samples = [
            s for s in self.nusc.sample
            if self.nusc.get('scene', s['scene_token'])['name'] in valid_names
        ]
        print(f'Validation samples: {len(val_samples)}')

        lut_warmed = False

        for idx, sample in enumerate(val_samples):
            print(f'\n[{idx+1}/{len(val_samples)}] {sample["token"]}')
            t_total0 = time.time()
            lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

            # ── Stage-A：点云加载 / 图像加载并行 ──────────────────────────
            def _points_task():
                t0 = time.time()
                out = self.load_points_with_sweeps(lidar_data, sweeps_num=9)
                return out, time.time() - t0

            def _imgs_task():
                t0 = time.time()
                out = self.load_images_and_metas(sample['token'], lidar_data)
                return out, time.time() - t0

            with ThreadPoolExecutor(max_workers=2) as exe:
                fut_points = exe.submit(_points_task)
                fut_imgs = exe.submit(_imgs_task)
                points, t_sweep = fut_points.result()
                (imgs, metas, lidar2image_np), t_imgs = fut_imgs.result()

            self.timing_stats['sweep_loading'].append(t_sweep)
            self.timing_stats['image_loading'].append(t_imgs)
            print(f'  Points: {points.shape[0]}')

            # ── Stage-B：体素化 / depth / (首帧)geometry 并行 ─────────────
            def _vox_task():
                t0 = time.time()
                out = self._voxelize_points(points)
                return out, time.time() - t0

            def _depth_task():
                t0 = time.time()
                out = self._compute_depth_only(points, lidar2image_np)
                return out, time.time() - t0

            def _geom_task():
                t0 = time.time()
                out = self.precompute_geometry_only(metas, B=1, N=len(self.cam_keys))
                return out, time.time() - t0

            with ThreadPoolExecutor(max_workers=3 if not lut_warmed else 2) as exe:
                fut_vox = exe.submit(_vox_task)
                fut_depth = exe.submit(_depth_task)
                fut_geom = exe.submit(_geom_task) if not lut_warmed else None

                (voxels, npts, coords), t_vox = fut_vox.result()
                depth_np, t_depth = fut_depth.result()
                if fut_geom is not None:
                    geom_np, t_geom = fut_geom.result()
                else:
                    geom_np, t_geom = None, 0.0

            self.timing_stats['voxelization'].append(t_vox)
            self.timing_stats['precompute_depth'].append(t_depth)
            self.timing_stats['precompute_geometry'].append(t_geom)
            print(f'  Voxels: {voxels.shape[0]}')

            # ── 首帧一次性 LUT 预热 ───────────────────────────────────────
            if not lut_warmed:
                net.precompute_lut(geom_np)
                lut_warmed = True
                print(f'  geom: {geom_np.shape}  [LUT precomputed and cached]')

            # ── NPU 并行推理 ───────────────────────────────────────────────
            t0 = time.time()
            outputs = net.forward(
                voxels=voxels,
                num_points=npts,
                coords=coords,
                imgs=imgs,
                depth=depth_np,
            )
            self.timing_stats['npu_inference'].append(time.time() - t0)

            boxes = self.decode_outputs(outputs, score_threshold=0.01)
            boxes = self.nms_3d(boxes, default_threshold=0.5)
            boxes = self.transform_to_global(boxes, lidar_data)
            boxes.sort(key=lambda x: x['detection_score'], reverse=True)
            self.results[sample['token']] = boxes

            sample_total = time.time() - t_total0
            self.timing_stats['total_per_sample'].append(sample_total)
            print(f'  Total={sample_total*1000:.1f}ms  Boxes={len(boxes)}')

        submission = {
            'meta': {
                'use_camera': True, 'use_lidar': True,
                'use_radar': False, 'use_map': False,
                'use_external_track': False,
            },
            'results': self.results,
        }
        import json
        from bevfusion_full_npu_evaluator import NpEncoder
        with open(output_path, 'w') as f:
            json.dump(submission, f, cls=NpEncoder, indent=2)
        print(f'\nResults saved: {output_path}')

        self._print_timing_summary()

        print('\nRunning NuScenes official evaluation...')
        try:
            os.makedirs('eval_output', exist_ok=True)
            from nuscenes.eval.detection.evaluate import NuScenesEval
            from nuscenes.eval.detection.config import config_factory
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
            import traceback
            traceback.print_exc()

        print('\nNet timing:')
        net.print_timing_summary()


def main():
    parser = argparse.ArgumentParser(description='BEVFusion 并行 NPU evaluator')
    parser.add_argument('--dataroot', default='data/nuscenes-mini')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--lidar-om', default='models/om/bevfusion_lidar_branch_dynamic.om')
    parser.add_argument('--camera-om', default='models/om/bevfusion_camera_branch.om')
    parser.add_argument('--fusion-om', default='models/om/bevfusion_fusion_head.om')
    parser.add_argument('--output', default='bevfusion_parallel_npu_results.json')
    parser.add_argument('--max-pts', type=int, default=16)
    parser.add_argument('--launch-order', default='lidar_first', choices=['lidar_first', 'camera_first'])
    args = parser.parse_args()

    print('Initializing ACL...')
    _ = init_acl(device_id=args.device_id)

    print('\nInitializing BEVFusionParallelNPUNet...')
    net = BEVFusionParallelNPUNet(
        lidar_model_path=args.lidar_om,
        camera_model_path=args.camera_om,
        fusion_model_path=args.fusion_om,
        gears=[6000, 8000, 10000],
        D=118,
        feature_size=[32, 88],
        image_size=[256, 704],
        num_cameras=6,
        xbound=(-54.0, 54.0, 0.3),
        ybound=(-54.0, 54.0, 0.3),
        zbound=(-10.0, 10.0, 20.0),
        max_pts=args.max_pts,
        launch_order=args.launch_order,
    )

    print('\nInitializing evaluator...')
    evaluator = BEVFusionParallelNPUEvaluator(
        dataroot=args.dataroot,
        version=args.version,
    )

    evaluator.evaluate(net, output_path=args.output)
    print('\nDone.')


if __name__ == '__main__':
    main()
