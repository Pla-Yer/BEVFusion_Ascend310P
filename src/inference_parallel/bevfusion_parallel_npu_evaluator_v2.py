"""
BEVFusion 并行 NPU 评估脚本（v2，pipeline 优化版）
================================================================================
在上一版 evaluator 的基础上，继续做三类“零精度风险”优化：

  1. Host 双缓冲（double buffering）
     - 预处理线程把下一帧结果先 stage 到 net 的空闲 host slot；
     - 主线程对当前 slot 做 NPU 推理时，下一帧可并行准备。

  2. 预处理-推理流水化（pipeline）
     - 通过 prefetch executor，把“下一帧”的点云加载 / 图像加载 / 体素化 /
       depth 预计算 与 “当前帧”的 NPU 推理 + 后处理重叠起来。

  3. 持久化线程池
     - 图像读取改为复用固定 image thread pool，避免每帧重复创建 6 个线程；
     - Stage-A/Stage-B 的 CPU 任务改为复用固定 prep thread pool。

注意：
  - 本版不改网络参数、不改量化策略，目标是先把 pipeline 压到更极致。
  - 精度路径与上一版一致；pool_mask 的 uint8 改动只发生在导出 / 运行时输入存储上。
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from pyquaternion import Quaternion

from bevfusion_parallel_npu_net_v2 import BEVFusionParallelNPUNet, init_acl
from bevfusion_full_npu_evaluator import (
    BEVFusionFullNPUEvaluator,
    NpEncoder,
    load_and_preprocess_image,
)


@dataclass
class PreparedSample:
    idx: int
    slot_id: int
    sample_token: str
    lidar_data: dict
    geom_np: Optional[np.ndarray]
    points_count: int
    voxels_count: int
    t_sweep: float
    t_imgs: float
    t_vox: float
    t_depth: float
    t_geom: float
    t_stage: float


class BEVFusionParallelPipelineNPUEvaluator(BEVFusionFullNPUEvaluator):
    """基于 full evaluator 的 pipeline 优化版。"""

    def __init__(self,
                 dataroot: str = 'data/nuscenes-mini',
                 version: str = 'v1.0-mini'):
        super().__init__(dataroot=dataroot, version=version)
        self._image_pool = ThreadPoolExecutor(max_workers=len(self.cam_keys))
        self._prep_pool = ThreadPoolExecutor(max_workers=3)

        if 'host_staging' not in self.timing_stats:
            self.timing_stats['host_staging'] = []
        if 'prefetch_wait' not in self.timing_stats:
            self.timing_stats['prefetch_wait'] = []

    def close(self):
        for name in ('_image_pool', '_prep_pool'):
            pool = getattr(self, name, None)
            if pool is not None:
                try:
                    pool.shutdown(wait=True)
                except Exception:
                    pass
                setattr(self, name, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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

    def load_images_and_metas(self, sample_token: str, lidar_data: dict):
        """
        复用持久化 image pool，避免每帧重复创建 6 个图像线程。
        其余逻辑保持与基类一致。
        """
        nusc = self.nusc
        sample = nusc.get('sample', sample_token)
        N = len(self.cam_keys)
        cache = self._calib_cache

        cam_data_list = [
            nusc.get('sample_data', sample['data'][ck]) for ck in self.cam_keys
        ]
        img_paths = [
            os.path.join(nusc.dataroot, cd['filename']) for cd in cam_data_list
        ]

        def _load(path):
            return load_and_preprocess_image(path, self.image_size)

        imgs_list = list(self._image_pool.map(_load, img_paths))

        pose_lidar = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        R_ego2g_lid = Quaternion(pose_lidar['rotation']).rotation_matrix
        t_ego2g_lid = np.array(pose_lidar['translation'], dtype=np.float64)

        R_l2ego = cache['R_l2ego']
        t_l2ego = cache['t_l2ego']
        R_l2g = R_ego2g_lid @ R_l2ego
        t_l2g = R_ego2g_lid @ t_l2ego + t_ego2g_lid

        lidar2image_list = []
        cam2img_list = []
        cam2lidar_list = []

        for i, cam_key in enumerate(self.cam_keys):
            cd = cam_data_list[i]
            pose_cam = nusc.get('ego_pose', cd['ego_pose_token'])
            R_ego2g_cam = Quaternion(pose_cam['rotation']).rotation_matrix
            t_ego2g_cam = np.array(pose_cam['translation'], dtype=np.float64)

            ccache = cache['cams'][cam_key]
            R_c2g = R_ego2g_cam @ ccache['R_c2ego']
            t_c2g = R_ego2g_cam @ ccache['t_c2ego'] + t_ego2g_cam

            R_c2g_inv = np.linalg.inv(R_c2g)
            R_l2c = R_c2g_inv @ R_l2g
            t_l2c = R_c2g_inv @ (t_l2g - t_c2g)

            K4 = ccache['K4']

            l2c = np.eye(4, dtype=np.float32)
            l2c[:3, :3] = R_l2c
            l2c[:3, 3] = t_l2c
            lidar2image_list.append((K4 @ l2c).astype(np.float32))

            c2l = np.eye(4, dtype=np.float32)
            c2l[:3, :3] = R_l2c.T
            c2l[:3, 3] = -R_l2c.T @ t_l2c
            cam2lidar_list.append(c2l)
            cam2img_list.append(K4.astype(np.float32))

        B = 1
        imgs = np.stack(imgs_list).reshape(B, N, 3, *self.image_size)
        lidar2image_np = np.stack(lidar2image_list).astype(np.float32)

        lidar2image_torch = torch.from_numpy(lidar2image_np.reshape(B, N, 4, 4))
        cam2img_torch = torch.from_numpy(np.stack(cam2img_list).reshape(B, N, 4, 4))
        cam2lidar_torch = torch.from_numpy(np.stack(cam2lidar_list).reshape(B, N, 4, 4))
        img_aug_torch = torch.from_numpy(
            np.tile(np.eye(4, dtype=np.float32), (B, N, 1, 1)))
        lidar_aug_torch = torch.from_numpy(
            np.eye(4, dtype=np.float32).reshape(B, 4, 4))

        metas = {
            'lidar2image': lidar2image_torch,
            'cam2img': cam2img_torch,
            'cam2lidar': cam2lidar_torch,
            'img_aug_matrix': img_aug_torch,
            'lidar_aug_matrix': lidar_aug_torch,
        }
        return imgs, metas, lidar2image_np

    def _prepare_and_stage_sample(
        self,
        idx: int,
        sample: dict,
        slot_id: int,
        need_geometry: bool,
        net: BEVFusionParallelNPUNet,
    ) -> PreparedSample:
        sample_token = sample['token']
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        def _points_task():
            t0 = time.time()
            out = self.load_points_with_sweeps(lidar_data, sweeps_num=9)
            return out, time.time() - t0

        def _imgs_task():
            t0 = time.time()
            out = self.load_images_and_metas(sample_token, lidar_data)
            return out, time.time() - t0

        fut_points = self._prep_pool.submit(_points_task)
        fut_imgs = self._prep_pool.submit(_imgs_task)
        points, t_sweep = fut_points.result()
        (imgs, metas, lidar2image_np), t_imgs = fut_imgs.result()

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

        fut_vox = self._prep_pool.submit(_vox_task)
        fut_depth = self._prep_pool.submit(_depth_task)
        fut_geom = self._prep_pool.submit(_geom_task) if need_geometry else None

        (voxels, npts, coords), t_vox = fut_vox.result()
        depth_np, t_depth = fut_depth.result()
        if fut_geom is not None:
            geom_np, t_geom = fut_geom.result()
        else:
            geom_np, t_geom = None, 0.0

        stage_info = net.stage_host_inputs(slot_id, voxels, npts, coords, imgs, depth_np)

        return PreparedSample(
            idx=idx,
            slot_id=stage_info['slot_id'],
            sample_token=sample_token,
            lidar_data=lidar_data,
            geom_np=geom_np,
            points_count=int(points.shape[0]),
            voxels_count=int(voxels.shape[0]),
            t_sweep=t_sweep,
            t_imgs=t_imgs,
            t_vox=t_vox,
            t_depth=t_depth,
            t_geom=t_geom,
            t_stage=float(stage_info['stage']),
        )

    def evaluate(self, net: BEVFusionParallelNPUNet,
                 output_path: str = 'results_parallel_npu_v2.json'):
        print('\n' + '=' * 68)
        print('BEVFusion Parallel-NPU Evaluation Start (v2 pipeline)')
        print('=' * 68)

        split_name = 'mini_val' if 'mini' in self.nusc.version else 'val'
        from nuscenes.utils.splits import create_splits_scenes
        valid_names = set(create_splits_scenes()[split_name])
        val_samples = [
            s for s in self.nusc.sample
            if self.nusc.get('scene', s['scene_token'])['name'] in valid_names
        ]
        print(f'Validation samples: {len(val_samples)}')

        if not val_samples:
            raise RuntimeError('No validation samples found.')

        pipeline_start = time.time()
        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        future = prefetch_pool.submit(
            self._prepare_and_stage_sample,
            0,
            val_samples[0],
            0,
            True,
            net,
        )

        try:
            for idx, sample in enumerate(val_samples):
                t_step0 = time.time()
                prepared = future.result()
                prefetch_wait = time.time() - t_step0
                self.timing_stats['prefetch_wait'].append(prefetch_wait)

                self.timing_stats['sweep_loading'].append(prepared.t_sweep)
                self.timing_stats['image_loading'].append(prepared.t_imgs)
                self.timing_stats['voxelization'].append(prepared.t_vox)
                self.timing_stats['precompute_depth'].append(prepared.t_depth)
                self.timing_stats['precompute_geometry'].append(prepared.t_geom)
                self.timing_stats['host_staging'].append(prepared.t_stage)

                if idx + 1 < len(val_samples):
                    future = prefetch_pool.submit(
                        self._prepare_and_stage_sample,
                        idx + 1,
                        val_samples[idx + 1],
                        (idx + 1) % 2,
                        False,
                        net,
                    )
                else:
                    future = None

                print(f'\n[{idx+1}/{len(val_samples)}] {prepared.sample_token}')
                print(f'  Points: {prepared.points_count}  Voxels: {prepared.voxels_count}  Slot: {prepared.slot_id}')
                print(
                    f'  PrepWait={prefetch_wait*1000:.1f}ms  '
                    f'Stage={prepared.t_stage*1000:.1f}ms'
                )

                if prepared.geom_np is not None:
                    net.precompute_lut(prepared.geom_np)
                    print(f'  geom: {prepared.geom_np.shape}  [LUT precomputed and cached]')

                t0 = time.time()
                outputs = net.forward_staged(prepared.slot_id)
                self.timing_stats['npu_inference'].append(time.time() - t0)

                boxes = self.decode_outputs(outputs, score_threshold=0.01)
                boxes = self.nms_3d(boxes, default_threshold=0.5)
                boxes = self.transform_to_global(boxes, prepared.lidar_data)
                boxes.sort(key=lambda x: x['detection_score'], reverse=True)
                self.results[prepared.sample_token] = boxes

                step_total = time.time() - t_step0
                self.timing_stats['total_per_sample'].append(step_total)
                print(f'  StepTotal={step_total*1000:.1f}ms  Boxes={len(boxes)}')

            pipeline_wall = time.time() - pipeline_start

        finally:
            try:
                prefetch_pool.shutdown(wait=True)
            except Exception:
                pass

        submission = {
            'meta': {
                'use_camera': True, 'use_lidar': True,
                'use_radar': False, 'use_map': False,
                'use_external_track': False,
            },
            'results': self.results,
        }
        import json
        with open(output_path, 'w') as f:
            json.dump(submission, f, cls=NpEncoder, indent=2)
        print(f'\nResults saved: {output_path}')

        self._print_timing_summary()
        print('\nPipeline summary:')
        print(f'  Total wall time      : {pipeline_wall:.3f}s')
        print(f'  Mean wall/sample     : {pipeline_wall / max(len(val_samples), 1) * 1000.0:.2f}ms')
        print(f'  Effective throughput : {len(val_samples) / max(pipeline_wall, 1e-9):.2f} samples/s')

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
    parser = argparse.ArgumentParser(description='BEVFusion 并行 NPU evaluator v2')
    parser.add_argument('--dataroot', default='data/nuscenes-mini')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--lidar-om', default='models/om/bevfusion_lidar_branch_dynamic.om')
    parser.add_argument('--camera-om', default='src/ptq/results/lidar/bevfusion_camera_branch_q.om')
    parser.add_argument('--fusion-om', default='models/om/bevfusion_fusion_head.om')
    parser.add_argument('--output', default='bevfusion_parallel_npu_v2_results.json')
    parser.add_argument('--max-pts', type=int, default=16)
    parser.add_argument('--launch-order', default='lidar_first', choices=['lidar_first', 'camera_first'])
    args = parser.parse_args()

    print('Initializing ACL...')
    _ = init_acl(device_id=args.device_id)

    print('\nInitializing BEVFusionParallelNPUNet-v2...')
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
        stage_slots=2,
    )

    print('\nInitializing evaluator...')
    evaluator = BEVFusionParallelPipelineNPUEvaluator(
        dataroot=args.dataroot,
        version=args.version,
    )

    try:
        evaluator.evaluate(net, output_path=args.output)
    finally:
        evaluator.close()

    print('\nDone.')


if __name__ == '__main__':
    main()
