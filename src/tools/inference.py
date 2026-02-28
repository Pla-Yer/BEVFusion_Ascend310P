#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""BEVFusion 推理脚本 - 用于单样本推理"""
import argparse
import os
import os.path as osp

import mmcv
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

# 注册 BEVFusion 模块
import bevfusion  # noqa: F401
from mmdet3d.apis import init_model
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


def parse_args():
    parser = argparse.ArgumentParser(description='BEVFusion inference script')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--points', help='Point cloud file path (for LiDAR-only inference)')
    parser.add_argument(
        '--images', nargs='+', help='Image file paths (multi-view images)')
    parser.add_argument(
        '--out-dir', default='results', help='Output directory for results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show visualization results')
    parser.add_argument(
        '--score-thr', type=float, default=0.3,
        help='Score threshold for filtering predictions')
    args = parser.parse_args()
    return args


def load_points(points_path):
    """Load point cloud from file."""
    if points_path.endswith('.bin'):
        points = np.fromfile(points_path, dtype=np.float32)
        points = points.reshape(-1, 5)  # x, y, z, intensity, ring
    elif points_path.endswith('.npy'):
        points = np.load(points_path)
    else:
        raise ValueError(f'Unsupported point cloud format: {points_path}')
    return points


def load_images(image_paths):
    """Load multi-view images."""
    images = []
    for path in image_paths:
        img = mmcv.imread(path)
        images.append(img)
    return images


def build_data_sample(points, images=None, metainfo=None):
    """Build data sample for inference."""
    # Create data sample
    data_sample = Det3DDataSample()
    
    # Set metainfo
    if metainfo is not None:
        data_sample.set_metainfo(metainfo)
    
    return data_sample


def main():
    args = parse_args()
    
    # Initialize default scope
    init_default_scope('mmdet3d')
    
    # Build model from config and checkpoint
    print(f'Loading model from {args.checkpoint}...')
    cfg = Config.fromfile(args.config)
    
    # Build model
    model = MODELS.build(cfg.model)
    model = model.to(args.device)
    model.eval()
    
    # Load checkpoint
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    
    print('Model loaded successfully!')
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Prepare input data
    if args.points is not None:
        print(f'Processing point cloud: {args.points}')
        points = load_points(args.points)
        points = torch.from_numpy(points).float().to(args.device)
        
        # Prepare input dict
        batch_inputs_dict = {
            'points': [points]
        }
        
        # Create dummy data sample
        data_sample = Det3DDataSample()
        data_sample.set_metainfo({
            'sample_idx': os.path.basename(args.points).split('.')[0],
            'lidar_path': args.points,
        })
        
        # Inference
        with torch.no_grad():
            results = model.predict(
                batch_inputs_dict, [data_sample])
        
        # Process results
        result = results[0]
        pred_instances = result.pred_instances_3d
        
        # Filter by score threshold
        scores = pred_instances.scores_3d
        labels = pred_instances.labels_3d
        bboxes = pred_instances.bboxes_3d
        
        mask = scores > args.score_thr
        scores = scores[mask]
        labels = labels[mask]
        bboxes = bboxes[mask]
        
        print(f'Detected {len(scores)} objects:')
        for i, (score, label, bbox) in enumerate(zip(scores, labels, bboxes)):
            print(f'  {i+1}. Class: {label.item()}, Score: {score.item():.3f}')
            print(f'      BBox: {bbox.cpu().numpy()}')
        
        # Save results
        output_file = osp.join(
            args.out_dir, 
            os.path.basename(args.points).split('.')[0] + '_result.pkl'
        )
        mmcv.dump(result.to_dict(), output_file)
        print(f'Results saved to {output_file}')
    
    else:
        print('Please provide point cloud file with --points argument')
        return


if __name__ == '__main__':
    main()
