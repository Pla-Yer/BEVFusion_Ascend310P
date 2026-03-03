#!/usr/bin/env python
"""
单样本测试脚本 - 验证推理流程
"""

import os
import sys
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bevfusion_evaluator import hard_voxelize, CLASS_NAMES

def test_single_inference():
    """测试单个样本的推理流程"""
    print("="*60)
    print("单样本推理测试")
    print("="*60)

    # 加载一个真实的点云文件
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataroot = os.path.join(project_root, "data/nuscenes-mini")

    # 使用第一个样本
    pcl_path = os.path.join(dataroot, "samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025.pcd.bin")

    if not os.path.exists(pcl_path):
        print(f"错误: 点云文件不存在: {pcl_path}")
        return False

    print(f"加载点云: {pcl_path}")
    points = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)
    print(f"点云形状: {points.shape}")

    # 体素化参数
    pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    voxel_size = [0.30, 0.30, 8.0]

    print(f"\n体素化参数:")
    print(f"  点云范围: {pc_range}")
    print(f"  体素大小: {voxel_size}")

    # 体素化
    print("\n执行体素化...")
    voxels, num_points, coords, count = hard_voxelize(
        points, voxel_size, pc_range, max_voxels=10000
    )

    print(f"体素化结果:")
    print(f"  体素数量: {count}")
    print(f"  voxels形状: {voxels.shape}")
    print(f"  coords形状: {coords.shape}")
    print(f"  num_points形状: {num_points.shape}")

    # 检查数据有效性
    if count == 0:
        print("\n✗ 错误: 没有生成任何体素")
        return False

    print(f"\n✓ 体素化成功")

    # 模拟模型输出（用于测试解码逻辑）
    print("\n" + "="*60)
    print("测试解码逻辑（使用模拟输出）")
    print("="*60)

    # 创建模拟输出
    B = 1
    K = 200
    num_classes = 10
    H, W = 180, 180

    # 模拟输出
    dense_heatmap = np.random.randn(B, num_classes, H, W).astype(np.float32) * 0.1
    top_cls = np.random.randint(0, num_classes, (K,)).astype(np.int32)
    query_heatmap_score = np.random.rand(B, num_classes, K).astype(np.float32) * 0.5 + 0.3  # 0.3-0.8
    heatmap_q = np.random.rand(B, num_classes, K).astype(np.float32)
    center = np.random.randn(B, 2, K).astype(np.float32) * 10
    height = np.random.randn(B, 1, K).astype(np.float32)
    dim = np.random.randn(B, 3, K).astype(np.float32) * 0.5 + 1.0
    rot = np.random.randn(B, 2, K).astype(np.float32)
    vel = np.random.randn(B, 2, K).astype(np.float32) * 0.5

    outputs = [
        dense_heatmap, top_cls, query_heatmap_score, heatmap_q,
        center, height, dim, rot, vel
    ]

    print(f"模拟输出形状:")
    for i, out in enumerate(outputs):
        print(f"  output[{i}]: {out.shape}")

    # 测试解码
    from bevfusion_evaluator import BEVFusionEvaluator

    class DummyEvaluator:
        def __init__(self):
            self.pc_range = pc_range
            self.voxel_size = voxel_size
            self.timing_stats = {'decode': []}

    evaluator = DummyEvaluator()
    evaluator.decode_outputs = BEVFusionEvaluator.decode_outputs.__get__(evaluator, DummyEvaluator)

    print(f"\n执行解码（score_threshold=0.1）...")
    boxes = evaluator.decode_outputs(outputs, score_threshold=0.1)

    print(f"\n解码结果:")
    print(f"  检测框数量: {len(boxes)}")

    if len(boxes) > 0:
        print(f"\n第一个检测框:")
        for key, value in boxes[0].items():
            print(f"    {key}: {value}")
        print(f"\n✓ 解码成功")
        return True
    else:
        print(f"\n✗ 解码失败: 没有检测到任何框")
        return False


if __name__ == "__main__":
    success = test_single_inference()
    sys.exit(0 if success else 1)
