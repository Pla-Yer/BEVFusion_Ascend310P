"""
获取OM模型的输入数据和输出结果
运行环境: Ascend
"""

import os
import sys
import time
import numpy as np
import acl
import torch
import json

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（根据你的路径结构调整，这里假设根目录是 BEVFusion_Ascend310P）
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# 将项目根目录添加到 Python 搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# # 添加项目路径
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
# sys.path.insert(0, project_root)
# sys.path.insert(0, os.path.join(project_root, 'src/inference'))
# sys.path.insert(0, os.path.join(project_root, 'src/bevfusion'))

from src.inference.bevfusion_net import Net, init_acl, check_ret
from src.bevfusion.ops.voxel import Voxelization


class OMDataCollector:
    """收集OM模型的输入输出数据"""

    def __init__(self, model_path, output_dir='./om_data'):
        """
        初始化数据收集器

        Args:
            model_path: OM模型路径
            output_dir: 输出数据保存目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 初始化ACL
        self.context = init_acl(0)

        # 初始化模型
        self.net = Net(model_path)

        # Voxelization参数
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.30, 0.30, 8.0]
        self.max_voxels = 6000
        self.max_points_per_voxel = 20

        # 初始化Voxelization算子
        self.voxelizer = Voxelization(
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range,
            max_num_points=self.max_points_per_voxel,
            max_voxels=self.max_voxels,
            deterministic=True
        )

    def process_sample(self, pcl_path, sample_token):
        """
        处理单个样本，获取输入输出数据

        Args:
            pcl_path: 点云文件路径
            sample_token: 样本token
        """
        print(f"\n处理样本: {sample_token}")

        # 1. 加载点云数据
        points = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)[:, :5]
        print(f"原始点云形状: {points.shape}")

        # 2. Voxelization
        points_tensor = torch.from_numpy(points).float()

        # 过滤点云范围
        valid_mask = (
            (points[:,0] >= -54) & (points[:,0] < 54) &
            (points[:,1] >= -54) & (points[:,1] < 54) &
            (points[:,2] >= -5)  & (points[:,2] < 3)
        )
        print(f"有效点数: {valid_mask.sum()} / {len(points)}")

        # Voxelization
        voxels, coords, num_points = self.voxelizer(points_tensor)

        # 转换为numpy
        voxels = voxels.numpy() if isinstance(voxels, torch.Tensor) else voxels
        coords = coords.numpy() if isinstance(coords, torch.Tensor) else coords
        num_points = num_points.numpy() if isinstance(num_points, torch.Tensor) else num_points

        print(f"Voxelization后 - voxels: {voxels.shape}, coords: {coords.shape}, num_points: {num_points.shape}")
        zero_col = np.zeros((coords.shape[0], 1), dtype=coords.dtype)
        # 步骤2：将全0列拼接到原始数组的最左侧（第一列），得到 (n, 4)
        new_coords = np.hstack([zero_col, coords])
        coords = new_coords
        # 打印前几行坐标以验证
        print("Sample coords (first 5 rows):")
        print(coords[:5])
        # # 3. Padding到固定大小
        # current_voxels = voxels.shape[0]

        # if current_voxels > self.max_voxels:
        #     voxels = voxels[:self.max_voxels]
        #     coords = coords[:self.max_voxels]
        #     num_points = num_points[:self.max_voxels]
        # elif current_voxels < self.max_voxels:
        #     pad_size = self.max_voxels - current_voxels

        #     # Pad voxels
        #     padded_voxels = np.zeros((self.max_voxels, self.max_points_per_voxel, voxels.shape[-1]), dtype=np.float32)
        #     padded_voxels[:current_voxels] = voxels
        #     voxels = padded_voxels

        #     # Pad coords
        #     padded_coords = np.full((self.max_voxels, 4), 0, dtype=np.int32)
        #     padded_coords[:current_voxels,1:] = coords
        #     padded_coords[:current_voxels, 0] = 0
        #     coords = padded_coords

        #     # Pad num_points
        #     padded_num_points = np.zeros(self.max_voxels, dtype=np.int64)
        #     padded_num_points[:current_voxels] = num_points
        #     num_points = padded_num_points

        print(f"Padding后 - voxels: {voxels.shape}, coords: {coords.shape}, num_points: {num_points.shape}")

        # 确保数据是连续的
        coords = coords[:, [0, 3, 1, 2]]
        voxels = np.ascontiguousarray(voxels.astype(np.float32))
        coords = np.ascontiguousarray(coords.astype(np.float32))
        num_points = np.ascontiguousarray(num_points.astype(np.int64))
        
        inputs = [voxels, num_points, coords]
        actual_voxel_num = voxels.shape[0]
        print(f"Actual voxel number: {actual_voxel_num}")

        # 4. OM模型推理
        print("开始OM模型推理...")
        start_time = time.time()
        outputs = self.net.forward(inputs, actual_voxel_num)
        inference_time = time.time() - start_time
        print(f"推理耗时: {inference_time:.4f}秒")

        # 5. 保存数据
        sample_dir = os.path.join(self.output_dir, sample_token)
        os.makedirs(sample_dir, exist_ok=True)

        # 保存输入数据
        np.save(os.path.join(sample_dir, 'voxels.npy'), voxels)
        np.save(os.path.join(sample_dir, 'coords.npy'), coords)
        np.save(os.path.join(sample_dir, 'num_points.npy'), num_points)

        # 保存输出数据
        output_names = [
            'dense_heatmap', 'top_cls', 'query_heatmap_score',
            'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel', 
            'top_idx', 'top', 'top_score','keep','hm','lm'
        ]

        # for i, (name, data) in enumerate(zip(output_names, outputs)):
        #     print(f"输出 {i} ({name}): shape={data.shape}, dtype={data.dtype}")
        #     np.save(os.path.join(sample_dir, f'{name}.npy'), data)
        # 定义每个输出对应的目标形状和数据类型
        output_shape_config = {
            "dense_heatmap": {"shape": (1, 10, 180, 180), "dtype": np.float32},
            "top_cls": {"shape": (1, 100), "dtype": np.int32},
            "query_heatmap_score": {"shape": (1, 10, 100), "dtype": np.float32},
            "heatmap_q": {"shape": (1, 10, 100), "dtype": np.float32},
            "center": {"shape": (1, 2, 100), "dtype": np.float32},
            "height": {"shape": (1, 1, 100), "dtype": np.float32},
            "dim": {"shape": (1, 3, 100), "dtype": np.float32},
            "rot": {"shape": (1, 2, 100), "dtype": np.float32},
            "vel": {"shape": (1, 2, 100), "dtype": np.float32},
            "top_idx": {"shape": (1, 100), "dtype": np.int32},
            "top": {"shape": (1, 100), "dtype": np.int32},
            "top_score": {"shape": (1, 1*10*180*180), "dtype": np.float32},
            "keep": {"shape": (1, 1*10*180*180), "dtype": np.int32},
            "hm": {"shape": (1, 10, 180, 180), "dtype": np.float32},
            "lm": {"shape": (1, 10, 180, 180), "dtype": np.float32},
        }

        # 遍历输出并重塑形状
        for i, (name, data) in enumerate(zip(output_names, outputs)):
            # 检查当前输出是否在配置中
            if name in output_shape_config:
                config = output_shape_config[name]
                # 计算目标形状的总元素数
                target_size = np.prod(config["shape"])
                # 展平数据（确保是一维）后重塑为目标形状
                reshaped_data = data.flatten()[:target_size].reshape(config["shape"])
                # 转换数据类型
                reshaped_data = reshaped_data.astype(config["dtype"])
                
                # 打印重塑后的信息
                print(f"输出 {i} ({name}): shape={reshaped_data.shape}, dtype={reshaped_data.dtype}")
                # 打印前10个元素
                print(name + "前10个元素:")
                # 打印top_idx的所有元素：
                if name == "top_cls":
                    print(reshaped_data.flatten())
                else:
                    print(f"前10个元素: {reshaped_data.flatten()[:10]}")
                #打印最大最小值
                print(f"最大值: {reshaped_data.max()}, 最小值: {reshaped_data.min()}")
                # 保存重塑后的数据
                np.save(os.path.join(sample_dir, f'{name}.npy'), reshaped_data)
            else:
                # 对于不在配置中的输出，保持原有逻辑
                print(f"输出 {i} ({name}): shape={data.shape}, dtype={data.dtype} (未配置目标形状)")
                np.save(os.path.join(sample_dir, f'{name}.npy'), data)
        # 保存元数据
        metadata = {
            'sample_token': sample_token,
            'pcl_path': pcl_path,
            'original_points': int(len(points)),
            'valid_points': int(valid_mask.sum()),
            'voxels_shape': list(voxels.shape),
            'coords_shape': list(coords.shape),
            'num_points_shape': list(num_points.shape),
            'inference_time': float(inference_time),
            'output_shapes': {name: list(data.shape) for name, data in zip(output_names, outputs)}
        }

        with open(os.path.join(sample_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"数据已保存到: {sample_dir}")

        return voxels, coords, num_points, outputs

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'net'):
            del self.net


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='获取OM模型输入输出数据')
    parser.add_argument('--model_path', type=str,
                       default='models/om/bevfusion_full_dynamic.om',
                       help='OM模型路径')
    parser.add_argument('--pcl_path', type=str,
                       default='data/nuscenes-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin',
                       help='点云文件路径')
    parser.add_argument('--sample_token', type=str,
                       default='sample_0',
                       help='样本标识')
    parser.add_argument('--output_dir', type=str,
                       default='./om_data',
                       help='输出数据保存目录')

    args = parser.parse_args()

    # 创建数据收集器
    collector = OMDataCollector(args.model_path, args.output_dir)

    # 处理样本
    collector.process_sample(args.pcl_path, args.sample_token)

    print("\n数据收集完成!")


if __name__ == '__main__':
    main()
