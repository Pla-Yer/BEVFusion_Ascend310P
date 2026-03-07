"""
获取ONNX模型的输入数据和输出结果
运行环境: CPU/GPU (onnxruntime)
适配新的ONNX模型（bevfusion_final_dynamic.onnx）
"""

import os
import sys
import time
import numpy as np
import onnxruntime as ort
import torch
import json

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# 将项目根目录添加到 Python 搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class ONNXDataCollector:
    """收集ONNX模型的输入输出数据"""

    def __init__(self, model_path, output_dir='./onnx_data', use_gpu=False):
        """
        初始化数据收集器

        Args:
            model_path: ONNX模型路径
            output_dir: 输出数据保存目录
            use_gpu: 是否使用GPU推理
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 初始化ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        print(f"加载ONNX模型: {model_path}")
        print(f"使用Provider: {providers}")
        self.session = ort.InferenceSession(model_path, providers=providers)

        # 打印模型输入输出信息
        print("\n模型输入信息:")
        for inp in self.session.get_inputs():
            print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

        print("\n模型输出信息:")
        for out in self.session.get_outputs():
            print(f"  {out.name}: shape={out.shape}, type={out.type}")

    def process_sample(self, pcl_path, sample_token, use_same_voxels=None):
        """
        处理单个样本，获取输入输出数据

        Args:
            pcl_path: 点云文件路径
            sample_token: 样本token
            use_same_voxels: 如果提供，使用相同的voxels数据（用于对比）
        """
        print(f"\n处理样本: {sample_token}")

        # 1. 加载点云数据
        points = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)[:, :5]
        print(f"原始点云形状: {points.shape}")

        # 2. Voxelization
        if use_same_voxels is not None:
            # 使用提供的voxels数据
            voxels, coords, num_points = use_same_voxels
            print(f"使用提供的voxels数据 - voxels: {voxels.shape}, coords: {coords.shape}")
        else:
            # 这里需要实际的voxelization实现
            # 暂时使用虚拟数据
            print("警告: 未提供voxels数据，使用虚拟数据")
            V = 6000
            M = 32
            Cin = 5
            voxels = np.random.randn(V, M, Cin).astype(np.float32)
            num_points = np.random.randint(1, M+1, V).astype(np.int64)
            coords = np.zeros((V, 4), dtype=np.float32)
            coords[:, 2] = np.random.randint(0, 360, V).astype(np.float32)
            coords[:, 3] = np.random.randint(0, 360, V).astype(np.float32)

        print(f"Voxelization后 - voxels: {voxels.shape}, coords: {coords.shape}, num_points: {num_points.shape}")

        # 确保数据是连续的
        # 注意：新的ONNX模型期望coords是float类型
        voxels = np.ascontiguousarray(voxels.astype(np.float32))
        coords = np.ascontiguousarray(coords.astype(np.float32))
        num_points = np.ascontiguousarray(num_points.astype(np.int64))
        # coords = coords[:, [0, 3, 1, 2]]
        # 3. ONNX模型推理
        print("开始ONNX模型推理...")

        # 准备输入
        inputs = {
            'voxels': voxels,
            'num_points': num_points,
            'coords': coords
        }

        start_time = time.time()
        outputs = self.session.run(None, inputs)
        inference_time = time.time() - start_time
        print(f"推理耗时: {inference_time:.4f}秒")

        # 4. 处理输出
        output_names = [
            'dense_heatmap', 'top_cls', 'query_heatmap_score',
            'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel', 
            'top_idx', 'top', 'top_score', 'keep', 'hm', 'lm'
        ]

        outputs_dict = {}
        for i, (name, data) in enumerate(zip(output_names, outputs)):
            print(f"输出 {i} ({name}): shape={data.shape}, dtype={data.dtype}")
                # 打印前10个元素
            # 如果为cls，则全部打印
            if 'cls' in name :
                print(f"  全部元素: {data.flatten()}")
            else:
                print(f"  前10个元素: {data.flatten()[:10]}")
            # 打印最大最小值
            print(f"  最大值: {data.max()}, 最小值: {data.min()}")
            outputs_dict[name] = data

        # 5. 保存数据
        sample_dir = os.path.join(self.output_dir, sample_token)
        os.makedirs(sample_dir, exist_ok=True)

        # 保存输入数据
        np.save(os.path.join(sample_dir, 'voxels.npy'), voxels)
        np.save(os.path.join(sample_dir, 'coords.npy'), coords)
        np.save(os.path.join(sample_dir, 'num_points.npy'), num_points)

        # 保存输出数据
        for name, data in outputs_dict.items():
            np.save(os.path.join(sample_dir, f'{name}.npy'), data)

        # 保存元数据
        metadata = {
            'sample_token': sample_token,
            'pcl_path': pcl_path,
            'original_points': int(len(points)),
            'voxels_shape': list(voxels.shape),
            'coords_shape': list(coords.shape),
            'num_points_shape': list(num_points.shape),
            'inference_time': float(inference_time),
            'output_shapes': {name: list(data.shape) for name, data in outputs_dict.items()},
            'model_path': self.session._model_path
        }

        with open(os.path.join(sample_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"数据已保存到: {sample_dir}")

        return voxels, coords, num_points, outputs_dict


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='获取ONNX模型输入输出数据')
    parser.add_argument('--model_path', type=str,
                       default='models/onnx_final/bevfusion_final_dynamic.onnx',
                       help='ONNX模型路径')
    parser.add_argument('--pcl_path', type=str,
                       default='data/nuscenes-mini/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402938197897.pcd.bin',
                       help='点云文件路径')
    parser.add_argument('--sample_token', type=str,
                       default='sample_0',
                       help='样本标识')
    parser.add_argument('--output_dir', type=str,
                       default='./onnx_data',
                       help='输出数据保存目录')
    parser.add_argument('--use_gpu', action='store_true',
                       help='是否使用GPU推理')
    parser.add_argument('--pytorch_data_dir', type=str,
                       default=None,
                       help='PyTorch数据目录（如果提供，将使用相同的voxels数据）')

    args = parser.parse_args()

    # 创建数据收集器
    collector = ONNXDataCollector(args.model_path, args.output_dir, args.use_gpu)

    # 如果提供了PyTorch数据目录，使用相同的voxels数据
    use_same_voxels = None
    if args.pytorch_data_dir:
        pytorch_sample_dir = os.path.join(args.pytorch_data_dir, args.sample_token)
        if os.path.exists(pytorch_sample_dir):
            print(f"使用PyTorch模型的voxels数据: {pytorch_sample_dir}")
            voxels = np.load(os.path.join(pytorch_sample_dir, 'voxels.npy'))
            coords = np.load(os.path.join(pytorch_sample_dir, 'coords.npy'))
            num_points = np.load(os.path.join(pytorch_sample_dir, 'num_points.npy'))
            use_same_voxels = (voxels, coords, num_points)

    # 处理样本
    collector.process_sample(args.pcl_path, args.sample_token, use_same_voxels)

    print("\n数据收集完成!")


if __name__ == '__main__':
    main()
