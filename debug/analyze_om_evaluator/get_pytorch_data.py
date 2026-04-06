"""
获取PyTorch模型的输入数据和输出结果
运行环境: OpenMMLab (GPU)
使用新的BEVFusionDeployFinal模块
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model


class OnnxPointPillarsScatter(nn.Module):
    """ONNX友好的Scatter实现"""
    def __init__(self, in_channels, ny, nx):
        super().__init__()
        self.C, self.ny, self.nx = in_channels, ny, nx

    def forward(self, voxel_features, coors, batch_size):
        C, ny, nx = self.C, self.ny, self.nx
        HW = ny * nx
        indices = coors[:,0].long() * HW + coors[:,2].long() * nx + coors[:,3].long()
        canvas = torch.zeros(C, batch_size * HW,
                            dtype=voxel_features.dtype,
                            device=voxel_features.device)
        canvas = canvas.scatter(1, indices.unsqueeze(0).expand(C, -1),
                               voxel_features.t())
        return canvas.view(C, batch_size, ny, nx).permute(1, 0, 2, 3).contiguous()


class BEVFusionDeployFinal(nn.Module):
    """最终版本的BEVFusion部署模块"""
    TIEBREAK_EPS = 1e-6

    def __init__(self, model, K=200):
        super().__init__()
        self.pts_voxel_encoder = model.pts_voxel_encoder
        self.pts_backbone = model.pts_backbone
        self.pts_neck = model.pts_neck
        self.head = model.bbox_head
        self.K = K
        orig = model.pts_middle_encoder
        self.pts_middle_encoder = OnnxPointPillarsScatter(
            orig.in_channels, orig.ny, orig.nx)

    def forward(self, voxels, num_points, coords):
        vf = self.pts_voxel_encoder(voxels, num_points, coords)
        batch_size = int(coords[-1, 0].item()) + 1
        bev = self.pts_middle_encoder(vf, coords, batch_size)
        neck = self.pts_neck(list(self.pts_backbone(bev)))[0]

        head = self.head
        B = neck.shape[0]
        ff = head.shared_conv(neck)
        _, C, H, W = ff.shape
        HW = H * W
        ff_flat = ff.view(B, C, HW)
        bev_pos = head.bev_pos.repeat(B, 1, 1).to(ff.device)

        dm = head.heatmap_head(ff)
        hm = torch.sigmoid(dm)
        p = head.nms_kernel_size // 2
        lm = F.max_pool2d(hm, kernel_size=head.nms_kernel_size, stride=1, padding=p)
        hmf = (hm * (hm == lm)).view(B, head.num_classes, HW)

        # tiebreak bias
        num_el_tensor = hmf.shape[2] * head.num_classes
        idx = torch.arange(num_el_tensor, dtype=hmf.dtype, device=hmf.device)
        bias = (self.TIEBREAK_EPS * (num_el_tensor - idx) / num_el_tensor).view(
                    1, head.num_classes, hmf.shape[2])
        _, top = torch.topk((hmf + bias).view(B, -1), k=self.K, dim=-1)
        top_cls = top // HW
        top_idx = top % HW

        qf = ff_flat.gather(-1, top_idx[:, None, :].expand(-1, C, -1))
        qf = qf + head.class_encoding(
                F.one_hot(top_cls, head.num_classes).permute(0, 2, 1).float())
        qp = bev_pos.gather(1, top_idx[:, :, None].expand(-1, -1, bev_pos.shape[-1]))

        for i in range(head.num_decoder_layers):
            qf = head.decoder[i](qf, key=ff_flat, query_pos=qp, key_pos=bev_pos)
            res = head.prediction_heads[i](qf)
            res['center'] = res['center'] + qp.permute(0, 2, 1)
            qp = res['center'].permute(0, 2, 1)

        qhs = hmf.gather(-1, top_idx[:, None, :].expand(-1, head.num_classes, -1))
        return (dm, top_cls.to(torch.int32), qhs,
                res['heatmap'], res['center'], res['height'], res['dim'], res['rot'],
                res.get('vel', torch.zeros(B, 2, self.K, device=voxels.device)))


class PyTorchDataCollector:
    """收集PyTorch模型的输入输出数据"""

    def __init__(self, config_path, checkpoint_path, output_dir='./pytorch_data'):
        """
        初始化数据收集器

        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型权重路径
            output_dir: 输出数据保存目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 注册模块
        register_all_modules(init_default_scope=True)
        cfg = Config.fromfile(config_path)

        if cfg.get("custom_imports", None):
            import_modules_from_strings(**cfg.custom_imports)

        # 避免重新加载backbone预训练
        if "img_backbone" in cfg.model and "init_cfg" in cfg.model["img_backbone"]:
            cfg.model["img_backbone"]["init_cfg"] = None

        # 加载模型
        print("加载PyTorch模型...")
        model = init_model(cfg, checkpoint_path)
        model.eval()
        
        # 创建部署模块
        self.deploy = BEVFusionDeployFinal(model, K=200).eval().cuda()
        print("模型加载完成")

        # Voxelization参数
        self.pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = [0.30, 0.30, 8.0]
        self.max_voxels = 6000
        self.max_points_per_voxel = 32

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

        # 3. PyTorch模型推理
        print("开始PyTorch模型推理...")

        # 转换为tensor并移到GPU
        voxels_tensor = torch.from_numpy(voxels).float().cuda()
        coords_tensor = torch.from_numpy(coords).float().cuda()
        num_points_tensor = torch.from_numpy(num_points).long().cuda()
        # coords_tensor = coords_tensor[:, [0, 3, 1, 2]].contiguous() 

        with torch.no_grad():
            start_time = time.time()
            outputs = self.deploy(voxels_tensor, num_points_tensor, coords_tensor)
            inference_time = time.time() - start_time
            print(f"推理耗时: {inference_time:.4f}秒")

        # 4. 转换输出为numpy
        output_names = [
            'dense_heatmap', 'top_cls', 'query_heatmap_score',
            'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel', 'top_idx'
        ]
        
        outputs_dict = {}
        for name, data in zip(output_names, outputs):
            data_np = data.cpu().numpy()
            outputs_dict[name] = data_np
            # 打印前10个元素以验证
            print(f"输出 {name} 前10个元素: {data_np[0,:10]}")
            print(f"输出 {name}: shape={data_np.shape}, dtype={data_np.dtype}")

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
            'output_shapes': {name: list(data.shape) for name, data in outputs_dict.items()}
        }

        with open(os.path.join(sample_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"数据已保存到: {sample_dir}")

        return voxels, coords, num_points, outputs_dict


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='获取PyTorch模型输入输出数据')
    parser.add_argument('--config_path', type=str,
                       default='src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
                       help='配置文件路径')
    parser.add_argument('--checkpoint_path', type=str,
                       default='work_dirs/bevfusion/epoch_20.pth',
                       help='模型权重路径')
    parser.add_argument('--pcl_path', type=str,
                       default='data/nuscenes-mini/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402938197897.pcd.bin',
                       help='点云文件路径')
    parser.add_argument('--sample_token', type=str,
                       default='sample_0',
                       help='样本标识')
    parser.add_argument('--output_dir', type=str,
                       default='./pytorch_data',
                       help='输出数据保存目录')
    parser.add_argument('--om_data_dir', type=str,
                       default=None,
                       help='OM数据目录（如果提供，将使用相同的voxels数据）')

    args = parser.parse_args()

    # 创建数据收集器
    collector = PyTorchDataCollector(args.config_path, args.checkpoint_path, args.output_dir)

    # 如果提供了OM数据目录，使用相同的voxels数据
    use_same_voxels = None
    if args.om_data_dir:
        om_sample_dir = os.path.join(args.om_data_dir, args.sample_token)
        if os.path.exists(om_sample_dir):
            print(f"使用OM模型的voxels数据: {om_sample_dir}")
            voxels = np.load(os.path.join(om_sample_dir, 'voxels.npy'))
            coords = np.load(os.path.join(om_sample_dir, 'coords.npy'))
            num_points = np.load(os.path.join(om_sample_dir, 'num_points.npy'))
            use_same_voxels = (voxels, coords, num_points)

    # 处理样本
    collector.process_sample(args.pcl_path, args.sample_token, use_same_voxels)

    print("\n数据收集完成!")


if __name__ == '__main__':
    main()
