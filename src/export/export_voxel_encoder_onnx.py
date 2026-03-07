import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os.path as osp
import sys
import numpy as np

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmengine.registry import DATASETS
from torch.utils.data import DataLoader

from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model
import onnxruntime as ort
import numpy as np
import torch
import os

# PyTorch 2.6+ weights_only 补丁
sys.path.insert(0, osp.dirname(__file__))
import fix_pytorch_weights_only  # noqa

# 获取当前执行位置
work_dir = os.getcwd()

class VoxelEncoderDeploy(nn.Module):
    """
    体素编码器导出模块
    输入: voxels [V, M, Cin], num_points [V], coords [V, 4]
    输出: voxel_features [V, Cout]
    """
    def __init__(self, voxel_encoder):
        super().__init__()
        self.voxel_encoder = voxel_encoder

    def forward(self, voxels, num_points, coords):
        """
        Args:
            voxels: [V, M, Cin] where V is number of voxels, M is max points per voxel, Cin is input channels
            num_points: [V] number of points in each voxel
            coords: [V, 4] batch_id, z, y, x coordinates
        Returns:
            voxel_features: [V, Cout] encoded features for each voxel
        """
        voxel_features = self.voxel_encoder(voxels, num_points, coords)
        return voxel_features


def main():
    cfg_path = "src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
    ckpt_path = "work_dirs/bevfusion/epoch_20.pth"

    # ---- 注册模块 ----
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(cfg_path)

    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg.custom_imports)

    # 避免重新加载 backbone 预训练
    if "img_backbone" in cfg.model and "init_cfg" in cfg.model["img_backbone"]:
        cfg.model["img_backbone"]["init_cfg"] = None

    model = init_model(cfg, ckpt_path)
    model.eval()

    # 创建部署模块
    deploy = VoxelEncoderDeploy(model.pts_voxel_encoder).eval().cuda()

    # 创建虚拟输入
    # V: number of voxels, M: max points per voxel, Cin: input channels (5)
    # 使用固定体素数量8000，导出静态shape模型
    V = 8000  # 固定体素数量
    M = 32
    Cin = 5
    Cout = 256  # output channels

    # point_cloud_range: [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    # voxel_size: [0.3, 0.3, 8.0]
    # coords范围: x,y: 0-360, z: 0-1
    dummy_voxels = torch.randn(V, M, Cin, device='cuda')
    dummy_num_points = torch.randint(1, M+1, (V,), device='cuda')
    dummy_coords = torch.zeros(V, 4, device='cuda')
    dummy_coords[:, 1] = torch.randint(0, 2, (V,), device='cuda').float()  # z: 0-1
    dummy_coords[:, 2] = torch.randint(0, 361, (V,), device='cuda').float()  # y: 0-360
    dummy_coords[:, 3] = torch.randint(0, 361, (V,), device='cuda').float()  # x: 0-360

    # 拼接相对路径
    target_dir = os.path.join(work_dir, "models/onnx/voxel_encoder.onnx")
    target_folder = os.path.dirname(target_dir)
    os.makedirs(target_folder, exist_ok=True)

    # 导出静态shape模型（不使用动态轴）
    torch.onnx.export(
        deploy,
        (dummy_voxels, dummy_num_points, dummy_coords),
        target_dir,
        opset_version=11,  # 使用更稳定的opset版本
        input_names=[
            "voxels",
            "num_points",
            "coords"
        ],
        output_names=["voxel_features"],
        # 不设置dynamic_axes，使用静态shape
        do_constant_folding=True,
        dynamo=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # 确保使用标准ONNX算子
        export_params=True
    )
    print("Exported:", target_dir)


if __name__ == "__main__":
    main()
