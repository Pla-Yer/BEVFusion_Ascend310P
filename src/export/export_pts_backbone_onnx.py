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

class PTSBackboneDeploy(nn.Module):
    """
    点云骨干网络导出模块
    输入: bev_feat [B, Cin, H_in, W_in]
    输出: features - tuple of multi-scale features
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        """
        Args:
            x: [B, Cin, H_in, W_in] BEV feature map
        Returns:
            features: tuple of feature maps at different scales
        """
        features = self.backbone(x)
        return features


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
    deploy = PTSBackboneDeploy(model.pts_backbone).eval().cuda()

    # 创建虚拟输入
    # 根据配置: input_shape = [360, 360], in_channels = 256
    B = 1
    Cin = 256
    H, W = 360, 360

    dummy_input = torch.randn(B, Cin, H, W, device='cuda')

    # 拼接相对路径
    target_dir = os.path.join(work_dir, "models/onnx/pts_backbone.onnx")
    target_folder = os.path.dirname(target_dir)
    os.makedirs(target_folder, exist_ok=True)

    # 根据SECOND backbone的输出通道数定义输出名称
    # out_channels = [128, 256]
    torch.onnx.export(
        deploy,
        dummy_input,
        target_dir,
        opset_version=18,
        input_names=["bev_feat"],
        output_names=[
            "feat_stride2",  # [B, 128, 180, 180]
            "feat_stride4"   # [B, 256, 90, 90]
        ],
        do_constant_folding=True,
        dynamo=False
    )
    print("Exported:", target_dir)


if __name__ == "__main__":
    main()
