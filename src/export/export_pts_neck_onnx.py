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

class PTSNeckDeploy(nn.Module):
    """
    点云颈部网络导出模块
    输入: multi-scale features from backbone
    输出: fused feature map
    """
    def __init__(self, neck):
        super().__init__()
        self.neck = neck

    def forward(self, x):
        """
        Args:
            x: tuple of feature maps from backbone
                - feat_stride2: [B, 128, 180, 180]
                - feat_stride4: [B, 256, 90, 90]
        Returns:
            fused_feat: [B, Cout, H_out, W_out] fused feature map
        """
        fused_feat = self.neck(x)
        return fused_feat


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
    deploy = PTSNeckDeploy(model.pts_neck).eval().cuda()

    # 创建虚拟输入
    # 根据SECONDFPN的配置:
    # in_channels = [128, 256]
    # out_channels = [256, 256]
    # upsample_strides = [1, 2]
    B = 1
    feat_stride2 = torch.randn(B, 128, 180, 180, device='cuda')
    feat_stride4 = torch.randn(B, 256, 90, 90, device='cuda')

    dummy_input = (feat_stride2, feat_stride4)

    # 拼接相对路径
    target_dir = os.path.join(work_dir, "models/onnx/pts_neck.onnx")
    target_folder = os.path.dirname(target_dir)
    os.makedirs(target_folder, exist_ok=True)

    torch.onnx.export(
        deploy,
        dummy_input,
        target_dir,
        opset_version=18,
        input_names=[
            "feat_stride2",
            "feat_stride4"
        ],
        output_names=["fused_feat"],
        do_constant_folding=True,
        dynamo=False
    )
    print("Exported:", target_dir)


if __name__ == "__main__":
    main()
