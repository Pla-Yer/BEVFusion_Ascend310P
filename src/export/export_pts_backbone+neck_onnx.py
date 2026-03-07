import torch
import torch.nn as nn
import os.path as osp
import sys

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model
import os
# PyTorch 2.6+ weights_only 补丁
sys.path.insert(0, osp.dirname(__file__))
import fix_pytorch_weights_only  # noqa

# 获取当前执行位置
work_dir = os.getcwd()

class PtsBackboneNeckDeploy(nn.Module):
    def __init__(self, pts_backbone, pts_neck):
        super().__init__()
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck

    def forward(self, x):  # x: [B,256,180,180]
        feats = self.pts_backbone(x)        # tuple(T0,T1)
        out_list = self.pts_neck(list(feats))  # neck返回 [out]
        out = out_list[0]                   # [B,512,180,180]
        return out


def main():
    cfg_path = "src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
    # ckpt_path = "work_dirs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_20.pth"
    ckpt_path = "work_dirs/bevfusion/epoch_20.pth"

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(cfg_path)
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg.custom_imports)

    # 避免加载预训练
    if "img_backbone" in cfg.model and "init_cfg" in cfg.model["img_backbone"]:
        cfg.model["img_backbone"]["init_cfg"] = None

    model = init_model(cfg, ckpt_path)
    model.eval()

    deploy = PtsBackboneNeckDeploy(model.pts_backbone, model.pts_neck).eval().cuda()

    dummy = torch.randn(1, 256, 180, 180, device="cuda")
    # 拼接相对路径（当前目录下的 onnx 子文件夹）
    target_dir = os.path.join(work_dir, "models/onnx/pts_backbone_neck.onnx")
    target_folder = os.path.dirname(target_dir)
    os.makedirs(target_folder, exist_ok=True)

    torch.onnx.export(
        deploy, dummy,
        target_dir,
        opset_version=18,
        input_names=["bev_256"],
        output_names=["bev_512"],
        do_constant_folding=True,
        dynamo=False
    )
    print("Exported:", target_dir)


if __name__ == "__main__":
    main()
