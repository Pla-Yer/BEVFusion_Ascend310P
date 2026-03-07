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


class TransFusionHeadDeploy(nn.Module):
    def __init__(self, head, K=200):
        super().__init__()
        self.head = head
        self.K = K

    def forward(self, inputs):  # inputs: [1,512,180,180]
        head = self.head
        B = inputs.shape[0]

        fusion_feat = head.shared_conv(inputs)  # [B,128,H,W]
        _, C, H, W = fusion_feat.shape
        HW = H * W

        # [B, C, HW]
        fusion_feat_flat = fusion_feat.view(B, C, HW)

        # bev_pos: [1, HW, 2] -> [B, HW, 2]
        bev_pos = head.bev_pos.repeat(B, 1, 1).to(fusion_feat.device)

        # dense heatmap: [B, num_cls, H, W]
        dense_heatmap = head.heatmap_head(fusion_feat)
        heatmap = torch.sigmoid(dense_heatmap)

        # NMS via maxpool (no inplace)
        p = head.nms_kernel_size // 2
        local_max = F.max_pool2d(heatmap, kernel_size=head.nms_kernel_size, stride=1, padding=p)
        heatmap_nms = heatmap * (heatmap == local_max)

        # flatten: [B, num_cls, HW]
        heatmap_flat = heatmap_nms.view(B, head.num_classes, HW)

        # topk across all classes+locations
        scores_flat = heatmap_flat.view(B, -1)  # [B, num_cls*HW]
        _, top = torch.topk(scores_flat, k=self.K, dim=-1)  # [B,K]
        top_cls = top // HW                                  # [B,K]
        top_idx = top % HW                                   # [B,K]

        # query_feat: gather from fusion_feat_flat [B,C,HW] -> [B,C,K]
        query_feat = fusion_feat_flat.gather(
            dim=-1, index=top_idx[:, None, :].expand(-1, C, -1)
        )

        # category embedding
        one_hot = F.one_hot(top_cls, num_classes=head.num_classes).permute(0, 2, 1).float()  # [B,num_cls,K]
        query_feat = query_feat + head.class_encoding(one_hot)  # [B,C,K]

        # query_pos: gather from bev_pos [B,HW,2] -> [B,K,2]
        query_pos = bev_pos.gather(
            dim=1, index=top_idx[:, :, None].expand(-1, -1, bev_pos.shape[-1])
        )

        # key/value positions are bev_pos, key is fusion_feat_flat
        # decoder layers
        for i in range(head.num_decoder_layers):
            query_feat = head.decoder[i](
                query_feat,
                key=fusion_feat_flat,
                query_pos=query_pos,
                key_pos=bev_pos
            )
            res = head.prediction_heads[i](query_feat)  # dict of tensors [B,*,K]
            # center residual add
            res_center = res['center'] + query_pos.permute(0, 2, 1)  # [B,2,K]
            res['center'] = res_center
            # next query_pos (no detach/clone for export)
            query_pos = res_center.permute(0, 2, 1)  # [B,K,2]

        # query_heatmap_score: [B,num_cls,K], gather from heatmap_flat
        query_heatmap_score = heatmap_flat.gather(
            dim=-1, index=top_idx[:, None, :].expand(-1, head.num_classes, -1)
        )

        # 输出全部改成 tensor tuple（ATC 友好）
        # 建议输出 raw heatmap（未 sigmoid）以保持与你现有后处理一致
        return (
            dense_heatmap,          # [B,num_cls,H,W]
            top_cls.to(torch.int32),# [B,K]  (给CPU后处理用)
            query_heatmap_score,    # [B,num_cls,K]
            res['heatmap'],         # [B,num_cls,K]  (query heatmap logits)
            res['center'],          # [B,2,K]
            res['height'],          # [B,1,K]
            res['dim'],             # [B,3,K]
            res['rot'],             # [B,2,K]
            res.get('vel', torch.zeros(B,2,self.K, device=inputs.device)) # [B,2,K]
        )
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

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model(cfg, ckpt_path)
    model.eval()
    deploy = TransFusionHeadDeploy(model.bbox_head, K=200).eval().cuda()
    dummy = torch.randn(1, 512, 180, 180, device='cuda')
    # 拼接相对路径（当前目录下的 onnx 子文件夹）
    target_dir = os.path.join(work_dir, "models/onnx/transfusion_head.onnx")
    target_folder = os.path.dirname(target_dir)
    os.makedirs(target_folder, exist_ok=True)


    torch.onnx.export(
        deploy, dummy,
        target_dir,
        opset_version=18,
        input_names=["bev_feat"],
        output_names=[
            "dense_heatmap",
            "top_cls",
            "query_heatmap_score",
            "heatmap_q",
            "center",
            "height",
            "dim",
            "rot",
            "vel"
        ],
        do_constant_folding=True,
        dynamo=False
    )
    print("Exported:", target_dir)

if __name__ == "__main__":
    main()