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
from copy import deepcopy

# PyTorch 2.6+ weights_only 补丁
sys.path.insert(0, osp.dirname(__file__))
import fix_pytorch_weights_only  # noqa

# 获取当前执行位置
work_dir = os.getcwd()

class BEVFusionFullDeploy(nn.Module):
    """
    完整的BEVFusion模型导出模块
    输入: voxels [V, M, Cin], num_points [V], coords [V, 4]
    输出: detection results including heatmap, topk indices, bbox predictions
    """
    def __init__(self, model, K=200):
        super().__init__()
        self.pts_voxel_encoder = model.pts_voxel_encoder
        self.pts_middle_encoder = model.pts_middle_encoder
        self.pts_backbone = model.pts_backbone
        self.pts_neck = model.pts_neck
        self.head = model.bbox_head
        self.K = K

    def forward(self, voxels, num_points, coords):
        """
        Args:
            voxels: [V, M, Cin] voxels
            num_points: [V] number of points in each voxel
            coords: [V, 4] batch_id, z, y, x coordinates
        Returns:
            tuple of detection outputs
        """
        # 1. Voxel Encoding
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coords)  # [V, Cout]

        # 2. Scatter to BEV
        # 如果当前 coords 是 (b, y, x, z)，转成 (b, z, y, x)
        coords = coords[:, [0, 3, 1, 2]].contiguous()
        batch_size = int(coords[-1, 0].item()) + 1
        bev_feat = self.pts_middle_encoder(voxel_features, coords, batch_size)  # [B, Cout, H, W]

        # 3. Backbone
        backbone_feats = self.pts_backbone(bev_feat)

        # 4. Neck
        neck_feat_list = self.pts_neck(list(backbone_feats))  # returns tuple
        neck_feat = neck_feat_list[0]  # [B, 512, 180, 180]

        # 5. Head (TransFusion)
        head = self.head
        B = neck_feat.shape[0]

        fusion_feat = head.shared_conv(neck_feat)  # [B,128,H,W]
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
        return (
            dense_heatmap,          # [B,num_cls,H,W]
            top_cls.to(torch.int32),# [B,K]
            query_heatmap_score,    # [B,num_cls,K]
            res['heatmap'],         # [B,num_cls,K]
            res['center'],          # [B,2,K]
            res['height'],          # [B,1,K]
            res['dim'],             # [B,3,K]
            res['rot'],             # [B,2,K]
            res.get('vel', torch.zeros(B,2,self.K, device=voxels.device)) # [B,2,K]
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

    model = init_model(cfg, ckpt_path)
    model.eval()

    # 创建部署模块
    deploy = BEVFusionFullDeploy(model, K=200).eval().cuda()

    # 创建虚拟输入
    V = 6000
    M = 32
    Cin = 5
    Cout = 256

    dummy_voxels = torch.randn(V, M, Cin, device='cuda')
    dummy_num_points = torch.randint(1, M+1, (V,), device='cuda')
    # coords 格式: [V, 4] -> [batch_id, z, y, x]
    dummy_coords = torch.zeros(V, 4, device='cuda')
    dummy_coords[:, 0] = 0  # batch_id = 0
    dummy_coords[:, 1] = torch.randint(0, 1, (V,)).float()  # z = 0
    dummy_coords[:, 2] = torch.randint(0, 360, (V,)).float()  # y
    dummy_coords[:, 3] = torch.randint(0, 360, (V,)).float()  # x

    # 创建输出目录
    target_folder = os.path.join(work_dir, "models/onnx")
    os.makedirs(target_folder, exist_ok=True)

    # 定义输入输出名称
    input_names = [
        "voxels",
        "num_points",
        "coords"
    ]
    output_names = [
        "dense_heatmap",
        "top_cls",
        "query_heatmap_score",
        "heatmap_q",
        "center",
        "height",
        "dim",
        "rot",
        "vel"
    ]

    # 1. 导出静态shape模型（用于对比）
    target_dir_static = os.path.join(target_folder, "bevfusion_full_static.onnx")
    with torch.no_grad():
        torch.onnx.export(
            deploy,
            (dummy_voxels, dummy_num_points, dummy_coords),
            target_dir_static,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False
        )
    print("Exported static model:", target_dir_static)

    # 2. 导出动态shape模型
    # 定义动态轴：第0维（体素数量）为动态
    dynamic_axes = {
        "voxels": {0: "num_voxels"},
        "num_points": {0: "num_voxels"},
        "coords": {0: "num_voxels"},
    }

    target_dir_dynamic = os.path.join(target_folder, "bevfusion_full_dynamic.onnx")
    with torch.no_grad():
        torch.onnx.export(
            deploy,
            (dummy_voxels, dummy_num_points, dummy_coords),
            target_dir_dynamic,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False
        )
    print("Exported dynamic model:", target_dir_dynamic)


if __name__ == "__main__":
    main()
