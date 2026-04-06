import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model

# PyTorch 2.6+ weights_only 补丁（你已有）
sys.path.insert(0, osp.dirname(__file__))
import fix_pytorch_weights_only  # noqa


class TransFusionHeadDeploy(nn.Module):
    """把 bbox_head 的 forward_single 里那些“动态图后处理”(topk/nms/queries) 固化为 ONNX-friendly 的 Tensor 输出"""
    def __init__(self, head, K=200):
        super().__init__()
        self.head = head
        self.K = K

    def forward(self, inputs):  # inputs: [B,512,180,180]
        head = self.head
        B = inputs.shape[0]

        fusion_feat = head.shared_conv(inputs)  # [B,128,H,W]
        _, C, H, W = fusion_feat.shape
        HW = H * W

        fusion_feat_flat = fusion_feat.view(B, C, HW)  # [B,C,HW]
        bev_pos = head.bev_pos.repeat(B, 1, 1).to(fusion_feat.device)  # [B,HW,2]

        dense_heatmap = head.heatmap_head(fusion_feat)  # [B,num_cls,H,W]
        heatmap = torch.sigmoid(dense_heatmap)

        # NMS via maxpool（ONNX 友好写法）
        p = head.nms_kernel_size // 2
        local_max = F.max_pool2d(heatmap, kernel_size=head.nms_kernel_size, stride=1, padding=p)
        heatmap_nms = heatmap * (heatmap == local_max)

        heatmap_flat = heatmap_nms.view(B, head.num_classes, HW)
        scores_flat = heatmap_flat.view(B, -1)  # [B, num_cls*HW]

        _, top = torch.topk(scores_flat, k=self.K, dim=-1)  # [B,K]
        top_cls = top // HW                                  # [B,K]
        top_idx = top % HW                                   # [B,K]

        # query_feat: [B,C,K]
        query_feat = fusion_feat_flat.gather(
            dim=-1, index=top_idx[:, None, :].expand(-1, C, -1)
        )

        # category embedding
        one_hot = F.one_hot(top_cls, num_classes=head.num_classes).permute(0, 2, 1).float()  # [B,num_cls,K]
        query_feat = query_feat + head.class_encoding(one_hot)  # [B,C,K]

        # query_pos: [B,K,2]
        query_pos = bev_pos.gather(
            dim=1, index=top_idx[:, :, None].expand(-1, -1, bev_pos.shape[-1])
        )

        # decoder
        for i in range(head.num_decoder_layers):
            query_feat = head.decoder[i](
                query_feat,
                key=fusion_feat_flat,
                query_pos=query_pos,
                key_pos=bev_pos
            )
            res = head.prediction_heads[i](query_feat)  # dict: [B,*,K]
            res_center = res['center'] + query_pos.permute(0, 2, 1)  # [B,2,K]
            res['center'] = res_center
            query_pos = res_center.permute(0, 2, 1)  # [B,K,2]

        query_heatmap_score = heatmap_flat.gather(
            dim=-1, index=top_idx[:, None, :].expand(-1, head.num_classes, -1)
        )  # [B,num_cls,K]

        vel = res.get('vel', None)
        if vel is None:
            vel = torch.zeros(B, 2, self.K, device=inputs.device, dtype=inputs.dtype)

        # 输出全部是 tensor tuple（ATC/OM 友好）
        return (
            dense_heatmap,                  # [B,10,180,180]
            top_cls.to(torch.int32),        # [B,K]
            query_heatmap_score,            # [B,10,K]
            res['heatmap'],                 # [B,10,K]  query heatmap logits
            res['center'],                  # [B,2,K]
            res['height'],                  # [B,1,K]
            res['dim'],                     # [B,3,K]
            res['rot'],                     # [B,2,K]
            vel,                            # [B,2,K]
        )


class BEVFusion4StageDeploy(nn.Module):
    """
    把这 4 段合成一个 ONNX：
      fusion_layer (ConvFuser)
      -> pts_backbone (SECOND)
      -> pts_neck (SECONDFPN)
      -> bbox_head (TransFusionHeadDeploy)
    """
    def __init__(self, model, K=200, fuse_order="img_then_pts"):
        super().__init__()
        self.fuser = model.fusion_layer
        self.backbone = model.pts_backbone
        self.neck = model.pts_neck
        self.head = TransFusionHeadDeploy(model.bbox_head, K=K)

        # 通道拼接顺序必须和训练时一致！默认按常见实现：先 pts 后 img
        assert fuse_order in ["pts_then_img", "img_then_pts"]
        self.fuse_order = fuse_order

    def forward(self, pts_bev, img_bev):
        # pts_bev: [B,256,H,W]  img_bev: [B,80,H,W]
        if self.fuse_order == "pts_then_img":
            fused = self.fuser([pts_bev, img_bev])  # ✅ 注意这里传 List[Tensor]
        else:
            fused = self.fuser([img_bev, pts_bev])

        # SECOND backbone: returns tuple of multi-scale feats
        feats = self.backbone(fused)        # tuple(Tensor,...)
        out_list = self.neck(list(feats))   # list([Tensor])
        bev512 = out_list[0]                # [B,512,180,180]

        return self.head(bev512)            # 9 个输出


def main():
    cfg_path = "src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
    ckpt_path = "work_dirs/bevfusion/epoch_20.pth"
    
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(cfg_path)
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg.custom_imports)

    # 避免再次加载 imagenet backbone 预训练
    if "img_backbone" in cfg.model and "init_cfg" in cfg.model["img_backbone"]:
        cfg.model["img_backbone"]["init_cfg"] = None

    model = init_model(cfg, ckpt_path)
    model.eval()

    deploy = BEVFusion4StageDeploy(model, K=200, fuse_order="pts_then_img").eval().cuda()

    # 你需要确保这两个 shape 跟你训练/推理时的真实 BEV 特征一致
    dummy_pts = torch.randn(1, 256, 180, 180, device="cuda")
    dummy_img = torch.randn(1, 80, 180, 180, device="cuda")

    torch.onnx.export(
        deploy,
        (dummy_pts, dummy_img),
        "bevfusion_4stage.onnx",
        opset_version=18,  # ✅ 强烈建议 18
        input_names=["pts_bev", "img_bev"],
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
    print("Exported:", "bevfusion_4stage.onnx")


if __name__ == "__main__":
    main()
