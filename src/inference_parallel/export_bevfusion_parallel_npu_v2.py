"""
BEVFusion 并行 NPU 导出脚本
================================================================================
将原本的单一 full-NPU ONNX/OM 拆为三个模型：

  1) LiDAR 分支   : voxels, num_points, coords                 -> lidar_bev
  2) Camera 分支  : imgs, depth, pool_lookup, pool_mask         -> camera_bev
  3) Fusion Head  : camera_bev, lidar_bev                       -> detections

运行时可通过两个 stream 并发执行 1) 和 2)，待二者结束后再执行 3)。

附加优化（v2）：
  - pool_mask 从 float32 改为 uint8 输入；在模型内再 cast 到特征 dtype。
  - 该改动只影响静态 LUT 掩码的存储/传输，不改变数值语义。
"""

import argparse
import os
from os import path as osp
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(__file__))

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model

from export_bevfusion_full_npu import (
    OnnxPointPillarsScatter,
    BEVFusionFullNPU,
    make_dummy_inputs,
)


class LidarBranchNPU(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pts_voxel_encoder = model.pts_voxel_encoder
        scatter = model.pts_middle_encoder
        self.pts_middle_encoder = OnnxPointPillarsScatter(
            scatter.in_channels, scatter.ny, scatter.nx)

    def forward(self, voxels, num_points, coords):
        vf = self.pts_voxel_encoder(voxels, num_points, coords)
        mask = (coords.sum(dim=1) != 0).float().unsqueeze(1)
        vf = vf * mask
        return self.pts_middle_encoder(vf, coords)


class CameraBranchNPU(BEVFusionFullNPU):
    def __init__(self, model, B=1, N=6, fH=32, fW=88, max_pts: int = 16):
        super().__init__(model, K=200, dataset='nuScenes', B=B, N=N, fH=fH, fW=fW, max_pts=max_pts)

    def forward(self, imgs, depth, pool_lookup, pool_mask):
        B, N, C_img, H, W = imgs.shape
        B_int = self._B
        N_int = self._N
        fH_int = self._fH
        fW_int = self._fW
        D_int = self.D
        C_int = self.C_cam

        imgs_flat = imgs.view(B_int * N_int, C_img, H, W)
        bb_feats = self.img_backbone(imgs_flat)
        neck_feats = self.img_neck(list(bb_feats))
        img_feat = neck_feats[0]

        d = depth.view(B_int * N_int, *depth.shape[2:])
        d = self.dtransform(d)

        x = torch.cat([d, img_feat], dim=1)
        x = self.depthnet(x)

        depth_logits = x[:, :D_int]
        feat = x[:, D_int:D_int + C_int]
        depth_prob = depth_logits.softmax(dim=1)
        cam_feats = depth_prob.unsqueeze(1) * feat.unsqueeze(2)
        cam_feats = cam_feats.view(B_int, N_int, C_int, D_int, fH_int, fW_int)
        return self._bev_pool(cam_feats, pool_lookup, pool_mask)


class FusionHeadNPU(BEVFusionFullNPU):
    def __init__(self, model, K=200, dataset='nuScenes', B=1, N=6, fH=32, fW=88, max_pts: int = 16):
        super().__init__(model, K=K, dataset=dataset, B=B, N=N, fH=fH, fW=fW, max_pts=max_pts)

    def forward(self, camera_bev, lidar_bev):
        fused_feat = self.fusion_layer([camera_bev, lidar_bev])
        backbone_feats = self.pts_backbone(fused_feat)
        neck_out = self.pts_neck(list(backbone_feats))[0]

        head = self.head
        batch_size = neck_out.shape[0]
        fusion_feat = head.shared_conv(neck_out)
        fusion_feat_flatten = fusion_feat.view(batch_size, fusion_feat.shape[1], -1)
        bev_pos = head.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)

        dense_heatmap = head.heatmap_head(fusion_feat.float())
        heatmap = dense_heatmap.sigmoid()

        padding = head.nms_kernel_size // 2
        local_max = F.max_pool2d(
            heatmap, kernel_size=head.nms_kernel_size, stride=1, padding=padding)

        if self.dataset == 'nuScenes':
            local_max_cls8 = F.max_pool2d(heatmap[:, 8:9], kernel_size=1, stride=1, padding=0)
            local_max_cls9 = F.max_pool2d(heatmap[:, 9:10], kernel_size=1, stride=1, padding=0)
            local_max = torch.cat([local_max[:, :8], local_max_cls8, local_max_cls9], dim=1)
        elif self.dataset == 'Waymo':
            local_max_cls1 = F.max_pool2d(heatmap[:, 1:2], kernel_size=1, stride=1, padding=0)
            local_max_cls2 = F.max_pool2d(heatmap[:, 2:3], kernel_size=1, stride=1, padding=0)
            local_max = torch.cat(
                [local_max[:, :1], local_max_cls1, local_max_cls2, local_max[:, 3:]], dim=1)

        heatmap = heatmap * ((heatmap + 1e-3) >= local_max).float()
        heatmap = heatmap.view(batch_size, head.num_classes, -1)

        scores_flat = heatmap.view(batch_size, -1)
        _, top_proposals = torch.topk(scores_flat, k=self.K, dim=-1)
        HW = heatmap.shape[-1]
        top_proposals_class = top_proposals // HW
        top_proposals_index = top_proposals % HW

        query_feat = fusion_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, fusion_feat_flatten.shape[1], -1),
            dim=-1)

        one_hot = F.one_hot(top_proposals_class, num_classes=head.num_classes).permute(0, 2, 1)
        query_cat_encoding = head.class_encoding(one_hot.float())
        query_feat = query_feat + query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1)

        ret_dicts = []
        for i in range(head.num_decoder_layers):
            query_feat = head.decoder[i](
                query_feat, key=fusion_feat_flatten,
                query_pos=query_pos, key_pos=bev_pos)
            res = head.prediction_heads[i](query_feat)
            res['center'] = res['center'] + query_pos.permute(0, 2, 1)
            ret_dicts.append(res)
            query_pos = res['center'].clone().permute(0, 2, 1)

        query_heatmap_score = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, head.num_classes, -1),
            dim=-1)

        last = ret_dicts[-1]
        vel = last.get('vel', torch.zeros(batch_size, 2, self.K, device=fused_feat.device))
        return (
            dense_heatmap,
            top_proposals_class,
            query_heatmap_score,
            last['heatmap'],
            last['center'],
            last['height'],
            last['dim'],
            last['rot'],
            vel,
            top_proposals_index,
        )


def print_atc_commands(outdir, max_voxels, B=1, N=6, H=256, W=704,
                       nx0=360, nx1=360, nx2=1, max_pts=16, soc='Ascend310P1',
                       camera_bev_shape=(1, 80, 360, 360), lidar_bev_shape=(1, 64, 360, 360)):
    out_cells = B * nx2 * nx0 * nx1
    cam_shape = ",".join(str(int(x)) for x in camera_bev_shape)
    lidar_shape = ",".join(str(int(x)) for x in lidar_bev_shape)
    print(f"""
ATC 编译命令:
─────────────────────────────────────────────────────────────────────────
# 1) LiDAR branch (dynamic voxels)
atc --model="{osp.join(outdir, 'bevfusion_lidar_branch.onnx')}" \\
    --framework=5 \\
    --output="models/om/bevfusion_lidar_branch_dynamic" \\
    --input_format=ND \\
    --input_shape="voxels:-1,32,5;num_points:-1;coords:-1,4" \\
    --dynamic_dims="6000,6000,6000;8000,8000,8000;10000,10000,10000" \\
    --soc_version={soc} \\
    --op_select_implmode=high_precision \\
    --precision_mode=allow_fp32_to_fp16 \\
    --log=warning

# 2) Camera branch (static)
atc --model="{osp.join(outdir, 'bevfusion_camera_branch.onnx')}" \\
    --framework=5 \\
    --output="models/om/bevfusion_camera_branch" \\
    --input_format=ND \\
    --input_shape="imgs:{B},{N},3,{H},{W};depth:{B},{N},1,{H},{W};pool_lookup:{out_cells},{max_pts};pool_mask:{out_cells},{max_pts}" \\
    --soc_version={soc} \\
    --op_select_implmode=high_precision \\
    --precision_mode=allow_fp32_to_fp16 \\
    --log=warning

# 3) Fusion head (static)
atc --model="{osp.join(outdir, 'bevfusion_fusion_head.onnx')}" \\
    --framework=5 \\
    --output="models/om/bevfusion_fusion_head" \\
    --input_format=ND \\
    --input_shape="camera_bev:{cam_shape};lidar_bev:{lidar_shape}" \\
    --soc_version={soc} \\
    --op_select_implmode=high_precision \\
    --precision_mode=allow_fp32_to_fp16 \\
    --log=warning
─────────────────────────────────────────────────────────────────────────
""")


def main():
    p = argparse.ArgumentParser(description='导出 BEVFusion 三段并行 NPU ONNX')
    p.add_argument('--config',
                   default='src/configs/bevfusion_lidar-cam_voxel03_second_secfpn_'
                           '8xb4-cyclic-20e_nus-3d_resnet50.py')
    p.add_argument('--ckpt',
                   default='work_dirs/bevfusion_lidar-cam_voxel03_second_secfpn_'
                           '8xb4-cyclic-20e_nus-3d_resnet50/epoch_6.pth')
    p.add_argument('--outdir', default='models/onnx_parallel_npu')
    p.add_argument('--max-voxels', type=int, default=10000)
    p.add_argument('--K', type=int, default=200)
    p.add_argument('--dataset', default='nuScenes', choices=['nuScenes', 'Waymo'])
    p.add_argument('--soc', default='Ascend310P1')
    p.add_argument('--max-pts', type=int, default=16)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = 'cuda'

    print('[1/4] 加载 BEVFusion 模型...')
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    if 'img_backbone' in cfg.model:
        cfg.model['img_backbone']['init_cfg'] = None
    model = init_model(cfg, args.ckpt)
    model.eval()

    D = model.view_transform.D
    enc = model.pts_middle_encoder
    H, W = 256, 704
    fH, fW = 32, 88
    B, N = 1, 6
    vt = model.view_transform
    nx0 = int(round(vt.nx[0].item()))
    nx1 = int(round(vt.nx[1].item()))
    nx2 = int(round(vt.nx[2].item()))

    print('\n[2/4] 生成 dummy inputs...')
    dummy = make_dummy_inputs(
        model,
        max_voxels=args.max_voxels,
        B=B, N=N, H=H, W=W, fH=fH, fW=fW,
        device=device,
        max_pts=args.max_pts,
    )
    voxels, num_points, coords, imgs, depth, pool_lookup, pool_mask = dummy
    pool_mask = pool_mask.to(torch.uint8)

    output_names = [
        'dense_heatmap', 'top_cls', 'query_heatmap_score',
        'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel', 'top_idx'
    ]

    print('\n[3/4] 导出三个 ONNX...')
    # LiDAR branch
    lidar_branch = LidarBranchNPU(model).eval().to(device)
    lidar_onnx = osp.join(args.outdir, 'bevfusion_lidar_branch.onnx')
    with torch.no_grad():
        lidar_bev = lidar_branch(voxels, num_points, coords)
        torch.onnx.export(
            lidar_branch,
            (voxels, num_points, coords),
            lidar_onnx,
            opset_version=16,
            input_names=['voxels', 'num_points', 'coords'],
            output_names=['lidar_bev'],
            dynamic_axes={
                'voxels': {0: 'max_V'},
                'num_points': {0: 'max_V'},
                'coords': {0: 'max_V'},
            },
            do_constant_folding=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False,
        )
    print(f'  ✓ {lidar_onnx}')
    del lidar_branch

    # Camera branch
    camera_branch = CameraBranchNPU(model, B=B, N=N, fH=fH, fW=fW, max_pts=args.max_pts).eval().to(device)
    camera_onnx = osp.join(args.outdir, 'bevfusion_camera_branch.onnx')
    with torch.no_grad():
        camera_bev = camera_branch(imgs, depth, pool_lookup, pool_mask)
        torch.onnx.export(
            camera_branch,
            (imgs, depth, pool_lookup, pool_mask),
            camera_onnx,
            opset_version=11,
            input_names=['imgs', 'depth', 'pool_lookup', 'pool_mask'],
            output_names=['camera_bev'],
            do_constant_folding=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False,
        )
    print(f'  ✓ {camera_onnx}')
    del camera_branch

    # Fusion head
    fusion_head = FusionHeadNPU(
        model,
        K=args.K,
        dataset=args.dataset,
        B=B,
        N=N,
        fH=fH,
        fW=fW,
        max_pts=args.max_pts,
    ).eval().to(device)
    fusion_onnx = osp.join(args.outdir, 'bevfusion_fusion_head.onnx')
    with torch.no_grad():
        _ = fusion_head(camera_bev, lidar_bev)
        torch.onnx.export(
            fusion_head,
            (camera_bev, lidar_bev),
            fusion_onnx,
            opset_version=11,
            input_names=['camera_bev', 'lidar_bev'],
            output_names=output_names,
            do_constant_folding=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False,
        )
    print(f'  ✓ {fusion_onnx}')

    print('\n[4/4] 打印 ATC 编译命令...')
    print_atc_commands(
        args.outdir,
        max_voxels=args.max_voxels,
        B=B, N=N, H=H, W=W,
        nx0=nx0, nx1=nx1, nx2=nx2,
        max_pts=args.max_pts,
        soc=args.soc,
        camera_bev_shape=tuple(camera_bev.shape),
        lidar_bev_shape=tuple(lidar_bev.shape),
    )

    print('=' * 72)
    print('导出完成！')
    print('=' * 72)
    print('运行时请配合 bevfusion_parallel_npu_net.py / bevfusion_parallel_npu_evaluator.py 使用。')


if __name__ == '__main__':
    main()
