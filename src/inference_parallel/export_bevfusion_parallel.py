"""
BEVFusion 并行分支导出脚本
================================================================================
导出 3 个独立 ONNX/OM：
  lidar_branch.onnx   : voxels/num_points/coords → lidar_bev [1,256,ny,nx]
  camera_branch.onnx  : imgs/depth/pool_lookup/pool_mask → camera_bev [1,C*nz,nx0,nx1]
  fusion_head.onnx    : camera_bev + lidar_bev → 10 个检测输出

推理时两条 ACL Stream 并行执行 lidar+camera，Barrier 后串行执行 fusion。

NPU 时延理论收益：
  串行旧版 ≈ 30(lidar) + 100(camera) + 52(fusion) = 182ms
  并行新版 ≈ max(30, 100)            + 52         = 152ms  (-30ms/帧)

BEVPool 使用 Gather+Mul+ReduceSum，完全消除 ScatterElements。
"""

import argparse
import os
import sys
from os import path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(__file__))
try:
    import fix_pytorch_weights_only  # noqa
except ModuleNotFoundError:
    pass

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model


# ══════════════════════════════════════════════════════════════════════════
#  OnnxPointPillarsScatter
# ══════════════════════════════════════════════════════════════════════════
class OnnxPointPillarsScatter(nn.Module):
    """ONNX 兼容的 PointPillarsScatter，用 scatter 替代原版 for 循环。"""

    def __init__(self, in_channels, ny, nx):
        super().__init__()
        self.C, self.ny, self.nx = in_channels, ny, nx

    def forward(self, voxel_features, coors):
        C, ny, nx = self.C, self.ny, self.nx
        indices = (coors[:, 2].long() * nx + coors[:, 3].long()).to(torch.int32)
        canvas  = torch.zeros(C, ny * nx,
                              dtype=voxel_features.dtype,
                              device=voxel_features.device)
        canvas  = canvas.scatter(1,
                                 indices.unsqueeze(0).expand(C, -1),
                                 voxel_features.t())
        return canvas.view(C, 1, ny, nx).permute(1, 0, 2, 3).contiguous()


# ══════════════════════════════════════════════════════════════════════════
#  分支 1：LiDAR Branch
# ══════════════════════════════════════════════════════════════════════════
class LidarBranchModel(nn.Module):
    """
    输入: voxels [V,32,5] + num_points [V] + coords [V,4]
    输出: lidar_bev [1, 256, ny, nx]
    """

    def __init__(self, model):
        super().__init__()
        self.pts_voxel_encoder = model.pts_voxel_encoder
        orig = model.pts_middle_encoder
        self.pts_middle_encoder = OnnxPointPillarsScatter(
            orig.in_channels, orig.ny, orig.nx)

    def forward(self, voxels, num_points, coords):
        vf   = self.pts_voxel_encoder(voxels, num_points, coords)
        mask = (coords.sum(dim=1) != 0).float().unsqueeze(1)
        vf   = vf * mask
        return self.pts_middle_encoder(vf, coords)   # [1, 256, ny, nx]


# ══════════════════════════════════════════════════════════════════════════
#  分支 2：Camera Branch
# ══════════════════════════════════════════════════════════════════════════
class CameraBranchModel(nn.Module):
    """
    输入: imgs [B,N,3,H,W] + depth [B,N,1,H,W]
        + pool_lookup [out_cells,max_pts] int64
        + pool_mask   [out_cells,max_pts] float32
    输出: camera_bev [B, C_cam*nz, nx0, nx1]

    BEVPool = Gather + Mul + ReduceSum（无 ScatterElements）
    """

    def __init__(self, model, B=1, N=6, fH=32, fW=88, max_pts=16):
        super().__init__()
        self.img_backbone = model.img_backbone
        self.img_neck     = model.img_neck
        vt = model.view_transform
        self.dtransform = vt.dtransform
        self.depthnet   = vt.depthnet
        self.D     = int(vt.D)
        self.C_cam = int(vt.C)

        self.register_buffer('_bx', vt.bx.detach().clone().float())
        self.register_buffer('_dx', vt.dx.detach().clone().float())

        self._B    = int(B)
        self._N    = int(N)
        self._fH   = int(fH)
        self._fW   = int(fW)
        self._nx0  = int(round(vt.nx[0].item()))
        self._nx1  = int(round(vt.nx[1].item()))
        self._nx2  = int(round(vt.nx[2].item()))
        self._out_cells = B * self._nx2 * self._nx0 * self._nx1
        self._max_pts   = int(max_pts)

    def _bev_pool(self, cam_feats, pool_lookup, pool_mask):
        B, C      = self._B, self.C_cam
        nx0, nx1, nx2        = self._nx0, self._nx1, self._nx2
        out_cells, max_pts   = self._out_cells, self._max_pts

        feats    = cam_feats.permute(0, 1, 3, 4, 5, 2).contiguous().reshape(-1, C)
        flat_idx = pool_lookup.reshape(-1)                          # Gather index
        gathered = feats[flat_idx].reshape(out_cells, max_pts, C)  # Gather
        gathered = gathered * pool_mask.to(feats.dtype).unsqueeze(-1)  # Mul
        out      = gathered.sum(dim=1)                              # ReduceSum
        out      = out.view(B, nx2, nx0, nx1, C).permute(0, 4, 1, 2, 3).contiguous()
        return out.reshape(B, C * nx2, nx0, nx1)

    def forward(self, imgs, depth, pool_lookup, pool_mask):
        B_int, N_int   = self._B, self._N
        fH_int, fW_int = self._fH, self._fW
        D_int, C_int   = self.D, self.C_cam
        _, _, C_img, H, W = imgs.shape

        imgs_flat = imgs.view(B_int * N_int, C_img, H, W)
        bb_feats  = self.img_backbone(imgs_flat)
        neck_feat = self.img_neck(list(bb_feats))[0]   # [B*N, 256, fH, fW]

        d = depth.view(B_int * N_int, *depth.shape[2:])
        d = self.dtransform(d)                          # [B*N, 64, fH, fW]

        x          = torch.cat([d, neck_feat], dim=1)
        x          = self.depthnet(x)                  # [B*N, D+C, fH, fW]
        depth_prob = x[:, :D_int].softmax(dim=1)
        feat       = x[:, D_int: D_int + C_int]

        cam_feats  = (depth_prob.unsqueeze(1) * feat.unsqueeze(2)
                      ).view(B_int, N_int, C_int, D_int, fH_int, fW_int)

        return self._bev_pool(cam_feats, pool_lookup, pool_mask)


# ══════════════════════════════════════════════════════════════════════════
#  融合头：Fusion Head
# ══════════════════════════════════════════════════════════════════════════
class FusionHeadModel(nn.Module):
    """
    输入: camera_bev [B, C*nz, nx0, nx1]
        + lidar_bev  [B, 256,  ny,  nx]
    输出: 10 个检测输出张量
    """

    def __init__(self, model, K=200, dataset='nuScenes'):
        super().__init__()
        self.K       = K
        self.dataset = dataset
        self.fusion_layer = model.fusion_layer
        self.pts_backbone = model.pts_backbone
        self.pts_neck     = model.pts_neck
        self.head         = model.bbox_head

    def forward(self, camera_bev, lidar_bev):
        fused      = self.fusion_layer([camera_bev, lidar_bev])
        bb_feats   = self.pts_backbone(fused)
        neck_out   = self.pts_neck(list(bb_feats))[0]

        head       = self.head
        batch_size = neck_out.shape[0]
        ff         = head.shared_conv(neck_out)
        ff_flat    = ff.view(batch_size, ff.shape[1], -1)
        bev_pos    = head.bev_pos.repeat(batch_size, 1, 1).to(ff.device)

        dense_heatmap = head.heatmap_head(ff.float())
        heatmap       = dense_heatmap.sigmoid()

        padding   = head.nms_kernel_size // 2
        local_max = F.max_pool2d(heatmap, kernel_size=head.nms_kernel_size,
                                 stride=1, padding=padding)
        if self.dataset == 'nuScenes':
            lm8 = F.max_pool2d(heatmap[:, 8:9], 1, 1, 0)
            lm9 = F.max_pool2d(heatmap[:, 9:10], 1, 1, 0)
            local_max = torch.cat([local_max[:, :8], lm8, lm9], dim=1)
        elif self.dataset == 'Waymo':
            lm1 = F.max_pool2d(heatmap[:, 1:2], 1, 1, 0)
            lm2 = F.max_pool2d(heatmap[:, 2:3], 1, 1, 0)
            local_max = torch.cat([local_max[:, :1], lm1, lm2,
                                   local_max[:, 3:]], dim=1)

        heatmap = heatmap * ((heatmap + 1e-3) >= local_max).float()
        heatmap = heatmap.view(batch_size, head.num_classes, -1)

        scores_flat  = heatmap.view(batch_size, -1)
        _, top_props = torch.topk(scores_flat, k=self.K, dim=-1)
        HW           = heatmap.shape[-1]
        top_cls      = top_props // HW
        top_idx      = top_props % HW

        query_feat = ff_flat.gather(
            index=top_idx[:, None, :].expand(-1, ff_flat.shape[1], -1), dim=-1)
        one_hot    = F.one_hot(top_cls, num_classes=head.num_classes).permute(0, 2, 1)
        query_feat = query_feat + head.class_encoding(one_hot.float())
        query_pos  = bev_pos.gather(
            index=top_idx[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1)

        ret_dicts = []
        for i in range(head.num_decoder_layers):
            query_feat = head.decoder[i](query_feat, key=ff_flat,
                                         query_pos=query_pos, key_pos=bev_pos)
            res = head.prediction_heads[i](query_feat)
            res['center'] = res['center'] + query_pos.permute(0, 2, 1)
            ret_dicts.append(res)
            query_pos = res['center'].clone().permute(0, 2, 1)

        qhs  = heatmap.gather(
            index=top_idx[:, None, :].expand(-1, head.num_classes, -1), dim=-1)
        last = ret_dicts[-1]
        vel  = last.get('vel', torch.zeros(batch_size, 2, self.K, device=fused.device))

        return (dense_heatmap, top_cls, qhs, last['heatmap'],
                last['center'], last['height'], last['dim'],
                last['rot'], vel, top_idx)


# ══════════════════════════════════════════════════════════════════════════
#  LUT 构建（向量化 numpy，无 Python 逐点循环）
# ══════════════════════════════════════════════════════════════════════════
def build_lut(geom_feats, bx, dx, B, N, D, fH, fW, nx0, nx1, nx2, max_pts):
    """
    从 geom_feats [B,N,D,fH,fW,3] 构建 BEV pool Gather 查找表。

    Returns:
        pool_lookup : LongTensor  [out_cells, max_pts]
        pool_mask   : FloatTensor [out_cells, max_pts]
    """
    ppb       = N * D * fH * fW
    out_cells = B * nx2 * nx0 * nx1

    coords = geom_feats.reshape(-1, 3).cpu().float().numpy()
    bx_np  = bx.cpu().float().numpy()
    dx_np  = dx.cpu().float().numpy()

    idx    = ((coords - (bx_np - dx_np * 0.5)) / dx_np).astype(np.int64)
    x_id, y_id, z_id = idx[:, 0], idx[:, 1], idx[:, 2]

    valid = ((x_id >= 0) & (x_id < nx0) &
             (y_id >= 0) & (y_id < nx1) &
             (z_id >= 0) & (z_id < nx2))

    b_id = np.repeat(np.arange(B, dtype=np.int64), ppb)
    lin  = (b_id * (nx2 * nx0 * nx1)
            + np.clip(z_id, 0, nx2 - 1) * (nx0 * nx1)
            + np.clip(x_id, 0, nx0 - 1) * nx1
            + np.clip(y_id, 0, nx1 - 1))

    lookup = np.zeros((out_cells, max_pts), np.int64)
    mask   = np.zeros((out_cells, max_pts), np.float32)

    vpt   = np.where(valid)[0]
    vvox  = lin[vpt]
    order = np.argsort(vvox, kind='stable')
    spt, svox = vpt[order], vvox[order]

    is_new    = np.concatenate([[True], svox[1:] != svox[:-1]])
    seg_start = np.maximum.accumulate(
        np.where(is_new, np.arange(len(spt), dtype=np.int64), 0))
    slot = np.arange(len(spt), dtype=np.int64) - seg_start

    keep = slot < max_pts
    lookup[svox[keep], slot[keep]] = spt[keep]
    mask  [svox[keep], slot[keep]] = 1.0

    return torch.from_numpy(lookup), torch.from_numpy(mask)


# ══════════════════════════════════════════════════════════════════════════
#  ONNX 导出辅助
# ══════════════════════════════════════════════════════════════════════════
def export_onnx(model, dummy_inputs, path, input_names, output_names,
                dynamic_axes=None):
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_inputs, path,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or {},
            do_constant_folding=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False,
        )
    sz = os.path.getsize(path) / 1024 / 1024
    print(f'  ✓ {path}  ({sz:.1f} MB)')


def save_bin(arr, path):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr.tofile(path)
    print(f'  bin: {path}  shape={arr.shape}  dtype={arr.dtype}')


def print_atc_cmds(outdir, om_dir, max_voxels, B, N, H, W,
                   out_cells, max_pts, ny, nx, C_cam, nz, nx0, nx1, soc):
    """打印三个独立 ATC 编译命令（可并行执行）。"""
    cam_c = C_cam * nz
    print(f"""
{'='*72}
ATC 编译命令（三个 OM 可同时并行编译）
{'='*72}

[1] lidar_branch  ──  动态体素数
─────────────────────────────────────────────────────────────────────────
atc --model="{outdir}/lidar_branch.onnx" \\
    --framework=5 \\
    --output="{om_dir}/lidar_branch" \\
    --input_format=ND \\
    --input_shape="voxels:-1,32,5;num_points:-1;coords:-1,4" \\
    --dynamic_dims="{max_voxels},{max_voxels},{max_voxels}" \\
    --soc_version={soc} \\
    --op_select_implmode=high_precision \\
    --precision_mode=allow_fp32_to_fp16 \\
    --log=warning

[2] camera_branch  ──  静态 shape（标定固定）
─────────────────────────────────────────────────────────────────────────
atc --model="{outdir}/camera_branch.onnx" \\
    --framework=5 \\
    --output="{om_dir}/camera_branch" \\
    --input_format=ND \\
    --input_shape="imgs:{B},{N},3,{H},{W};depth:{B},{N},1,{H},{W};pool_lookup:{out_cells},{max_pts};pool_mask:{out_cells},{max_pts}" \\
    --soc_version={soc} \\
    --op_select_implmode=high_precision \\
    --precision_mode=allow_fp32_to_fp16 \\
    --log=warning

[3] fusion_head  ──  静态 shape
─────────────────────────────────────────────────────────────────────────
atc --model="{outdir}/fusion_head.onnx" \\
    --framework=5 \\
    --output="{om_dir}/fusion_head" \\
    --input_format=ND \\
    --input_shape="camera_bev:{B},{cam_c},{nx0},{nx1};lidar_bev:{B},256,{ny},{nx}" \\
    --soc_version={soc} \\
    --op_select_implmode=high_precision \\
    --precision_mode=allow_fp32_to_fp16 \\
    --log=warning
{'='*72}
""")


# ══════════════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description='BEVFusion 并行分支 ONNX 导出')
    p.add_argument('--config',
                   default='src/configs/bevfusion_lidar-cam_voxel03_second_secfpn_'
                           '8xb4-cyclic-20e_nus-3d_resnet50.py')
    p.add_argument('--ckpt',
                   default='work_dirs/bevfusion_lidar-cam_voxel03_second_secfpn_'
                           '8xb4-cyclic-20e_nus-3d_resnet50/epoch_6.pth')
    p.add_argument('--outdir',     default='models/onnx_parallel')
    p.add_argument('--om-dir',     default='models/om_parallel')
    p.add_argument('--bindir',     default='bins_parallel')
    p.add_argument('--max-voxels', type=int, default=10000)
    p.add_argument('--max-pts',    type=int, default=16)
    p.add_argument('--K',          type=int, default=200)
    p.add_argument('--dataset',    default='nuScenes', choices=['nuScenes', 'Waymo'])
    p.add_argument('--save-bins',  action='store_true')
    p.add_argument('--soc',        default='Ascend310P1')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = 'cuda'

    # ── 1. 加载模型 ───────────────────────────────────────────────────────
    print('[1/5] 加载 BEVFusion 模型...')
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    if 'img_backbone' in cfg.model:
        cfg.model['img_backbone']['init_cfg'] = None
    model = init_model(cfg, args.ckpt)
    model.eval()

    vt  = model.view_transform
    enc = model.pts_middle_encoder
    B, N, H, W, fH, fW = 1, 6, 256, 704, 32, 88
    D     = int(vt.D)
    nx0   = int(round(vt.nx[0].item()))
    nx1   = int(round(vt.nx[1].item()))
    nx2   = int(round(vt.nx[2].item()))
    C_cam = int(vt.C)
    bx    = vt.bx.detach().cpu().float()
    dx    = vt.dx.detach().cpu().float()
    MAX_PTS   = args.max_pts
    out_cells = B * nx2 * nx0 * nx1

    print(f'\n  D={D}  C_cam={C_cam}  nx={nx0}×{nx1}×{nx2}'
          f'  out_cells={out_cells}  max_pts={MAX_PTS}')
    print(f'  Scatter grid: ny={enc.ny}  nx_scatter={enc.nx}')
    print(f'  Max voxels: {args.max_voxels}')

    # ── 2. 构建哑元输入 ────────────────────────────────────────────────────
    print('\n[2/5] 构建哑元输入...')
    torch.manual_seed(0)
    V = args.max_voxels

    voxels     = torch.randn(V, 32, 5, device=device)
    num_points = torch.randint(1, 33, (V,), device=device).float()
    fi = torch.randperm(enc.ny * enc.nx, device=device)[:V]
    coords = torch.zeros(V, 4, device=device)
    coords[:, 2] = (fi // enc.nx).float()
    coords[:, 3] = (fi %  enc.nx).float()

    imgs  = torch.randn(B, N, 3, H, W, device=device)
    depth = torch.rand(B, N, 1, H, W, device=device) * 59.0 + 1.0

    geom_feats = torch.zeros(B, N, D, fH, fW, 3)
    geom_feats[..., 0] = torch.rand(B, N, D, fH, fW) * 108.0 - 54.0
    geom_feats[..., 1] = torch.rand(B, N, D, fH, fW) * 108.0 - 54.0
    geom_feats[..., 2] = torch.rand(B, N, D, fH, fW) *  20.0 - 10.0

    print('  构建 BEV pool LUT...')
    pool_lookup, pool_mask = build_lut(
        geom_feats, bx, dx, B, N, D, fH, fW, nx0, nx1, nx2, MAX_PTS)
    pool_lookup = pool_lookup.to(device)
    pool_mask   = pool_mask.to(device)
    print(f'  pool_lookup: {pool_lookup.shape}  pool_mask: {pool_mask.shape}')

    # ── 3. 构建三个部署模型并验证 ─────────────────────────────────────────
    print('\n[3/5] 构建三个部署模型并前向验证...')
    lidar_model  = LidarBranchModel(model).eval().to(device)
    camera_model = CameraBranchModel(
        model, B=B, N=N, fH=fH, fW=fW, max_pts=MAX_PTS).eval().to(device)
    fusion_model = FusionHeadModel(
        model, K=args.K, dataset=args.dataset).eval().to(device)

    with torch.no_grad():
        try:
            lidar_bev  = lidar_model(voxels, num_points, coords)
            camera_bev = camera_model(imgs, depth, pool_lookup, pool_mask)
            outputs    = fusion_model(camera_bev, lidar_bev)

            print(f'  lidar_bev  : {lidar_bev.shape}')
            print(f'  camera_bev : {camera_bev.shape}')
            print(f'  dense_heatmap: {outputs[0].shape}')
            print('  前向验证成功!')
        except Exception as e:
            print(f'  [WARNING] 前向验证失败: {e}')
            import traceback; traceback.print_exc()
            print('  继续尝试 ONNX 导出...')

    # 可选：保存测试 bin
    if args.save_bins:
        os.makedirs(args.bindir, exist_ok=True)
        print(f'\n  保存测试 bin 到 {args.bindir}/')
        save_bin(voxels,      osp.join(args.bindir, 'voxels.bin'))
        save_bin(num_points,  osp.join(args.bindir, 'num_points.bin'))
        save_bin(coords,      osp.join(args.bindir, 'coords.bin'))
        save_bin(imgs,        osp.join(args.bindir, 'imgs.bin'))
        save_bin(depth,       osp.join(args.bindir, 'depth.bin'))
        save_bin(pool_lookup, osp.join(args.bindir, 'pool_lookup.bin'))
        save_bin(pool_mask,   osp.join(args.bindir, 'pool_mask.bin'))
        with torch.no_grad():
            save_bin(lidar_bev,  osp.join(args.bindir, 'lidar_bev.bin'))
            save_bin(camera_bev, osp.join(args.bindir, 'camera_bev.bin'))

    # ── 4. 导出三个 ONNX ─────────────────────────────────────────────────
    print('\n[4/5] 导出三个 ONNX...')

    export_onnx(
        lidar_model,
        (voxels, num_points, coords),
        osp.join(args.outdir, 'lidar_branch.onnx'),
        input_names=['voxels', 'num_points', 'coords'],
        output_names=['lidar_bev'],
        dynamic_axes={
            'voxels':     {0: 'max_V'},
            'num_points': {0: 'max_V'},
            'coords':     {0: 'max_V'},
        },
    )

    export_onnx(
        camera_model,
        (imgs, depth, pool_lookup, pool_mask),
        osp.join(args.outdir, 'camera_branch.onnx'),
        input_names=['imgs', 'depth', 'pool_lookup', 'pool_mask'],
        output_names=['camera_bev'],
    )

    with torch.no_grad():
        export_onnx(
            fusion_model,
            (camera_bev, lidar_bev),
            osp.join(args.outdir, 'fusion_head.onnx'),
            input_names=['camera_bev', 'lidar_bev'],
            output_names=[
                'dense_heatmap', 'top_cls', 'query_heatmap_score',
                'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel', 'top_idx',
            ],
        )

    # ── 5. 打印 ATC 命令 ─────────────────────────────────────────────────
    print('[5/5] ATC 编译命令:')
    print_atc_cmds(
        outdir=args.outdir,
        om_dir=args.om_dir,
        max_voxels=V,
        B=B, N=N, H=H, W=W,
        out_cells=out_cells,
        max_pts=MAX_PTS,
        ny=enc.ny, nx=enc.nx,
        C_cam=C_cam, nz=nx2,
        nx0=nx0, nx1=nx1,
        soc=args.soc,
    )

    print('导出完成!\n')
    print('推理流程说明:')
    print('  Stream 1 → lidar_branch.om  (async) ─┐')
    print('  Stream 2 → camera_branch.om (async) ─┤→ sync barrier → fusion_head.om')
    print(f'  理论 NPU 延迟: max(lidar, camera) + fusion = 100 + 52 ≈ 152ms/帧')


if __name__ == '__main__':
    main()
