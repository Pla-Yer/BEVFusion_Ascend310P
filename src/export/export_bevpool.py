"""
BEVPool 独立 ONNX 导出脚本
================================================================================
输入:
    cam_feats  : [B, N, C, D, fH, fW]   e.g. [1, 6, 80, 118, 32, 88]
    geom_feats : [B, N, D, fH, fW, 3]   e.g. [1, 6, 118, 32, 88, 3]

输出:
    bev_feat   : [B, C*nz, nx0, nx1]    e.g. [1, 80, 360, 360]

用途: 评估 BEVPool 在 NPU OM 上的独立推理耗时
"""

import argparse
import os
from os import path as osp
import sys
import torch
import torch.nn as nn

sys.path.insert(0, osp.dirname(__file__))
try:
    import fix_pytorch_weights_only  # noqa
except ModuleNotFoundError:
    pass

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model


class BEVPoolOnnx(nn.Module):
    """
    独立 BEVPool 模块，仅包含 scatter_add 投影逻辑。
    所有标量在 __init__ 固化为 Python int，forward 内无 .item() 调用。
    """

    def __init__(self, vt, B=1, N=6, fH=32, fW=88):
        super().__init__()

        # BEV 网格参数：register_buffer 跟随 .to(device)
        self.register_buffer('_bx', vt.bx.detach().clone().float())
        self.register_buffer('_dx', vt.dx.detach().clone().float())

        # Python int 常量
        self._nx0 = int(round(vt.nx[0].item()))
        self._nx1 = int(round(vt.nx[1].item()))
        self._nx2 = int(round(vt.nx[2].item()))
        self._C   = int(vt.C)
        self._D   = int(vt.D)
        self._B   = int(B)
        self._N   = int(N)
        self._fH  = int(fH)
        self._fW  = int(fW)

        # 预计算常量
        self._pts_per_batch = int(N) * self._D * int(fH) * int(fW)
        self._out_cells     = int(B) * self._nx2 * self._nx0 * self._nx1

    def forward(self, cam_feats, geom_feats):
        """
        Args:
            cam_feats  : [B, N, C, D, fH, fW]
            geom_feats : [B, N, D, fH, fW, 3]
        Returns:
            bev_feat   : [B, C*nz, nx0, nx1]
        """
        B   = self._B
        C   = self._C
        nx0 = self._nx0
        nx1 = self._nx1
        nx2 = self._nx2
        ppb = self._pts_per_batch
        out_cells = self._out_cells

        # [B,N,C,D,fH,fW] → [Nprime, C]
        feats  = cam_feats.permute(0, 1, 3, 4, 5, 2).contiguous().reshape(-1, C)
        coords = geom_feats.reshape(-1, 3)

        # 格网索引
        idx  = ((coords - (self._bx - self._dx * 0.5)) / self._dx).to(torch.int64)
        x_id = idx[:, 0]
        y_id = idx[:, 1]
        z_id = idx[:, 2]

        # batch id：先在 CPU 建 arange，再迁移到 feats.device
        b_id = (torch.arange(B, dtype=torch.int64)
                    .to(feats.device)
                    .repeat_interleave(ppb))

        # 有效 mask
        valid = ((x_id >= 0) & (x_id < nx0) &
                 (y_id >= 0) & (y_id < nx1) &
                 (z_id >= 0) & (z_id < nx2))
        feats = feats * valid.to(feats.dtype).unsqueeze(1)

        x_id = x_id.clamp(0, nx0 - 1)
        y_id = y_id.clamp(0, nx1 - 1)
        z_id = z_id.clamp(0, nx2 - 1)

        # 线性索引
        lin = (b_id * (nx2 * nx0 * nx1)
               + z_id * (nx0 * nx1)
               + x_id * nx1
               + y_id)

        # scatter_add（ONNX opset-11 兼容）
        out = torch.zeros(out_cells, C, device=feats.device, dtype=feats.dtype)
        out = out.scatter_add(0, lin.view(-1, 1).expand(-1, C), feats)

        out = out.view(B, nx2, nx0, nx1, C).permute(0, 4, 1, 2, 3).contiguous()
        return out.reshape(B, C * nx2, nx0, nx1)


def main():
    p = argparse.ArgumentParser(description='BEVPool 独立 ONNX 导出')
    p.add_argument('--config',
                   default='src/configs/bevfusion_lidar-cam_voxel03_second_secfpn_'
                           '8xb4-cyclic-20e_nus-3d_resnet50.py')
    p.add_argument('--ckpt',
                   default='work_dirs/bevfusion_lidar-cam_voxel03_second_secfpn_'
                           '8xb4-cyclic-20e_nus-3d_resnet50/epoch_6.pth')
    p.add_argument('--outdir', default='models/onnx_bevpool')
    p.add_argument('--soc',    default='Ascend310P1')
    p.add_argument('--B',  type=int, default=1)
    p.add_argument('--N',  type=int, default=6)
    p.add_argument('--fH', type=int, default=32)
    p.add_argument('--fW', type=int, default=88)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = 'cuda'

    # ── 加载模型取参数 ────────────────────────────────────────────────────
    print('[1/3] 加载模型...')
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    if 'img_backbone' in cfg.model:
        cfg.model['img_backbone']['init_cfg'] = None
    model = init_model(cfg, args.ckpt)
    model.eval()

    vt = model.view_transform
    D  = int(vt.D)
    C  = int(vt.C)
    B, N, fH, fW = args.B, args.N, args.fH, args.fW

    print(f'  D={D}  C={C}  grid=({vt.nx[0].item():.0f},'
          f'{vt.nx[1].item():.0f},{vt.nx[2].item():.0f})')

    # ── 构建模块 & 生成哑元输入 ───────────────────────────────────────────
    print('[2/3] 构建 BEVPoolOnnx 并运行前向验证...')
    bev_pool = BEVPoolOnnx(vt, B=B, N=N, fH=fH, fW=fW).eval().to(device)

    torch.manual_seed(0)
    cam_feats  = torch.randn(B, N, C, D, fH, fW, device=device)
    geom_feats = torch.zeros(B, N, D, fH, fW, 3, device=device)
    geom_feats[..., 0] = torch.rand(B, N, D, fH, fW, device=device) * 108 - 54
    geom_feats[..., 1] = torch.rand(B, N, D, fH, fW, device=device) * 108 - 54
    geom_feats[..., 2] = torch.rand(B, N, D, fH, fW, device=device) * 20  - 10

    with torch.no_grad():
        out = bev_pool(cam_feats, geom_feats)
    print(f'  前向验证: cam_feats {tuple(cam_feats.shape)}'
          f' → bev_feat {tuple(out.shape)}')

    # ── 导出 ONNX ─────────────────────────────────────────────────────────
    print('[3/3] 导出 ONNX...')
    onnx_path = osp.join(args.outdir, 'bevpool.onnx')

    with torch.no_grad():
        torch.onnx.export(
            bev_pool,
            (cam_feats, geom_feats),
            onnx_path,
            opset_version=11,
            input_names=['cam_feats', 'geom_feats'],
            output_names=['bev_feat'],
            do_constant_folding=True,
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            export_params=True,
            verbose=False,
        )
    print(f'  ✓ 已保存: {onnx_path}')

    # ── 保存测试 bin ──────────────────────────────────────────────────────
    bindir = osp.join(args.outdir, 'test_bins')
    os.makedirs(bindir, exist_ok=True)
    cam_feats.cpu().numpy().tofile(osp.join(bindir, 'cam_feats.bin'))
    geom_feats.cpu().numpy().tofile(osp.join(bindir, 'geom_feats.bin'))
    out.cpu().numpy().tofile(osp.join(bindir, 'bev_feat.bin'))
    print(f'  ✓ 测试 bin 已保存到 {bindir}/')

    # ── ATC 编译命令 ───────────────────────────────────────────────────────
    print(f"""
ATC 编译命令:
─────────────────────────────────────────────────────────────────────────
atc --model="{onnx_path}" \\
    --framework=5 \\
    --output="models/om/bevpool" \\
    --input_format=ND \\
    --input_shape="cam_feats:{B},{N},{C},{D},{fH},{fW};geom_feats:{B},{N},{D},{fH},{fW},3" \\
    --soc_version={args.soc} \\
    --op_select_implmode=high_precision \\
    --precision_mode=allow_fp32_to_fp16 \\
    --log=warning
─────────────────────────────────────────────────────────────────────────
""")


if __name__ == '__main__':
    main()