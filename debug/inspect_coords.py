"""
诊断 coords 实际列顺序 + PointPillarsScatter 源码
用法: python inspect_coords.py --config ... --ckpt ...
"""
import argparse, sys, os, inspect
import argparse, sys, os
from os import path as osp
sys.path.insert(0, osp.dirname(__file__))
try:
    import fix_pytorch_weights_only  # noqa
except ModuleNotFoundError:
    pass

import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
    p.add_argument("--ckpt",   default="work_dirs/bevfusion/epoch_20.pth")
    args = p.parse_args()

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    model = init_model(cfg, args.ckpt)
    model.eval()

    # ── 打印 PointPillarsScatter 源码 ────────────────────────────────────
    enc = model.pts_middle_encoder
    print("=" * 70)
    print("PointPillarsScatter 完整源码:")
    print("=" * 70)
    print(inspect.getsource(type(enc)))

    # ── 用真实数据跑一次，打印 coords 实际内容 ───────────────────────────
    print("=" * 70)
    print("实际推理时 coords 的内容（前10行）:")
    print("=" * 70)

    # Hook 拦截 pts_middle_encoder 的输入
    captured = {}
    def hook(module, inputs, output):
        captured['voxel_features'] = inputs[0].detach().cpu()
        captured['coors']          = inputs[1].detach().cpu()

    handle = enc.register_forward_hook(hook)

    V, M, Cin = 200, 32, 5
    torch.manual_seed(0)
    device = "cuda"
    voxels     = torch.randn(V, M, Cin, device=device)
    num_points = torch.randint(1, M+1, (V,), device=device)
    # 模拟导出脚本中的 coords 格式
    coords_raw = torch.zeros(V, 4, device=device)
    coords_raw[:, 1] = torch.randint(0, 1,   (V,)).float()
    coords_raw[:, 2] = torch.randint(0, 360, (V,)).float()
    coords_raw[:, 3] = torch.randint(0, 360, (V,)).float()

    # 导出脚本中先做了 coords[:, [0,3,1,2]] 再传入 middle_encoder
    coords_bzyx = coords_raw[:, [0, 3, 1, 2]].contiguous()

    with torch.no_grad():
        vf = model.pts_voxel_encoder(voxels, num_points, coords_raw)
        _ = enc(vf, coords_bzyx, 1)
    handle.remove()

    coors = captured['coors']
    print(f"coors shape: {coors.shape}")
    print(f"coors[:10]:\n{coors[:10]}")
    print(f"\ncoors 各列范围:")
    for i in range(coors.shape[1]):
        col = coors[:, i]
        print(f"  col[{i}]: min={col.min().item():.0f}  max={col.max().item():.0f}  "
              f"unique_count={col.unique().numel()}")

    print()
    print("enc.ny (H):", enc.ny)
    print("enc.nx (W):", enc.nx)

    # ── 同时打印原始 forward 源码的变量名，看它用哪列做 y/x ────────────
    print()
    print("提示：请对照上面源码确认 coors[:, ?] 是 batch_id / z / y / x")

if __name__ == "__main__":
    main()