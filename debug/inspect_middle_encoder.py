"""
检查 pts_middle_encoder 的具体类型和内部算子
用法: python inspect_middle_encoder.py --config ... --ckpt ...
"""
import argparse, sys, os
from os import path as osp
sys.path.insert(0, os.path.dirname(__file__))
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

    enc = model.pts_middle_encoder
    print("=" * 60)
    print(f"pts_middle_encoder 类型: {type(enc)}")
    print(f"模块名: {enc.__class__.__module__}.{enc.__class__.__name__}")
    print()
    print("完整结构:")
    print(enc)
    print("=" * 60)

    # 打印 config 中的定义
    print("\nconfig 中 pts_middle_encoder 配置:")
    import json
    try:
        mid_cfg = cfg.model.get("pts_middle_encoder", {})
        print(json.dumps(dict(mid_cfg), indent=2, default=str))
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()