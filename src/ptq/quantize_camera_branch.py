"""
BEVFusion Camera 分支 AMCT 量化脚本
================================================================================
目标 ONNX  : bevfusion_camera_branch.onnx
输入       :
    imgs        [B, N, 3,  H,  W]   float32
    depth       [B, N, 1,  H,  W]   float32
    pool_lookup [OUT_CELLS, MAX_PTS] int32
    pool_mask   [OUT_CELLS, MAX_PTS] uint8
输出       : camera_bev [B, 80, 360, 360]

修复说明（v3）：
  run_calibration() 从 Session 读取每个输入的期望 dtype，
  喂数据前自动 cast，避免 "Unexpected input data type" 错误。
"""

import os
import sys
import argparse
import numpy as np
import amct_onnx as amct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amct_session_utils import make_amct_session  # noqa: E402

B, N     = 1, 6
H, W     = 256, 704
fH, fW   = 32, 88
MAX_PTS  = 16
NX0, NX1, NX2 = 360, 360, 1
OUT_CELLS = B * NX2 * NX0 * NX1   # 129600

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
TMP    = os.path.join(SCRIPT_DIR, "tmp/camera")
RESULT = os.path.join(SCRIPT_DIR, "results/camera")

ORT_DTYPE_MAP = {
    'tensor(float)'   : np.float32,
    'tensor(float16)' : np.float16,
    'tensor(float64)' : np.float64,
    'tensor(int8)'    : np.int8,
    'tensor(int16)'   : np.int16,
    'tensor(int32)'   : np.int32,
    'tensor(int64)'   : np.int64,
    'tensor(uint8)'   : np.uint8,
    'tensor(uint16)'  : np.uint16,
    'tensor(uint32)'  : np.uint32,
    'tensor(uint64)'  : np.uint64,
    'tensor(bool)'    : np.bool_,
}


def build_dummy_inputs(seed=0):
    rng = np.random.default_rng(seed)
    imgs        = rng.standard_normal((B, N, 3, H, W)).astype(np.float32)
    depth       = rng.standard_normal((B, N, 1, H, W)).astype(np.float32)
    total_pts   = B * N * fH * fW
    pool_lookup = rng.integers(0, total_pts, size=(OUT_CELLS, MAX_PTS)).astype(np.int32)
    pool_mask   = rng.integers(0, 2, size=(OUT_CELLS, MAX_PTS)).astype(np.uint8)
    return imgs, depth, pool_lookup, pool_mask


def run_calibration(modified_onnx, calib_data_list, batch_num):
    session = make_amct_session(modified_onnx)

    expected_dtypes = {}
    print("  modified model input specs:")
    for inp in session.get_inputs():
        np_dtype = ORT_DTYPE_MAP.get(inp.type)
        expected_dtypes[inp.name] = np_dtype
        print(f"    {inp.name:20s}  ort_type={inp.type:20s}  shape={inp.shape}")

    for i in range(batch_num):
        imgs, depth, pool_lookup, pool_mask = calib_data_list[i % len(calib_data_list)]
        raw_feed = {
            'imgs'       : imgs,
            'depth'      : depth,
            'pool_lookup': pool_lookup,
            'pool_mask'  : pool_mask,
        }
        feed = {}
        for name, arr in raw_feed.items():
            target = expected_dtypes.get(name)
            if target is not None and arr.dtype != target:
                print(f"  [cast] {name}: {arr.dtype} -> {np.dtype(target)}")
                arr = arr.astype(target)
            feed[name] = arr

        session.run(None, feed)
        print(f"  校准推理 [{i+1}/{batch_num}] done，imgs shape={imgs.shape}")


def main():
    parser = argparse.ArgumentParser(description="Camera 分支 AMCT 量化")
    parser.add_argument("--onnx",        default="models/onnx_parallel_npu/bevfusion_camera_branch.onnx")
    parser.add_argument("--batch-num",   type=int, default=1)
    parser.add_argument("--skip-layers", nargs="*", default=[])
    args = parser.parse_args()

    os.makedirs(TMP, exist_ok=True)
    os.makedirs(RESULT, exist_ok=True)

    ori_model      = os.path.realpath(args.onnx)
    config_file    = os.path.join(TMP, "config.json")
    modified_model = os.path.join(TMP, "modified_model.onnx")
    record_file    = os.path.join(TMP, "record.txt")
    save_path      = os.path.join(RESULT, "camera_branch_quant")

    print("[1/4] 生成量化配置 config.json ...")
    amct.create_quant_config(
        config_file=config_file, model_file=ori_model,
        skip_layers=args.skip_layers, batch_num=args.batch_num,
        activation_offset=True,
    )
    print(f"  ✓ {config_file}")

    print("[2/4] 量化图修改，插入校准算子 ...")
    amct.quantize_model(
        config_file=config_file, model_file=ori_model,
        modified_onnx_file=modified_model, record_file=record_file,
    )
    print(f"  ✓ {modified_model}")

    print("[3/4] 执行校准推理 ...")
    calib_data_list = [build_dummy_inputs(seed=i) for i in range(args.batch_num)]
    run_calibration(modified_model, calib_data_list, batch_num=args.batch_num)
    print("  ✓ 校准推理完成")

    print("[4/4] 保存量化模型 ...")
    amct.save_model(
        modified_onnx_file=modified_model,
        record_file=record_file,
        save_path=save_path,
    )
    print(f"  ✓ deploy  : {save_path}_deploy_model.onnx")
    print(f"  ✓ fakequant: {save_path}_fake_quant_model.onnx")

    deploy_onnx = f"{save_path}_deploy_model.onnx"
    cam_shape  = f"{B},{N},3,{H},{W}"
    dep_shape  = f"{B},{N},1,{H},{W}"
    lkp_shape  = f"{OUT_CELLS},{MAX_PTS}"
    mask_shape = f"{OUT_CELLS},{MAX_PTS}"
    print(f"""
后续 ATC 编译命令（Camera 分支，静态形状）：
  atc --model="{deploy_onnx}" --framework=5 \\
      --output="models/om/bevfusion_camera_branch_quant" \\
      --input_format=ND \\
      --input_shape="imgs:{cam_shape};depth:{dep_shape};pool_lookup:{lkp_shape};pool_mask:{mask_shape}" \\
      --soc_version=Ascend310P1 --precision_mode=allow_fp32_to_fp16 --log=warning
""")


if __name__ == "__main__":
    main()
