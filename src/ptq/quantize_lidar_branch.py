"""
BEVFusion LiDAR 分支 AMCT 量化脚本
================================================================================
目标 ONNX  : bevfusion_lidar_branch.onnx
输入       : voxels [V,32,5]  num_points [V]  coords [V,4]   (V 为动态轴)
输出       : lidar_bev [1,64,360,360]

AMCT 均匀量化流程（手工量化）：
  1. create_quant_config  -> config.json
  2. quantize_model       -> modified_model.onnx  (插入校准算子)
  3. ONNX Runtime 校准推理（batch_num 次）
  4. save_model           -> deploy / fake_quant 两个结果模型

修复说明（v3）：
  AMCT 在插入校准算子时会对图做若干优化 pass，可能将原本 int32 的输入
  （如 num_points / coords）的期望 dtype 改为 float32。
  run_calibration() 现在会从 Session 读取每个输入的期望 dtype，
  并在喂数据前自动 cast，避免 "Unexpected input data type" 错误。
"""

import os
import sys
import argparse
import numpy as np
import amct_onnx as amct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amct_session_utils import make_amct_session  # noqa: E402

# ──────────────────────────── 目录 ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
TMP    = os.path.join(SCRIPT_DIR, "tmp/lidar")
RESULT = os.path.join(SCRIPT_DIR, "results/lidar")

# ORT dtype 字符串 -> numpy dtype 映射
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


def build_dummy_inputs(max_voxels=8000, max_points=32, point_dim=5):
    """
    生成随机 LiDAR dummy 校准数据（原始 dtype，run_calibration 会按需 cast）。
    生产环境中请替换为真实 nuScenes 点云加载逻辑。
    """
    V = max_voxels
    voxels     = np.random.randn(V, max_points, point_dim).astype(np.float32)
    num_points = np.random.randint(1, max_points + 1, size=(V,)).astype(np.int32)
    coords_zy  = np.random.randint(0, 360, size=(V, 2)).astype(np.int32)
    batch_col  = np.zeros((V, 1), dtype=np.int32)
    z_col      = np.zeros((V, 1), dtype=np.int32)
    coords     = np.concatenate([batch_col, z_col, coords_zy], axis=1)
    return voxels, num_points, coords


def run_calibration(modified_onnx, calib_data_list, batch_num):
    """
    使用 ONNX Runtime 对 modified_model 执行 batch_num 次校准推理。

    修复要点：
      1. make_amct_session() 注册 AMCT 自定义 op，避免 "No opset for amct.customop"。
      2. 从 session.get_inputs() 读取每个输入的期望 dtype，喂数据前自动 cast，
         避免 AMCT pass 将 int32 输入改为 float 后报 "Unexpected input data type"。
    """
    session = make_amct_session(modified_onnx)

    # 建立 name -> 期望 numpy dtype 的映射
    expected_dtypes = {}
    print("  modified model input specs:")
    for inp in session.get_inputs():
        np_dtype = ORT_DTYPE_MAP.get(inp.type)
        expected_dtypes[inp.name] = np_dtype
        print(f"    {inp.name:20s}  ort_type={inp.type:20s}  shape={inp.shape}")

    for i in range(batch_num):
        voxels, num_points, coords = calib_data_list[i % len(calib_data_list)]

        # 原始 feed（使用模型原始语义 dtype）
        raw_feed = {
            'voxels'    : voxels,
            'num_points': num_points,
            'coords'    : coords,
        }

        # 按 modified model 的实际期望 dtype 进行 cast
        feed = {}
        for name, arr in raw_feed.items():
            target = expected_dtypes.get(name)
            if target is not None and arr.dtype != target:
                print(f"  [cast] {name}: {arr.dtype} -> {np.dtype(target)}")
                arr = arr.astype(target)
            feed[name] = arr

        session.run(None, feed)
        print(f"  校准推理 [{i+1}/{batch_num}] done，voxels shape={voxels.shape}")


def main():
    parser = argparse.ArgumentParser(description="LiDAR 分支 AMCT 量化")
    parser.add_argument("--onnx",        default="models/onnx_parallel_npu/bevfusion_lidar_branch.onnx")
    parser.add_argument("--max-voxels",  type=int, default=8000)
    parser.add_argument("--batch-num",   type=int, default=1)
    parser.add_argument("--skip-layers", nargs="*", default=[])
    args = parser.parse_args()

    os.makedirs(TMP, exist_ok=True)
    os.makedirs(RESULT, exist_ok=True)

    ori_model      = os.path.realpath(args.onnx)
    config_file    = os.path.join(TMP, "config.json")
    modified_model = os.path.join(TMP, "modified_model.onnx")
    record_file    = os.path.join(TMP, "record.txt")
    save_path      = os.path.join(RESULT, "lidar_branch_quant")

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
    # *** 生产环境：替换为真实 nuScenes 点云数据加载 ***
    calib_data_list = [build_dummy_inputs(args.max_voxels) for _ in range(args.batch_num)]
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
    print(f"""
后续 ATC 编译命令（LiDAR 分支，动态 voxel）：
  atc --model="{deploy_onnx}" --framework=5 \\
      --output="models/om/bevfusion_lidar_branch_quant_dynamic" \\
      --input_format=ND \\
      --input_shape="voxels:-1,32,5;num_points:-1;coords:-1,4" \\
      --dynamic_dims="6000,6000,6000;8000,8000,8000;10000,10000,10000" \\
      --soc_version=Ascend310P1 --precision_mode=allow_fp32_to_fp16 --log=warning
""")


if __name__ == "__main__":
    main()
