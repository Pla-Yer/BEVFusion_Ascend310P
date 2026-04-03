"""
BEVFusion Fusion Head AMCT 量化脚本
================================================================================
目标 ONNX  : bevfusion_fusion_head.onnx
输入       :
    camera_bev [1, 80,  360, 360]   float32
    lidar_bev  [1, 256, 360, 360]   float32   ← 注意实际通道数以模型为准
输出（10项）:
    dense_heatmap, top_cls, query_heatmap_score,
    heatmap_q, center, height, dim, rot, vel, top_idx

修复说明（v4）：
  ATC 编译量化后 deploy_model 时报两类错误，均通过量化阶段 skip_layers 解决：

  1. SwinAttentionScoreFusionPass 报错
     ATC 图融合 pass 尝试融合 Decoder 的 BMM 为 SwinAttention 格式，
     但 BEVFusion Decoder 的 attention 张量为 3D（不满足 >3D 的要求）。
     → 在 ATC 侧通过 --fusion_switch_file 关闭此 pass（见 print_atc_cmd）。

  2. requant_/.../Conv.dequant 编译失败
     position_embedding_head 和 prediction_heads 里的 Conv 量化后
     产生的 AscendDequant 算子在 310P 上无法编译（channel shape 不支持）。
     → 将这些 Conv 层自动加入 skip_layers，保持 FP16 精度。
"""

import os
import sys
import argparse
import numpy as np
import amct_onnx as amct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amct_session_utils import make_amct_session  # noqa: E402

# ──────────────────────────── 超参 ────────────────────────────
B              = 1
CAM_C, CAM_H, CAM_W       = 80,  360, 360
LIDAR_C, LIDAR_H, LIDAR_W = 256, 360, 360   # 以实际导出模型为准

TMP    = "./tmp/fusion"
RESULT = "./results/fusion"

ORT_DTYPE_MAP = {
    'tensor(float)'  : np.float32, 'tensor(float16)': np.float16,
    'tensor(float64)': np.float64, 'tensor(int8)'   : np.int8,
    'tensor(int16)'  : np.int16,   'tensor(int32)'  : np.int32,
    'tensor(int64)'  : np.int64,   'tensor(uint8)'  : np.uint8,
    'tensor(uint16)' : np.uint16,  'tensor(uint32)' : np.uint32,
    'tensor(uint64)' : np.uint64,  'tensor(bool)'   : np.bool_,
}


# ──────────────────────── skip_layers 自动收集 ────────────────
def collect_skip_layers(onnx_path: str, soc: str) -> list:
    """
    收集需要跳过量化的算子名，包含两类：

    1. 双变量 MatMul（Transformer Decoder attention）：
       310P 上量化无收益，且会触发 SwinAttentionScoreFusionPass 报错。

    2. position_embedding_head / prediction_heads 里的 Conv：
       量化后对应的 AscendDequant 在 310P 上无法编译。
       通过关键词匹配节点名自动识别。
    """
    import onnx
    model  = onnx.load(onnx_path)
    graph  = model.graph
    consts = {init.name for init in graph.initializer}
    consts |= {n.output[0] for n in graph.node if n.op_type == 'Constant'}

    skip = []

    # ── 关键词匹配：dequant 编译失败的 Conv ──────────────────
    # 匹配 position_embedding_head 和 prediction_heads 下的所有 Conv
    PROBLEM_KEYWORDS = [
        'position_embedding_head',  # cross/self posembed Conv
        'prediction_heads',         # center / height / dim / rot / vel Conv
        'heatmap_head',             # dense heatmap Conv（通常 1×1，channel 敏感）
    ]
    for node in graph.node:
        if node.op_type == 'Conv':
            name = node.name or ''
            if any(kw in name for kw in PROBLEM_KEYWORDS):
                if node.name and node.name not in skip:
                    skip.append(node.name)

    # ── 双变量 MatMul（310P 无收益）──────────────────────────
    if '310P' in soc:
        for node in graph.node:
            if node.op_type == 'MatMul' and len(node.input) >= 2:
                if node.input[0] not in consts and node.input[1] not in consts:
                    if node.name and node.name not in skip:
                        skip.append(node.name)

    return skip


# ──────────────────────────── 校准 ────────────────────────────
def build_dummy_inputs(seed=0):
    rng = np.random.default_rng(seed)
    camera_bev = rng.standard_normal(
        (B, CAM_C,   CAM_H,   CAM_W  )).astype(np.float32)
    lidar_bev  = rng.standard_normal(
        (B, LIDAR_C, LIDAR_H, LIDAR_W)).astype(np.float32)
    return camera_bev, lidar_bev


def run_calibration(modified_onnx, calib_data_list, batch_num):
    session = make_amct_session(modified_onnx)
    expected = {inp.name: ORT_DTYPE_MAP.get(inp.type)
                for inp in session.get_inputs()}
    print("  modified model input specs:")
    for inp in session.get_inputs():
        print(f"    {inp.name:20s}  type={inp.type}  shape={inp.shape}")

    for i in range(batch_num):
        camera_bev, lidar_bev = calib_data_list[i % len(calib_data_list)]
        raw = {'camera_bev': camera_bev, 'lidar_bev': lidar_bev}
        feed = {}
        for name, arr in raw.items():
            tgt = expected.get(name)
            feed[name] = arr.astype(tgt) if (tgt and arr.dtype != tgt) else arr
        session.run(None, feed)
        print(f"  校准推理 [{i+1}/{batch_num}] done")


# ──────────────────────── ATC 命令打印 ────────────────────────
def print_atc_cmd(deploy_onnx, soc, cam_shape, lidar_shape, outdir):
    switch_file = os.path.join(outdir, 'fusion_switch.json')
    print(f"""
后续 ATC 编译命令（Fusion Head 量化，静态形状）：
─────────────────────────────────────────────────────────────────────
atc --model="{deploy_onnx}" \\
    --framework=5 \\
    --output="models/om/bevfusion_fusion_head_quant" \\
    --input_format=ND \\
    --input_shape="camera_bev:{cam_shape};lidar_bev:{lidar_shape}" \\
    --fusion_switch_file="{switch_file}" \\
    --soc_version={soc} \\
    --op_select_implmode=high_precision \\
    --precision_mode=allow_fp32_to_fp16 \\
    --log=warning
─────────────────────────────────────────────────────────────────────
说明：
  --fusion_switch_file 关闭了以下 ATC 图融合 pass，避免 BEVFusion
  Transformer Decoder 的 3D BMM 触发只支持 4D 的 Swin 融合 pass：
    · SwinAttentionScoreFusionPass
    · AttentionQKVFusionPass
    · AttentionLnQKVFusionPass
""")


# ──────────────────────────── main ────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fusion Head AMCT 量化")
    parser.add_argument("--onnx",        default="models/onnx_parallel_npu/bevfusion_fusion_head.onnx")
    parser.add_argument("--batch-num",   type=int, default=1)
    parser.add_argument("--skip-layers", nargs="*", default=None,
                        help="手动指定 skip_layers；默认自动收集（推荐保持默认）")
    parser.add_argument("--soc",         default="Ascend310P1")
    parser.add_argument("--outdir",      default="models/onnx_parallel_npu",
                        help="fusion_switch.json 输出目录")
    args = parser.parse_args()

    os.makedirs(TMP, exist_ok=True)
    os.makedirs(RESULT, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    ori_model      = os.path.realpath(args.onnx)
    config_file    = os.path.join(TMP, "config.json")
    modified_model = os.path.join(TMP, "modified_model.onnx")
    record_file    = os.path.join(TMP, "record.txt")
    save_path      = os.path.join(RESULT, "fusion_head_quant")

    # ── 确定 skip_layers ──────────────────────────────────────
    if args.skip_layers is not None:
        skip_layers = args.skip_layers
        print(f"[*] 使用手动 skip_layers（{len(skip_layers)} 个）")
    else:
        print("[*] 自动收集 skip_layers ...")
        skip_layers = collect_skip_layers(ori_model, args.soc)
        print(f"  共跳过 {len(skip_layers)} 个算子：")
        for name in skip_layers:
            print(f"    · {name}")

    # ── 生成 fusion_switch.json ───────────────────────────────
    switch_file = os.path.join(args.outdir, "fusion_switch.json")
    import json
    switch_cfg = {
        "Switch": {
            "GraphFusion": {
                "SwinAttentionScoreFusionPass": "off",
                "AttentionQKVFusionPass":       "off",
                "AttentionLnQKVFusionPass":     "off",
            }
        }
    }
    with open(switch_file, 'w') as f:
        json.dump(switch_cfg, f, indent=4)
    print(f"\n[*] fusion_switch.json 已生成: {switch_file}")

    # ── Step 1 ────────────────────────────────────────────────
    print("\n[1/4] 生成量化配置 config.json ...")
    amct.create_quant_config(
        config_file=config_file, model_file=ori_model,
        skip_layers=skip_layers, batch_num=args.batch_num,
        activation_offset=True,
    )
    print(f"  ✓ {config_file}")

    # ── Step 2 ────────────────────────────────────────────────
    print("[2/4] 量化图修改，插入校准算子 ...")
    amct.quantize_model(
        config_file=config_file, model_file=ori_model,
        modified_onnx_file=modified_model, record_file=record_file,
    )
    print(f"  ✓ {modified_model}")

    # ── Step 3 ────────────────────────────────────────────────
    print("[3/4] 执行校准推理 ...")
    # *** 生产环境：替换为 LiDAR/Camera 分支真实推理得到的 BEV 特征 ***
    calib_data_list = [build_dummy_inputs(seed=i) for i in range(args.batch_num)]
    run_calibration(modified_model, calib_data_list, batch_num=args.batch_num)
    print("  ✓ 校准推理完成")

    # ── Step 4 ────────────────────────────────────────────────
    print("[4/4] 保存量化模型 ...")
    amct.save_model(
        modified_onnx_file=modified_model,
        record_file=record_file,
        save_path=save_path,
    )
    print(f"  ✓ deploy  : {save_path}_deploy_model.onnx")
    print(f"  ✓ fakequant: {save_path}_fake_quant_model.onnx")

    # ── 打印 ATC 命令 ─────────────────────────────────────────
    deploy_onnx = f"{save_path}_deploy_model.onnx"
    cam_shape   = f"{B},{CAM_C},{CAM_H},{CAM_W}"
    lidar_shape = f"{B},{LIDAR_C},{LIDAR_H},{LIDAR_W}"
    print_atc_cmd(deploy_onnx, args.soc, cam_shape, lidar_shape, args.outdir)


if __name__ == "__main__":
    main()
