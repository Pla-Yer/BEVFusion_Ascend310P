"""
BEVFusion LiDAR 分支精度对比脚本
================================================================================
对比原始 ONNX 与 AMCT 量化后 fake_quant ONNX 的推理精度。

输出指标（逐输出张量）：
  - 余弦相似度  (Cosine Similarity)
  - 最大绝对误差 (Max Absolute Error)
  - 平均绝对误差 (Mean Absolute Error)
  - 相对误差    (Relative Error = MAE / mean(|origin|))
  - 信噪比      (SNR, dB)

fake_quant 模型在 CPU/ORT 环境下模拟量化精度，结果可直接反映
部署到昇腾 NPU 后的精度损失。
"""

import os
import sys
import argparse
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amct_session_utils import make_amct_session  # noqa: E402

# ──────────────────────── dtype 映射 ──────────────────────────
ORT_DTYPE_MAP = {
    'tensor(float)'  : np.float32, 'tensor(float16)': np.float16,
    'tensor(float64)': np.float64, 'tensor(int8)'   : np.int8,
    'tensor(int16)'  : np.int16,   'tensor(int32)'  : np.int32,
    'tensor(int64)'  : np.int64,   'tensor(uint8)'  : np.uint8,
    'tensor(uint16)' : np.uint16,  'tensor(uint32)' : np.uint32,
    'tensor(uint64)' : np.uint64,  'tensor(bool)'   : np.bool_,
}


# ──────────────────────── Session 工厂 ────────────────────────
def make_session(onnx_path, use_amct=False):
    """
    use_amct=True  → 注册 AMCT 自定义 op（用于 modified / fake_quant 模型）
    use_amct=False → 原生 ORT（用于原始 ONNX）
    """
    if use_amct:
        return make_amct_session(onnx_path)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=sess_opts,
                                providers=['CPUExecutionProvider'])


# ──────────────────────── 自动 cast 并推理 ────────────────────
def run_inference(session, raw_feed: dict):
    """按 session 期望 dtype 自动 cast 后执行推理，返回输出列表。"""
    expected = {inp.name: ORT_DTYPE_MAP.get(inp.type) for inp in session.get_inputs()}
    feed = {}
    for name, arr in raw_feed.items():
        tgt = expected.get(name)
        feed[name] = arr.astype(tgt) if (tgt is not None and arr.dtype != tgt) else arr
    out_names = [o.name for o in session.get_outputs()]
    return session.run(None, feed), out_names


# ──────────────────────── 指标计算 ────────────────────────────
def cosine_similarity(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float('nan')


def snr_db(origin, quant):
    signal  = np.mean(origin.astype(np.float64) ** 2)
    noise   = np.mean((origin.astype(np.float64) - quant.astype(np.float64)) ** 2)
    if noise < 1e-30:
        return float('inf')
    return float(10 * np.log10(signal / noise))


def compare_outputs(ori_outs, qnt_outs, out_names):
    """逐张量打印精度指标，返回各张量余弦相似度列表。"""
    cos_list = []
    header = f"{'Output':<30s} {'Shape':<20s} {'CosSim':>8s} {'MaxAE':>10s} {'MAE':>10s} {'RelErr':>8s} {'SNR(dB)':>10s}"
    print(header)
    print("─" * len(header))
    for name, ori, qnt in zip(out_names, ori_outs, qnt_outs):
        ori = np.array(ori, dtype=np.float64)
        qnt = np.array(qnt, dtype=np.float64)
        cos  = cosine_similarity(ori, qnt)
        mae  = float(np.mean(np.abs(ori - qnt)))
        maxae= float(np.max(np.abs(ori - qnt)))
        base = float(np.mean(np.abs(ori)))
        rel  = mae / base if base > 1e-12 else float('nan')
        snr  = snr_db(ori, qnt)
        cos_list.append(cos)
        print(f"{name:<30s} {str(ori.shape):<20s} {cos:>8.5f} {maxae:>10.4e} {mae:>10.4e} {rel:>8.4f} {snr:>10.2f}")
    return cos_list


# ──────────────────────── dummy 数据 ──────────────────────────
def build_dummy_inputs(max_voxels=8000, seed=42):
    np.random.seed(seed)
    V = max_voxels
    voxels     = np.random.randn(V, 32, 5).astype(np.float32)
    num_points = np.random.randint(1, 33, size=(V,)).astype(np.int32)
    coords_zy  = np.random.randint(0, 360, size=(V, 2)).astype(np.int32)
    coords     = np.concatenate(
        [np.zeros((V, 1), np.int32), np.zeros((V, 1), np.int32), coords_zy], axis=1)
    return {'voxels': voxels, 'num_points': num_points, 'coords': coords}


# ──────────────────────────── main ────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LiDAR 分支精度对比")
    parser.add_argument("--ori-onnx",  default="models/onnx_parallel_npu/bevfusion_lidar_branch.onnx",
                        help="原始 ONNX 路径")
    parser.add_argument("--qnt-onnx",  default="results/lidar/lidar_branch_quant_fake_quant_model.onnx",
                        help="量化 fake_quant ONNX 路径")
    parser.add_argument("--max-voxels", type=int, default=8000)
    parser.add_argument("--num-iters",  type=int, default=5,
                        help="对比推理次数（取平均指标）")
    args = parser.parse_args()

    print("=" * 72)
    print("BEVFusion LiDAR 分支精度对比")
    print("=" * 72)
    print(f"  原始 ONNX : {args.ori_onnx}")
    print(f"  量化 ONNX : {args.qnt_onnx}")
    print()

    ori_sess = make_session(args.ori_onnx, use_amct=False)
    # fake_quant 模型通常为纯 ORT 算子；若报 amct.customop 错误请改为 use_amct=True
    try:
        qnt_sess = make_session(args.qnt_onnx, use_amct=False)
    except Exception:
        print("  [INFO] fake_quant 含 AMCT 自定义 op，切换为 make_amct_session")
        qnt_sess = make_session(args.qnt_onnx, use_amct=True)

    all_cos = []
    for it in range(args.num_iters):
        raw_feed = build_dummy_inputs(args.max_voxels, seed=it)
        ori_outs, out_names = run_inference(ori_sess, raw_feed)
        qnt_outs, _         = run_inference(qnt_sess, raw_feed)

        print(f"\n── Iter {it+1}/{args.num_iters} ────────────────────────────────────────")
        cos_list = compare_outputs(ori_outs, qnt_outs, out_names)
        all_cos.append(cos_list)

    # 汇总
    all_cos = np.array(all_cos)          # [num_iters, num_outputs]
    mean_cos = all_cos.mean(axis=0)
    print(f"\n{'═'*72}")
    print(f"平均余弦相似度（{args.num_iters} 次）：")
    for name, c in zip(out_names, mean_cos):
        flag = "✓" if c >= 0.999 else ("△" if c >= 0.99 else "✗")
        print(f"  {flag} {name:<30s} {c:.6f}")
    print(f"{'═'*72}")
    overall = float(mean_cos.mean())
    print(f"  整体平均余弦相似度: {overall:.6f}  "
          f"{'[PASS ≥0.999]' if overall >= 0.999 else '[WARN <0.999]'}")


if __name__ == "__main__":
    main()
