"""
BEVFusion Fusion Head 精度对比脚本
================================================================================
对比原始 ONNX 与 AMCT 量化后 fake_quant ONNX 的推理精度。

输入：
    camera_bev [1, 80, 360, 360]   float32
    lidar_bev  [1, 256, 360, 360]   float32
输出（10项）：
    dense_heatmap, top_cls, query_heatmap_score,
    heatmap_q, center, height, dim, rot, vel, top_idx

输出指标：
  余弦相似度 / 最大绝对误差 / 平均绝对误差 / 相对误差 / SNR(dB)

附加指标（检测头专项）：
  - heatmap 的 Top-K 命中率：量化前后 Top-K proposal index 的重合比例
  - center / dim / rot 的平均位置偏差
"""

import os
import sys
import argparse
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amct_session_utils import make_amct_session  # noqa: E402

ORT_DTYPE_MAP = {
    'tensor(float)'  : np.float32, 'tensor(float16)': np.float16,
    'tensor(float64)': np.float64, 'tensor(int8)'   : np.int8,
    'tensor(int16)'  : np.int16,   'tensor(int32)'  : np.int32,
    'tensor(int64)'  : np.int64,   'tensor(uint8)'  : np.uint8,
    'tensor(uint16)' : np.uint16,  'tensor(uint32)' : np.uint32,
    'tensor(uint64)' : np.uint64,  'tensor(bool)'   : np.bool_,
}

# 与导出脚本保持一致
B              = 1
CAM_C, CAM_H, CAM_W       = 80, 360, 360
LIDAR_C, LIDAR_H, LIDAR_W = 256, 360, 360

# 输出名称列表（与导出脚本 output_names 对齐）
OUTPUT_NAMES = [
    'dense_heatmap', 'top_cls', 'query_heatmap_score',
    'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel', 'top_idx'
]


def make_session(onnx_path, use_amct=False):
    if use_amct:
        return make_amct_session(onnx_path)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=sess_opts,
                                providers=['CPUExecutionProvider'])


def run_inference(session, raw_feed):
    expected = {inp.name: ORT_DTYPE_MAP.get(inp.type) for inp in session.get_inputs()}
    feed = {}
    for name, arr in raw_feed.items():
        tgt = expected.get(name)
        feed[name] = arr.astype(tgt) if (tgt is not None and arr.dtype != tgt) else arr
    out_names = [o.name for o in session.get_outputs()]
    return session.run(None, feed), out_names


def cosine_similarity(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float('nan')


def snr_db(origin, quant):
    signal = np.mean(origin.astype(np.float64) ** 2)
    noise  = np.mean((origin.astype(np.float64) - quant.astype(np.float64)) ** 2)
    if noise < 1e-30:
        return float('inf')
    return float(10 * np.log10(signal / noise))


def compare_outputs(ori_outs, qnt_outs, out_names):
    """通用逐张量精度表。"""
    cos_list = []
    header = f"{'Output':<26s} {'Shape':<22s} {'CosSim':>8s} {'MaxAE':>10s} {'MAE':>10s} {'RelErr':>8s} {'SNR(dB)':>10s}"
    print(header)
    print("─" * len(header))
    for name, ori, qnt in zip(out_names, ori_outs, qnt_outs):
        ori_f = np.array(ori, dtype=np.float64)
        qnt_f = np.array(qnt, dtype=np.float64)
        cos   = cosine_similarity(ori_f, qnt_f)
        mae   = float(np.mean(np.abs(ori_f - qnt_f)))
        maxae = float(np.max(np.abs(ori_f - qnt_f)))
        base  = float(np.mean(np.abs(ori_f)))
        rel   = mae / base if base > 1e-12 else float('nan')
        snr   = snr_db(ori_f, qnt_f)
        cos_list.append(cos)
        print(f"{name:<26s} {str(ori_f.shape):<22s} {cos:>8.5f} {maxae:>10.4e} {mae:>10.4e} {rel:>8.4f} {snr:>10.2f}")
    return cos_list


def detection_specific_metrics(ori_outs, qnt_outs, out_names):
    """
    检测头专项指标：
      1. Top-K index 命中率：ori 与 qnt 的 top_idx 重合比例
      2. center / dim / rot 平均 L2 偏差（仅对重合 proposal 计算）
    """
    name2ori = dict(zip(out_names, ori_outs))
    name2qnt = dict(zip(out_names, qnt_outs))

    print("\n  ── 检测头专项指标 ─────────────────────────────────────────")

    # 1. Top-K proposal index 命中率
    if 'top_idx' in name2ori:
        ori_idx = np.array(name2ori['top_idx']).flatten()
        qnt_idx = np.array(name2qnt['top_idx']).flatten()
        K = len(ori_idx)
        hit = len(set(ori_idx.tolist()) & set(qnt_idx.tolist()))
        print(f"  Top-K index 命中率 : {hit}/{K} = {hit/K*100:.2f}%")

    # 2. center / dim / rot L2 偏差（全 proposal 平均）
    for key in ('center', 'dim', 'rot'):
        if key in name2ori:
            o = np.array(name2ori[key], dtype=np.float64)
            q = np.array(name2qnt[key], dtype=np.float64)
            l2 = float(np.mean(np.sqrt(np.sum((o - q) ** 2, axis=1))))
            print(f"  {key:<8s} 平均 L2 偏差 : {l2:.6f}")

    # 3. dense_heatmap sigmoid 响应分布对比
    if 'dense_heatmap' in name2ori:
        oh = 1 / (1 + np.exp(-np.array(name2ori['dense_heatmap'], dtype=np.float64)))
        qh = 1 / (1 + np.exp(-np.array(name2qnt['dense_heatmap'], dtype=np.float64)))
        print(f"  dense_heatmap sigmoid 统计：")
        print(f"    ori  mean={oh.mean():.5f}  max={oh.max():.5f}")
        print(f"    qnt  mean={qh.mean():.5f}  max={qh.max():.5f}")
        print(f"    MAE={float(np.mean(np.abs(oh-qh))):.6f}")


def build_dummy_inputs(seed=42):
    rng = np.random.default_rng(seed)
    camera_bev = rng.standard_normal((B, CAM_C,   CAM_H,   CAM_W  )).astype(np.float32)
    lidar_bev  = rng.standard_normal((B, LIDAR_C, LIDAR_H, LIDAR_W)).astype(np.float32)
    return {'camera_bev': camera_bev, 'lidar_bev': lidar_bev}


def main():
    parser = argparse.ArgumentParser(description="Fusion Head 精度对比")
    parser.add_argument("--ori-onnx",  default="models/onnx_parallel_npu/bevfusion_fusion_head.onnx")
    parser.add_argument("--qnt-onnx",  default="results/fusion/fusion_head_quant_fake_quant_model.onnx")
    parser.add_argument("--num-iters", type=int, default=5)
    args = parser.parse_args()

    print("=" * 72)
    print("BEVFusion Fusion Head 精度对比")
    print("=" * 72)
    print(f"  原始 ONNX : {args.ori_onnx}")
    print(f"  量化 ONNX : {args.qnt_onnx}")
    print()

    ori_sess = make_session(args.ori_onnx, use_amct=False)
    try:
        qnt_sess = make_session(args.qnt_onnx, use_amct=False)
    except Exception:
        print("  [INFO] fake_quant 含 AMCT 自定义 op，切换为 make_amct_session")
        qnt_sess = make_session(args.qnt_onnx, use_amct=True)

    all_cos = []
    for it in range(args.num_iters):
        raw_feed = build_dummy_inputs(seed=it)
        ori_outs, out_names = run_inference(ori_sess, raw_feed)
        qnt_outs, _         = run_inference(qnt_sess, raw_feed)

        print(f"\n── Iter {it+1}/{args.num_iters} ────────────────────────────────────────")
        cos_list = compare_outputs(ori_outs, qnt_outs, out_names)
        all_cos.append(cos_list)

        # 仅第一次打印检测头专项指标（避免重复冗长）
        if it == 0:
            detection_specific_metrics(ori_outs, qnt_outs, out_names)

    all_cos  = np.array(all_cos)
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

    # 精度不达标时给出建议
    for name, c in zip(out_names, mean_cos):
        if c < 0.99:
            print(f"\n  [建议] '{name}' 余弦相似度 {c:.4f} < 0.99，"
                  f"可将对应敏感层加入 --skip-layers 后重新量化。")


if __name__ == "__main__":
    main()
