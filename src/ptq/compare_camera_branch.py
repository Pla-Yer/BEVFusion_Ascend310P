"""
BEVFusion Camera 分支精度对比脚本
================================================================================
对比原始 ONNX 与 AMCT 量化后 fake_quant ONNX 的推理精度。

输入：
    imgs        [B, N, 3,  H,  W]   float32
    depth       [B, N, 1,  H,  W]   float32
    pool_lookup [OUT_CELLS, MAX_PTS] int32
    pool_mask   [OUT_CELLS, MAX_PTS] uint8
输出：camera_bev [B, 80, 360, 360]

输出指标（逐输出张量）：
  余弦相似度 / 最大绝对误差 / 平均绝对误差 / 相对误差 / SNR(dB)
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
B, N     = 1, 6
H, W     = 256, 704
fH, fW   = 32, 88
MAX_PTS  = 16
OUT_CELLS = B * 1 * 360 * 360   # 129600


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

    # BEV 特征额外统计：逐通道余弦相似度分布
    if len(ori_outs) == 1:
        bev_ori = np.array(ori_outs[0])   # [B, C, H, W]
        bev_qnt = np.array(qnt_outs[0])
        ch_cos = []
        for c in range(bev_ori.shape[1]):
            ch_cos.append(cosine_similarity(bev_ori[0, c], bev_qnt[0, c]))
        ch_cos = np.array(ch_cos)
        print(f"\n  逐通道余弦相似度统计（共 {len(ch_cos)} 通道）：")
        print(f"    min={ch_cos.min():.5f}  max={ch_cos.max():.5f}  "
              f"mean={ch_cos.mean():.5f}  "
              f"<0.99 通道数: {int((ch_cos < 0.99).sum())}")
    return cos_list


def build_dummy_inputs(seed=42):
    rng = np.random.default_rng(seed)
    imgs        = rng.standard_normal((B, N, 3, H, W)).astype(np.float32)
    depth       = rng.standard_normal((B, N, 1, H, W)).astype(np.float32)
    total_pts   = B * N * fH * fW
    pool_lookup = rng.integers(0, total_pts, size=(OUT_CELLS, MAX_PTS)).astype(np.int32)
    pool_mask   = rng.integers(0, 2, size=(OUT_CELLS, MAX_PTS)).astype(np.uint8)
    return {'imgs': imgs, 'depth': depth,
            'pool_lookup': pool_lookup, 'pool_mask': pool_mask}


def main():
    parser = argparse.ArgumentParser(description="Camera 分支精度对比")
    parser.add_argument("--ori-onnx", default="models/onnx_parallel_npu/bevfusion_camera_branch.onnx")
    parser.add_argument("--qnt-onnx", default="results/camera/camera_branch_quant_fake_quant_model.onnx")
    parser.add_argument("--num-iters", type=int, default=5)
    args = parser.parse_args()

    print("=" * 72)
    print("BEVFusion Camera 分支精度对比")
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


if __name__ == "__main__":
    main()
