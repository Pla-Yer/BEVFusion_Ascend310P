import numpy as np


def safe_kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)

    mask = p > 0
    p_safe = p[mask]
    q_safe = q[mask] + eps
    return np.sum(p_safe * np.log(p_safe / q_safe))


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)
    m = 0.5 * (p + q)

    return 0.5 * safe_kl_divergence(p, m, eps) + 0.5 * safe_kl_divergence(q, m, eps)


def hist_distribution_metrics(x_ref, x_recon, num_bins=2048):
    """
    比较原始张量和反量化张量的分布相似性
    """
    x_ref = x_ref.astype(np.float32).reshape(-1)
    x_recon = x_recon.astype(np.float32).reshape(-1)

    vmin = min(np.min(x_ref), np.min(x_recon))
    vmax = max(np.max(x_ref), np.max(x_recon))

    hist_ref, _ = np.histogram(x_ref, bins=num_bins, range=(vmin, vmax))
    hist_recon, _ = np.histogram(x_recon, bins=num_bins, range=(vmin, vmax))

    hist_ref = hist_ref.astype(np.float64)
    hist_recon = hist_recon.astype(np.float64)

    kl = safe_kl_divergence(hist_ref, hist_recon)
    js = js_divergence(hist_ref, hist_recon)

    # 直方图余弦相似度
    denom = (np.linalg.norm(hist_ref) * np.linalg.norm(hist_recon) + 1e-12)
    hist_cos = np.dot(hist_ref, hist_recon) / denom

    return {
        "hist_kl": kl,
        "hist_js": js,
        "hist_cosine": hist_cos,
    }


def sqnr(x_ref, x_recon, eps=1e-12):
    """
    Signal-to-Quantization-Noise Ratio
    """
    x_ref = x_ref.astype(np.float32)
    x_recon = x_recon.astype(np.float32)

    signal_power = np.mean(x_ref ** 2)
    noise_power = np.mean((x_ref - x_recon) ** 2)

    return 10.0 * np.log10((signal_power + eps) / (noise_power + eps))


def cosine_similarity(x_ref, x_recon, eps=1e-12):
    x_ref = x_ref.astype(np.float32).reshape(-1)
    x_recon = x_recon.astype(np.float32).reshape(-1)

    denom = np.linalg.norm(x_ref) * np.linalg.norm(x_recon) + eps
    return float(np.dot(x_ref, x_recon) / denom)


def clipping_stats(x, threshold):
    x = x.astype(np.float32).reshape(-1)
    clipped_mask = np.abs(x) > threshold

    clip_ratio = np.mean(clipped_mask)

    total_energy = np.sum(x ** 2) + 1e-12
    clipped_energy = np.sum((x[clipped_mask]) ** 2)

    return {
        "clip_ratio": float(clip_ratio),
        "clipped_energy_ratio": float(clipped_energy / total_energy),
    }


def linear_output_error(weight_ref, weight_recon, in_features=256, batch=64, seed=123):
    """
    用一层 fake linear 来评估量化后输出误差
    y = Wx
    这通常比直接看权重 MSE 更能体现量化对推理的影响
    """
    rng = np.random.default_rng(seed)

    weight_ref = weight_ref.astype(np.float32).reshape(-1)
    weight_recon = weight_recon.astype(np.float32).reshape(-1)

    out_features = len(weight_ref) // in_features
    if out_features == 0:
        raise ValueError("weight size too small for the selected in_features")

    usable = out_features * in_features
    w_ref = weight_ref[:usable].reshape(out_features, in_features)
    w_recon = weight_recon[:usable].reshape(out_features, in_features)

    x = rng.normal(0, 1, size=(batch, in_features)).astype(np.float32)

    y_ref = x @ w_ref.T
    y_recon = x @ w_recon.T

    err = y_ref - y_recon
    mse = np.mean(err ** 2)
    mae = np.mean(np.abs(err))
    cos = cosine_similarity(y_ref, y_recon)
    rel_mse = mse / (np.mean(y_ref ** 2) + 1e-12)

    return {
        "out_mse": float(mse),
        "out_mae": float(mae),
        "out_cosine": float(cos),
        "out_rel_mse": float(rel_mse),
    }

def safe_kl_divergence(p, q, eps=1e-12):
    """
    计算 KL(P || Q)
    只在 p > 0 的位置参与计算
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)

    mask = p > 0
    p_safe = p[mask]
    q_safe = q[mask] + eps

    return np.sum(p_safe * np.log(p_safe / q_safe))


def build_symmetric_histogram(x_abs, num_bins, max_val):
    """
    对 |x| 构建 [0, max_val] 的直方图
    """
    hist, bin_edges = np.histogram(
        x_abs,
        bins=num_bins,
        range=(0.0, max_val)
    )
    return hist.astype(np.float64), bin_edges


def expand_quantized_distribution(quant_bins, num_merged_bins):
    """
    将压缩后的量化分布展开回原始高分辨率 bin 数
    这里采用“均匀展开”的近似方式：
    每个量化 bin 的概率均匀分配给它覆盖的原始细 bin
    """
    expanded = np.zeros(num_merged_bins, dtype=np.float64)

    num_quant_bins = len(quant_bins)
    edges = np.linspace(0, num_merged_bins, num_quant_bins + 1, dtype=int)

    for i in range(num_quant_bins):
        start = edges[i]
        end = edges[i + 1]
        width = end - start
        if width <= 0:
            continue
        if quant_bins[i] > 0:
            expanded[start:end] = quant_bins[i] / width

    return expanded


def merge_hist_to_quant_bins(ref_hist, num_quant_bins):
    """
    把高分辨率参考直方图压缩成 num_quant_bins 个量化 bin
    """
    num_ref_bins = len(ref_hist)
    quant_bins = np.zeros(num_quant_bins, dtype=np.float64)

    edges = np.linspace(0, num_ref_bins, num_quant_bins + 1, dtype=int)

    for i in range(num_quant_bins):
        start = edges[i]
        end = edges[i + 1]
        quant_bins[i] = np.sum(ref_hist[start:end])

    return quant_bins


def kl_search_symmetric_threshold(x_fp16, num_bits=8, num_hist_bins=2048, debug=False):
    """
    用 KL 散度搜索对称量化阈值 T
    仅对 |x| 做统计，最终量化范围为 [-T, T]

    返回:
        best_threshold
        best_scale
        best_kl
    """
    x = x_fp16.astype(np.float32).reshape(-1)
    x_abs = np.abs(x)

    max_val = np.max(x_abs)
    if max_val == 0:
        return 1.0, 1.0 / 127.0, 0.0

    hist, bin_edges = build_symmetric_histogram(x_abs, num_hist_bins, max_val)

    # 对称 int8，通常有效正半轴可视作 128 个等级
    # 这里只对 |x| 的正半轴直方图做 KL，因此 quant bins 用 128
    num_quant_bins = 2 ** (num_bits - 1)

    # 至少要有足够多的参考 bin 才能合并到量化 bin
    start_bin = num_quant_bins
    best_kl = float("inf")
    best_threshold_bin = num_hist_bins - 1

    for threshold_bin in range(start_bin, num_hist_bins):
        # 参考分布 P：截断到 threshold_bin，对尾部做合并
        sliced = hist[:threshold_bin].copy()
        tail_sum = np.sum(hist[threshold_bin:])
        sliced[-1] += tail_sum

        if np.sum(sliced) == 0:
            continue

        # 压缩到低 bit 量化 bin
        quant_bins = merge_hist_to_quant_bins(sliced, num_quant_bins)

        # 再展开回 threshold_bin 个细 bin，得到近似分布 Q
        expanded = expand_quantized_distribution(quant_bins, threshold_bin)

        # 归一化前的参考分布 P 和近似分布 Q
        p = sliced
        q = expanded

        # 避免 q 中某些位置为 0 而 p > 0 导致 KL 发散
        nonzero_mask = p > 0
        q[(q == 0) & nonzero_mask] = 1e-12

        kl = safe_kl_divergence(p, q)

        if kl < best_kl:
            best_kl = kl
            best_threshold_bin = threshold_bin

    # bin 边界映射回阈值 T
    best_threshold = bin_edges[best_threshold_bin]
    best_scale = best_threshold / 127.0 if best_threshold > 0 else 1.0 / 127.0

    if debug:
        print(f"[KL] best_threshold_bin = {best_threshold_bin}")
        print(f"[KL] best_threshold     = {best_threshold}")
        print(f"[KL] best_scale         = {best_scale}")
        print(f"[KL] best_kl            = {best_kl}")

    return best_threshold, best_scale, best_kl


def quantize_symmetric_int8(x_fp16, scale):
    """
    对称量化到 int8: [-127, 127]
    """
    x = x_fp16.astype(np.float32)
    q = np.round(x / scale)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q


def dequantize_symmetric_int8(q_int8, scale):
    """
    对称反量化
    """
    x_hat = q_int8.astype(np.float32) * np.float32(scale)
    return x_hat.astype(np.float16)


def minmax_symmetric_scale(x_fp16):
    """
    普通 min-max 对称量化 scale
    """
    x = x_fp16.astype(np.float32)
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return 1.0 / 127.0
    return max_abs / 127.0


def calc_error(x_ref, x_recon):
    """
    误差指标
    """
    x_ref = x_ref.astype(np.float32)
    x_recon = x_recon.astype(np.float32)

    err = x_ref - x_recon
    mse = np.mean(err ** 2)
    mae = np.mean(np.abs(err))
    max_err = np.max(np.abs(err))
    return mse, mae, max_err


def print_better_compare(name, x_ref, x_recon, threshold=None):
    x_ref = x_ref.astype(np.float32)
    x_recon = x_recon.astype(np.float32)

    err = x_ref - x_recon
    mse = np.mean(err ** 2)
    mae = np.mean(np.abs(err))
    max_err = np.max(np.abs(err))
    sqnr_db = sqnr(x_ref, x_recon)
    cos = cosine_similarity(x_ref, x_recon)

    dist_metrics = hist_distribution_metrics(x_ref, x_recon, num_bins=512)
    out_metrics = linear_output_error(x_ref, x_recon, in_features=64, batch=32, seed=123)

    print(f"===== {name} =====")
    print(f"Weight MSE        = {mse:.8f}")
    print(f"Weight MAE        = {mae:.8f}")
    print(f"Weight Max Error  = {max_err:.8f}")
    print(f"Weight Cosine     = {cos:.8f}")
    print(f"SQNR(dB)          = {sqnr_db:.4f}")
    print(f"Hist KL           = {dist_metrics['hist_kl']:.8f}")
    print(f"Hist JS           = {dist_metrics['hist_js']:.8f}")
    print(f"Hist Cosine       = {dist_metrics['hist_cosine']:.8f}")
    print(f"Output MSE        = {out_metrics['out_mse']:.8f}")
    print(f"Output MAE        = {out_metrics['out_mae']:.8f}")
    print(f"Output Rel MSE    = {out_metrics['out_rel_mse']:.8f}")
    print(f"Output Cosine     = {out_metrics['out_cosine']:.8f}")

    if threshold is not None:
        clip_metrics = clipping_stats(x_ref, threshold)
        print(f"Clip Ratio        = {clip_metrics['clip_ratio']:.8f}")
        print(f"Clipped Energy    = {clip_metrics['clipped_energy_ratio']:.8f}")

    print()


def main():
    np.random.seed(42)

    # 构造一个更接近真实情况的分布：
    # 大部分值在 0 附近，少量离群点较大
    main_part = np.random.normal(loc=0.0, scale=0.35, size=12000)
    outliers = np.random.normal(loc=0.0, scale=2.5, size=80)
    x = np.concatenate([main_part, outliers]).astype(np.float16)

    print("输入张量信息:")
    print(f"shape            = {x.shape}")
    print(f"min              = {x.min()}")
    print(f"max              = {x.max()}")
    print(f"mean(abs(x))      = {np.mean(np.abs(x.astype(np.float32))):.6f}")
    print()

    # 1) KL 搜索 scale
    kl_threshold, kl_scale, best_kl = kl_search_symmetric_threshold(
        x,
        num_bits=8,
        num_hist_bins=2048,
        debug=True
    )

    q_kl = quantize_symmetric_int8(x, kl_scale)
    x_kl_recon = dequantize_symmetric_int8(q_kl, kl_scale)

    # 2) 普通 min-max scale
    mm_scale = minmax_symmetric_scale(x)
    q_mm = quantize_symmetric_int8(x, mm_scale)
    x_mm_recon = dequantize_symmetric_int8(q_mm, mm_scale)

    print()
    print(f"KL threshold = {kl_threshold}")
    print(f"KL scale     = {kl_scale}")
    print(f"Best KL      = {best_kl}")
    print()

    print_better_compare("KL Quant", x, x_kl_recon, threshold=kl_threshold)
    print_better_compare("MinMax Quant", x, x_mm_recon, threshold=np.max(np.abs(x.astype(np.float32))))

    # 打印前若干个值对比
    print("对比: 原始 -> KL量化值 -> KL反量化 -> MinMax量化值 -> MinMax反量化")
    for i in range(20):
        print(
            f"{i:02d}: "
            f"{float(x[i]):9.5f} -> "
            f"{int(q_kl[i]):4d} -> {float(x_kl_recon[i]):9.5f} -> "
            f"{int(q_mm[i]):4d} -> {float(x_mm_recon[i]):9.5f}"
        )


if __name__ == "__main__":
    main()