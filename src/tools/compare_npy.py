import numpy as np
import argparse
import os


def print_diff(data1, data2, atol):

    diff = np.abs(data1 - data2)

    max_diff = diff.max()
    mean_diff = diff.mean()

    print("Max diff:", max_diff)
    print("Mean diff:", mean_diff)

    equal = np.allclose(data1, data2, atol=atol)

    if equal:
        print("✅ 两个文件基本一致 (within atol)")
        return

    print("❌ 存在差异")

    idx = np.where(diff > atol)
    # 打印有多少个元素不同
    print(f"Number of different elements: {len(idx[0])}")

    print("First 10 different positions:")

    for i in range(min(10, len(idx[0]))):
        pos = tuple(axis[i] for axis in idx)
        print(
            "Index:", pos,
            "file1:", data1[pos],
            "file2:", data2[pos],
            "diff:", diff[pos]
        )


def compare_npy(file1, file2, atol=1e-5):

    print("File1:", file1)
    print("File2:", file2)

    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)

    print("Size1:", size1, "bytes")
    print("Size2:", size2, "bytes")

    data1 = np.load(file1)
    data2 = np.load(file2)

    print("\n===== Data Info =====")

    print("dtype1:", data1.dtype)
    print("dtype2:", data2.dtype)

    print("shape1:", data1.shape)
    print("shape2:", data2.shape)

    print("element count1:", data1.size)
    print("element count2:", data2.size)

    # flatten
    data1_flat = data1.flatten()
    data2_flat = data2.flatten()

    if data1_flat.size == data2_flat.size:

        print("\n=== Direct Compare ===")

        print_diff(data1_flat, data2_flat, atol)
        return

    print("\n⚠ 元素数量不同，执行两种对比方式")

    # =========================
    # truncate
    # =========================

    min_size = min(data1_flat.size, data2_flat.size)

    print("\n=== Compare Mode 1: truncate ===")
    print("Aligned size:", min_size)

    d1 = data1_flat[:min_size]
    d2 = data2_flat[:min_size]

    print_diff(d1, d2, atol)

    # =========================
    # pad0
    # =========================

    print("\n=== Compare Mode 2: pad0 ===")

    max_size = max(data1_flat.size, data2_flat.size)

    p1 = np.zeros(max_size, dtype=data1.dtype)
    p2 = np.zeros(max_size, dtype=data2.dtype)

    p1[:data1_flat.size] = data1_flat
    p2[:data2_flat.size] = data2_flat

    print("Padded size:", max_size)

    print_diff(p1, p2, atol)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("npy1")
    parser.add_argument("npy2")
    parser.add_argument("--atol", default=1e-5, type=float)

    args = parser.parse_args()

    compare_npy(args.npy1, args.npy2, args.atol)