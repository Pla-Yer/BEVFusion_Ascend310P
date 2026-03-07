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

    print("First 10 different positions:")

    for i in range(min(10, len(idx[0]))):
        pos = tuple(axis[i] for axis in idx)
        print(
            "Index:", pos,
            "file1:", data1[pos],
            "file2:", data2[pos],
            "diff:", diff[pos]
        )


def compare_bin(file1, file2, dtype="float32", shape=None, atol=1e-5):
    dtype = np.dtype(dtype)

    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)

    print("File1:", file1)
    print("File2:", file2)

    print("Size1:", size1, "bytes")
    print("Size2:", size2, "bytes")

    data1 = np.fromfile(file1, dtype=dtype)
    data2 = np.fromfile(file2, dtype=dtype)

    print("Element count1:", data1.size)
    print("Element count2:", data2.size)

    # 如果大小一致
    if data1.size == data2.size:

        if shape is not None:
            shape = tuple(map(int, shape.split(',')))
            data1 = data1.reshape(shape)
            data2 = data2.reshape(shape)
            print("Reshaped to:", shape)

        print("\n=== Direct Compare ===")
        print_diff(data1, data2, atol)
        return

    print("\n⚠ 元素数量不同，执行两种对比方式")

    # =========================
    # 1. truncate 对齐小的
    # =========================

    min_size = min(data1.size, data2.size)

    print("\n=== Compare Mode 1: truncate to min size ===")

    d1 = data1[:min_size]
    d2 = data2[:min_size]

    print("Aligned size:", min_size)

    print_diff(d1, d2, atol)

    # =========================
    # 2. pad 0
    # =========================

    print("\n=== Compare Mode 2: pad smaller with zeros ===")

    max_size = max(data1.size, data2.size)

    p1 = np.zeros(max_size, dtype=dtype)
    p2 = np.zeros(max_size, dtype=dtype)

    p1[:data1.size] = data1
    p2[:data2.size] = data2

    print("Padded size:", max_size)

    print_diff(p1, p2, atol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("bin1")
    parser.add_argument("bin2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--shape", default=None)
    parser.add_argument("--atol", default=1e-1, type=float)

    args = parser.parse_args()

    compare_bin(args.bin1, args.bin2, args.dtype, args.shape, args.atol)