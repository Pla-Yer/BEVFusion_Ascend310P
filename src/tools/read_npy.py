import numpy as np
import argparse
import os

def inspect_npy(npy_path, shape=None):
    # 文件大小
    file_size = os.path.getsize(npy_path)

    print("File:", npy_path)
    print("File size:", file_size, "bytes")

    # 读取 npy
    data = np.load(npy_path)

    print("Dtype:", data.dtype)
    print("Original shape:", data.shape)
    print("Element count:", data.size)

    # 如果用户想 reshape
    if shape is not None:
        shape = tuple(map(int, shape.split(',')))
        try:
            data = data.reshape(shape)
            print("Reshaped to:", data.shape)
        except:
            print("⚠ reshape失败，shape与数据大小不匹配")

    # 数值范围
    if data.size > 0:
        print("Min:", data.min())
        print("Max:", data.max())
        print("Mean:", data.mean())

    print("First 10 values:", data.flatten()[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npy", help="npy file path")
    parser.add_argument("--shape", default=None, help="reshape like 4000,32,5")

    args = parser.parse_args()

    inspect_npy(args.npy, args.shape)