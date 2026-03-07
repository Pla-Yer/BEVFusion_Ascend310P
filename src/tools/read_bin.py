import numpy as np
import argparse
import os

def inspect_bin(bin_path, dtype="float32", shape=None):
    # dtype 转换
    dtype = np.dtype(dtype)

    # 文件大小
    file_size = os.path.getsize(bin_path)

    print("File:", bin_path)
    print("File size:", file_size, "bytes")
    print("Dtype:", dtype)

    # 读取数据
    data = np.fromfile(bin_path, dtype=dtype)

    print("Element count:", data.size)

    # 如果提供shape
    if shape is not None:
        shape = tuple(map(int, shape.split(',')))
        try:
            data = data.reshape(shape)
            print("Reshaped to:", data.shape)
        except:
            print("⚠ reshape失败，shape与数据大小不匹配")

    # 输出数值范围
    if data.size > 0:
        print("Min:", data.min())
        print("Max:", data.max())
        print("Mean:", data.mean())

    print("First 10 values:", data[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bin", help="bin file path")
    parser.add_argument("--dtype", default="float32", help="data type")
    parser.add_argument("--shape", default=None, help="shape like 4000,32,5")

    args = parser.parse_args()

    inspect_bin(args.bin, args.dtype, args.shape)