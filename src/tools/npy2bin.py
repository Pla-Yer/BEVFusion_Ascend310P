import numpy as np
import os

input_dir = "om_data/sample_0"
output_dir = "bin_data/sample_0"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".npy"):
        npy_path = os.path.join(input_dir, file)
        bin_path = os.path.join(output_dir, file.replace(".npy", ".bin"))

        data = np.load(npy_path)
        data.tofile(bin_path)

        print(f"{file} -> {bin_path}  shape={data.shape} dtype={data.dtype}")