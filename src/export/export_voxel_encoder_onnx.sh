#!/bin/bash

# 导出 voxel_encoder ONNX 模型（支持动态shape）

# 切换到conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openmmlab

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORK_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 切换到项目根目录
cd "$WORK_DIR"

echo "开始导出 voxel_encoder ONNX 模型（支持动态shape）..."
echo "工作目录: $WORK_DIR"

# 运行导出脚本
python src/export/export_voxel_encoder_onnx.py

if [ $? -eq 0 ]; then
    echo "✓ ONNX 模型导出成功（支持动态体素数量）"
else
    echo "✗ ONNX 模型导出失败"
    exit 1
fi
