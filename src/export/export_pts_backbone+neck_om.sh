#!/bin/bash

# BEVFusion 点云骨干网络 OM 模型导出脚本
# 使用方法: bash export_pts_backbone_om.sh

# 切换到conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ascend-py3.7.10

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录（脚本在 src/export/ 下，需要向上两级）
WORK_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 设置输入输出路径
ONNX_MODEL="$WORK_DIR/models/onnx/pts_backbone_neck.onnx"
OUTPUT_DIR="$WORK_DIR/models/om"
OUTPUT_NAME="pts_backbone_neck"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查ONNX模型是否存在
if [ ! -f "$ONNX_MODEL" ]; then
    echo "错误: ONNX模型不存在: $ONNX_MODEL"
    echo "请先运行 export_pts_backbone_neck_onnx.py 导出ONNX模型"
    exit 1
fi

echo "开始导出 OM 模型..."
echo "输入模型: $ONNX_MODEL"
echo "输出目录: $OUTPUT_DIR"
echo "输出名称: $OUTPUT_NAME"

# 使用atc工具导出OM模型
atc \
  --model="$ONNX_MODEL" \
  --framework=5 \
  --output="$OUTPUT_DIR/$OUTPUT_NAME" \
  --input_format=NCHW \
  --input_shape="bev_256:1,256,360,360" \
  --soc_version=Ascend310P1 \
  --log=error \
  --output_type=FP32

# 检查导出结果
if [ $? -eq 0 ]; then
    echo "✓ OM 模型导出成功: $OUTPUT_DIR/${OUTPUT_NAME}.om"
else
    echo "✗ OM 模型导出失败"
    exit 1
fi

echo "导出完成!"
