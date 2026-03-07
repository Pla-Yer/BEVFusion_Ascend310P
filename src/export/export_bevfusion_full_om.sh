#!/bin/bash

# BEVFusion 完整模型 OM 模型导出脚本（支持动态shape）
# 使用方法: bash export_bevfusion_full_om.sh

# 切换到conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ascend-py3.7.10

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录（脚本在 src/export/ 下，需要向上两级）
WORK_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 设置输入输出路径
ONNX_MODEL="$WORK_DIR/models/onnx_final/bevfusion_final_dynamic.onnx"
OUTPUT_DIR="$WORK_DIR/models/om"
OUTPUT_NAME="bevfusion_full_dynamic"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查ONNX模型是否存在
if [ ! -f "$ONNX_MODEL" ]; then
    echo "错误: ONNX模型不存在: $ONNX_MODEL"
    echo "请先运行 export_bevfusion_full_onnx.py 导出动态ONNX模型"
    exit 1
fi

echo "开始导出动态shape OM 模型..."
echo "输入模型: $ONNX_MODEL"
echo "输出目录: $OUTPUT_DIR"
echo "输出名称: $OUTPUT_NAME"

# 动态shape配置
# 定义支持的体素数量范围：最小值~最大值，步长
# 例如：4000~8000~1000 表示支持4000, 5000, 6000, 7000, 8000五个档位
MIN_VOXELS=4000
MAX_VOXELS=8000
STEP=1000

# 使用atc工具导出动态shape OM模型
# input_shape_range: 定义每个输入的shape范围
# input_format: 输入数据格式
# dynamic_dims: 动态维度配置
# atc \
#   --model="$ONNX_MODEL" \
#   --framework=5 \
#   --output="$OUTPUT_DIR/$OUTPUT_NAME" \
#   --input_format=NCHW \
#   --input_shape_range="voxels:[4000,8000],[32,32],[5,5];num_points:[4000,8000];coords:[4000,8000],[4,4]" \
#   --dynamic_dims="${MIN_VOXELS}~${MAX_VOXELS}~${STEP}" \
#   --soc_version=Ascend310P1 \
#   --log=error
atc --model="$ONNX_MODEL"     --framework=5     --output="$OUTPUT_DIR/$OUTPUT_NAME"     --input_format=ND     --input_shape="voxels:-1,20,5;num_points:-1;coords:-1,4"     --dynamic_dims="4000,4000,4000;5000,5000,5000;6000,6000,6000"     --soc_version=Ascend310P1     --log=info     --precision_mode=force_fp32
# 检查导出结果
if [ $? -eq 0 ]; then
    echo "✓ 动态shape OM 模型导出成功: $OUTPUT_DIR/${OUTPUT_NAME}.om"
    echo "支持的体素数量档位: $(seq $MIN_VOXELS $STEP $MAX_VOXELS | tr '\n' ' ')"
else
    echo "✗ OM 模型导出失败"
    exit 1
fi

echo "导出完成!"
