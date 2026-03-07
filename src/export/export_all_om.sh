#!/bin/bash

# BEVFusion 所有OM模型一键导出脚本
# 使用方法: bash export_all_om.sh

echo "=========================================="
echo "BEVFusion OM 模型批量导出工具"
echo "=========================================="
echo ""

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 定义要导出的模型列表
declare -a models=(
    "voxel_encoder"
    "pts_backbone"
    "pts_neck"
    "transfusion_head"
    "bevfusion_full"
)

# 计数器
success_count=0
fail_count=0

# 遍历导出每个模型
for model in "${models[@]}"; do
    echo "----------------------------------------"
    echo "正在导出: $model"
    echo "----------------------------------------"

    # 执行对应的导出脚本
    bash "$SCRIPT_DIR/export_${model}_om.sh"

    # 检查导出结果
    if [ $? -eq 0 ]; then
        ((success_count++))
        echo "✓ $model 导出成功"
    else
        ((fail_count++))
        echo "✗ $model 导出失败"
    fi
    echo ""
done

# 打印汇总信息
echo "=========================================="
echo "导出汇总"
echo "=========================================="
echo "成功: $success_count"
echo "失败: $fail_count"
echo "总计: ${#models[@]}"
echo "=========================================="

if [ $fail_count -eq 0 ]; then
    echo "✓ 所有模型导出成功!"
    exit 0
else
    echo "✗ 部分模型导出失败,请检查错误信息"
    exit 1
fi
