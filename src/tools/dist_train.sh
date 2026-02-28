#!/usr/bin/env bash
# BEVFusion 分布式训练脚本

CONFIG=$1
GPUS=$2
WORK_DIR=${3:-'./work_dirs/bevfusion'}

if [ -z "$CONFIG" ] || [ -z "$GPUS" ]; then
    echo "Usage: $0 CONFIG_FILE GPUS [WORK_DIR]"
    echo "Example: $0 configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py 4"
    exit 1
fi

# 设置 CUDA 可见设备
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# 训练命令
if [ $GPUS -lt 8 ]; then
    python -m torch.distributed.launch \
        --nproc_per_node=$GPUS \
        --master_port=29500 \
        tools/train.py $CONFIG \
        --launcher pytorch \
        --work-dir $WORK_DIR
else
    python -m torch.distributed.launch \
        --nproc_per_node=$GPUS \
        --master_port=29500 \
        tools/train.py $CONFIG \
        --launcher pytorch \
        --work-dir $WORK_DIR
fi
