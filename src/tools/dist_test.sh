#!/usr/bin/env bash
# BEVFusion 分布式测试脚本

CONFIG=$1
CHECKPOINT=$2
GPUS=$3


if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ] || [ -z "$GPUS" ]; then
    echo "Usage: $0 CONFIG_FILE CHECKPOINT GPUS"
    echo "Example: $0 configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py work_dirs/bevfusion/epoch_20.pth 4"
    exit 1
fi

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=29500 \
    src/tools/test.py $CONFIG $CHECKPOINT --work-dir work_dirs/bevfusion --show --task lidar_det\
    --launcher pytorch
