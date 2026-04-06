#!/usr/bin/env bash
# BEVFusion 分布式训练脚本

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 检查必要参数
if [ -z "$CONFIG" ] || [ -z "$GPUS" ]; then
    echo "Usage: $0 CONFIG_FILE GPUS [额外train.py参数]"
    echo "Example: $0 configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py 4 --cfg-options load_from=work_dirs/bevfusion/epoch_20.pth"
    exit 1
fi

# 设置 PYTHONPATH 并执行训练
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
