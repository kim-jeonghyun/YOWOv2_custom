#!/usr/bin/env bash

set -x


GPUS=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
 --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train_custom.py  --cuda -dist -d ava_v2.2 --world_size 2 ${@:3}
