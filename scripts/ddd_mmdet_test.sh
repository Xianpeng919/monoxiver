#!/usr/bin/env bash

if [ "$#" -lt 2 ]; then
    echo "Usage: me.sh relative_config_filename [ckpt_dir] [mode] [gpus] [nb_gpus] [port]"
    exit
fi

PYTHON=${PYTHON:-"python"}

if [ "$#" -gt 4 ]; then
  GPUS=$4
  NUM_GPUS=$5
else
  GPUS=0,1,2,3,4,5,6,7
  NUM_GPUS=8
fi

if [ "$#" -gt 5 ]; then
  PORT=${PORT:-$6}
else
  PORT=${PORT:-23700}
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CONFIG_FILE=$1
CKPT_FILE=$2
MODE=$3


OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPUS $PYTHON -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS --master_port=$PORT \
    $DIR/../tools/mmdet3d_test.py $CONFIG_FILE $CKPT_FILE\
    --launcher pytorch \
    --eval $MODE
