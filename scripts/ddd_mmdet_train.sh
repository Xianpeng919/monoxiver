#!/usr/bin/env bash

if [ "$#" -lt 2 ]; then
    echo "Usage: me.sh relative_config_filename remove_old_if_exist_0_or_1 [name_tag] [gpus] [nb_gpus] [port] [resume_dir]"
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

if [ "$#" -gt 6 ]; then
  RESUME=${RESUME:-$7}
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CONFIG_FILE=$1
CONFIG_FILENAME=${CONFIG_FILE##*/}
CONFIG_BASE="${CONFIG_FILENAME%.*}"

if [ "$#" -gt 2 ]; then
  WORK_DIR=$DIR/../work_dirs/ddd_detection/$CONFIG_BASE-$3
else
  WORK_DIR=$DIR/../work_dirs/ddd_detection/$CONFIG_BASE
fi

if [ -d $WORK_DIR ]; then
  echo "$WORK_DIR --- Already exists"
  if [ $2 -gt 0 ]; then
    while true; do
        read -p "Are you sure to delete this result directory? " yn
        case $yn in
            [Yy]* ) rm -r $WORK_DIR; mkdir -p $WORK_DIR; break;;
            [Nn]* ) exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
  else
    echo "Skip. Delete it first if retraining is needed"
    exit
  fi
else
    mkdir -p $WORK_DIR
fi


if [ "$#" -gt 6 ]; then
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPUS $PYTHON -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS --master_port=$PORT \
      $DIR/../tools/mmdet3d_train.py $CONFIG_FILE \
      --launcher pytorch \
      --work-dir $WORK_DIR \
      --resume-from $RESUME
else
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPUS $PYTHON -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS --master_port=$PORT \
      $DIR/../tools/mmdet3d_train.py $CONFIG_FILE \
      --launcher pytorch \
      --work-dir $WORK_DIR
fi