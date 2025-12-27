#!/bin/bash

# 显卡设置
export CUDA_VISIBLE_DEVICES=4

# chmod +x scripts/run_Phase3.sh
# ./scripts/run_Phase3.sh

python train_Phase3.py \
    --experiment_name Phase3_original \
    --model MS_CAFNet \
    --dataset Pohang-Canal \
    --train_batch_size 4 \
    --epochs 50 \
    --optimizer Adam \
    --lr 0.0005 \
    --in_channels 2 \
    --deep_supervision False