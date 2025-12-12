#!/bin/bash

# 显卡设置
export CUDA_VISIBLE_DEVICES=4

# 运行 Phase 3 训练 (修正版)
# 1. 增加 --optimizer Adam (关键修正！)
# 2. 保持 --lr 0.0005 (对 Adam 来说这是黄金数值)

python train_Phase3.py \
    --model MS_CAFNet \
    --dataset Pohang-Canal \
    --train_batch_size 4 \
    --epochs 500 \
    --optimizer Adam \
    --lr 0.0005 \
    --in_channels 2 \
    --deep_supervision False