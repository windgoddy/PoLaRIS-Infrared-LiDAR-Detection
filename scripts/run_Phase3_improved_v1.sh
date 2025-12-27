#!/bin/bash

# 显卡设置
export CUDA_VISIBLE_DEVICES=4

# Phase3 改进版本 v1 (保守改进)
# 主要改进：
# 1. 提高学习率 (0.0005 -> 0.001)
# 2. 添加 L2 正则化 (weight_decay=1e-4)
# 3. 减少训练轮次 (50 -> 35)
# 4. 添加学习率调度器 (CosineAnnealingLR)

python train_Phase3.py \
    --experiment_name Phase3_improved_v1 \
    --model MS_CAFNet \
    --dataset Pohang-Canal \
    --train_batch_size 4 \
    --epochs 35 \
    --optimizer Adam \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --scheduler CosineAnnealingLR \
    --min_lr 1e-6 \
    --in_channels 2 \
    --deep_supervision False
