#!/bin/bash

# 显卡设置
export CUDA_VISIBLE_DEVICES=4

# chmod +x scripts/run_Phase3_improved_v3.sh
# ./scripts/run_Phase3_improved_v3.sh

# Phase3 改进版本 v3 (折中方案 - 推荐)
# 主要改进：
# 1. 中等学习率 (0.0005 -> 0.003)
# 2. 添加 L2 正则化 (weight_decay=1e-4)
# 3. 适中训练轮次 (50 -> 30)
# 4. 添加学习率调度器 (CosineAnnealingLR)
# 5. 更强的正则化 (weight_decay=5e-4)

python train_Phase3.py \
    --experiment_name Phase3_improved_v3 \
    --model MS_CAFNet \
    --dataset Pohang-Canal \
    --train_batch_size 4 \
    --epochs 30 \
    --optimizer Adam \
    --lr 0.003 \
    --weight_decay 5e-4 \
    --scheduler CosineAnnealingLR \
    --min_lr 1e-6 \
    --in_channels 2 \
    --deep_supervision False
