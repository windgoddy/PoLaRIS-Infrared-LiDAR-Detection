#!/bin/bash

# 显卡设置
export CUDA_VISIBLE_DEVICES=4

# chmod +x scripts/run_Phase3_improved_v3.sh
# ./scripts/run_Phase3_improved_v3.sh

# Phase3 原始最佳配置（已验证效果好）
# 关键配置：
# 1. 学习率：0.0001（原始最佳值，不要改动！）
# 2. L2 正则化：weight_decay=5e-4
# 3. 训练轮次：30 epochs
# 4. 学习率调度器：CosineAnnealingLR
# 5. 随机种子：42（确保可复现）

python train_Phase3.py \
    --experiment_name Phase3_improved_v3 \
    --model MS_CAFNet \
    --dataset Pohang-Canal \
    --train_batch_size 4 \
    --epochs 200 \
    --optimizer Adam \
    --lr 0.0001 \
    --weight_decay 5e-4 \
    --scheduler CosineAnnealingLR \
    --min_lr 1e-6 \
    --in_channels 2 \
    --deep_supervision False \
    --seed 42