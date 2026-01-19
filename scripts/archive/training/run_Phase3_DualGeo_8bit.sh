#!/bin/bash

# 显卡设置
export CUDA_VISIBLE_DEVICES=5

# chmod +x scripts/run_Phase3_DualGeo_8bit.sh
# ./scripts/run_Phase3_DualGeo_8bit.sh

# ============================================================
# 8-bit Mode: 使用原始 8-bit DataLoader
# ============================================================
# 特性:
# - 8-bit 图像支持
# - 使用 TrainSetLoader (model/utils.py)
# - 硬标签训练 (Binary: 0/1)
# - 适用于传统 8-bit 数据集
# ============================================================

python train_Phase3.py \
    --experiment_name Phase3_DualGeo_8bit \
    --model MS_CAFNet_DualGeo \
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
    --seed 42 \
    --use_lidar_dataloader False \
    --use_soft_labels False
