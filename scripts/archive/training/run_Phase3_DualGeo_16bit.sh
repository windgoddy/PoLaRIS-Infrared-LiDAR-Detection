#!/bin/bash

# 显卡设置
export CUDA_VISIBLE_DEVICES=5

# chmod +x scripts/run_Phase3_DualGeo_16bit.sh
# ./scripts/run_Phase3_DualGeo_16bit.sh

# ============================================================
# Enhanced Mode: 使用 PoLaRIS LiDAR DataLoader（新版）
# ============================================================
# 特性:
# ✅ 16-bit 红外图像支持（Min-Max 归一化）
# ✅ LiDAR 点云加载 (lidar_roi/*.bin)
# ✅ 软标签训练 (Soft Labels: 0.0, 0.6, 1.0)
#    - 1.0: 有 LiDAR 点云的目标（高置信度）
#    - 0.6: 无 LiDAR 点云的目标（中置信度）
#    - 0.0: 背景
# ✅ 深度图支持 (depth_maps/*.npy)
# ============================================================

python train_Phase3.py \
    --experiment_name Phase3_DualGeo_16bit \
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
    --use_lidar_dataloader True \
    --normalize_16bit True \
    --use_soft_labels True

# ============================================================
# 参数说明:
# ============================================================
# --use_lidar_dataloader True   : 使用新的 PoLaRIS LiDAR DataLoader
# --normalize_16bit True         : 使用 Min-Max 归一化（推荐）
#                                  False: 使用简单缩放 /65535
# --use_soft_labels True         : 使用软标签（oracle_masks）训练
#                                  False: 使用硬标签（GT masks）
# --in_channels 2                : 2通道输入（红外 + 深度图）
#                                  1: 仅红外图像
# ============================================================
