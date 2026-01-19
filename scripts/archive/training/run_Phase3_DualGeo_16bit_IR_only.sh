#!/bin/bash

# 显卡设置
export CUDA_VISIBLE_DEVICES=5

# chmod +x scripts/run_Phase3_DualGeo_16bit_IR_only.sh
# ./scripts/run_Phase3_DualGeo_16bit_IR_only.sh

# ============================================================
# IR-Only Mode: 仅红外图像（无深度图）
# ============================================================
# 特性:
# ✅ 16-bit 红外图像支持
# ✅ 软标签训练
# ❌ 不使用 LiDAR 深度图（in_channels=1）
# ============================================================
# 适用场景:
# - 测试纯红外图像的性能
# - LiDAR 深度图未生成或不可用
# - 对比实验：IR vs IR+Depth
# ============================================================

python train_Phase3.py \
    --experiment_name Phase3_DualGeo_16bit_IR_only \
    --model MS_CAFNet_DualGeo \
    --dataset Pohang-Canal \
    --train_batch_size 4 \
    --epochs 200 \
    --optimizer Adam \
    --lr 0.0001 \
    --weight_decay 5e-4 \
    --scheduler CosineAnnealingLR \
    --min_lr 1e-6 \
    --in_channels 1 \
    --deep_supervision False \
    --seed 42 \
    --use_lidar_dataloader True \
    --normalize_16bit True \
    --use_soft_labels True
