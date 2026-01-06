#!/bin/bash

# ========================================
# Phase 3: LiDAR 残差增强 FPN 训练脚本
# ========================================
# 模型改进：
# 1. 残差增强机制 (feat × (1 + att_map))
# 2. 保留远距离目标的红外特征
# 3. 浅层-深层特征融合
# ========================================
# chmod +x scripts/run_Phase3_ResidualFPN.sh
# ./scripts/run_Phase3_ResidualFPN.sh

# GPU 设置 (根据你的设备调整)
export CUDA_VISIBLE_DEVICES=4

# 训练参数
python train_Phase3.py \
    --experiment_name Phase3_ResidualFPN \
    --model MS_CAFNet \
    --dataset Pohang-Canal \
    --split_method 50_50 \
    --mode TXT \
    --in_channels 2 \
    --train_batch_size 4 \
    --test_batch_size 4 \
    --epochs 50 \
    --optimizer Adam \
    --lr 0.0005 \
    --min_lr 1e-5 \
    --scheduler CosineAnnealingLR \
    --deep_supervision False \
    --base_size 256 \
    --crop_size 256 \
    --workers 4 \
    --channel_size three \
    --backbone resnet_34

# 参数说明:
# --in_channels 2          : 双通道输入 (红外 + 深度)
# --train_batch_size 4     : 批次大小 (根据 GPU 内存调整)
# --epochs 500             : 训练轮数
# --optimizer Adam         : 优化器
# --lr 0.0005             : 初始学习率
# --deep_supervision False : 不使用深度监督 (MS_CAFNet 已有置信度分支)
