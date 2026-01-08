#!/bin/bash

# 显卡设置
export CUDA_VISIBLE_DEVICES=4

# chmod +x scripts/run_Phase3_DualGeo.sh
# ./scripts/run_Phase3_DualGeo.sh

# Phase3 双流几何增强版本 (MS_CAFNet_DualGeo)
# 核心升级：
# 1. ✅ 保留原有 MS_CAFNet 的所有优势（置信度门控、MSBlock、残差增强FPN）
# 2. 🆕 新增视觉结构提取器 (VisualStructureExtractor)
#    - 从红外图像提取边缘/纹理结构先验
#    - 作为 LiDAR 失效时的几何补偿
# 3. 🆕 自适应双流融合机制
#    - LiDAR 几何分支 (电子锁): alpha 可学习权重
#    - 视觉结构分支 (机械锁): beta 可学习权重
#    - 公式: feat_enhanced = feat × (1 + α·lidar_conf + β·visual_struct)
# 4. 参数量增加: 仅 +82 参数 (<0.003%)

# 训练配置：与原版 MS_CAFNet 保持一致，便于公平对比
python train_Phase3.py \
    --experiment_name Phase3_DualGeo \
    --model MS_CAFNet_DualGeo \
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
