#!/bin/bash

# 显卡设置
export CUDA_VISIBLE_DEVICES=0

# chmod +x scripts/run_Phase3_DualGeo_all_dataset.sh
# ./scripts/run_Phase3_DualGeo_all_dataset.sh

# Phase3 双流几何增强版本 (MS_CAFNet_DualGeo) - 适配新数据集 Pohang-Canal-all
# 以及支持 6000 张图的大规模训练

# 1. 检查 Oracle Masks 是否生成 (防止忘记生成导致训练报错)
ORACLE_DIR="dataset/Pohang-Canal-all/oracle_masks"
if [ ! -d "$ORACLE_DIR" ]; then
    echo "Warning: $ORACLE_DIR not found. Generating Oracle Masks first..."
    python scripts/generate_oracle_masks.py
fi

# 2. 启动训练
# 注意：
# - dataset: 改为 Pohang-Canal-all
# - split_method: 改为 split_data (匹配您新数据集的划分文件夹名)
# - train_batch_size: 保持 4 (或根据显存调整为 8/16)
python train_Phase3.py \
    --experiment_name Phase3_DualGeo_All \
    --model MS_CAFNet_DualGeo \
    --dataset Pohang-Canal-all \
    --split_method split_data \
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
    --workers 8  # 增加 DataLoader 线程数以加快 6000 张图的加载速度