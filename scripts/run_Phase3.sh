#!/bin/bash

# 显卡设置
export CUDA_VISIBLE_DEVICES=0

# 运行 Phase 3 训练
# 注意：这里调用的是 train_Phase3.py
# 确保 dataset_dir 指向您的 Pohang-Canal 数据集路径
python train_Phase3.py \
    --model MS_CAFNet \
    --dataset Pohang-Canal \
    --train_batch_size 4 \
    --test_batch_size 4 \
    --epochs 500 \
    --lr 0.0005 \
    --in_channels 2 \
    --deep_supervision False