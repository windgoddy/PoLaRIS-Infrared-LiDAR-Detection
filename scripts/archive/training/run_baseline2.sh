#!/bin/bash

# chmod +x scripts/run_baseline2.sh
# ./scripts/run_baseline2.sh

# GPU 设置 (根据你的设备调整)
export CUDA_VISIBLE_DEVICES=4

# Baseline 2: Naive Fusion (IR + LiDAR Depth)
# Input: Infrared Images + LiDAR Depth Maps (2 channels)
# Label: Original GT Masks (Loose Bounding Boxes)

python train.py \
    --experiment_name baseline2_IR_Depth_naive \
    --dataset Pohang-Canal \
    --split_method 50_50 \
    --root dataset \
    --model DNANet \
    --in_channels 2 \
    --epochs 30 \
    --train_batch_size 4 \
    --test_batch_size 4 \
    --base_size 256 \
    --crop_size 256 \
    --deep_supervision True \
    --optimizer Adam \
    --lr 0.05 \
    --scheduler CosineAnnealingLR \
    --backbone resnet_34 \
