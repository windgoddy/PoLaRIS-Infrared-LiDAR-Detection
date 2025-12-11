#!/bin/bash

# Baseline 2: Naive Fusion (IR + LiDAR Depth)
# Input: Infrared Images + LiDAR Depth Maps (2 channels)
# Label: Original GT Masks (Loose Bounding Boxes)

python train.py \
    --dataset Pohang-Canal \
    --split_method 50_50 \
    --root dataset \
    --model DNANet \
    --in_channels 2 \
    --epochs 100 \
    --train_batch_size 4 \
    --test_batch_size 4 \
    --base_size 256 \
    --crop_size 256 \
    --deep_supervision True \
    --optimizer Adam \
    --lr 0.05 \
    --scheduler CosineAnnealingLR
