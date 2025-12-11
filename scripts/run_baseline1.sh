#!/bin/bash

# Baseline 1: Pure Infrared DNANet
# Input: Infrared Images (1 channel)
# Label: Original GT Masks (Loose Bounding Boxes)

python train.py \
    --dataset Pohang-Canal \
    --split_method 50_50 \
    --root dataset \
    --model DNANet \
    --in_channels 1 \
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