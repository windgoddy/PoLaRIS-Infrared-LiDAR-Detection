#!/bin/bash


# chmod +x scripts/run_baseline1.sh
# ./scripts/run_baseline1.sh

# GPU 设置 (根据你的设备调整)
export CUDA_VISIBLE_DEVICES=6

# Baseline 1: Pure Infrared DNANet
# Input: Infrared Images (1 channel)
# Label: Original GT Masks (Loose Bounding Boxes)

python train.py \
    --experiment_name baseline1_NUDT-SIRST \
    --dataset NUDT-SIRST \
    --split_method split_data \
    --root dataset \
    --model DNANet \
    --in_channels 1 \
    --epochs 1500 \
    --train_batch_size 16 \
    --test_batch_size 16 \
    --base_size 256 \
    --crop_size 256 \
    --deep_supervision True \
    --optimizer Adam \
    --lr 0.05 \
    --scheduler CosineAnnealingLR \
    --backbone resnet_18 \