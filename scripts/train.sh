#!/bin/bash

# ============================================================
# 统一训练脚本 - PoLaRIS Infrared-LiDAR Detection
# ============================================================
# 用法:
#   ./scripts/train.sh <mode> [options]
#
# 模式:
#   auto         - 自动选择（根据数据集：Pohang-Canal-3k 用 16bit，其他用 8bit）
#   8bit         - 8-bit 图像训练（旧版 DataLoader）
#   16bit        - 16-bit 图像训练（新版 DataLoader + 软标签）
#   16bit-ir     - 16-bit 仅红外（无深度图）
#   baseline1    - DNANet 原始论文配置（对比基准）
#
# 示例:
#   ./scripts/train.sh auto                      # 自动选择模式（推荐）
#   ./scripts/train.sh auto --dataset Pohang-Canal-3k  # 16-bit 模式
#   ./scripts/train.sh auto --dataset Pohang-Canal     # 8-bit 模式
#   ./scripts/train.sh 16bit --gpu 0             # 手动指定 16-bit
# ============================================================

# 默认参数
MODE="auto"
GPU=5
DATASET="Pohang-Canal"
EPOCHS=2000

# 解析第一个参数作为模式（如果提供）
if [[ $# -gt 0 && $1 =~ ^(auto|8bit|16bit|16bit-ir|baseline1)$ ]]; then
    MODE="$1"
    shift
fi

# 解析其他命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "警告: 未知参数 '$1'"
            shift
            ;;
    esac
done

# 自动选择模式（根据数据集）
if [[ "$MODE" == "auto" ]]; then
    if [[ "$DATASET" == *"3k"* ]]; then
        MODE="16bit"
        echo "🤖 自动选择: 16-bit 模式（检测到数据集包含 '3k'）"
    else
        MODE="8bit"
        echo "🤖 自动选择: 8-bit 模式（数据集: $DATASET）"
    fi
fi

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU

echo "============================================================"
echo "训练模式: $MODE"
echo "GPU: $GPU"
echo "============================================================"

# 根据模式选择配置
case $MODE in
    8bit)
        echo "🔹 8-bit 模式（旧版 DataLoader）"
        python train_Phase3.py \
            --experiment_name Phase3_DualGeo_8bit \
            --model MS_CAFNet_DualGeo \
            --dataset Pohang-Canal \
            --train_batch_size 4 \
            --epochs 2000 \
            --optimizer Adam \
            --lr 0.0001 \
            --weight_decay 5e-4 \
            --scheduler CosineAnnealingLR \
            --min_lr 1e-6 \
            --in_channels 2 \
            --deep_supervision False \
            --seed 42 \
            --use_lidar_dataloader False \
            --use_soft_labels False \
            --train_batch_size 16 \
            --test_batch_size 16 \
            --backbone resnet_18 \
        ;;

    16bit)
        echo "🔸 16-bit 模式（推荐）"
        python train_Phase3.py \
            --experiment_name Phase3_DualGeo_16bit \
            --model MS_CAFNet_DualGeo \
            --dataset Pohang-Canal-3k \
            --epochs 2000 \
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
            --use_soft_labels True \
            --train_batch_size 16 \
            --test_batch_size 16 \
            --backbone resnet_18 \
        ;;

    16bit-ir)
        echo "🔹 16-bit 仅红外模式（无深度图）"
        python train_Phase3.py \
            --experiment_name Phase3_DualGeo_16bit_IR_only \
            --model MS_CAFNet_DualGeo \
            --dataset Pohang-Canal-3k \
            --epochs $EPOCHS \
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
            --use_soft_labels True \
            --backbone resnet_18 \
            --train_batch_size 16 \
            --test_batch_size 16 \
        ;;

    baseline1)
        echo "🔹 Baseline1 (DNANet)"
        python train.py \
            --experiment_name baseline1 \
            --model DNANet \
            --dataset $DATASET \
            --train_batch_size 8 \
            --epochs $EPOCHS \
            --optimizer Adagrad \
            --lr 0.05 \
            --deep_supervision True \
            --seed 42
        ;;

    *)
        echo "❌ 未知模式: $MODE"
        echo "支持的模式: auto, 8bit, 16bit, 16bit-ir, baseline1"
        exit 1
        ;;
esac

echo "============================================================"
echo "✅ 训练完成!"
echo "============================================================"
