#!/bin/bash

# ============================================================
# 统一训练脚本 - PoLaRIS Infrared-LiDAR Detection
# ============================================================
# 用法:
#   ./scripts/train.sh <mode> [options]
#
# 模式:
#   baseline1    - DNANet + 8-bit images（对比基准）
#   16bit-ir     - PoLaRIS + 16-bit + 仅红外（无深度图）
#   16bit        - PoLaRIS + 16-bit + 深度图（完整模型）
#
# 选项:
#   --gpu <N>              指定GPU编号（默认：5）
#   --dataset <name>       指定数据集（默认：Pohang-Canal-3k）
#   --epochs <N>           训练轮数（默认：2000）
#   --oracle-masks <name>  Oracle masks文件夹名称（默认：oracle_masks，可选：oracle_masks2, oracle_masks3）
#
# 示例:
#   ./scripts/train.sh baseline1 --gpu 0                                # DNANet baseline (8-bit)
#   ./scripts/train.sh 16bit-ir --gpu 1                                 # PoLaRIS 16-bit 无深度图
#   ./scripts/train.sh 16bit --gpu 2 --oracle-masks oracle_masks2       # PoLaRIS 完整模型使用oracle_masks2
#   ./scripts/train.sh 16bit --oracle-masks oracle_masks3 --epochs 1000 # 使用oracle_masks3训练1000轮
# ============================================================

# 默认参数
MODE="16bit"
GPU=5
DATASET="Pohang-Canal-3k"
EPOCHS=2000
ORACLE_MASKS="oracle_masks"

# 解析第一个参数作为模式（如果提供）
if [[ $# -gt 0 && $1 =~ ^(baseline1|16bit-ir|16bit)$ ]]; then
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
        --oracle-masks)
            ORACLE_MASKS="$2"
            shift 2
            ;;
        *)
            echo "警告: 未知参数 '$1'"
            shift
            ;;
    esac
done

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU

echo "============================================================"
echo "训练模式: $MODE"
echo "GPU: $GPU"
echo "============================================================"

# 根据模式选择配置
case $MODE in
    baseline1)
        echo "🔹 Baseline: DNANet + 8-bit images"
        python train.py \
            --experiment_name DNANet_baseline_8bit \
            --model DNANet \
            --dataset Pohang-Canal-3k \
            --image_folder images-8bit \
            --train_batch_size 8 \
            --epochs $EPOCHS \
            --optimizer Adagrad \
            --lr 0.05 \
            --deep_supervision True \
            --backbone resnet_18 \
            --channel_size three \
            --seed 42 \
            --suffix .png \
            --split_method 50_50 \
            --workers 4
        ;;
    
    baseline2)
        echo "🔹 Baseline: MS_CAFNet_DualGeo + 8-bit images"
        python train.py \
            --experiment_name MS_CAFNet_baseline_8bit \
            --model MS_CAFNet_DualGeo \
            --dataset Pohang-Canal-3k \
            --image_folder images-8bit \
            --train_batch_size 8 \
            --epochs $EPOCHS \
            --optimizer Adagrad \
            --lr 0.05 \
            --deep_supervision True \
            --backbone resnet_18 \
            --channel_size three \
            --seed 42 \
            --suffix .png \
            --split_method 50_50 \
            --workers 4
        ;;

    16bit)
        echo "🔸 PoLaRIS: 16-bit + 深度图（完整模型）"
        python train_Phase3.py \
            --experiment_name PoLaRIS_16bit_full \
            --model MS_CAFNet_DualGeo \
            --dataset Pohang-Canal-3k \
            --epochs $EPOCHS \
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
            --oracle_masks_folder $ORACLE_MASKS \
            --train_batch_size 16 \
            --test_batch_size 16 \
            --backbone resnet_34 \
            --suffix .png \
            --split_method 50_50 \
            --workers 4
        ;;

    16bit-ir)
        echo "🔹 PoLaRIS: 16-bit + 仅红外（无深度图）"
        python train_Phase3.py \
            --experiment_name PoLaRIS_16bit_IR_only \
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
            --oracle_masks_folder $ORACLE_MASKS \
            --backbone resnet_34 \
            --train_batch_size 16 \
            --test_batch_size 16 \
            --suffix .png \
            --split_method 50_50 \
            --workers 4
        ;;

    *)
        echo "❌ 未知模式: $MODE"
        echo "支持的模式: baseline1, 16bit-ir, 16bit"
        echo ""
        echo "实验设计："
        echo "  baseline1  - DNANet + 8-bit images (Pohang-Canal-3k-8bit)"
        echo "  16bit-ir   - PoLaRIS + 16-bit + 仅红外（无深度图）"
        echo "  16bit      - PoLaRIS + 16-bit + 深度图（完整模型）"
        exit 1
        ;;
esac

echo "============================================================"
echo "✅ 训练完成!"
echo "============================================================"
