#!/bin/bash

# ============================================================
# PoLaRIS 完整训练配置文件
# ============================================================
# 此文件包含所有可用的训练参数和选项
# 使用方法：
#   1. 修改下面的参数配置
#   2. 运行：chmod +x run_config.sh && ./run_config.sh
# ============================================================

# ============================================================
# 基础配置
# ============================================================

# GPU 配置
export CUDA_VISIBLE_DEVICES=5              # GPU ID（可以是 0,1,2 等多GPU）

# 数据集配置
DATASET="Pohang-Canal"                     # 数据集名称
# 可选数据集：
#   - Pohang-Canal          # 8-bit 数据集
#   - Pohang-Canal-3k       # 16-bit 数据集（推荐）
#   - NUDT-SIRST            # 公开数据集
#   - NUAA-SIRST            # 公开数据集
#   - NUST-SIRST            # 公开数据集

ROOT="dataset/"                            # 数据集根目录

# ============================================================
# 模型配置
# ============================================================

MODEL="MS_CAFNet_DualGeo"                  # 模型名称
# 可选模型：
#   - DNANet                # 原始 DNANet
#   - MS_CAFNet_DualGeo     # PoLaRIS 模型（推荐）
#   - ResidualFPN           # ResidualFPN 变体

BACKBONE="resnet_18"                       # 骨干网络
# 可选骨干：
#   - vgg10                 # VGG-10
#   - resnet_10             # ResNet-10
#   - resnet_18             # ResNet-18（推荐）
#   - resnet_34             # ResNet-34

CHANNEL_SIZE="three"                       # 通道大小（仅 DNANet）
# 可选：one, two, three, four

DEEP_SUPERVISION="False"                   # 深度监督
# 可选：True, False

IN_CHANNELS=2                              # 输入通道数
# 可选：
#   - 1: 仅红外图像
#   - 2: 红外 + 深度图（推荐）
#   - 3: RGB（仅用于某些模型）

# ============================================================
# 训练超参数
# ============================================================

EPOCHS=200                                 # 训练轮数
START_EPOCH=0                              # 起始轮数
TRAIN_BATCH_SIZE=4                         # 训练批次大小
TEST_BATCH_SIZE=4                          # 测试批次大小

# 学习率配置
LR=0.0001                                  # 初始学习率
MIN_LR=0.000001                            # 最小学习率
OPTIMIZER="Adam"                           # 优化器
# 可选优化器：
#   - Adam                  # Adam（推荐）
#   - Adagrad               # Adagrad（DNANet 原始）
#   - SGD                   # SGD

SCHEDULER="CosineAnnealingLR"              # 学习率调度器
# 可选调度器：
#   - CosineAnnealingLR     # 余弦退火（推荐）
#   - ReduceLROnPlateau     # 平台降低

WEIGHT_DECAY=0.0005                        # 权重衰减（L2正则化）

# ============================================================
# 数据预处理
# ============================================================

BASE_SIZE=512                              # 基础图像尺寸
CROP_SIZE=480                              # 裁剪图像尺寸
SUFFIX=".png"                              # 图像后缀
SPLIT_METHOD="split_data"                  # 数据划分方法
# 可选划分方法：
#   - split_data            # 使用 split_data 文件夹（推荐）
#   - 50_50                 # 50% 训练 / 50% 测试
#   - 10000_100             # 用于 NUST-SIRST

MODE="TXT"                                 # 模式
# 可选：TXT, Ratio

WORKERS=4                                  # DataLoader 线程数

# ============================================================
# PoLaRIS 特定配置
# ============================================================

USE_LIDAR_DATALOADER="True"                # 使用 PoLaRIS DataLoader
# 可选：
#   - True: 使用新版 DataLoader（支持 16-bit + LiDAR + 软标签）
#   - False: 使用旧版 DataLoader（8-bit）

NORMALIZE_16BIT="True"                     # 16-bit 归一化方式
# 可选：
#   - True: Min-Max 归一化（推荐）
#   - False: 简单缩放

USE_SOFT_LABELS="True"                     # 使用软标签
# 可选：
#   - True: 使用 Oracle Masks（软标签：0.0, 0.6, 1.0）
#   - False: 使用硬标签（0.0, 1.0）

# ============================================================
# 实验管理
# ============================================================

EXPERIMENT_NAME="Phase3_DualGeo_16bit"     # 实验名称
# 建议命名规则：
#   - Phase3_DualGeo_16bit  # 16-bit 模式
#   - Phase3_DualGeo_8bit   # 8-bit 模式
#   - baseline1             # Baseline 实验
#   - ablation_no_depth     # 消融实验

SEED=42                                    # 随机种子

# ============================================================
# 高级配置（可选）
# ============================================================

# 损失函数权重（在代码中配置）
# IOU_WEIGHT=0.5
# BCE_WEIGHT=0.5
# POS_WEIGHT=2.0

# 数据增强（在代码中配置）
# RANDOM_FLIP=True
# RANDOM_ROTATION=True
# COLOR_JITTER=True

# ============================================================
# 快速配置模板
# ============================================================

# 模板 1: 16-bit 模式（推荐）
run_16bit_mode() {
    python train_Phase3.py \
        --experiment_name "Phase3_DualGeo_16bit" \
        --model MS_CAFNet_DualGeo \
        --dataset Pohang-Canal-3k \
        --root dataset/ \
        --train_batch_size 4 \
        --test_batch_size 4 \
        --epochs 200 \
        --optimizer Adam \
        --lr 0.0001 \
        --weight_decay 0.0005 \
        --scheduler CosineAnnealingLR \
        --min_lr 0.000001 \
        --in_channels 2 \
        --base_size 512 \
        --crop_size 480 \
        --deep_supervision False \
        --seed 42 \
        --use_lidar_dataloader True \
        --normalize_16bit True \
        --use_soft_labels True \
        --suffix .png \
        --split_method split_data \
        --workers 4
}

# 模板 2: 8-bit 模式（旧版）
run_8bit_mode() {
    python train_Phase3.py \
        --experiment_name "Phase3_DualGeo_8bit" \
        --model MS_CAFNet_DualGeo \
        --dataset Pohang-Canal \
        --root dataset/ \
        --train_batch_size 4 \
        --test_batch_size 4 \
        --epochs 200 \
        --optimizer Adam \
        --lr 0.0001 \
        --weight_decay 0.0005 \
        --scheduler CosineAnnealingLR \
        --min_lr 0.000001 \
        --in_channels 2 \
        --base_size 512 \
        --crop_size 480 \
        --deep_supervision False \
        --seed 42 \
        --use_lidar_dataloader False \
        --use_soft_labels False \
        --suffix .png \
        --split_method split_data \
        --workers 4
}

# 模板 3: 仅红外模式（无深度图）
run_ir_only_mode() {
    python train_Phase3.py \
        --experiment_name "Phase3_DualGeo_16bit_IR_only" \
        --model MS_CAFNet_DualGeo \
        --dataset Pohang-Canal-3k \
        --root dataset/ \
        --train_batch_size 4 \
        --test_batch_size 4 \
        --epochs 200 \
        --optimizer Adam \
        --lr 0.0001 \
        --weight_decay 0.0005 \
        --scheduler CosineAnnealingLR \
        --min_lr 0.000001 \
        --in_channels 1 \
        --base_size 512 \
        --crop_size 480 \
        --deep_supervision False \
        --seed 42 \
        --use_lidar_dataloader True \
        --normalize_16bit True \
        --use_soft_labels True \
        --suffix .png \
        --split_method split_data \
        --workers 4
}

# 模板 4: DNANet Baseline
run_baseline1() {
    python train.py \
        --experiment_name "baseline1" \
        --model DNANet \
        --dataset Pohang-Canal \
        --root dataset/ \
        --train_batch_size 8 \
        --test_batch_size 8 \
        --epochs 200 \
        --optimizer Adagrad \
        --lr 0.05 \
        --weight_decay 0 \
        --scheduler CosineAnnealingLR \
        --min_lr 0.00001 \
        --in_channels 3 \
        --base_size 256 \
        --crop_size 256 \
        --deep_supervision True \
        --backbone resnet_18 \
        --channel_size three \
        --seed 42 \
        --suffix .png \
        --split_method split_data \
        --workers 4
}

# 模板 5: DNANet 变体
run_baseline2() {
    python train.py \
        --experiment_name "baseline2" \
        --model DNANet \
        --dataset Pohang-Canal \
        --root dataset/ \
        --train_batch_size 8 \
        --test_batch_size 8 \
        --epochs 200 \
        --optimizer Adagrad \
        --lr 0.05 \
        --weight_decay 0 \
        --scheduler CosineAnnealingLR \
        --min_lr 0.00001 \
        --in_channels 3 \
        --base_size 256 \
        --crop_size 256 \
        --deep_supervision False \
        --backbone resnet_18 \
        --channel_size one \
        --seed 42 \
        --suffix .png \
        --split_method split_data \
        --workers 4
}

# ============================================================
# 自定义训练（使用上面的配置变量）
# ============================================================

run_custom() {
    python train_Phase3.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --root "$ROOT" \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --test_batch_size $TEST_BATCH_SIZE \
        --epochs $EPOCHS \
        --start_epoch $START_EPOCH \
        --optimizer "$OPTIMIZER" \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --scheduler "$SCHEDULER" \
        --min_lr $MIN_LR \
        --in_channels $IN_CHANNELS \
        --base_size $BASE_SIZE \
        --crop_size $CROP_SIZE \
        --deep_supervision $DEEP_SUPERVISION \
        --backbone "$BACKBONE" \
        --seed $SEED \
        --use_lidar_dataloader "$USE_LIDAR_DATALOADER" \
        --normalize_16bit "$NORMALIZE_16BIT" \
        --use_soft_labels "$USE_SOFT_LABELS" \
        --suffix "$SUFFIX" \
        --split_method "$SPLIT_METHOD" \
        --workers $WORKERS
}

# ============================================================
# 主函数
# ============================================================

main() {
    echo "============================================================"
    echo "PoLaRIS 训练配置"
    echo "============================================================"
    echo "数据集: $DATASET"
    echo "模型: $MODEL"
    echo "实验名称: $EXPERIMENT_NAME"
    echo "GPU: $CUDA_VISIBLE_DEVICES"
    echo "============================================================"
    echo ""

    # 选择要运行的模式（取消注释一个）

    # run_16bit_mode        # 16-bit 模式（推荐）
    # run_8bit_mode         # 8-bit 模式
    # run_ir_only_mode      # 仅红外模式
    # run_baseline1         # DNANet Baseline
    # run_baseline2         # DNANet 变体
    run_custom              # 自定义配置（默认）

    echo ""
    echo "============================================================"
    echo "✅ 训练完成！"
    echo "============================================================"
}

# 运行主函数
main

# ============================================================
# 使用说明
# ============================================================
#
# 方法 1: 使用预定义模板
#   1. 在 main() 函数中取消注释想要的模式
#   2. 运行：./run_config.sh
#
# 方法 2: 自定义配置
#   1. 修改顶部的配置变量
#   2. 在 main() 中使用 run_custom
#   3. 运行：./run_config.sh
#
# 方法 3: 直接运行模板
#   ./run_config.sh && run_16bit_mode
#
# ============================================================
# 参数说明
# ============================================================
#
# 数据集相关：
#   --dataset          数据集名称
#   --root             数据集根目录
#   --suffix           图像后缀（.png, .jpg 等）
#   --split_method     数据划分方法
#   --base_size        基础图像尺寸
#   --crop_size        裁剪图像尺寸
#
# 模型相关：
#   --model            模型名称
#   --backbone         骨干网络
#   --in_channels      输入通道数
#   --deep_supervision 深度监督
#   --channel_size     通道大小（DNANet）
#
# 训练相关：
#   --epochs           训练轮数
#   --train_batch_size 训练批次大小
#   --test_batch_size  测试批次大小
#   --optimizer        优化器
#   --lr               学习率
#   --scheduler        学习率调度器
#   --weight_decay     权重衰减
#
# PoLaRIS 特定：
#   --use_lidar_dataloader  使用新版 DataLoader
#   --normalize_16bit       16-bit 归一化方式
#   --use_soft_labels       使用软标签
#
# ============================================================
