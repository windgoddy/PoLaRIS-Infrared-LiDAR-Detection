#!/bin/bash
# =============================================================================
# Mamba-UNet++ 混合架构训练脚本
# =============================================================================
#
# 功能：
#   训练 Mamba-UNet++ 混合架构模型
#   - Encoder: CNN（浅层，局部特征）+ Mamba（深层，全局上下文）
#   - Decoder: UNet++ 密集跳跃连接（多尺度特征融合）
#   - 目标：结合两者优势，提升小目标检测性能
#
# 用法：
#   bash model_Mamba/scripts/train_mamba_unetpp.sh [mode] [gpu] [batch_size] [dataset] [split_method]
#
#   mode 选项：
#     ir_only     - 单通道 IR（默认）
#     ir_depth    - 双通道 IR + Depth
#     rgb         - 三通道 RGB（与 DNANet 完全匹配）
#
#   gpu 选项：
#     auto        - 自动选择 GPU（默认）
#     0,1,2...    - 手动指定 GPU ID
#
#   batch_size 选项：
#     数值        - 训练 batch size（默认：8）
#
#   dataset 选项：
#     Pohang-Canal-3k - 浦项运河数据集（默认）
#
#   split_method 选项：
#     50_50_2k    - 标准划分（默认）
#     50_50_cat2  - Cat2 小目标专用划分
#
# 示例：
#   bash model_Mamba/scripts/train_mamba_unetpp.sh                           # IR-only，自动GPU
#   bash model_Mamba/scripts/train_mamba_unetpp.sh rgb auto 8 Pohang-Canal-3k 50_50_cat2  # RGB模式，Cat2数据集
#   bash model_Mamba/scripts/train_mamba_unetpp.sh ir_only 0 16              # IR-only，指定GPU 0，batch=16
# =============================================================================

# 自动GPU选择函数
select_gpu() {
    local manual_gpu=$1

    if [ "$manual_gpu" != "auto" ]; then
        echo "$manual_gpu"
        return 0
    fi

    if ! command -v nvidia-smi &> /dev/null; then
        echo "0"
        return 0
    fi

    # 获取GPU信息（ID, 显存总量MB, 已用显存MB）
    gpu_info=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
               awk -F', ' '{free=$2-$3; print $1, $2, free}')

    if [ -z "$gpu_info" ]; then
        echo "0"
        return 0
    fi

    # 选择空闲显存最大的GPU
    best_gpu=$(echo "$gpu_info" | sort -k3 -rn | head -1 | awk '{print $1}')
    echo "$best_gpu"
}

# 默认参数
MODE="${1:-ir_only}"
MANUAL_GPU="${2:-auto}"
BATCH_SIZE="${3:-8}"
DATASET="${4:-Pohang-Canal-3k}"
SPLIT_METHOD="${5:-50_50_2k}"

# 自动选择GPU
GPU=$(select_gpu "$MANUAL_GPU")
export CUDA_VISIBLE_DEVICES=$GPU

# 根据模式设置输入通道
case $MODE in
    ir_only)
        IN_CHANNELS=1
        EXPERIMENT_NAME="MambaUNetPP_IR_only"
        IMAGE_FOLDER="images-8bit"
        echo "🔹 模式: IR-only (单通道)"
        ;;
    ir_depth)
        IN_CHANNELS=2
        EXPERIMENT_NAME="MambaUNetPP_IR_Depth"
        IMAGE_FOLDER="images"
        echo "🔸 模式: IR + Depth (双通道)"
        ;;
    rgb)
        IN_CHANNELS=3
        EXPERIMENT_NAME="MambaUNetPP_RGB"
        IMAGE_FOLDER="images-8bit"
        echo "🎨 模式: RGB (三通道，与 DNANet 完全匹配)"
        ;;
    *)
        echo "❌ 未知模式: $MODE"
        echo "支持的模式: ir_only, ir_depth, rgb"
        exit 1
        ;;
esac

echo "============================================================"
echo "🚀 Mamba-UNet++ 混合架构训练"
echo "============================================================"
if [ "$MANUAL_GPU" == "auto" ]; then
    echo "GPU: $GPU (自动选择)"
else
    echo "GPU: $GPU"
fi
echo "输入通道: $IN_CHANNELS"
echo "Batch Size: $BATCH_SIZE"
echo "数据集: $DATASET"
echo "划分方法: $SPLIT_METHOD"
echo "实验名称: $EXPERIMENT_NAME"
echo "============================================================"

# 开始训练
cd model_Mamba

python train.py \
    --model mamba_unetpp \
    --experiment_name "$EXPERIMENT_NAME" \
    --dataset "$DATASET" \
    --split_method "$SPLIT_METHOD" \
    --in_channels $IN_CHANNELS \
    --image_folder "$IMAGE_FOLDER" \
    --suffix .png \
    --base_size 256 \
    --crop_size 256 \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $BATCH_SIZE \
    --epochs 2000 \
    --optimizer Adagrad \
    --lr 0.05 \
    --scheduler CosineAnnealingLR \
    --min_lr 1e-6 \
    --seed 42 \
    --use_polaris_loader False \
    --normalize_16bit False \
    --loss_type hybrid \
    --dice_weight 1.0 \
    --projection_weight 5.0 \
    --thres 0.3 \
    --workers 4 \
    --use_lidar False \
    --use_deep_supervision False

echo "============================================================"
echo "✅ 训练完成!"
echo "============================================================"
