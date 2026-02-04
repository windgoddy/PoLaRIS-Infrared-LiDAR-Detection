#!/bin/bash
# ============================================================
# 测试已训练模型的 Mask-to-Box IoU
# ============================================================
# 用法:
#   bash test_checkpoint.sh <checkpoint_path> [gpu_id]

# 示例:
#   bash test_checkpoint.sh result/DNANet_baseline_8bit_Pohang-Canal-3k_DNANet_28_01_2026_17_37_58_wDS/latest_best_model.pth.tar
#   bash test_checkpoint.sh result/xxx/best_model_epoch0100_mIoU0.5678.pth.tar 0
# ============================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 默认参数
CHECKPOINT=""
GPU=0
THRESHOLD=0.5
BATCH_SIZE=4
EVAL_STRATEGY="auto"

# 解析参数
if [ $# -lt 1 ]; then
    echo "❌ 错误: 缺少 checkpoint 路径"
    echo ""
    echo "用法: bash test_checkpoint.sh <checkpoint_path> [gpu_id] [OPTIONS]"
    echo ""
    echo "选项:"
    echo "  --threshold VALUE        固定阈值 (default: 0.5)"
    echo "  --batch_size VALUE       批次大小 (default: 4)"
    echo "  --eval_strategy STRATEGY 评估策略 (auto|fixed|dynamic, default: auto)"
    echo "                          auto: Mamba用dynamic，其他用fixed"
    echo "                          fixed: 所有模型用固定阈值"
    echo "                          dynamic: 所有模型用动态阈值扫描"
    echo ""
    echo "示例:"
    echo "  bash test_checkpoint.sh result/DNANet_baseline_8bit_Pohang-Canal-3k_DNANet_28_01_2026_17_37_58_wDS/latest_best_model.pth.tar"
    echo "  bash test_checkpoint.sh result/xxx/best_model.pth.tar 0 --threshold 0.3"
    echo "  bash test_checkpoint.sh result/DNANet_xxx.pth.tar 0 --eval_strategy dynamic  # DNANet也用动态扫描"
    echo "  bash test_checkpoint.sh model_Mamba/result/xxx.pth 0 --eval_strategy fixed --threshold 0.5  # Mamba用固定阈值"
    exit 1
fi

CHECKPOINT="$1"
shift  # 移除第一个参数

# 可选参数: GPU ID (位置参数)
if [[ $# -ge 1 && ! "$1" =~ ^-- ]]; then
    GPU="$1"
    shift
fi

# 解析命名参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --eval_strategy)
            EVAL_STRATEGY="$2"
            shift 2
            ;;
        *)
            echo "⚠️  未知参数: $1（忽略）"
            shift
            ;;
    esac
done

# 检查 checkpoint 是否存在
if [ ! -f "$SCRIPT_DIR/$CHECKPOINT" ]; then
    echo "❌ 错误: Checkpoint 文件不存在: $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "测试 Mask-to-Box IoU"
echo "=========================================="
echo "  Checkpoint: $CHECKPOINT"
echo "  GPU: $GPU"
echo "  Threshold: $THRESHOLD"
echo "  Eval Strategy: $EVAL_STRATEGY"
echo "=========================================="
echo ""

MODEL=""
USE_LIDAR=""
NORMALIZE_16BIT=""
IN_CHANNELS_OVERRIDE=""

# 从 checkpoint 路径推断数据集配置（优先 Mamba，其次 16bit，再处理 8bit 基线）
if [[ "$CHECKPOINT" == *"mamba"* || "$CHECKPOINT" == *"Mamba"* ]]; then
    DATASET="Pohang-Canal-3k"
    SPLIT_METHOD="50_50_2k_new"
    IMAGE_FOLDER="images"
    USE_LIDAR="True"
    NORMALIZE_16BIT="True"
    MODEL="mamba_tiny"
elif [[ "$CHECKPOINT" == *"16bit"* ]]; then
    DATASET="Pohang-Canal-3k"
    SPLIT_METHOD="50_50_2k_new"
    IMAGE_FOLDER="images"
    USE_LIDAR="True"
    NORMALIZE_16BIT="True"
elif [[ "$CHECKPOINT" == *"8bit"* || "$CHECKPOINT" == *"baseline1"* || "$CHECKPOINT" == *"DNANet_baseline_8bit"* ]]; then
    DATASET="Pohang-Canal-3k"
    SPLIT_METHOD="50_50_2k_new"
    IMAGE_FOLDER="images-8bit"
else
    DATASET="Pohang-Canal-3k"
    SPLIT_METHOD="50_50_2k_new"
    IMAGE_FOLDER="images"
fi

# 运行测试
cd "$SCRIPT_DIR"

EXTRA_ARGS=()
if [[ -n "$MODEL" ]]; then
    EXTRA_ARGS+=(--model "$MODEL")
fi
if [[ -n "$USE_LIDAR" ]]; then
    EXTRA_ARGS+=(--use_lidar_dataloader "$USE_LIDAR")
fi
if [[ -n "$NORMALIZE_16BIT" ]]; then
    EXTRA_ARGS+=(--normalize_16bit "$NORMALIZE_16BIT")
fi
if [[ -n "$IN_CHANNELS_OVERRIDE" ]]; then
    EXTRA_ARGS+=(--in_channels "$IN_CHANNELS_OVERRIDE")
fi

python test_box_iou.py \
    --checkpoint "$CHECKPOINT" \
    --gpu "$GPU" \
    --threshold "$THRESHOLD" \
    --batch_size "$BATCH_SIZE" \
    --eval_strategy "$EVAL_STRATEGY" \
    --dataset "$DATASET" \
    --split_method "$SPLIT_METHOD" \
    --image_folder "$IMAGE_FOLDER" \
    --base_size 512 \
    --crop_size 480 \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=========================================="
echo "✅ 测试完成!"
echo "=========================================="
echo ""
echo "结果已保存到:"
echo "  $(dirname "$CHECKPOINT")/box_iou_test_results.txt"
echo ""
