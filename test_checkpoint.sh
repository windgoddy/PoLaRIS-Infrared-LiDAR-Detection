#!/bin/bash
# ============================================================
# 测试已训练模型的 Mask-to-Box IoU
# ============================================================
# 用法:
#   bash test_checkpoint.sh <checkpoint_path> [gpu_id]
#
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

# 解析参数
if [ $# -lt 1 ]; then
    echo "❌ 错误: 缺少 checkpoint 路径"
    echo ""
    echo "用法: bash test_checkpoint.sh <checkpoint_path> [gpu_id]"
    echo ""
    echo "示例:"
    echo "  bash test_checkpoint.sh result/DNANet_baseline_8bit_Pohang-Canal-3k_DNANet_28_01_2026_17_37_58_wDS/latest_best_model.pth.tar"
    echo "  bash test_checkpoint.sh result/xxx/best_model_epoch0100_mIoU0.5678.pth.tar 0"
    exit 1
fi

CHECKPOINT="$1"

# 可选参数: GPU ID
if [ $# -ge 2 ]; then
    GPU="$2"
fi

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
echo "=========================================="
echo ""

# 从 checkpoint 路径推断数据集配置
if [[ "$CHECKPOINT" == *"Pohang-Canal-3k"* ]]; then
    DATASET="Pohang-Canal-3k"
    SPLIT_METHOD="50_50_2k_new"
    IMAGE_FOLDER="images"
elif [[ "$CHECKPOINT" == *"16bit"* ]]; then
    DATASET="Pohang-Canal-3k"
    SPLIT_METHOD="50_50_2k_new"
    IMAGE_FOLDER="images"
    USE_LIDAR="True"
    NORMALIZE_16BIT="True"
else
    DATASET="Pohang-Canal-3k"
    SPLIT_METHOD="50_50_2k_new"
    IMAGE_FOLDER="images-8bit"
fi

# 运行测试
cd "$SCRIPT_DIR"

python test_box_iou.py \
    --checkpoint "$CHECKPOINT" \
    --gpu "$GPU" \
    --threshold "$THRESHOLD" \
    --batch_size "$BATCH_SIZE" \
    --dataset "$DATASET" \
    --split_method "$SPLIT_METHOD" \
    --image_folder "$IMAGE_FOLDER" \
    --in_channels 1 \
    --base_size 512 \
    --crop_size 480

echo ""
echo "=========================================="
echo "✅ 测试完成!"
echo "=========================================="
echo ""
echo "结果已保存到:"
echo "  $(dirname "$CHECKPOINT")/box_iou_test_results.txt"
echo ""
