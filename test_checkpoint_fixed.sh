#!/bin/bash
# Test Mask-to-Box IoU for trained checkpoint

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default parameters
CHECKPOINT=""
GPU=0
THRESHOLD=0.5
BATCH_SIZE=4

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash test_checkpoint.sh <checkpoint_path> [gpu_id]"
    echo ""
    echo "Example:"
    echo "  bash test_checkpoint.sh result/xxx/latest_best_model.pth.tar 5"
    exit 1
fi

CHECKPOINT="$1"

if [ $# -ge 2 ]; then
    GPU="$2"
fi

# Check if checkpoint exists
if [ ! -f "$SCRIPT_DIR/$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "Testing Mask-to-Box IoU"
echo "=========================================="
echo "  Checkpoint: $CHECKPOINT"
echo "  GPU: $GPU"
echo "  Threshold: $THRESHOLD"
echo "=========================================="
echo ""

# Infer dataset configuration from checkpoint path
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

# Run test
cd "$SCRIPT_DIR"

# Note: in_channels and deep_supervision will be auto-detected from checkpoint
python test_box_iou.py \
    --checkpoint "$CHECKPOINT" \
    --gpu "$GPU" \
    --threshold "$THRESHOLD" \
    --batch_size "$BATCH_SIZE" \
    --dataset "$DATASET" \
    --split_method "$SPLIT_METHOD" \
    --image_folder "$IMAGE_FOLDER" \
    --base_size 512 \
    --crop_size 480

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  $(dirname "$CHECKPOINT")/box_iou_test_results.txt"
echo ""
