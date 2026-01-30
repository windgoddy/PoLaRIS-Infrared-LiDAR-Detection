#!/bin/bash
# Dataset Setup Helper for PoLaRIS-Mamba
# Usage: bash model_Mamba/scripts/check_dataset.sh

set -e

echo "========================================"
echo "PoLaRIS Dataset Structure Checker"
echo "========================================"

DATASET_ROOT="dataset/Pohang-Canal-3k"
SPLIT_METHOD="50_50_2k_new"  # 根据实际存在的 split 目录修改

echo ""
echo "📁 Checking dataset structure..."
echo ""

# Check if dataset root exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "❌ Dataset root not found: $DATASET_ROOT"
    exit 1
fi

echo "✓ Dataset root: $DATASET_ROOT"

# Check required directories
REQUIRED_DIRS=("images" "masks" "labels")
OPTIONAL_DIRS=("lidar_roi" "depth_maps" "oracle_masks")
MISSING_DIRS=()

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$DATASET_ROOT/$dir" ]; then
        count=$(ls -1 "$DATASET_ROOT/$dir" 2>/dev/null | wc -l)
        echo "✓ $DATASET_ROOT/$dir/ ($count files)"
    else
        echo "❌ $DATASET_ROOT/$dir/ (NOT FOUND)"
        MISSING_DIRS+=("$dir")
    fi
done

for dir in "${OPTIONAL_DIRS[@]}"; do
    if [ -d "$DATASET_ROOT/$dir" ]; then
        count=$(ls -1 "$DATASET_ROOT/$dir" 2>/dev/null | wc -l)
        echo "✓ $DATASET_ROOT/$dir/ ($count files, optional)"
    else
        echo "⚠️  $DATASET_ROOT/$dir/ (optional, not found)"
    fi
done

# Check split files
echo ""
echo "📄 Checking split files..."
SPLIT_DIR="$DATASET_ROOT/$SPLIT_METHOD"

if [ ! -d "$SPLIT_DIR" ]; then
    echo "❌ Split directory not found: $SPLIT_DIR"
    echo ""
    echo "Available splits:"
    ls -d $DATASET_ROOT/50_50_* 2>/dev/null || echo "  (none found)"
    exit 1
fi

if [ -f "$SPLIT_DIR/train.txt" ]; then
    train_count=$(wc -l < "$SPLIT_DIR/train.txt")
    echo "✓ $SPLIT_DIR/train.txt ($train_count samples)"
    echo "  First 5 samples:"
    head -5 "$SPLIT_DIR/train.txt" | sed 's/^/    /'
else
    echo "❌ $SPLIT_DIR/train.txt NOT FOUND"
    exit 1
fi

if [ -f "$SPLIT_DIR/test.txt" ]; then
    test_count=$(wc -l < "$SPLIT_DIR/test.txt")
    echo "✓ $SPLIT_DIR/test.txt ($test_count samples)"
else
    echo "❌ $SPLIT_DIR/test.txt NOT FOUND"
    exit 1
fi

# Verify sample files
echo ""
echo "🔍 Verifying sample files..."
SAMPLE_ID=$(head -1 "$SPLIT_DIR/train.txt")
echo "  Checking sample: $SAMPLE_ID"

for ext in "images/$SAMPLE_ID.png" "masks/$SAMPLE_ID.png" "labels/$SAMPLE_ID.txt"; do
    if [ -f "$DATASET_ROOT/$ext" ]; then
        echo "  ✓ $ext"
    else
        echo "  ✗ $ext (NOT FOUND)"
    fi
done

# Summary
echo ""
echo "========================================"
if [ ${#MISSING_DIRS[@]} -eq 0 ]; then
    echo "✅ Dataset structure looks good!"
    echo ""
    echo "Next steps:"
    echo "  1. Run verification: python model_Mamba/scripts/verify_dataset.py"
    echo "  2. Start training: bash model_Mamba/scripts/auto_train_gpu.sh"
else
    echo "⚠️  Missing directories: ${MISSING_DIRS[*]}"
    echo ""
    echo "Please ensure all required directories exist."
fi
echo "========================================"
