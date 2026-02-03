#!/bin/bash
# ============================================================================
# Ellipse vs Box Annotation Comparison Experiment
# ============================================================================
#
# Purpose: Compare training performance with different fill modes:
#   1. Box mode (baseline) - full rectangular annotation
#   2. Ellipse mode (proposed) - inscribed ellipse annotation
#
# Expected Results:
#   - Ellipse mode should achieve:
#     * Lower Best_Threshold (0.3-0.5 vs 0.9)
#     * Higher IoU (60%+ vs 50%)
#     * Better precision (fewer false positives)
#
# Author: PoLaRIS Team
# Date: 2026-02-03
# ============================================================================

set -e  # Exit on error

DATASET="Pohang-Canal-3k"
MODEL="mamba_tiny"
EPOCHS=50  # Quick test (use 200 for full training)
BATCH_SIZE=4
GPU="0"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "  Ellipse vs Box Annotation - Training Comparison"
echo "========================================================================"
echo ""

# ============================================================================
# Experiment 1: Baseline (Box mode - current)
# ============================================================================
echo -e "${BLUE}[Experiment 1/2] Training with Box annotation (baseline)${NC}"
echo "------------------------------------------------------------------------"

EXP_NAME="box_baseline_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="model_Mamba/experiments/${EXP_NAME}"

mkdir -p "${SAVE_DIR}"

# Temporarily modify train.py to use 'box' mode
echo -e "${YELLOW}  Configuring fill_mode='box'...${NC}"
sed -i.bak "s/fill_mode='ellipse'/fill_mode='box'/g" model_Mamba/train.py

python model_Mamba/train.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --epochs ${EPOCHS} \
    --train_batch_size ${BATCH_SIZE} \
    --gpus ${GPU} \
    --save_dir ${SAVE_DIR} \
    --experiment_name ${EXP_NAME}

# Restore original train.py
mv model_Mamba/train.py.bak model_Mamba/train.py

echo -e "${GREEN}✓ Box mode training complete!${NC}"
echo -e "  Results saved to: ${SAVE_DIR}"
echo ""

# ============================================================================
# Experiment 2: Proposed (Ellipse mode)
# ============================================================================
echo -e "${BLUE}[Experiment 2/2] Training with Ellipse annotation (proposed)${NC}"
echo "------------------------------------------------------------------------"

EXP_NAME="ellipse_proposed_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="model_Mamba/experiments/${EXP_NAME}"

mkdir -p "${SAVE_DIR}"

echo -e "${YELLOW}  Using fill_mode='ellipse' (already configured in train.py)${NC}"

python model_Mamba/train.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --epochs ${EPOCHS} \
    --train_batch_size ${BATCH_SIZE} \
    --gpus ${GPU} \
    --save_dir ${SAVE_DIR} \
    --experiment_name ${EXP_NAME}

echo -e "${GREEN}✓ Ellipse mode training complete!${NC}"
echo -e "  Results saved to: ${SAVE_DIR}"
echo ""

# ============================================================================
# Compare Results
# ============================================================================
echo "========================================================================"
echo "  Training Complete - Comparing Results"
echo "========================================================================"
echo ""

echo "Box mode results:"
tail -1 model_Mamba/experiments/box_baseline_*/train_log.csv

echo ""
echo "Ellipse mode results:"
tail -1 model_Mamba/experiments/ellipse_proposed_*/train_log.csv

echo ""
echo "========================================================================"
echo "  Analysis Instructions:"
echo "========================================================================"
echo "1. Check Best_Threshold column (should be lower for ellipse mode)"
echo "2. Compare final IoU (should be higher for ellipse mode)"
echo "3. Check Precision (should be higher for ellipse mode)"
echo ""
echo "Expected improvements with ellipse mode:"
echo "  - Best_Threshold: 0.9 → 0.3-0.5"
echo "  - IoU: 50% → 60%+"
echo "  - Precision: 60% → 70%+"
echo "========================================================================"
