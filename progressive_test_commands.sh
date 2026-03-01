#!/bin/bash
# Progressive Cat3 Testing Commands
# Run these sequentially to see performance degradation with Cat3

CHECKPOINT="model_Mamba/result/tiny_LiDAR_16bit_PoLaRIS_d2.5p2.0_20260212_221812/best_model_epoch1148_IoU0.7071_BoxIoU0.7928.pth"

# Test 1: Baseline (Cat1+Cat2)
echo "Testing: Baseline (Cat1+Cat2)"
python test_box_iou.py \
  --checkpoint $CHECKPOINT \
  --split_method progressive_test_1_baseline

# Test 2: Baseline + 20 Easy Cat3
echo "Testing: Baseline + 20 Easy Cat3"
python test_box_iou.py \
  --checkpoint $CHECKPOINT \
  --split_method progressive_test_2_add_easy

# Test 3: Baseline + Easy + 50 Medium Cat3
echo "Testing: Baseline + Easy + 50 Medium Cat3"
python test_box_iou.py \
  --checkpoint $CHECKPOINT \
  --split_method progressive_test_3_add_medium

# Test 4: Baseline + All Cat3
echo "Testing: Baseline + All Cat3"
python test_box_iou.py \
  --checkpoint $CHECKPOINT \
  --split_method progressive_test_4_all_cat3

