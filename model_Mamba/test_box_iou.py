"""
Quick Test for Mask-to-Box IoU Function
========================================

验证 calculate_mask_to_box_iou 函数是否正常工作。

Author: PoLaRIS Team
Date: 2026-02-03
"""

import torch
import sys
import os

# Add path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_Mamba.core.metrics import calculate_mask_to_box_iou

print("=" * 60)
print("Testing Mask-to-Box IoU Function")
print("=" * 60)

# Test 1: Perfect match (box vs box)
print("\n[Test 1] Perfect box match")
pred = torch.zeros(2, 1, 128, 128)
gt = torch.zeros(2, 1, 128, 128)
pred[:, :, 40:60, 40:60] = 1.0  # 20x20 box
gt[:, :, 40:60, 40:60] = 1.0    # Same box

box_iou = calculate_mask_to_box_iou(pred, gt, threshold=0.5)
print(f"  Pred: 20x20 box at [40:60, 40:60]")
print(f"  GT:   20x20 box at [40:60, 40:60]")
print(f"  Box IoU: {box_iou:.4f}")
print(f"  ✅ PASS" if abs(box_iou - 1.0) < 0.01 else f"❌ FAIL (expected ~1.0)")

# Test 2: Ellipse vs Box (弱监督场景)
print("\n[Test 2] Ellipse vs Box (Weak Supervision)")
import numpy as np

def create_ellipse_mask(h, w, center, a, b):
    """创建椭圆 mask"""
    y, x = np.ogrid[:h, :w]
    cy, cx = center
    mask = ((x - cx)**2 / a**2 + (y - cy)**2 / b**2) <= 1
    return mask.astype(np.float32)

pred_np = np.zeros((128, 128), dtype=np.float32)
gt_np = np.zeros((128, 128), dtype=np.float32)

# Pred: 实心矩形 20x20
pred_np[40:60, 40:60] = 1.0

# GT: 椭圆 (长轴30, 短轴20)
gt_np = create_ellipse_mask(128, 128, center=(50, 50), a=15, b=10)

pred_ellipse = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)
gt_ellipse = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0)

seg_iou = ((pred_ellipse * gt_ellipse).sum() / 
           (pred_ellipse + gt_ellipse).clamp(0, 1).sum()).item()
box_iou_ellipse = calculate_mask_to_box_iou(pred_ellipse, gt_ellipse, threshold=0.5)

print(f"  Pred: 20x20 solid box")
print(f"  GT:   30x20 ellipse")
print(f"  Segmentation IoU: {seg_iou:.4f} (低，因为形状不同)")
print(f"  Mask-to-Box IoU : {box_iou_ellipse:.4f} (高，因为外接矩形相似)")
print(f"  💡 这就是 Box IoU 的价值！")

# Test 3: Partial overlap
print("\n[Test 3] Partial Overlap (50% area)")
pred_partial = torch.zeros(1, 1, 128, 128)
gt_partial = torch.zeros(1, 1, 128, 128)
pred_partial[:, :, 40:60, 40:60] = 1.0  # 20x20
gt_partial[:, :, 50:70, 50:70] = 1.0    # 20x20, shifted

box_iou_partial = calculate_mask_to_box_iou(pred_partial, gt_partial, threshold=0.5)
print(f"  Pred: [40:60, 40:60]")
print(f"  GT:   [50:70, 50:70]")
print(f"  Box IoU: {box_iou_partial:.4f}")
print(f"  ✅ PASS" if 0.3 < box_iou_partial < 0.5 else f"❌ FAIL")

# Test 4: No overlap (miss detection)
print("\n[Test 4] No Overlap (Missed Detection)")
pred_miss = torch.zeros(1, 1, 128, 128)
gt_miss = torch.zeros(1, 1, 128, 128)
pred_miss[:, :, 10:20, 10:20] = 1.0
gt_miss[:, :, 100:110, 100:110] = 1.0

box_iou_miss = calculate_mask_to_box_iou(pred_miss, gt_miss, threshold=0.5)
print(f"  Pred: [10:20, 10:20]")
print(f"  GT:   [100:110, 100:110]")
print(f"  Box IoU: {box_iou_miss:.4f}")
print(f"  ✅ PASS" if box_iou_miss == 0.0 else f"❌ FAIL (expected 0.0)")

# Test 5: Empty prediction
print("\n[Test 5] Empty Prediction (All Background)")
pred_empty = torch.zeros(1, 1, 128, 128)  # All 0
gt_empty = torch.zeros(1, 1, 128, 128)
gt_empty[:, :, 50:60, 50:60] = 1.0

box_iou_empty = calculate_mask_to_box_iou(pred_empty, gt_empty, threshold=0.5)
print(f"  Pred: All zeros")
print(f"  GT:   [50:60, 50:60]")
print(f"  Box IoU: {box_iou_empty:.4f}")
print(f"  ✅ PASS" if box_iou_empty == 0.0 else f"❌ FAIL (expected 0.0)")

print("\n" + "=" * 60)
print("All tests completed! ✓")
print("=" * 60)
print("\n💡 Key Insight:")
print("   Mask-to-Box IoU is robust to GT shape variations")
print("   (box vs ellipse), making it ideal for weak supervision!")
print("=" * 60)
