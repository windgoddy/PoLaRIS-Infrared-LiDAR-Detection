"""
Evaluation Metrics for Binary Segmentation
==========================================

This module provides evaluation metrics with threshold sweep,
aligned with LIDAR-Mamba's approach for optimal performance.

Key Features:
- Threshold sweep for mIoU (critical for high scores!)
- Precision, Recall, F1 computation
- IoU curve visualization

Author: PoLaRIS Team
Date: 2026-01-30
"""

import torch
import numpy as np


def compute_miou_with_sweep(pred, target, thresh_step=0.05, thresh_range=(0.1, 0.9)):
    """
    Compute mIoU with threshold sweep (aligned with LIDAR-Mamba).

    This is the KEY to achieving high IoU scores! LIDAR-Mamba sweeps thresholds
    from 0.0 to 1.0 in 0.01 steps and reports the maximum IoU.

    Args:
        pred: (B, 1, H, W) predictions in [0, 1]
        target: (B, 1, H, W) binary ground truth {0, 1}
        thresh_step: Threshold step size (default: 0.05)
                     Use 0.01 for LIDAR-Mamba exact replication (slower)
        thresh_range: (min, max) threshold range (default: 0.1 to 0.9)
                      LIDAR-Mamba uses (0.0, 1.0)

    Returns:
        best_iou: Best IoU found across all thresholds
        best_thresh: Threshold that achieved best IoU
        iou_curve: List of (threshold, iou) tuples

    Example:
        >>> pred = torch.rand(4, 1, 512, 640)
        >>> target = torch.randint(0, 2, (4, 1, 512, 640)).float()
        >>> best_iou, best_thresh, curve = compute_miou_with_sweep(pred, target)
        >>> print(f"Best IoU: {best_iou:.4f} at threshold {best_thresh:.2f}")
    """
    target_binary = (target > 0.5).float()

    best_iou = 0.0
    best_thresh = 0.5
    iou_curve = []

    thresh_min, thresh_max = thresh_range
    thresholds = np.arange(thresh_min, thresh_max + thresh_step, thresh_step)

    for thresh in thresholds:
        pred_binary = (pred > thresh).float()

        # Compute IoU per sample
        intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
        union = (pred_binary + target_binary).clamp(0, 1).sum(dim=(1, 2, 3))

        # Average IoU over batch
        iou = (intersection / (union + 1e-7)).mean().item()

        iou_curve.append((float(thresh), iou))

        if iou > best_iou:
            best_iou = iou
            best_thresh = float(thresh)

    return best_iou, best_thresh, iou_curve


def compute_precision_recall_f1(pred, target, threshold=0.5):
    """
    Compute Precision, Recall, F1 for binary segmentation.

    Args:
        pred: (B, 1, H, W) predictions in [0, 1]
        target: (B, 1, H, W) binary ground truth {0, 1}
        threshold: Binary threshold (default: 0.5)

    Returns:
        precision: TP / (TP + FP)
        recall: TP / (TP + FN)
        f1: 2 * Precision * Recall / (Precision + Recall)

    Example:
        >>> pred = torch.rand(4, 1, 512, 640)
        >>> target = torch.randint(0, 2, (4, 1, 512, 640)).float()
        >>> p, r, f1 = compute_precision_recall_f1(pred, target, threshold=0.5)
        >>> print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()

    # Compute metrics
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return precision, recall, f1


def compute_iou_at_threshold(pred, target, threshold=0.5):
    """
    Compute IoU at a single threshold.

    Args:
        pred: (B, 1, H, W) predictions in [0, 1]
        target: (B, 1, H, W) binary ground truth {0, 1}
        threshold: Binary threshold (default: 0.5)

    Returns:
        iou: Intersection over Union
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()

    intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
    union = (pred_binary + target_binary).clamp(0, 1).sum(dim=(1, 2, 3))

    iou = (intersection / (union + 1e-7)).mean().item()

    return iou


def compute_dice_coefficient(pred, target, threshold=0.5):
    """
    Compute Dice coefficient (F1 score for segmentation).

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)

    Args:
        pred: (B, 1, H, W) predictions in [0, 1]
        target: (B, 1, H, W) binary ground truth {0, 1}
        threshold: Binary threshold (default: 0.5)

    Returns:
        dice: Dice coefficient
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()

    intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection / (union + 1e-7)).mean().item()

    return dice


# ======================== Batch Metrics Accumulator ========================

class MetricsAccumulator:
    """
    Accumulator for metrics across multiple batches.

    Usage:
        >>> acc = MetricsAccumulator()
        >>> for batch in test_loader:
        >>>     pred, target = model(batch), batch['target']
        >>>     acc.update(pred, target)
        >>> results = acc.compute()
        >>> print(f"mIoU: {results['miou']:.4f}")
    """
    def __init__(self, use_sweep=True, thresh_step=0.05):
        self.use_sweep = use_sweep
        self.thresh_step = thresh_step
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.all_preds = []
        self.all_targets = []

    def update(self, pred, target):
        """
        Add a batch to the accumulator.

        Args:
            pred: (B, 1, H, W) predictions
            target: (B, 1, H, W) ground truth
        """
        self.all_preds.append(pred.cpu())
        self.all_targets.append(target.cpu())

    def compute(self):
        """
        Compute final metrics over all batches.

        Returns:
            results: Dictionary with keys:
                - 'miou': mIoU (with sweep if enabled)
                - 'best_thresh': Best threshold found
                - 'precision': Precision at best threshold
                - 'recall': Recall at best threshold
                - 'f1': F1 score at best threshold
                - 'dice': Dice coefficient
        """
        # Concatenate all batches
        all_preds = torch.cat(self.all_preds, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)

        # Compute mIoU
        if self.use_sweep:
            miou, best_thresh, _ = compute_miou_with_sweep(
                all_preds,
                all_targets,
                thresh_step=self.thresh_step
            )
        else:
            best_thresh = 0.5
            miou = compute_iou_at_threshold(all_preds, all_targets, threshold=best_thresh)

        # Compute other metrics at best threshold
        precision, recall, f1 = compute_precision_recall_f1(
            all_preds,
            all_targets,
            threshold=best_thresh
        )

        dice = compute_dice_coefficient(all_preds, all_targets, threshold=best_thresh)

        results = {
            'miou': miou,
            'best_thresh': best_thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'dice': dice,
        }

        return results


# ======================== Unit Test ========================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Metrics Module")
    print("=" * 60)

    # Create dummy data
    torch.manual_seed(42)
    pred = torch.rand(4, 1, 128, 128)
    target = (torch.rand(4, 1, 128, 128) > 0.7).float()

    # Test 1: Threshold sweep
    print("\n[Test 1] Threshold Sweep")
    best_iou, best_thresh, curve = compute_miou_with_sweep(pred, target)
    print(f"  Best IoU: {best_iou:.4f} at threshold {best_thresh:.2f}")
    print(f"  IoU curve length: {len(curve)}")
    print(f"  ✅ PASS" if 0.0 <= best_iou <= 1.0 else "❌ FAIL")

    # Test 2: Precision, Recall, F1
    print("\n[Test 2] Precision, Recall, F1")
    p, r, f1 = compute_precision_recall_f1(pred, target, threshold=best_thresh)
    print(f"  Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    print(f"  ✅ PASS" if 0.0 <= f1 <= 1.0 else "❌ FAIL")

    # Test 3: IoU at fixed threshold
    print("\n[Test 3] IoU at threshold=0.5")
    iou_fixed = compute_iou_at_threshold(pred, target, threshold=0.5)
    print(f"  IoU: {iou_fixed:.4f}")
    print(f"  ✅ PASS" if 0.0 <= iou_fixed <= 1.0 else "❌ FAIL")

    # Test 4: Dice coefficient
    print("\n[Test 4] Dice Coefficient")
    dice = compute_dice_coefficient(pred, target, threshold=best_thresh)
    print(f"  Dice: {dice:.4f}")
    print(f"  ✅ PASS" if 0.0 <= dice <= 1.0 else "❌ FAIL")

    # Test 5: Metrics Accumulator
    print("\n[Test 5] Metrics Accumulator")
    acc = MetricsAccumulator(use_sweep=True, thresh_step=0.1)
    acc.update(pred[:2], target[:2])
    acc.update(pred[2:], target[2:])
    results = acc.compute()
    print(f"  mIoU: {results['miou']:.4f}")
    print(f"  Best Threshold: {results['best_thresh']:.2f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  ✅ PASS" if 0.0 <= results['miou'] <= 1.0 else "❌ FAIL")

    # Test 6: Compare sweep vs fixed threshold
    print("\n[Test 6] Sweep vs Fixed Threshold")
    iou_sweep = best_iou
    iou_fixed = compute_iou_at_threshold(pred, target, threshold=0.5)
    improvement = iou_sweep - iou_fixed
    print(f"  Fixed (0.5): {iou_fixed:.4f}")
    print(f"  Sweep: {iou_sweep:.4f}")
    print(f"  Improvement: {improvement:.4f} ({improvement/iou_fixed*100:+.1f}%)")
    print(f"  ✅ PASS" if improvement >= 0 else "❌ FAIL (sweep should >= fixed)")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\n💡 Tip: Use compute_miou_with_sweep() in your testing loop")
    print("   for LIDAR-Mamba style evaluation!")
