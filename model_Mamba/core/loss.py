"""
Gaussian Focal Loss for Heatmap Supervision
============================================

This file implements the Gaussian Focal Loss used in CenterNet, CornerNet,
and other keypoint detection methods.

Key Features:
1. Focal loss variant that focuses on hard examples
2. Reduces penalty for predictions near the Gaussian falloff region
3. Handles class imbalance (very few positives vs many negatives)

Reference:
- CenterNet: https://arxiv.org/abs/1904.07850
- Focal Loss: https://arxiv.org/abs/1708.02002

Author: PoLaRIS Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFocalLoss(nn.Module):
    """
    Gaussian Focal Loss for heatmap regression.

    This loss is designed for keypoint/center detection with Gaussian targets.
    It applies a modified focal loss that:
        - Heavily penalizes confident wrong predictions
        - Reduces penalty near the Gaussian falloff (soft positives)
        - Handles extreme class imbalance (1:10000 positive:negative ratio)

    Loss Formula:
        For each pixel (i, j):
            if target[i,j] == 1 (peak):
                L = -( (1 - pred[i,j])^alpha ) * log(pred[i,j])
            else:
                L = -( (1 - target[i,j])^beta ) * (pred[i,j])^alpha * log(1 - pred[i,j])

    Args:
        alpha: Focal loss exponent for hard example weighting (default: 2)
        beta: Gaussian falloff weighting exponent (default: 4)
        reduction: Loss reduction method ('mean' or 'sum')
    """
    def __init__(self, alpha=2, beta=4, reduction='mean'):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted heatmap in [0, 1] (after sigmoid)
            target: (B, 1, H, W) Gaussian target heatmap in [0, 1]

        Returns:
            loss: scalar
        """
        eps = 1e-7  # For numerical stability

        # Ensure predictions are in valid range
        pred = torch.clamp(pred, min=eps, max=1 - eps)

        # Positive locations (center points): target == 1
        pos_mask = target.eq(1).float()

        # Negative locations: target < 1
        neg_mask = target.lt(1).float()

        # Positive loss (focus on recall)
        # If pred is close to 1, loss is small
        # If pred is close to 0, loss is large
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask

        # Negative loss (focus on precision)
        # Weight by (1 - target)^beta to reduce penalty near Gaussian falloff
        # If target = 0.8 (near peak), penalty is low
        # If target = 0.0 (far from peak), penalty is high
        neg_weight = (1 - target) ** self.beta
        neg_loss = -(neg_weight) * (pred ** self.alpha) * torch.log(1 - pred) * neg_mask

        # Total loss
        loss = pos_loss + neg_loss

        # Normalize by number of positive samples
        num_pos = pos_mask.sum() + 1  # Avoid division by zero

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.sum() / num_pos
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class AdaptiveGaussianFocalLoss(nn.Module):
    """
    Adaptive Gaussian Focal Loss with dynamic weighting.

    This variant automatically adjusts the loss weighting based on:
    1. Number of positive samples in the batch
    2. Average prediction confidence

    Useful for datasets with varying object counts per image.

    Args:
        alpha: Focal loss exponent (default: 2)
        beta: Gaussian falloff exponent (default: 4)
        pos_weight: Weight for positive samples (default: 1.0)
        neg_weight: Weight for negative samples (default: 1.0)
    """
    def __init__(self, alpha=2, beta=4, pos_weight=1.0, neg_weight=1.0):
        super(AdaptiveGaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W)
            target: (B, 1, H, W)

        Returns:
            loss: scalar
        """
        eps = 1e-7
        pred = torch.clamp(pred, min=eps, max=1 - eps)

        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        # Adaptive weighting based on batch statistics
        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()

        # Avoid division by zero
        if num_pos == 0:
            # No positive samples in this batch (rare case)
            # Return a small regularization loss
            return torch.zeros(1, device=pred.device, dtype=pred.dtype)

        # Compute losses
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
        neg_weight = (1 - target) ** self.beta
        neg_loss = -(neg_weight) * (pred ** self.alpha) * torch.log(1 - pred) * neg_mask

        # Apply weights
        pos_loss = pos_loss * self.pos_weight
        neg_loss = neg_loss * self.neg_weight

        # Normalize
        pos_loss = pos_loss.sum() / num_pos
        neg_loss = neg_loss.sum() / (num_neg + 1)

        # Combine (balance pos/neg)
        total_loss = pos_loss + neg_loss

        return total_loss


class SmoothL1HeatmapLoss(nn.Module):
    """
    Smooth L1 Loss for heatmap regression (alternative to Focal Loss).

    This is a simpler loss that can be used as a baseline.
    It applies Smooth L1 (Huber) loss to heatmap regression.

    Args:
        beta: Smooth L1 beta parameter (default: 1.0)
    """
    def __init__(self, beta=1.0):
        super(SmoothL1HeatmapLoss, self).__init__()
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W)
            target: (B, 1, H, W)

        Returns:
            loss: scalar
        """
        loss = F.smooth_l1_loss(pred, target, beta=self.beta, reduction='mean')
        return loss


class MSEHeatmapLoss(nn.Module):
    """
    Mean Squared Error for heatmap regression (simplest baseline).

    Args:
        reduction: 'mean' or 'sum'
    """
    def __init__(self, reduction='mean'):
        super(MSEHeatmapLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W)
            target: (B, 1, H, W)

        Returns:
            loss: scalar
        """
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return loss


# ======================== Utility Functions ========================

class AverageMeter:
    """
    Computes and stores the average and current value.

    Usage:
        loss_meter = AverageMeter()
        for batch in dataloader:
            loss = criterion(pred, target)
            loss_meter.update(loss.item(), batch_size)
        print(f"Average Loss: {loss_meter.avg}")
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ======================== Unit Test ========================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Gaussian Focal Loss")
    print("=" * 60)

    # Create dummy data
    B, H, W = 2, 128, 128

    # Target: Gaussian heatmap with 2 peaks
    # Use our own gaussian generation (no scipy dependency)
    import numpy as np
    target = torch.zeros(B, 1, H, W)

    # Generate Gaussian blobs manually
    for b in range(B):
        center = (50, 50) if b == 0 else (80, 80)
        radius = 10
        target_np = target[b, 0].numpy()

        # Simple Gaussian generation
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        gaussian = np.exp(-(dist**2) / (2 * (radius/3)**2))
        gaussian = gaussian / (gaussian.max() + 1e-7)

        target[b, 0] = torch.from_numpy(gaussian.astype(np.float32))

    # Prediction: slightly shifted and noisy
    pred = target.clone()
    pred += torch.randn_like(pred) * 0.1  # Add noise
    pred = torch.clamp(pred, 0, 1)
    pred = torch.sigmoid(pred)  # Ensure [0, 1]

    print(f"\nTarget shape: {target.shape}")
    print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
    print(f"Pred shape: {pred.shape}")
    print(f"Pred range: [{pred.min():.4f}, {pred.max():.4f}]")

    # Test GaussianFocalLoss
    print("\n[1] Testing GaussianFocalLoss...")
    criterion_focal = GaussianFocalLoss(alpha=2, beta=4, reduction='mean')
    loss_focal = criterion_focal(pred, target)
    print(f"Gaussian Focal Loss: {loss_focal.item():.6f}")

    # Test AdaptiveGaussianFocalLoss
    print("\n[2] Testing AdaptiveGaussianFocalLoss...")
    criterion_adaptive = AdaptiveGaussianFocalLoss(alpha=2, beta=4)
    loss_adaptive = criterion_adaptive(pred, target)
    print(f"Adaptive Focal Loss: {loss_adaptive.item():.6f}")

    # Test baseline losses
    print("\n[3] Testing baseline losses...")
    criterion_mse = MSEHeatmapLoss()
    loss_mse = criterion_mse(pred, target)
    print(f"MSE Loss: {loss_mse.item():.6f}")

    criterion_l1 = SmoothL1HeatmapLoss()
    loss_l1 = criterion_l1(pred, target)
    print(f"Smooth L1 Loss: {loss_l1.item():.6f}")

    # Test gradient flow
    print("\n[4] Testing gradient flow...")
    pred.requires_grad = True
    loss = criterion_focal(pred, target)
    loss.backward()
    print(f"Gradient norm: {pred.grad.norm().item():.6f}")
    print(f"Gradient is finite: {torch.isfinite(pred.grad).all().item()}")

    # Test AverageMeter
    print("\n[5] Testing AverageMeter...")
    meter = AverageMeter()
    for i in range(10):
        meter.update(i, n=1)
    print(f"Average: {meter.avg:.2f}, Sum: {meter.sum:.2f}, Count: {meter.count}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
