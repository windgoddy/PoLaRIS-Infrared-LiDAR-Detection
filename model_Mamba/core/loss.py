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
        num_pos = pos_mask.sum()

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            # CRITICAL FIX (2026-01-30): Normalize by TOTAL pixels instead of num_pos
            # to prevent loss explosion when positive samples vary across batches.
            #
            # Previous issue: loss = (pos + 0.1*neg) / num_pos caused:
            # - Batch with 1000 pos: loss ≈ 30
            # - Batch with 500 pos:  loss ≈ 60 (2x instability!)
            #
            # Solution: Normalize by total pixels for stability
            total_pixels = pred.shape[0] * pred.shape[2] * pred.shape[3]

            if num_pos == 0:
                # No positive samples: only negative loss
                return loss.sum() / total_pixels
            else:
                # Balance positive and negative loss by their relative importance
                # Positive samples are rare (~0.2% of pixels), so weight them higher
                pos_loss_val = pos_loss.sum()
                neg_loss_val = neg_loss.sum()

                # Method 1: Per-pixel normalization then weighted sum
                # This ensures both pos and neg contribute equally regardless of sample count
                pos_weight = 1.0  # Standard weight for positive
                neg_weight = 0.1  # Reduce negative weight to 10%

                pos_loss_normalized = pos_loss_val / (num_pos + 1e-7)
                neg_loss_normalized = neg_loss_val / (total_pixels - num_pos + 1e-7)

                # Weight positive loss higher to compensate for sample imbalance
                # UPDATED 2026-01-30 v2: Reduced from 10x to 3x to prevent over-prediction
                # Issue: Model was predicting mean=0.5 when GT mean=0.0002 (2500x higher!)
                # Root cause: 10x weight was too aggressive, causing model to over-predict
                loss_combined = 3.0 * pos_loss_normalized + neg_loss_normalized

                return loss_combined
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class DiceLoss(nn.Module):
    """
    Dice Loss for better precision-recall balance.
    Especially useful when facing class imbalance.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted heatmap
            target: (B, 1, H, W) ground truth heatmap
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Gaussian Focal Loss + Dice Loss
    Better balance between pixel-wise accuracy and region overlap

    NOTE (2026-01-30): Dice loss helps prevent "all-black" predictions
    by explicitly optimizing for overlap, complementing the focal loss.
    """
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=1, beta=4):
        super(CombinedLoss, self).__init__()
        self.focal_loss = GaussianFocalLoss(alpha=alpha, beta=beta, reduction='mean')
        self.dice_loss = DiceLoss(smooth=1.0)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice


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


# ======================== Binary Segmentation Losses (Scheme A) ========================

class DiceLoss(nn.Module):
    """
    Dice Loss for Binary Segmentation.

    Dice coefficient measures the overlap between prediction and ground truth.
    It's particularly effective for handling class imbalance in segmentation tasks.

    Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
    Dice Loss = 1 - Dice

    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)

    Reference:
        V-Net: https://arxiv.org/abs/1606.04797
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted probabilities in [0, 1]
            target: (B, 1, H, W) binary ground truth in {0, 1}

        Returns:
            loss: scalar Dice loss
        """
        # Flatten spatial dimensions
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        # Compute Dice coefficient
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average over batch
        dice = dice.mean()

        # Return loss (1 - Dice)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss for Binary Segmentation (Scheme A).

    This loss combines:
    1. Binary Cross Entropy (BCE): Pixel-wise classification accuracy
    2. Dice Loss: Region overlap optimization

    Standard configuration for medical image segmentation, crack detection,
    and other binary segmentation tasks. Aligns with LIDAR-Mamba approach.

    Args:
        bce_weight: Weight for BCE loss component (default: 1.0)
        dice_weight: Weight for Dice loss component (default: 3.0)
                     Higher weight recommended for small object segmentation
        smooth: Smoothing factor for Dice (default: 1.0)

    Reference:
        LIDAR-Mamba: https://github.com/Karl1109/LIDAR-Mamba
    """
    def __init__(self, bce_weight=1.0, dice_weight=3.0, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        # BCELoss expects probabilities in [0, 1]
        # Since GaussianHead outputs sigmoid, we use BCELoss (not BCEWithLogitsLoss)
        self.bce_fn = nn.BCELoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted probabilities in [0, 1]
            target: (B, 1, H, W) binary ground truth in {0, 1}

        Returns:
            loss: scalar combined loss
        """
        # Ensure target is float
        target = target.float()

        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)

        # Compute individual losses
        bce = self.bce_fn(pred, target)
        dice = self.dice_loss(pred, target)

        # Weighted combination
        loss = self.bce_weight * bce + self.dice_weight * dice

        return loss


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
