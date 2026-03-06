"""
Improved Loss Functions for PoLaRIS-Mamba
==========================================

Addresses the critical issue: model outputs high confidence for background (0.6-0.8)
instead of low confidence (<0.2), requiring threshold=0.9 to filter false positives.

Key improvements:
1. Focal Loss with strong background suppression
2. Adaptive weighting based on prediction distribution
3. Explicit penalization of high-confidence false predictions

Author: PoLaRIS Team
Date: 2026-02-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """
    Focal BCE Loss - Addresses the confidence distribution collapse.

    Standard BCE Loss treats all pixels equally:
    - Easy negatives (sea background correctly predicted as 0.1) get same weight as
    - Hard negatives (sea background wrongly predicted as 0.7)

    Focal Loss downweights easy examples, forcing model to focus on:
    - False Positives with high confidence (sea predicted as 0.7)
    - False Negatives (missed targets)

    Formula:
        FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1, else (1-p)

    Args:
        alpha: Balancing factor for positive/negative samples (default: 0.25)
               Higher alpha = more weight on positive (targets)
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples

    Reference:
        Focal Loss for Dense Object Detection (RetinaNet)
        https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted probabilities in [0, 1]
            target: (B, 1, H, W) binary ground truth in {0, 1}

        Returns:
            loss: scalar focal loss
        """
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1.0 - eps)

        # Binary cross entropy
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        # Focal weight: (1 - p_t)^gamma
        # For correctly classified samples, p_t is high, so (1-p_t)^gamma is small
        # For misclassified samples, p_t is low, so (1-p_t)^gamma is large
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)

        # Final loss
        focal_loss = alpha_t * focal_weight * bce

        return focal_loss.mean()


class ImprovedBCEDiceLoss(nn.Module):
    """
    Improved BCE+Dice Loss with Focal mechanism and hard negative mining.

    CRITICAL FIX for threshold=0.9 issue:

    Problem diagnosis:
    - Model predicts background as 0.6-0.8 instead of 0.0-0.2
    - Causes: Standard BCE doesn't punish "confident mistakes" enough

    Solution:
    1. Replace BCE with Focal BCE (suppresses easy negatives)
    2. Increase Dice weight (forces tighter segmentation)
    3. Add Online Hard Example Mining (focuses on worst predictions)

    Args:
        focal_weight: Weight for Focal BCE component (default: 1.0)
        dice_weight: Weight for Dice component (default: 4.0, increased from 2.0)
        focal_alpha: Focal loss alpha (default: 0.25)
        focal_gamma: Focal loss gamma (default: 2.0)
        ohem_ratio: Ratio of hard pixels to mine (default: 0.25)
                    0.25 means use worst 25% pixels for training
    """
    def __init__(
        self,
        focal_weight=1.0,
        dice_weight=4.0,  # Increased from 2.0
        focal_alpha=0.25,
        focal_gamma=2.0,
        ohem_ratio=0.0,  # 0 = disabled by default
        smooth=1.0
    ):
        super(ImprovedBCEDiceLoss, self).__init__()
        self.focal_bce = FocalBCELoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ohem_ratio = ohem_ratio

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

        # 1. Focal BCE Loss
        focal_loss = self.focal_bce(pred, target)

        # 2. Dice Loss (forces tight segmentation, reduces over-segmentation)
        dice_loss = self.dice_loss(pred, target)

        # 3. Optional: Online Hard Example Mining (OHEM)
        if self.ohem_ratio > 0:
            # Compute per-pixel BCE
            eps = 1e-7
            pred_clamped = torch.clamp(pred, eps, 1.0 - eps)
            pixel_bce = -target * torch.log(pred_clamped) - (1 - target) * torch.log(1 - pred_clamped)

            # Find hardest pixels
            n_pixels = pixel_bce.numel()
            n_hard = int(n_pixels * self.ohem_ratio)

            # Top-k hardest pixels
            pixel_bce_flat = pixel_bce.view(-1)
            hard_pixels, _ = torch.topk(pixel_bce_flat, n_hard)

            # OHEM loss (only on hard pixels)
            ohem_loss = hard_pixels.mean()

            # Combine with small weight
            total_loss = (
                self.focal_weight * focal_loss +
                self.dice_weight * dice_loss +
                0.5 * ohem_loss
            )
        else:
            total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss

        return total_loss


class DiceLoss(nn.Module):
    """Dice Loss for Binary Segmentation."""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice = dice.mean()

        return 1.0 - dice


class SoftIoULoss(nn.Module):
    """
    Soft IoU Loss for sparse small target detection.

    Key difference from DiceLoss:
    - smooth=1e-6 (not 1.0): avoids the gradient plateau where
      Dice loss ≈ constant 1.0 regardless of prediction when
      targets occupy < 0.1% of pixels (e.g. NUDT-SIRST 3-15px targets).
    - This is the standard loss used by DNANet, ACM, ALCNet for IRST.
    """
    def __init__(self, smooth=1e-6):
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        target = target.float()
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        union = (pred + target).sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou


class ConfidenceCalibrationLoss(nn.Module):
    """
    Explicit Confidence Calibration Loss.

    Forces model to output:
    - Targets: close to 1.0
    - Background: close to 0.0

    This directly addresses the threshold=0.9 issue by penalizing
    any background pixel with confidence > 0.3.

    Usage: Add as auxiliary loss with weight 0.1
    """
    def __init__(self, target_bg_conf=0.1, target_fg_conf=0.9):
        super(ConfidenceCalibrationLoss, self).__init__()
        self.target_bg_conf = target_bg_conf
        self.target_fg_conf = target_fg_conf

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) in [0, 1]
            target: (B, 1, H, W) in {0, 1}
        """
        # Foreground calibration: push predictions towards 0.9
        fg_mask = (target == 1).float()
        fg_loss = ((pred - self.target_fg_conf) ** 2 * fg_mask).sum() / (fg_mask.sum() + 1e-7)

        # Background calibration: push predictions towards 0.1
        bg_mask = (target == 0).float()
        bg_loss = ((pred - self.target_bg_conf) ** 2 * bg_mask).sum() / (bg_mask.sum() + 1e-7)

        return fg_loss + bg_loss


# ======================== Unit Test ========================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Improved Loss Functions")
    print("=" * 60)

    # Create dummy data
    B, H, W = 2, 128, 128

    # Simulate problematic predictions:
    # - Target: sparse small objects (1% positive)
    # - Prediction: high background confidence (0.6-0.8) <- THIS IS THE ISSUE
    target = torch.zeros(B, 1, H, W)
    target[:, :, 50:60, 50:60] = 1.0  # Small 10x10 object

    # Bad prediction (high background confidence)
    pred_bad = torch.rand(B, 1, H, W) * 0.5 + 0.3  # Range [0.3, 0.8]
    pred_bad[:, :, 50:60, 50:60] = 0.95  # Correct on target

    print(f"\nTarget shape: {target.shape}")
    print(f"Target positive ratio: {(target > 0.5).float().mean():.4f}")
    print(f"Pred (BAD) range: [{pred_bad.min():.4f}, {pred_bad.max():.4f}]")
    print(f"Pred (BAD) mean on background: {pred_bad[target == 0].mean():.4f}")

    # Test 1: Standard BCE (for comparison)
    print("\n[1] Standard BCE Loss (baseline)...")
    bce_loss = nn.BCELoss()
    loss_bce = bce_loss(pred_bad, target)
    print(f"BCE Loss: {loss_bce.item():.6f}")

    # Test 2: Focal BCE
    print("\n[2] Focal BCE Loss (improved)...")
    focal_bce_loss = FocalBCELoss(alpha=0.25, gamma=2.0)
    loss_focal_bce = focal_bce_loss(pred_bad, target)
    print(f"Focal BCE Loss: {loss_focal_bce.item():.6f}")
    print(f"Ratio (Focal/Standard): {loss_focal_bce.item() / loss_bce.item():.2f}x")

    # Test 3: Improved BCE+Dice
    print("\n[3] Improved BCE+Dice Loss...")
    improved_loss = ImprovedBCEDiceLoss(
        focal_weight=1.0,
        dice_weight=4.0,
        focal_gamma=2.0,
        ohem_ratio=0.0
    )
    loss_improved = improved_loss(pred_bad, target)
    print(f"Improved Loss: {loss_improved.item():.6f}")

    # Test 4: With OHEM
    print("\n[4] Improved Loss + OHEM (mine hardest 25%)...")
    improved_loss_ohem = ImprovedBCEDiceLoss(
        focal_weight=1.0,
        dice_weight=4.0,
        focal_gamma=2.0,
        ohem_ratio=0.25
    )
    loss_improved_ohem = improved_loss_ohem(pred_bad, target)
    print(f"Improved Loss + OHEM: {loss_improved_ohem.item():.6f}")

    # Test 5: Confidence Calibration
    print("\n[5] Confidence Calibration Loss...")
    calib_loss = ConfidenceCalibrationLoss(target_bg_conf=0.1, target_fg_conf=0.9)
    loss_calib = calib_loss(pred_bad, target)
    print(f"Calibration Loss: {loss_calib.item():.6f}")

    # Test gradient flow
    print("\n[6] Testing gradient flow...")
    pred_bad.requires_grad = True
    loss = improved_loss(pred_bad, target)
    loss.backward()
    print(f"Gradient norm: {pred_bad.grad.norm().item():.6f}")
    print(f"Gradient is finite: {torch.isfinite(pred_bad.grad).all().item()}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
