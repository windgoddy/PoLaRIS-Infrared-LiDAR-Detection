"""
Training Diagnosis Script for PoLaRIS-Mamba
===========================================

This script performs quick sanity checks on the model and loss function
to diagnose training issues.

Usage:
    python diagnose_training.py

Author: PoLaRIS Team
Date: 2026-01-30
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_Mamba.core.polaris_mamba import polaris_mamba_tiny
from model_Mamba.core.loss import GaussianFocalLoss, CombinedLoss, MSEHeatmapLoss


def test_loss_stability():
    """Test loss stability with varying positive sample counts."""
    print("=" * 70)
    print("Test 1: Loss Stability with Varying Positive Sample Counts")
    print("=" * 70)

    criterion = CombinedLoss()

    # Simulate realistic scenario
    B, H, W = 4, 512, 512

    results = []
    for num_pos in [500, 1000, 2000, 3000]:
        # Create ground truth with specific number of positive pixels
        gt = torch.zeros(B, 1, H, W)
        indices = torch.randperm(H * W)[:num_pos]
        for b in range(B):
            flat_gt = gt[b, 0].view(-1)
            batch_indices = indices[b * (num_pos // B):(b + 1) * (num_pos // B)]
            flat_gt[batch_indices] = 1.0

        # Create prediction (simulate "all-black" scenario)
        pred = torch.rand(B, 1, H, W) * 0.05  # Mean ≈ 0.025

        loss = criterion(pred, gt)

        results.append({
            'num_pos': num_pos,
            'loss': loss.item(),
            'pred_mean': pred.mean().item(),
            'pred_max': pred.max().item(),
        })

        print(f"\nPositive pixels: {num_pos:4d}")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Pred mean: {pred.mean().item():.6f}, max: {pred.max().item():.6f}")

    # Check stability
    losses = [r['loss'] for r in results]
    loss_std = np.std(losses)
    loss_mean = np.mean(losses)
    cv = loss_std / loss_mean  # Coefficient of variation

    print(f"\n{'=' * 70}")
    print(f"Loss Stability Analysis:")
    print(f"  Mean: {loss_mean:.6f}, Std: {loss_std:.6f}, CV: {cv:.4f}")
    if cv < 0.2:
        print(f"  ✅ PASS: Loss is stable (CV < 0.2)")
    else:
        print(f"  ❌ FAIL: Loss is unstable (CV >= 0.2)")
    print(f"{'=' * 70}\n")

    return cv < 0.2


def test_gradient_flow():
    """Test gradient flow through the model."""
    print("=" * 70)
    print("Test 2: Gradient Flow")
    print("=" * 70)

    model = polaris_mamba_tiny(use_lidar=True)
    criterion = CombinedLoss()

    # Create dummy data
    ir = torch.randn(2, 1, 512, 512)
    lidar = torch.randn(2, 1, 512, 512)
    gt = torch.zeros(2, 1, 512, 512)

    # Add some positive samples
    for b in range(2):
        for _ in range(10):
            y, x = np.random.randint(0, 512, 2)
            gt[b, 0, y-10:y+10, x-10:x+10] = 1.0

    # Forward
    pred = model(ir, lidar)
    loss = criterion(pred, gt)

    print(f"\nForward pass:")
    print(f"  Pred shape: {pred.shape}")
    print(f"  Pred range: [{pred.min():.6f}, {pred.max():.6f}]")
    print(f"  Pred mean: {pred.mean():.6f}")
    print(f"  Loss: {loss.item():.6f}")

    # Backward
    loss.backward()

    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if 'head.conv_out.bias' in name:
                print(f"\n  Output bias gradient: {grad_norm:.6f}")

    mean_grad = np.mean(grad_norms)
    max_grad = np.max(grad_norms)

    print(f"\nGradient Statistics:")
    print(f"  Mean grad norm: {mean_grad:.6f}")
    print(f"  Max grad norm: {max_grad:.6f}")

    if mean_grad > 1e-6 and mean_grad < 100:
        print(f"  ✅ PASS: Gradients are in healthy range")
        result = True
    else:
        print(f"  ❌ FAIL: Gradients are too {'small' if mean_grad <= 1e-6 else 'large'}")
        result = False

    print(f"{'=' * 70}\n")
    return result


def test_prediction_range():
    """Test if model can produce diverse predictions."""
    print("=" * 70)
    print("Test 3: Prediction Range")
    print("=" * 70)

    model = polaris_mamba_tiny(use_lidar=True)

    # Create diverse inputs
    ir_black = torch.zeros(1, 1, 512, 512)
    ir_white = torch.ones(1, 1, 512, 512)
    ir_random = torch.randn(1, 1, 512, 512)
    lidar = torch.randn(1, 1, 512, 512)

    model.eval()
    with torch.no_grad():
        pred_black = model(ir_black, lidar)
        pred_white = model(ir_white, lidar)
        pred_random = model(ir_random, lidar)

    print(f"\nPredictions for different inputs:")
    print(f"  All-black input:  mean={pred_black.mean():.6f}, std={pred_black.std():.6f}")
    print(f"  All-white input:  mean={pred_white.mean():.6f}, std={pred_white.std():.6f}")
    print(f"  Random input:     mean={pred_random.mean():.6f}, std={pred_random.std():.6f}")

    # Check if predictions are diverse
    means = [pred_black.mean().item(), pred_white.mean().item(), pred_random.mean().item()]
    diversity = max(means) - min(means)

    print(f"\nPrediction diversity: {diversity:.6f}")
    if diversity > 0.01:
        print(f"  ✅ PASS: Model produces diverse predictions")
        result = True
    else:
        print(f"  ⚠️  WARNING: Model predictions are too similar (may need more training)")
        result = False

    print(f"{'=' * 70}\n")
    return result


def test_loss_comparison():
    """Compare different loss functions."""
    print("=" * 70)
    print("Test 4: Loss Function Comparison")
    print("=" * 70)

    # Create realistic scenario
    B, H, W = 4, 512, 512
    pred = torch.rand(B, 1, H, W) * 0.1
    gt = torch.zeros(B, 1, H, W)

    # Add Gaussian blobs
    for b in range(B):
        for _ in range(5):  # 5 targets per image
            cy, cx = np.random.randint(50, H-50), np.random.randint(50, W-50)
            radius = 20
            y, x = np.ogrid[:H, :W]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            gaussian = np.exp(-(dist**2) / (2 * (radius/3)**2))
            gt[b, 0] = torch.maximum(gt[b, 0], torch.from_numpy(gaussian.astype(np.float32)))

    # Test different losses
    focal = GaussianFocalLoss()
    combined = CombinedLoss()
    mse = MSEHeatmapLoss()

    loss_focal = focal(pred, gt)
    loss_combined = combined(pred, gt)
    loss_mse = mse(pred, gt)

    print(f"\nLoss comparison:")
    print(f"  Gaussian Focal Loss: {loss_focal.item():.6f}")
    print(f"  Combined Loss:       {loss_combined.item():.6f}")
    print(f"  MSE Loss:            {loss_mse.item():.6f}")

    print(f"\nRecommendation:")
    if loss_combined.item() < 100:
        print(f"  ✅ Combined loss is in reasonable range, should work well")
    else:
        print(f"  ⚠️  Loss values are high, consider using MSE as baseline first")

    print(f"{'=' * 70}\n")


def test_optimizer_step():
    """Test one optimizer step to see if loss decreases."""
    print("=" * 70)
    print("Test 5: Optimizer Step Test")
    print("=" * 70)

    model = polaris_mamba_tiny(use_lidar=True)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # Create training data
    ir = torch.randn(2, 1, 512, 512)
    lidar = torch.randn(2, 1, 512, 512)
    gt = torch.zeros(2, 1, 512, 512)

    # Add targets
    for b in range(2):
        for _ in range(5):
            cy, cx = np.random.randint(100, 400), np.random.randint(100, 400)
            gt[b, 0, cy-20:cy+20, cx-20:cx+20] = 1.0

    # Initial loss
    model.train()
    pred = model(ir, lidar)
    loss_before = criterion(pred, gt).item()

    # Training steps
    losses = [loss_before]
    for step in range(10):
        optimizer.zero_grad()
        pred = model(ir, lidar)
        loss = criterion(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    loss_after = losses[-1]

    print(f"\nTraining progress (10 steps):")
    print(f"  Initial loss: {loss_before:.6f}")
    print(f"  Final loss:   {loss_after:.6f}")
    print(f"  Change:       {loss_after - loss_before:.6f} ({(loss_after/loss_before - 1)*100:+.1f}%)")

    # Check predictions
    model.eval()
    with torch.no_grad():
        pred_final = model(ir, lidar)
    print(f"\nFinal predictions:")
    print(f"  Mean: {pred_final.mean():.6f}")
    print(f"  Max:  {pred_final.max():.6f}")
    print(f"  Min:  {pred_final.min():.6f}")

    if loss_after < loss_before:
        print(f"\n  ✅ PASS: Loss decreased after training")
        result = True
    else:
        print(f"\n  ❌ FAIL: Loss did not decrease")
        result = False

    print(f"{'=' * 70}\n")
    return result


def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 70)
    print("PoLaRIS-Mamba Training Diagnosis")
    print("=" * 70)
    print("\nThis script performs sanity checks on the model and loss function.")
    print("All tests use synthetic data to quickly identify issues.\n")

    # Run tests
    results = {}

    try:
        results['loss_stability'] = test_loss_stability()
    except Exception as e:
        print(f"❌ Test 1 failed with error: {e}\n")
        results['loss_stability'] = False

    try:
        results['gradient_flow'] = test_gradient_flow()
    except Exception as e:
        print(f"❌ Test 2 failed with error: {e}\n")
        results['gradient_flow'] = False

    try:
        results['prediction_range'] = test_prediction_range()
    except Exception as e:
        print(f"❌ Test 3 failed with error: {e}\n")
        results['prediction_range'] = False

    try:
        test_loss_comparison()
    except Exception as e:
        print(f"❌ Test 4 failed with error: {e}\n")

    try:
        results['optimizer_step'] = test_optimizer_step()
    except Exception as e:
        print(f"❌ Test 5 failed with error: {e}\n")
        results['optimizer_step'] = False

    # Summary
    print("=" * 70)
    print("Diagnosis Summary")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name.replace('_', ' ').title()}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your model is ready for training.")
    elif passed >= total * 0.6:
        print("\n⚠️  Some tests failed, but the model should work with careful monitoring.")
    else:
        print("\n❌ Multiple tests failed. Please review the issues above.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
