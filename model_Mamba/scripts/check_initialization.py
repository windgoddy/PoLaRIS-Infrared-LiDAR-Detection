#!/usr/bin/env python3
"""
Quick script to verify model initialization is correct.
Checks:
1. Head bias is -2.19
2. Initial predictions are ~0.1 (not 0.5)
3. Initial loss is reasonable (<2.0)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from core.polaris_mamba import polaris_mamba_tiny
from core.loss import GaussianFocalLoss, CombinedLoss

def check_initialization():
    print("=" * 60)
    print("Model Initialization Check")
    print("=" * 60)
    
    # Create model
    model = polaris_mamba_tiny(use_lidar=True)
    model.eval()
    
    # Check 1: Head bias
    print("\n1. Checking Gaussian Head bias...")
    if hasattr(model, 'head') and hasattr(model.head, 'conv_out'):
        bias_value = model.head.conv_out.bias.data.item()
        print(f"   Head bias: {bias_value:.4f}")
        if abs(bias_value - (-2.19)) < 0.01:
            print("   ✅ PASS: Bias correctly initialized to -2.19")
        else:
            print(f"   ❌ FAIL: Expected -2.19, got {bias_value:.4f}")
    else:
        print("   ❌ FAIL: Could not find head.conv_out")
    
    # Check 2: Initial predictions
    print("\n2. Checking initial predictions...")
    dummy_ir = torch.randn(2, 1, 480, 480)
    dummy_lidar = torch.randn(2, 1, 480, 480)
    
    with torch.no_grad():
        pred = model(dummy_ir, dummy_lidar)
    
    pred_min = pred.min().item()
    pred_max = pred.max().item()
    pred_mean = pred.mean().item()
    
    print(f"   Pred range: [{pred_min:.4f}, {pred_max:.4f}]")
    print(f"   Pred mean: {pred_mean:.4f}")
    
    if 0.05 < pred_mean < 0.15:
        print("   ✅ PASS: Predictions centered around 0.1 (healthy)")
    elif pred_mean < 0.01:
        print("   ⚠️  WARNING: Predictions too low (may struggle to detect)")
    elif pred_mean > 0.4:
        print("   ⚠️  WARNING: Predictions too high (will trigger false positives)")
    else:
        print("   ✅ OK: Predictions in acceptable range")
    
    # Check 3: Initial loss magnitude
    print("\n3. Checking initial loss magnitude...")
    
    # Create dummy target (single small gaussian blob)
    target = torch.zeros(2, 1, 480, 480)
    target[0, 0, 240:250, 240:250] = 1.0  # Small hot spot
    target[1, 0, 100:110, 100:110] = 1.0
    
    # Test Focal Loss
    focal_loss = GaussianFocalLoss(alpha=1, beta=4)
    loss_focal = focal_loss(pred, target)
    print(f"   GaussianFocalLoss: {loss_focal.item():.4f}")
    
    if loss_focal.item() < 2.0:
        print("   ✅ PASS: Loss is reasonable (should decrease quickly)")
    else:
        print(f"   ❌ FAIL: Loss too high ({loss_focal.item():.2f}), training may be unstable")
    
    # Test Combined Loss
    combined_loss = CombinedLoss(focal_weight=0.7, dice_weight=0.3, alpha=1, beta=4)
    loss_combined = combined_loss(pred, target)
    print(f"   CombinedLoss: {loss_combined.item():.4f}")
    
    # Check 4: Count parameters
    print("\n4. Model statistics...")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {num_params / 1e6:.2f}M")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    print("\n" + "=" * 60)
    print("Initialization check complete!")
    print("=" * 60)

if __name__ == '__main__':
    check_initialization()
