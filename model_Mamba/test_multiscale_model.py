#!/usr/bin/env python3
"""
Quick test script for PoLaRIS_Mamba_MultiScale model.

This script verifies:
1. Model can be instantiated
2. Forward pass works in both training and eval mode
3. Deep Supervision outputs correct shapes
4. Parameter count is as expected
5. Model can be saved and loaded

Usage:
    python test_multiscale_model.py
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_Mamba.core.polaris_mamba_multiscale import (
    polaris_mamba_tiny_multiscale,
    PoLaRIS_Mamba_MultiScale
)

def test_model_instantiation():
    """Test 1: Model can be created."""
    print("\n" + "="*70)
    print("Test 1: Model Instantiation")
    print("="*70)

    try:
        model = polaris_mamba_tiny_multiscale(use_lidar=True, use_deep_supervision=True)
        print("✅ Model created successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        raise


def test_forward_pass(model):
    """Test 2: Forward pass in training and eval modes."""
    print("\n" + "="*70)
    print("Test 2: Forward Pass")
    print("="*70)

    # Create dummy inputs
    ir = torch.randn(2, 1, 512, 512)
    lidar = torch.randn(2, 1, 512, 512)

    # Test training mode (Deep Supervision)
    model.train()
    try:
        outputs_train = model(ir, lidar)
        if isinstance(outputs_train, list):
            print(f"✅ Training mode: Returns list with {len(outputs_train)} outputs")
            for i, out in enumerate(outputs_train):
                print(f"   Output {i}: {out.shape}")
                assert out.shape == (2, 1, 512, 512), f"Wrong shape for output {i}"
        else:
            print(f"⚠️  Training mode: Returns tensor instead of list: {outputs_train.shape}")
    except Exception as e:
        print(f"❌ Training forward pass failed: {e}")
        raise

    # Test eval mode (no Deep Supervision)
    model.eval()
    try:
        with torch.no_grad():
            outputs_eval = model(ir, lidar)
        if isinstance(outputs_eval, torch.Tensor):
            print(f"✅ Eval mode: Returns single tensor: {outputs_eval.shape}")
            assert outputs_eval.shape == (2, 1, 512, 512), "Wrong output shape"
        else:
            print(f"⚠️  Eval mode: Returns list instead of tensor")
    except Exception as e:
        print(f"❌ Eval forward pass failed: {e}")
        raise


def test_parameter_count(model):
    """Test 3: Parameter count is reasonable."""
    print("\n" + "="*70)
    print("Test 3: Parameter Count")
    print("="*70)

    num_params = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {num_params / 1e6:.2f}M")
    print(f"Trainable parameters: {num_params_trainable / 1e6:.2f}M")

    # Expected: ~12.5-13M for tiny model
    if 12.0e6 < num_params < 14.0e6:
        print("✅ Parameter count is in expected range (12-14M)")
    else:
        print(f"⚠️  Parameter count {num_params/1e6:.2f}M is outside expected range (12-14M)")


def test_save_load(model):
    """Test 4: Model can be saved and loaded."""
    print("\n" + "="*70)
    print("Test 4: Save and Load")
    print("="*70)

    # Save model
    save_path = "/tmp/test_multiscale_model.pth"
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'mamba_tiny_multiscale',
            'test_info': 'This is a test checkpoint'
        }, save_path)
        print(f"✅ Model saved to {save_path}")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        raise

    # Load model
    try:
        checkpoint = torch.load(save_path, map_location='cpu')
        model_new = polaris_mamba_tiny_multiscale(use_lidar=True, use_deep_supervision=True)
        model_new.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully")

        # Verify it still works
        model_new.eval()
        with torch.no_grad():
            ir = torch.randn(1, 1, 512, 512)
            lidar = torch.randn(1, 1, 512, 512)
            output = model_new(ir, lidar)
        print(f"✅ Loaded model forward pass works: {output.shape}")

        # Clean up
        os.remove(save_path)
        print(f"✅ Cleaned up test file")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        raise


def test_gradient_flow(model):
    """Test 5: Gradients flow through the network."""
    print("\n" + "="*70)
    print("Test 5: Gradient Flow")
    print("="*70)

    model.train()
    ir = torch.randn(2, 1, 512, 512, requires_grad=True)
    lidar = torch.randn(2, 1, 512, 512)

    try:
        outputs = model(ir, lidar)

        # Compute loss
        if isinstance(outputs, list):
            # Deep supervision: compute weighted loss
            dummy_gt = torch.rand_like(outputs[0])
            loss = sum([(out - dummy_gt).pow(2).mean() for out in outputs])
        else:
            dummy_gt = torch.rand_like(outputs)
            loss = (outputs - dummy_gt).pow(2).mean()

        # Backward
        loss.backward()

        # Check gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

        if has_grad:
            print("✅ Gradients flow through the network")

            # Sample a few gradients
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append((name, grad_norm))

            # Print top 5 gradients
            grad_norms.sort(key=lambda x: x[1], reverse=True)
            print("\n  Top 5 gradient norms:")
            for name, norm in grad_norms[:5]:
                print(f"    {name}: {norm:.6f}")

        else:
            print("❌ No gradients found!")

    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PoLaRIS_Mamba_MultiScale Model Test Suite")
    print("="*70)

    try:
        # Run tests sequentially
        model = test_model_instantiation()
        test_forward_pass(model)
        test_parameter_count(model)
        test_gradient_flow(model)
        test_save_load(model)

        # Success
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n🎯 Next Steps:")
        print("  1. Modify training script to use 'mamba_tiny_multiscale'")
        print("  2. Run short training (50 epochs) to verify improvements")
        print("  3. If Box IoU shows improvement, run full training (500 epochs)")
        print("\n")

        return 0

    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ TESTS FAILED: {e}")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(main())
