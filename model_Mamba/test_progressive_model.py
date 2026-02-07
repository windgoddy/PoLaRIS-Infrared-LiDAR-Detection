#!/usr/bin/env python
"""
Test script for PoLaRIS_Mamba_Progressive model.

This script verifies:
1. Model can be imported correctly
2. Model can be initialized
3. Forward pass works correctly
4. Output dimensions are correct
5. No CUDA errors with different batch sizes
"""

import torch
import sys
import os

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_Mamba.core.polaris_mamba_progressive import (
    polaris_mamba_tiny_progressive,
    polaris_mamba_small_progressive,
)


def test_model(model_name, model_fn, use_lidar=True):
    """Test a single model variant."""
    print(f"\n{'='*70}")
    print(f"Testing {model_name}")
    print(f"{'='*70}")

    # Create model
    print("1. Creating model...")
    model = model_fn(use_lidar=use_lidar, use_deep_supervision=False)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Parameters: {num_params / 1e6:.2f}M")

    # Test on CPU first
    print("2. Testing forward pass (CPU)...")
    model.eval()

    # Test different batch sizes
    for batch_size in [1, 2]:
        ir = torch.randn(batch_size, 1, 512, 512)
        lidar = torch.randn(batch_size, 1, 512, 512) if use_lidar else None

        with torch.no_grad():
            output = model(ir, lidar)

        assert output.shape == (batch_size, 1, 512, 512), \
            f"Output shape mismatch: expected ({batch_size}, 1, 512, 512), got {output.shape}"

        print(f"   ✓ Batch size {batch_size}: Input {ir.shape} → Output {output.shape}")

    # Test with different input sizes
    print("3. Testing different input sizes...")
    for size in [256, 512]:
        ir = torch.randn(1, 1, size, size)
        lidar = torch.randn(1, 1, size, size) if use_lidar else None

        with torch.no_grad():
            output = model(ir, lidar)

        assert output.shape == (1, 1, size, size), \
            f"Output shape mismatch for size {size}: got {output.shape}"

        print(f"   ✓ Input size {size}x{size}: Output {output.shape}")

    # Test CUDA if available
    if torch.cuda.is_available():
        print("4. Testing on CUDA...")
        model = model.cuda()

        ir = torch.randn(2, 1, 512, 512).cuda()
        lidar = torch.randn(2, 1, 512, 512).cuda() if use_lidar else None

        with torch.no_grad():
            output = model(ir, lidar)

        assert output.shape == (2, 1, 512, 512), \
            f"CUDA output shape mismatch: got {output.shape}"

        print(f"   ✓ CUDA forward pass: Output {output.shape}")

        # Clean up
        del model, ir, lidar, output
        torch.cuda.empty_cache()
    else:
        print("4. CUDA not available, skipping GPU test")

    print(f"✅ All tests passed for {model_name}!")
    return True


def main():
    print("="*70)
    print("PoLaRIS_Mamba_Progressive Model Test Suite")
    print("="*70)

    all_passed = True

    # Test tiny progressive (with LiDAR)
    try:
        test_model(
            "polaris_mamba_tiny_progressive (LiDAR)",
            polaris_mamba_tiny_progressive,
            use_lidar=True
        )
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test tiny progressive (IR-only)
    try:
        test_model(
            "polaris_mamba_tiny_progressive (IR-only)",
            polaris_mamba_tiny_progressive,
            use_lidar=False
        )
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test small progressive (with LiDAR)
    try:
        test_model(
            "polaris_mamba_small_progressive (LiDAR)",
            polaris_mamba_small_progressive,
            use_lidar=True
        )
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n📝 Next steps:")
        print("  1. Start IR-only training:")
        print("     bash model_Mamba/scripts/train_multiscale_full.sh ir_only auto mamba_tiny_progressive")
        print("\n  2. Or start LiDAR training:")
        print("     bash model_Mamba/scripts/train_multiscale_full.sh lidar auto mamba_tiny_progressive")
    else:
        print("❌ SOME TESTS FAILED - Please check the errors above")
        print("="*70)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
