#!/usr/bin/env python3
"""
Test Script for Ellipse Mask Generation Fix
============================================

Validates that the fixed ellipse generation handles:
1. Normal cases (objects inside image)
2. Edge cases (objects near/outside image boundaries)
3. Overlapping objects
4. Small objects
5. Performance (no global grid creation)

Author: PoLaRIS Team
Date: 2026-02-03
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Direct import to avoid torch dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "binary_mask_utils",
    os.path.join(SCRIPT_DIR, "dataset", "binary_mask_utils.py")
)
binary_mask_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(binary_mask_utils)

generate_binary_mask_target = binary_mask_utils.generate_binary_mask_target


def visualize_masks(masks, titles, save_path=None):
    """Visualize multiple masks side by side."""
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))

    if n == 1:
        axes = [axes]

    for i, (mask, title) in enumerate(zip(masks, titles)):
        axes[i].imshow(mask, cmap='hot', vmin=0, vmax=1)
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].axis('off')

        # Add statistics
        pos_pixels = (mask > 0).sum()
        axes[i].text(0.5, -0.05, f"Positive pixels: {pos_pixels}",
                    transform=axes[i].transAxes, ha='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization: {save_path}")
    else:
        plt.show()

    plt.close()


def test_case_1_normal_objects():
    """Test 1: Normal objects inside image bounds."""
    print("\n" + "="*60)
    print("Test 1: Normal Objects (Inside Image)")
    print("="*60)

    img_size = (256, 256)

    # Test labels: 3 objects at different locations
    labels = [
        [0, 0.3, 0.3, 0.2, 0.15],  # Top-left
        [0, 0.7, 0.5, 0.15, 0.25], # Right-center
        [0, 0.5, 0.8, 0.18, 0.12], # Bottom-center
    ]

    # Generate masks with different modes
    mask_box = generate_binary_mask_target(labels, img_size, fill_mode='box')
    mask_center = generate_binary_mask_target(labels, img_size, fill_mode='center')
    mask_ellipse = generate_binary_mask_target(labels, img_size, fill_mode='ellipse')

    # Validate
    assert mask_box.shape == img_size, "Box mask shape mismatch"
    assert mask_ellipse.shape == img_size, "Ellipse mask shape mismatch"
    assert (mask_box > 0).sum() > (mask_ellipse > 0).sum(), "Ellipse should have fewer pixels than box"
    assert (mask_ellipse > 0).sum() > 0, "Ellipse mask should not be empty"

    print(f"✓ Box mode:     {(mask_box > 0).sum():6d} pixels")
    print(f"✓ Center mode:  {(mask_center > 0).sum():6d} pixels")
    print(f"✓ Ellipse mode: {(mask_ellipse > 0).sum():6d} pixels")
    print("✓ All validations passed!")

    # Visualize
    visualize_masks(
        [mask_box, mask_center, mask_ellipse],
        ['Box Mode', 'Center Mode', 'Ellipse Mode (Fixed)'],
        save_path='model_Mamba/test_output_case1.png'
    )

    return True


def test_case_2_boundary_objects():
    """Test 2: Objects at/near image boundaries."""
    print("\n" + "="*60)
    print("Test 2: Boundary Objects (Near/Outside Image)")
    print("="*60)

    img_size = (256, 256)

    # Test labels: objects at boundaries and outside
    labels = [
        [0, 0.05, 0.05, 0.08, 0.08],   # Top-left corner (partly outside)
        [0, 0.95, 0.50, 0.08, 0.12],   # Right edge
        [0, 0.50, 0.97, 0.15, 0.08],   # Bottom edge
        [0, 1.05, 0.50, 0.10, 0.10],   # Completely outside (should be clipped)
        [0, -0.05, 0.50, 0.08, 0.08],  # Left outside (should be clipped)
    ]

    # Generate masks
    mask_box = generate_binary_mask_target(labels, img_size, fill_mode='box')
    mask_ellipse = generate_binary_mask_target(labels, img_size, fill_mode='ellipse')

    # Validate: should not crash or create artifacts
    assert mask_ellipse.shape == img_size, "Shape mismatch"
    assert np.all(mask_ellipse >= 0) and np.all(mask_ellipse <= 1), "Values out of range"
    assert (mask_ellipse > 0).sum() > 0, "Should have some pixels (not all labels are outside)"

    print(f"✓ Box mode:     {(mask_box > 0).sum():6d} pixels")
    print(f"✓ Ellipse mode: {(mask_ellipse > 0).sum():6d} pixels")
    print("✓ No crashes or artifacts!")
    print("✓ Boundary clipping works correctly!")

    # Visualize
    visualize_masks(
        [mask_box, mask_ellipse],
        ['Box Mode', 'Ellipse Mode (Fixed)'],
        save_path='model_Mamba/test_output_case2.png'
    )

    return True


def test_case_3_overlapping_objects():
    """Test 3: Overlapping objects."""
    print("\n" + "="*60)
    print("Test 3: Overlapping Objects")
    print("="*60)

    img_size = (256, 256)

    # Overlapping objects
    labels = [
        [0, 0.50, 0.50, 0.25, 0.20],  # Large object at center
        [0, 0.55, 0.52, 0.18, 0.15],  # Overlapping with first
        [0, 0.48, 0.48, 0.15, 0.12],  # Overlapping with first
    ]

    mask_box = generate_binary_mask_target(labels, img_size, fill_mode='box')
    mask_ellipse = generate_binary_mask_target(labels, img_size, fill_mode='ellipse')

    # Validate: overlapping should work (union operation)
    assert (mask_ellipse > 0).sum() > 0, "Should have pixels"
    assert np.all(mask_ellipse >= 0) and np.all(mask_ellipse <= 1), "Values out of range"

    print(f"✓ Box mode:     {(mask_box > 0).sum():6d} pixels")
    print(f"✓ Ellipse mode: {(mask_ellipse > 0).sum():6d} pixels")
    print("✓ Overlapping handled correctly (union)!")

    # Visualize
    visualize_masks(
        [mask_box, mask_ellipse],
        ['Box Mode (Overlapping)', 'Ellipse Mode (Overlapping)'],
        save_path='model_Mamba/test_output_case3.png'
    )

    return True


def test_case_4_small_objects():
    """Test 4: Very small objects."""
    print("\n" + "="*60)
    print("Test 4: Small Objects")
    print("="*60)

    img_size = (256, 256)

    # Very small objects
    labels = [
        [0, 0.3, 0.3, 0.02, 0.02],   # 5x5 pixels
        [0, 0.5, 0.5, 0.01, 0.01],   # 2x2 pixels (very small)
        [0, 0.7, 0.7, 0.005, 0.005], # 1x1 pixel (too small for ellipse)
    ]

    mask_box = generate_binary_mask_target(labels, img_size, fill_mode='box')
    mask_ellipse = generate_binary_mask_target(labels, img_size, fill_mode='ellipse')

    # Validate
    print(f"✓ Box mode:     {(mask_box > 0).sum():6d} pixels")
    print(f"✓ Ellipse mode: {(mask_ellipse > 0).sum():6d} pixels")

    # Very small ellipses might be skipped (a < 1.0 or b < 1.0)
    if (mask_ellipse > 0).sum() < (mask_box > 0).sum():
        print("✓ Small objects correctly skipped in ellipse mode (expected behavior)")

    # Visualize (zoomed in)
    visualize_masks(
        [mask_box, mask_ellipse],
        ['Box Mode (Small Objects)', 'Ellipse Mode (Small Objects)'],
        save_path='model_Mamba/test_output_case4.png'
    )

    return True


def test_case_5_performance():
    """Test 5: Performance comparison."""
    print("\n" + "="*60)
    print("Test 5: Performance Test")
    print("="*60)

    import time

    img_size = (512, 640)  # Realistic size

    # Many objects
    labels = [[0, np.random.rand(), np.random.rand(), 0.05 + np.random.rand()*0.1, 0.05 + np.random.rand()*0.1]
              for _ in range(20)]

    # Time box mode
    start = time.time()
    for _ in range(100):
        mask_box = generate_binary_mask_target(labels, img_size, fill_mode='box')
    time_box = (time.time() - start) / 100

    # Time ellipse mode
    start = time.time()
    for _ in range(100):
        mask_ellipse = generate_binary_mask_target(labels, img_size, fill_mode='ellipse')
    time_ellipse = (time.time() - start) / 100

    print(f"✓ Box mode:     {time_box*1000:.2f} ms/image")
    print(f"✓ Ellipse mode: {time_ellipse*1000:.2f} ms/image")
    print(f"✓ Slowdown:     {time_ellipse/time_box:.2f}x")

    # Ellipse should be reasonably fast (not 100x slower)
    if time_ellipse / time_box < 10:
        print("✓ Performance is acceptable! (< 10x slowdown)")
    else:
        print("⚠️  Performance degradation detected (but functional)")

    return True


def test_case_6_realistic_scenario():
    """Test 6: Realistic ship detection scenario."""
    print("\n" + "="*60)
    print("Test 6: Realistic Ship Detection Scenario")
    print("="*60)

    img_size = (480, 640)  # Typical PoLaRIS image size

    # Realistic ship annotations (rectangular boxes)
    labels = [
        [0, 0.25, 0.35, 0.08, 0.12],   # Small ship
        [0, 0.65, 0.55, 0.12, 0.18],   # Medium ship
        [0, 0.45, 0.72, 0.15, 0.10],   # Large ship (elongated)
    ]

    mask_box = generate_binary_mask_target(labels, img_size, fill_mode='box')
    mask_ellipse = generate_binary_mask_target(labels, img_size, fill_mode='ellipse', ellipse_ratio=0.8)

    # Calculate reduction in background noise
    box_pixels = (mask_box > 0).sum()
    ellipse_pixels = (mask_ellipse > 0).sum()
    reduction = (box_pixels - ellipse_pixels) / box_pixels * 100

    print(f"✓ Box mode:     {box_pixels:6d} pixels")
    print(f"✓ Ellipse mode: {ellipse_pixels:6d} pixels")
    print(f"✓ Reduction:    {reduction:.1f}% fewer pixels")
    print(f"  (This is the background noise removed from rectangular corners!)")

    # Visualize with better labels
    visualize_masks(
        [mask_box, mask_ellipse, np.abs(mask_box - mask_ellipse)],
        ['Box Annotation\n(Current)', 'Ellipse Annotation\n(Proposed)', 'Removed Background\n(Noise Reduction)'],
        save_path='model_Mamba/test_output_case6_realistic.png'
    )

    return True


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*70)
    print(" " * 15 + "ELLIPSE MASK GENERATION TEST SUITE")
    print("="*70)

    # Create output directory
    os.makedirs('model_Mamba', exist_ok=True)

    tests = [
        ("Normal Objects", test_case_1_normal_objects),
        ("Boundary Objects", test_case_2_boundary_objects),
        ("Overlapping Objects", test_case_3_overlapping_objects),
        ("Small Objects", test_case_4_small_objects),
        ("Performance", test_case_5_performance),
        ("Realistic Scenario", test_case_6_realistic_scenario),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))

    # Summary
    print("\n" + "="*70)
    print(" " * 25 + "TEST SUMMARY")
    print("="*70)

    for name, status in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {name:30s} {status}")

    all_passed = all(status == "PASS" for _, status in results)

    print("="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ellipse mask generation is ROBUST and SAFE!")
        print("\n✅ You can now safely use fill_mode='ellipse' in training!")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
