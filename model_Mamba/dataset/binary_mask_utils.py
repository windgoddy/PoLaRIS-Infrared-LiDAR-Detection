"""
Binary Mask Target Generation from YOLO Labels
==============================================
Convert YOLO bbox annotations to binary segmentation masks.

This module replaces Gaussian heatmap generation for Scheme A (Binary Segmentation).
Aligns with LIDAR-Mamba's approach for crack detection.

Key Functions:
- generate_binary_mask_target: Convert YOLO labels to binary mask
- load_yolo_labels: Load YOLO format annotations

Author: PoLaRIS Team
Date: 2026-01-30
"""

import numpy as np
import torch
import os


def generate_binary_mask_target(labels, img_size, downscale=1, fill_mode='box', ellipse_ratio=0.8):
    """
    Generate binary mask from YOLO labels.

    Args:
        labels: List of [class_id, cx, cy, w, h] (normalized 0-1)
        img_size: (H, W) tuple - original image size
        downscale: Output downscaling factor (default: 1)
        fill_mode: Mask filling strategy
            'box' - Fill entire bbox (default, most conservative)
            'center' - Fill only center 50% region
            'ellipse' - Fill inscribed ellipse (better for irregular shapes)
        ellipse_ratio: Ratio of ellipse to bbox (default: 0.8)

    Returns:
        mask: (H//downscale, W//downscale) binary mask in {0, 1}

    Example:
        >>> labels = [[0, 0.5, 0.5, 0.1, 0.1]]  # One object at center
        >>> mask = generate_binary_mask_target(labels, (512, 640))
        >>> print(mask.shape, mask.max())  # (512, 640) 1.0
    """
    H, W = img_size
    H_out, W_out = H // downscale, W // downscale

    # Initialize empty mask
    mask = np.zeros((H_out, W_out), dtype=np.float32)

    for label in labels:
        if len(label) < 5:
            continue  # Skip invalid labels

        # YOLO format: class_id, center_x, center_y, width, height (all normalized 0-1)
        class_id, cx_norm, cy_norm, w_norm, h_norm = label[:5]

        # Convert to pixel coordinates in the output resolution
        cx = cx_norm * W_out
        cy = cy_norm * H_out
        w = w_norm * W_out
        h = h_norm * H_out

        # Skip invalid boxes
        if w <= 0 or h <= 0:
            continue

        # Compute box coordinates (top-left, bottom-right)
        x1 = int(max(0, cx - w/2))
        y1 = int(max(0, cy - h/2))
        x2 = int(min(W_out, cx + w/2))
        y2 = int(min(H_out, cy + h/2))

        # Ensure valid box
        if x2 <= x1 or y2 <= y1:
            continue

        # Fill the mask
        if fill_mode == 'box':
            # Fill the entire bounding box (standard approach)
            mask[y1:y2, x1:x2] = 1.0

        elif fill_mode == 'center':
            # Only fill the center 50% region (conservative strategy)
            # Useful if boxes are loosely annotated
            w_center = max(1, int(w * 0.5))
            h_center = max(1, int(h * 0.5))
            x1_c = int(max(0, cx - w_center/2))
            y1_c = int(max(0, cy - h_center/2))
            x2_c = int(min(W_out, cx + w_center/2))
            y2_c = int(min(H_out, cy + h_center/2))

            if x2_c > x1_c and y2_c > y1_c:
                mask[y1_c:y2_c, x1_c:x2_c] = 1.0

        elif fill_mode == 'ellipse':
            # Fill inscribed ellipse (better for irregular shapes like L-shaped ships)
            # Reduces background noise from box corners
            a = w * ellipse_ratio / 2  # Semi-major axis
            b = h * ellipse_ratio / 2  # Semi-minor axis

            # Create coordinate grid
            y_coords, x_coords = np.ogrid[0:H_out, 0:W_out]

            # Ellipse equation: ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1
            ellipse_mask = ((x_coords - cx) / (a + 1e-7))**2 + \
                          ((y_coords - cy) / (b + 1e-7))**2 <= 1.0

            mask[ellipse_mask] = 1.0

    return mask


def load_yolo_labels(label_path):
    """
    Load YOLO format labels from txt file.

    Args:
        label_path: Path to YOLO label file (*.txt)

    Returns:
        labels: List of [class_id, cx, cy, w, h]
                Empty list if file not found or invalid

    Example:
        >>> labels = load_yolo_labels('dataset/labels/00_0001.txt')
        >>> print(len(labels), labels[0])  # 3 [0, 0.5, 0.4, 0.1, 0.15]
    """
    labels = []

    if not os.path.exists(label_path):
        return labels  # Return empty list for images without labels

    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Parse as floats
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    # Validate ranges (YOLO format should be in [0, 1])
                    # Clamp out-of-bounds values instead of warning (common for edge objects)
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))
                    
                    labels.append([class_id, cx, cy, w, h])

    except Exception as e:
        print(f"Error reading labels {label_path}: {e}")

    return labels


# ======================== Batch Utilities ========================

def generate_binary_mask_batch(labels_batch, img_size, downscale=1, fill_mode='box', device='cpu'):
    """
    Generate binary mask targets for a batch of images.

    Args:
        labels_batch: List of label lists (length = batch_size)
                      Each element is a list of [class_id, cx, cy, w, h]
        img_size: (H, W) tuple
        downscale: Downscaling factor
        fill_mode: 'box' or 'center'
        device: torch device

    Returns:
        masks: (B, 1, H//downscale, W//downscale) tensor
    """
    B = len(labels_batch)
    H, W = img_size
    H_out, W_out = H // downscale, W // downscale

    masks = np.zeros((B, 1, H_out, W_out), dtype=np.float32)

    for i, labels in enumerate(labels_batch):
        mask = generate_binary_mask_target(labels, img_size, downscale, fill_mode)
        masks[i, 0] = mask

    masks = torch.from_numpy(masks).to(device)
    return masks


# ======================== Visualization ========================

def visualize_binary_mask(mask, colormap_mode='red'):
    """
    Visualize binary mask with color overlay.

    Args:
        mask: (H, W) binary mask in [0, 1]
        colormap_mode: 'red' (default), 'green', or 'blue'

    Returns:
        mask_color: (H, W, 3) BGR image
    """
    import cv2

    H, W = mask.shape
    mask_color = np.zeros((H, W, 3), dtype=np.uint8)

    # Create colored overlay
    if colormap_mode == 'red':
        mask_color[:, :, 2] = (mask * 255).astype(np.uint8)  # Red channel
    elif colormap_mode == 'green':
        mask_color[:, :, 1] = (mask * 255).astype(np.uint8)  # Green channel
    elif colormap_mode == 'blue':
        mask_color[:, :, 0] = (mask * 255).astype(np.uint8)  # Blue channel

    return mask_color


def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlay binary mask on image.

    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask in [0, 1]
        alpha: Blending weight (default: 0.5)

    Returns:
        overlaid: (H, W, 3) BGR image
    """
    import cv2

    # Resize mask to match image size if needed
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Visualize mask
    mask_color = visualize_binary_mask(mask, colormap_mode='red')

    # Blend
    overlaid = cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)

    return overlaid


# ======================== Unit Test ========================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Binary Mask Generation")
    print("=" * 60)

    # Test 1: Single object
    print("\n[Test 1] Single object at center")
    labels = [[0, 0.5, 0.5, 0.1, 0.15]]
    mask = generate_binary_mask_target(labels, img_size=(512, 640), downscale=1, fill_mode='box')
    pos_pixels = (mask > 0).sum()
    print(f"  Mask shape: {mask.shape}")
    print(f"  Positive pixels: {pos_pixels}")
    print(f"  Expected: ~51 × 96 = 4896 pixels")
    print(f"  ✓ PASS" if 4800 <= pos_pixels <= 5000 else f"  ✗ FAIL")

    # Test 2: Multiple objects
    print("\n[Test 2] Multiple objects")
    labels = [
        [0, 0.3, 0.4, 0.08, 0.12],
        [0, 0.7, 0.6, 0.06, 0.10],
    ]
    mask = generate_binary_mask_target(labels, img_size=(512, 640), downscale=1, fill_mode='box')
    pos_pixels = (mask > 0).sum()
    print(f"  Positive pixels: {pos_pixels}")
    print(f"  ✓ PASS" if pos_pixels > 5000 else f"  ✗ FAIL")

    # Test 3: Downscaling
    print("\n[Test 3] Downscaling by 4x")
    mask_4x = generate_binary_mask_target(labels, img_size=(512, 640), downscale=4, fill_mode='box')
    print(f"  Mask shape: {mask_4x.shape}")
    print(f"  Expected: (128, 160)")
    print(f"  ✓ PASS" if mask_4x.shape == (128, 160) else f"  ✗ FAIL")

    # Test 4: Edge case - empty labels
    print("\n[Test 4] Empty labels")
    mask_empty = generate_binary_mask_target([], img_size=(512, 640))
    print(f"  Positive pixels: {(mask_empty > 0).sum()}")
    print(f"  ✓ PASS" if (mask_empty > 0).sum() == 0 else f"  ✗ FAIL")

    # Test 5: Load YOLO labels (simulated)
    print("\n[Test 5] Load YOLO labels (simulated)")
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("0 0.5 0.5 0.1 0.1\n")
        f.write("0 0.3 0.7 0.15 0.2\n")
        temp_path = f.name

    loaded_labels = load_yolo_labels(temp_path)
    os.remove(temp_path)
    print(f"  Loaded {len(loaded_labels)} labels")
    print(f"  ✓ PASS" if len(loaded_labels) == 2 else f"  ✗ FAIL")

    # Test 6: Batch generation
    print("\n[Test 6] Batch generation")
    labels_batch = [labels, labels]
    masks_batch = generate_binary_mask_batch(labels_batch, img_size=(512, 640), downscale=1)
    print(f"  Batch shape: {masks_batch.shape}")
    print(f"  Expected: (2, 1, 512, 640)")
    print(f"  ✓ PASS" if masks_batch.shape == (2, 1, 512, 640) else f"  ✗ FAIL")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
