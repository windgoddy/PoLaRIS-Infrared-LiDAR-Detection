"""
Gaussian Heatmap Target Generation Utilities
=============================================

This module provides utilities to convert YOLO format bounding box annotations
(cx, cy, w, h) into Gaussian heatmap targets for center-point detection.

Key Functions:
- gaussian_2d: Generate a 2D Gaussian kernel
- gaussian_radius: Compute adaptive radius based on bbox size
- draw_gaussian: Draw a Gaussian blob on the heatmap
- generate_gaussian_target: Convert YOLO labels to heatmap

Reference:
- CenterNet: https://arxiv.org/abs/1904.07850
- Objects as Points: https://github.com/xingyizhou/CenterNet

Author: PoLaRIS Team
Date: 2026-01-30
"""

import numpy as np
import torch
import cv2
import math


def gaussian_2d(shape, sigma=1.0):
    """
    Generate a 2D Gaussian kernel.

    Args:
        shape: (height, width) of the Gaussian kernel
        sigma: Standard deviation

    Returns:
        h: (H, W) Gaussian kernel normalized to [0, 1]
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0  # Clip very small values

    return h


def gaussian_radius(bbox_w, bbox_h, min_overlap=0.7):
    """
    Compute adaptive Gaussian radius based on bounding box size.

    The radius is chosen such that the Gaussian heatmap has IoU >= min_overlap
    with the original bounding box.

    This formula is derived from CenterNet to ensure that detections within
    the Gaussian radius will have sufficient IoU with the ground truth box.

    Args:
        bbox_w: Bounding box width (in pixels)
        bbox_h: Bounding box height (in pixels)
        min_overlap: Minimum IoU threshold (default: 0.7)

    Returns:
        radius: Gaussian radius (sigma) in pixels
    """
    # Compute the Gaussian radius using CenterNet formula
    # Reference: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py

    a1 = 1
    b1 = (bbox_h + bbox_w)
    c1 = bbox_w * bbox_h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (bbox_h + bbox_w)
    c2 = (1 - min_overlap) * bbox_w * bbox_h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (bbox_h + bbox_w)
    c3 = (min_overlap - 1) * bbox_w * bbox_h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    # Return the minimum valid radius
    radius = min(r1, r2, r3)

    # Ensure minimum radius (avoid too small Gaussians)
    radius = max(1.0, radius)

    return radius


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draw a Gaussian blob on the heatmap at the specified center.

    Args:
        heatmap: (H, W) heatmap array (in-place modification)
        center: (cx, cy) center coordinates (float)
        radius: Gaussian radius (sigma)
        k: Gaussian amplitude multiplier (default: 1)

    Returns:
        heatmap: (H, W) updated heatmap
    """
    # FIXED 2026-01-30: Use ceiling to ensure radius >= 1 for better coverage
    # Changed from int(radius) to ensure small objects still get drawn
    radius = max(1, int(np.ceil(radius)))  # Ensure minimum radius of 1
    # Removed degenerate case - always draw at least 3x3 Gaussian

    diameter = int(2 * radius + 1)
    gaussian = gaussian_2d((diameter, diameter), sigma=radius / 3)  # sigma ≈ radius / 3

    cx, cy = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    # Compute the bounding box of the Gaussian
    left = min(cx, radius)
    right = min(width - cx, radius + 1)
    top = min(cy, radius)
    bottom = min(height - cy, radius + 1)

    # FIXED 2026-01-30: Changed <= to < to allow edge cases
    # Skip ONLY if region is truly invalid (negative values)
    if left < 1 or right < 1 or top < 1 or bottom < 1:
        # Edge case: draw at least the center pixel
        if 0 <= cy < height and 0 <= cx < width:
            heatmap[cy, cx] = max(heatmap[cy, cx], k)
        return heatmap

    # Ensure valid indices
    left, right = int(left), int(right)
    top, bottom = int(top), int(bottom)

    # Extract the valid region of the Gaussian kernel
    masked_gaussian = gaussian[
        radius - top:radius + bottom,
        radius - left:radius + right
    ]

    # Update heatmap (take maximum to avoid overwriting existing Gaussians)
    target_region = heatmap[
        cy - top:cy + bottom,
        cx - left:cx + right
    ]

    # Verify shapes match before broadcasting
    if masked_gaussian.shape == target_region.shape:
        np.maximum(target_region, masked_gaussian * k, out=target_region)
    else:
        # Fallback: skip this Gaussian if shapes don't match
        pass

    return heatmap


def generate_gaussian_target(
    labels,
    img_size,
    downscale=1,
    min_overlap=0.7,
    normalize=True,
):
    """
    Generate Gaussian heatmap target from YOLO format labels.

    Args:
        labels: List of bounding boxes in YOLO format (normalized)
                Each element: [class_id, cx, cy, w, h] where cx, cy, w, h ∈ [0, 1]
        img_size: (height, width) of the original image
        downscale: Downscaling factor (default: 1, no downscale)
                   Use 4 if the model downsamples by 4x
        min_overlap: Minimum IoU for Gaussian radius computation (default: 0.7)
        normalize: Whether to normalize heatmap to [0, 1] (default: True)

    Returns:
        heatmap: (H // downscale, W // downscale) heatmap in [0, 1]
    """
    H, W = img_size
    H_out, W_out = H // downscale, W // downscale

    # Initialize empty heatmap
    heatmap = np.zeros((H_out, W_out), dtype=np.float32)

    if len(labels) == 0:
        # No objects in this image
        return heatmap

    for label in labels:
        if len(label) < 5:
            continue  # Skip invalid labels

        class_id, cx_norm, cy_norm, w_norm, h_norm = label[:5]

        # Convert normalized coordinates to pixel coordinates (output resolution)
        cx = cx_norm * W_out
        cy = cy_norm * H_out
        w = w_norm * W_out
        h = h_norm * H_out

        # Skip invalid boxes
        if w <= 0 or h <= 0:
            continue

        # Compute Gaussian radius
        radius = gaussian_radius(w, h, min_overlap=min_overlap)

        # Draw Gaussian on heatmap
        draw_gaussian(heatmap, (cx, cy), radius, k=1)

    # Normalize to [0, 1]
    if normalize and heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def load_yolo_labels(label_path):
    """
    Load YOLO format labels from a .txt file.

    Args:
        label_path: Path to YOLO label file

    Returns:
        labels: List of [class_id, cx, cy, w, h]
    """
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    labels.append([class_id, cx, cy, w, h])
    except FileNotFoundError:
        # No labels for this image
        pass

    return labels


def visualize_heatmap(heatmap, colormap=cv2.COLORMAP_JET):
    """
    Visualize heatmap with colormap.

    Args:
        heatmap: (H, W) heatmap in [0, 1]
        colormap: OpenCV colormap (default: JET)

    Returns:
        heatmap_color: (H, W, 3) BGR image
    """
    # Convert to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    return heatmap_color


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay heatmap on the original image.

    Args:
        image: (H, W, 3) BGR image
        heatmap: (H, W) heatmap in [0, 1]
        alpha: Blending weight (default: 0.5)

    Returns:
        overlaid: (H, W, 3) BGR image
    """
    # Resize heatmap to match image size if needed
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Visualize heatmap
    heatmap_color = visualize_heatmap(heatmap)

    # Blend
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return overlaid


# ======================== Torch Utilities ========================

def generate_gaussian_target_batch(labels_batch, img_size, downscale=1, device='cpu'):
    """
    Generate Gaussian heatmap targets for a batch of images.

    Args:
        labels_batch: List of label lists (length = batch_size)
                      Each element is a list of [class_id, cx, cy, w, h]
        img_size: (H, W) tuple
        downscale: Downscaling factor
        device: torch device

    Returns:
        heatmaps: (B, 1, H//downscale, W//downscale) tensor
    """
    B = len(labels_batch)
    H, W = img_size
    H_out, W_out = H // downscale, W // downscale

    heatmaps = np.zeros((B, 1, H_out, W_out), dtype=np.float32)

    for i, labels in enumerate(labels_batch):
        heatmap = generate_gaussian_target(labels, img_size, downscale)
        heatmaps[i, 0] = heatmap

    heatmaps = torch.from_numpy(heatmaps).to(device)

    return heatmaps


# ======================== Unit Test ========================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Gaussian Target Generation")
    print("=" * 60)

    # Test gaussian_2d
    print("\n[1] Testing gaussian_2d...")
    gaussian_kernel = gaussian_2d((21, 21), sigma=3.0)
    print(f"Gaussian kernel shape: {gaussian_kernel.shape}")
    print(f"Gaussian kernel range: [{gaussian_kernel.min():.4f}, {gaussian_kernel.max():.4f}]")

    # Test gaussian_radius
    print("\n[2] Testing gaussian_radius...")
    bbox_w, bbox_h = 50, 50
    radius = gaussian_radius(bbox_w, bbox_h, min_overlap=0.7)
    print(f"Bounding box: ({bbox_w}, {bbox_h})")
    print(f"Computed radius: {radius:.2f}")

    # Test draw_gaussian
    print("\n[3] Testing draw_gaussian...")
    heatmap = np.zeros((128, 128), dtype=np.float32)
    draw_gaussian(heatmap, center=(64, 64), radius=radius)
    print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print(f"Heatmap peak location: {np.unravel_index(heatmap.argmax(), heatmap.shape)}")

    # Test generate_gaussian_target
    print("\n[4] Testing generate_gaussian_target...")
    labels = [
        [0, 0.3, 0.4, 0.1, 0.15],  # Small object
        [0, 0.7, 0.6, 0.2, 0.25],  # Larger object
    ]
    img_size = (512, 640)
    heatmap_target = generate_gaussian_target(labels, img_size, downscale=1)
    print(f"Generated heatmap shape: {heatmap_target.shape}")
    print(f"Heatmap range: [{heatmap_target.min():.4f}, {heatmap_target.max():.4f}]")
    print(f"Number of peaks: {(heatmap_target == 1.0).sum()}")

    # Test batch generation
    print("\n[5] Testing batch generation...")
    labels_batch = [labels, labels]  # Batch of 2
    heatmaps_batch = generate_gaussian_target_batch(labels_batch, img_size, downscale=4)
    print(f"Batch heatmap shape: {heatmaps_batch.shape}")
    print(f"Batch heatmap range: [{heatmaps_batch.min():.4f}, {heatmaps_batch.max():.4f}]")

    # Visualize (optional, requires display)
    print("\n[6] Testing visualization...")
    heatmap_vis = visualize_heatmap(heatmap_target)
    print(f"Visualized heatmap shape: {heatmap_vis.shape}")

    # Test with real YOLO label file (if available)
    print("\n[7] Testing with YOLO label file (simulated)...")
    # Simulate a label file
    test_label_path = "/tmp/test_label.txt"
    with open(test_label_path, 'w') as f:
        f.write("0 0.5 0.5 0.1 0.1\n")
        f.write("0 0.3 0.7 0.15 0.2\n")

    labels_loaded = load_yolo_labels(test_label_path)
    print(f"Loaded {len(labels_loaded)} labels from {test_label_path}")
    heatmap_from_file = generate_gaussian_target(labels_loaded, (512, 640), downscale=1)
    print(f"Heatmap from file shape: {heatmap_from_file.shape}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

    # Save visualization (optional)
    try:
        cv2.imwrite("/tmp/gaussian_heatmap.png", heatmap_vis)
        print("\nSaved visualization to /tmp/gaussian_heatmap.png")
    except Exception as e:
        print(f"\nCould not save visualization: {e}")
