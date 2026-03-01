"""
Visualization Script for PoLaRIS-Gaussian-Mamba
===============================================

This script visualizes the predictions of the PoLaRIS_Mamba model,
including:
1. Input infrared images
2. LiDAR depth maps
3. Predicted Gaussian heatmaps
4. Ground truth heatmaps
5. Detected peaks (local maxima)

Usage:
    # Basic model
    python visualize.py --checkpoint result/Pohang-Canal-3k/mamba_tiny/best_model.pth \
                        --model mamba_tiny \
                        --dataset Pohang-Canal-3k \
                        --num_samples 10 \
                        --output_dir visualizations/

    # Progressive model with CBAM (recommended)
    python visualize.py --checkpoint result/tiny_LiDAR_16bit_PoLaRIS_d1.5p2.5_CBAM/best_model.pth \
                        --model mamba_tiny_progressive \
                        --use_cbam full \
                        --use_deep_supervision True \
                        --dataset Pohang-Canal-3k \
                        --split_method 50_50_2k_new \
                        --num_samples 10 \
                        --output_dir visualizations/

Author: PoLaRIS Team
Date: 2026-01-30
Updated: 2026-02-28 (Added Progressive/Multiscale/CBAM support)
"""

import torch
import numpy as np
import cv2
import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# PoLaRIS utilities (from project root)
from model.utils_lidar import PoLaRISTestLoader
from model.load_param_data import load_dataset
from model_Mamba.dataset.gaussian_utils import (
    generate_gaussian_target,
    load_yolo_labels,
    visualize_heatmap,
    overlay_heatmap,
)

# Mamba models
from model_Mamba.core.polaris_mamba import polaris_mamba_tiny, polaris_mamba_small, polaris_mamba_base
from model_Mamba.core.polaris_mamba_multiscale import polaris_mamba_tiny_multiscale, polaris_mamba_small_multiscale
from model_Mamba.core.polaris_mamba_progressive import polaris_mamba_tiny_progressive, polaris_mamba_small_progressive


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='PoLaRIS-Gaussian-Mamba Visualization')

    # Model and checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--model', type=str, default='mamba_tiny_progressive',
                        choices=['mamba_tiny', 'mamba_small', 'mamba_base',
                                'mamba_tiny_multiscale', 'mamba_small_multiscale',
                                'mamba_tiny_progressive', 'mamba_small_progressive'],
                        help='Model variant')
    parser.add_argument('--use_lidar', type=str, default='True',
                        help='Whether model uses LiDAR')
    parser.add_argument('--use_cbam', type=str, default='none',
                        choices=['none', 'spatial', 'full'],
                        help='CBAM attention type (for progressive models)')
    parser.add_argument('--use_deep_supervision', type=str, default='True',
                        help='Whether model uses deep supervision (for multiscale/progressive)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k',
                        help='Dataset name')
    parser.add_argument('--root', type=str, default='dataset/',
                        help='Dataset root directory')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='Train/test split method')
    parser.add_argument('--image_folder', type=str, default='images',
                        help='Image folder name')
    parser.add_argument('--suffix', type=str, default='.png',
                        help='Image file suffix')

    # Visualization settings
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations/mamba',
                        help='Output directory for visualizations')
    parser.add_argument('--peak_threshold', type=float, default=0.5,
                        help='Threshold for peak detection')
    parser.add_argument('--nms_radius', type=int, default=10,
                        help='NMS radius for peak detection')

    # Hardware
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU ID')

    args = parser.parse_args()
    return args


def detect_peaks(heatmap, threshold=0.5, nms_radius=10):
    """
    Detect peaks (local maxima) in the heatmap.

    Args:
        heatmap: (H, W) numpy array
        threshold: Minimum value for peak detection
        nms_radius: Non-maximum suppression radius

    Returns:
        peaks: List of (x, y, confidence) tuples
    """
    # Apply threshold
    heatmap_thresh = (heatmap > threshold).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(heatmap_thresh, connectivity=8)

    peaks = []
    for i in range(1, num_labels):  # Skip background (label 0)
        # Get region mask
        mask = (labels == i).astype(np.uint8)

        # Find local maximum within this region
        masked_heatmap = heatmap * mask
        max_val = masked_heatmap.max()
        max_loc = np.unravel_index(masked_heatmap.argmax(), masked_heatmap.shape)

        # Store peak (x, y, confidence)
        y, x = max_loc
        peaks.append((x, y, max_val))

    # Apply NMS (remove peaks too close to each other)
    peaks_nms = []
    peaks = sorted(peaks, key=lambda p: p[2], reverse=True)  # Sort by confidence

    for peak in peaks:
        x, y, conf = peak
        # Check distance to existing peaks
        too_close = False
        for (x_nms, y_nms, _) in peaks_nms:
            dist = np.sqrt((x - x_nms) ** 2 + (y - y_nms) ** 2)
            if dist < nms_radius:
                too_close = True
                break
        if not too_close:
            peaks_nms.append(peak)

    return peaks_nms


def visualize_sample(
    ir_img,
    lidar_img,
    heatmap_pred,
    heatmap_gt,
    peaks_pred,
    img_id,
    output_dir,
):
    """
    Visualize a single sample.

    Args:
        ir_img: (H, W) infrared image
        lidar_img: (H, W) LiDAR depth map
        heatmap_pred: (H, W) predicted heatmap
        heatmap_gt: (H, W) ground truth heatmap
        peaks_pred: List of (x, y, conf) detected peaks
        img_id: Image ID (for saving)
        output_dir: Output directory
    """
    H, W = ir_img.shape

    # Create figure
    fig_height = H
    fig_width = W * 4  # 4 columns
    canvas = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)

    # Column 1: IR image
    ir_uint8 = (ir_img * 255).astype(np.uint8)
    ir_bgr = cv2.cvtColor(ir_uint8, cv2.COLOR_GRAY2BGR)
    canvas[:, 0:W, :] = ir_bgr

    # Column 2: LiDAR depth map (colorized)
    lidar_uint8 = (lidar_img / (lidar_img.max() + 1e-7) * 255).astype(np.uint8)
    lidar_color = cv2.applyColorMap(lidar_uint8, cv2.COLORMAP_JET)
    canvas[:, W:2*W, :] = lidar_color

    # Column 3: Predicted heatmap
    heatmap_pred_vis = visualize_heatmap(heatmap_pred)
    canvas[:, 2*W:3*W, :] = heatmap_pred_vis

    # Column 4: Ground truth heatmap
    heatmap_gt_vis = visualize_heatmap(heatmap_gt)
    canvas[:, 3*W:4*W, :] = heatmap_gt_vis

    # Overlay predicted peaks on Column 3
    for (x, y, conf) in peaks_pred:
        cv2.circle(canvas, (2*W + x, y), radius=5, color=(0, 255, 0), thickness=2)
        cv2.putText(
            canvas,
            f'{conf:.2f}',
            (2*W + x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

    # Add column labels
    font_scale = 0.6
    font_thickness = 2
    cv2.putText(canvas, 'IR Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(canvas, 'LiDAR Depth', (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(canvas, 'Predicted Heatmap', (2*W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(canvas, 'Ground Truth', (3*W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # Save
    output_path = os.path.join(output_dir, f'{img_id}.png')
    cv2.imwrite(output_path, canvas)

    return output_path


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    use_lidar = (args.use_lidar == 'True')
    use_deep_supervision = (args.use_deep_supervision == 'True')

    if args.model == 'mamba_tiny':
        model = polaris_mamba_tiny(use_lidar=use_lidar)
    elif args.model == 'mamba_small':
        model = polaris_mamba_small(use_lidar=use_lidar)
    elif args.model == 'mamba_base':
        model = polaris_mamba_base(use_lidar=use_lidar)
    elif args.model == 'mamba_tiny_multiscale':
        model = polaris_mamba_tiny_multiscale(use_lidar=use_lidar, use_deep_supervision=use_deep_supervision)
    elif args.model == 'mamba_small_multiscale':
        model = polaris_mamba_small_multiscale(use_lidar=use_lidar, use_deep_supervision=use_deep_supervision)
    elif args.model == 'mamba_tiny_progressive':
        model = polaris_mamba_tiny_progressive(use_lidar=use_lidar, use_deep_supervision=use_deep_supervision, use_cbam=args.use_cbam)
    elif args.model == 'mamba_small_progressive':
        model = polaris_mamba_small_progressive(use_lidar=use_lidar, use_deep_supervision=use_deep_supervision, use_cbam=args.use_cbam)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✅ Loaded checkpoint from: {args.checkpoint}")
    if 'best_iou' in checkpoint:
        print(f"   Best IoU: {checkpoint['best_iou']:.4f}")

    # Load dataset
    dataset_dir = os.path.join(args.root, args.dataset)
    _, test_img_ids, _ = load_dataset(args.root, args.dataset, args.split_method)

    # Limit to num_samples
    test_img_ids = test_img_ids[:args.num_samples]

    base_test_loader = PoLaRISTestLoader(
        dataset_dir=dataset_dir,
        img_id=test_img_ids,
        base_size=512,
        crop_size=512,
        transform=None,
        suffix=args.suffix,
        normalize_16bit=True,
        in_channels=1,
        image_folder=args.image_folder,
    )

    print(f"✅ Loaded {len(base_test_loader)} test samples")

    # Visualize
    print(f"\nGenerating visualizations...")
    for i in tqdm(range(len(base_test_loader))):
        sample = base_test_loader[i]

        ir_img = sample['image'][0:1, :, :].unsqueeze(0).to(device)  # (1, 1, H, W)
        lidar_depth = sample.get('lidar_depth', None)
        img_id = sample['img_id']

        # Prepare LiDAR input
        if lidar_depth is not None:
            lidar_img = lidar_depth.unsqueeze(0).to(device)  # (1, 1, H, W)
        else:
            lidar_img = torch.zeros_like(ir_img).to(device)

        # Inference
        with torch.no_grad():
            heatmap_pred = model(ir_img, lidar_img)  # (1, 1, H, W)

        # Convert to numpy
        ir_np = ir_img[0, 0].cpu().numpy()
        lidar_np = lidar_img[0, 0].cpu().numpy()
        heatmap_pred_np = heatmap_pred[0, 0].cpu().numpy()

        # Generate ground truth heatmap
        label_path = os.path.join(dataset_dir, 'labels', f'{img_id}.txt')
        labels = load_yolo_labels(label_path)
        H, W = ir_np.shape
        heatmap_gt_np = generate_gaussian_target(labels, (H, W), downscale=1, min_overlap=0.7)

        # Detect peaks
        peaks_pred = detect_peaks(
            heatmap_pred_np,
            threshold=args.peak_threshold,
            nms_radius=args.nms_radius,
        )

        # Visualize
        output_path = visualize_sample(
            ir_np,
            lidar_np,
            heatmap_pred_np,
            heatmap_gt_np,
            peaks_pred,
            img_id,
            args.output_dir,
        )

    print(f"\n✅ Saved {len(base_test_loader)} visualizations to: {args.output_dir}")


if __name__ == '__main__':
    main()
