#!/usr/bin/env python3
"""
消融实验伪掩码离线批量生成器
==============================

生成两组伪掩码，用于对比：

  Mode A: bsnr_pstar   (HALO p* 锚点)
      高斯圆心 = argmax B-SNR = 目标内最亮像素（物理热点）
      与 bsnr_mask_utils.compute_bsnr_weight(spatial_gaussian=True) 等价

  Mode B: geocenter    (几何中心基线)
      高斯圆心 = YOLO box 几何中心 (cx, cy)
      σ_G 与 Mode A 相同（FWHM B-SNR 加权自适应）
      排除几何假设优势，纯粹对比锚点选择策略

两种方法均保持 B-SNR 权重乘积结构：
  P(i) = W_bsnr(i) × Gaussian(i; anchor, σ_G)

运行方式：
  # 生成 p* 锚点掩码（HALO 方法）
  python paper/gen_ablation_masks.py --dataset_dir dataset/NUDT-SIRST \
      --mode pstar --output_folder masks_bsnr_pstar

  # 生成几何中心掩码（基线方法）
  python paper/gen_ablation_masks.py --dataset_dir dataset/NUDT-SIRST \
      --mode geocenter --output_folder masks_geocenter
"""

import os
import sys
import argparse
import numpy as np
import cv2
from typing import List, Tuple, Optional

# 允许从项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# 核心计算函数
# ─────────────────────────────────────────────────────────────────────────────

def compute_weight(
    image_gray: np.ndarray,
    box: Tuple[int, int, int, int],
    mode: str = 'pstar',
    expand_ratio: float = 1.5,
    temperature: float = 3.0,
    epsilon: float = 1e-4,
    gauss_sigma_ratio: float = 1.5,
) -> np.ndarray:
    """
    计算单个 Box 内的 B-SNR 乘积权重图。

    Args:
        image_gray:       (H, W) float32 灰度图，值域 [0, 1]
        box:              (x1, y1, x2, y2) 像素坐标
        mode:             'pstar'    → 高斯圆心 = argmax(B-SNR)
                          'geocenter' → 高斯圆心 = box 几何中心
        expand_ratio:     Context Box 膨胀倍率（默认 1.5）
        temperature:      sigmoid 温度系数 τ
        epsilon:          防零小常数
        gauss_sigma_ratio: σ_G = ratio × FWHM_B-SNR_加权标准差

    Returns:
        weight: (box_h, box_w) float32，值域 [0, 1]
    """
    H, W = image_gray.shape
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return np.zeros((max(1, box_h), max(1, box_w)), dtype=np.float32)

    # ── Step 1: Context Box（Window 方法，与当前 bsnr_mask_utils 一致）──
    pad_x = max(int((expand_ratio - 1.0) / 2.0 * box_w), 3)
    pad_y = max(int((expand_ratio - 1.0) / 2.0 * box_h), 3)
    ctx_x1 = max(0, x1 - pad_x)
    ctx_y1 = max(0, y1 - pad_y)
    ctx_x2 = min(W, x2 + pad_x)
    ctx_y2 = min(H, y2 + pad_y)

    # ── Step 2: 背景统计量 μ, σ ──
    ctx_region = image_gray[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
    mu = ctx_region.mean()
    sigma = ctx_region.std()

    # ── Step 3: Box 内 B-SNR ──
    target_region = image_gray[y1:y2, x1:x2]  # (box_h, box_w)
    snr = (target_region - mu) / (sigma + epsilon)
    weight = 1.0 / (1.0 + np.exp(-temperature * snr))  # sigmoid

    # 归一化到 [0, 1]
    w_min, w_max = weight.min(), weight.max()
    if w_max - w_min > 1e-8:
        weight = (weight - w_min) / (w_max - w_min)
    else:
        weight = np.full_like(weight, 0.5)

    # ── Step 4: 物理锚定高斯乘积 ──
    # 先用 B-SNR argmax 求 peak（用于计算自适应 σ_G）
    peak_flat = snr.argmax()
    peak_y_snr = peak_flat // box_w
    peak_x_snr = peak_flat % box_w

    # 自适应 σ_G：FWHM 区域 B-SNR 加权方差（与 bsnr_mask_utils 一致）
    peak_snr_val = float(snr[peak_y_snr, peak_x_snr])
    half_max = peak_snr_val * 0.5
    snr_fwhm = np.where(snr >= half_max, np.maximum(snr, 0.0), 0.0).astype(np.float32)
    snr_sum = snr_fwhm.sum()

    yy, xx = np.mgrid[0:box_h, 0:box_w]

    if snr_sum > 1e-6:
        dist_sq_for_sigma = ((yy - peak_y_snr) ** 2 + (xx - peak_x_snr) ** 2).astype(np.float32)
        snr_weighted_var = (snr_fwhm * dist_sq_for_sigma).sum() / snr_sum
        sigma_px = max(gauss_sigma_ratio * np.sqrt(snr_weighted_var), 1.0)
    else:
        sigma_px = 1.0

    # 确定高斯圆心（两种模式的唯一差异）
    if mode == 'pstar':
        # HALO 方法：物理热点（argmax B-SNR）
        anchor_y = peak_y_snr
        anchor_x = peak_x_snr
    elif mode == 'geocenter':
        # 基线方法：YOLO box 几何中心（在 box 局部坐标系中）
        anchor_y = (box_h - 1) / 2.0
        anchor_x = (box_w - 1) / 2.0
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'pstar' or 'geocenter'.")

    dist_sq = ((yy - anchor_y) ** 2 + (xx - anchor_x) ** 2).astype(np.float32)
    gauss = np.exp(-dist_sq / (2 * sigma_px ** 2)).astype(np.float32)
    weight = weight * gauss

    return weight.astype(np.float32)


def generate_mask(
    labels: List[List[float]],
    img_size: Tuple[int, int],
    image: Optional[np.ndarray],
    mode: str = 'pstar',
    expand_ratio: float = 1.5,
    temperature: float = 3.0,
    fg_threshold: float = 0.5,
    soft_label: bool = True,
    max_value: float = 1.0,
    gauss_sigma_ratio: float = 1.5,
) -> np.ndarray:
    """从 YOLO 标签生成单张伪掩码。"""
    H, W = img_size
    mask = np.zeros((H, W), dtype=np.float32)

    if image is None or len(labels) == 0:
        return mask

    # 灰度化
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype == np.uint16:
        gray = gray.astype(np.float32) / 65535.0
    elif gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255.0
    else:
        gray = gray.astype(np.float32)

    for label in labels:
        if len(label) < 5:
            continue
        class_id, cx_norm, cy_norm, w_norm, h_norm = label[:5]

        cx = cx_norm * W
        cy = cy_norm * H
        bw = w_norm * W
        bh = h_norm * H

        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(W, cx + bw / 2))
        y2 = int(min(H, cy + bh / 2))

        if x2 <= x1 or y2 <= y1:
            continue

        weight = compute_weight(
            gray, (x1, y1, x2, y2),
            mode=mode,
            expand_ratio=expand_ratio,
            temperature=temperature,
            gauss_sigma_ratio=gauss_sigma_ratio,
        )

        if fg_threshold > 0:
            weight[weight < fg_threshold] = 0.0

        refined = weight * max_value if soft_label else (weight >= fg_threshold).astype(np.float32) * max_value
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], refined)

    return mask


def _load_yolo_labels(label_path: str) -> List[List[float]]:
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([int(parts[0])] + list(map(float, parts[1:5])))
    except FileNotFoundError:
        pass
    return labels


def batch_generate(
    dataset_dir: str,
    img_ids: List[str],
    output_dir: str,
    mode: str,
    label_folder: str = 'labels_box',
    image_folder: str = 'images',
    suffix: str = '.png',
    expand_ratio: float = 1.5,
    temperature: float = 3.0,
    fg_threshold: float = 0.5,
    soft_label: bool = True,
    max_value: float = 1.0,
    gauss_sigma_ratio: float = 1.5,
):
    """离线批量生成并保存伪掩码 PNG。"""
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(dataset_dir, label_folder)
    images_dir = os.path.join(dataset_dir, image_folder)

    total = len(img_ids)
    success = 0
    empty = 0

    for i, img_id in enumerate(img_ids):
        img_path = os.path.join(images_dir, img_id + suffix)
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        H, W = image.shape[:2]
        labels = _load_yolo_labels(os.path.join(labels_dir, img_id + '.txt'))

        mask = generate_mask(
            labels, (H, W), image,
            mode=mode,
            expand_ratio=expand_ratio,
            temperature=temperature,
            fg_threshold=fg_threshold,
            soft_label=soft_label,
            max_value=max_value,
            gauss_sigma_ratio=gauss_sigma_ratio,
        )

        mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, img_id + '.png'), mask_uint8)

        if mask.max() > 0:
            success += 1
        else:
            empty += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{total}] success={success}, empty={empty}")

    print(f"Done ({mode}): {success} masks saved to {output_dir}, {empty} empty")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation Pseudo-Mask Generator')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset root dir (e.g. dataset/NUDT-SIRST)')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['pstar', 'geocenter'],
                        help='pstar: HALO p* anchor | geocenter: geometric center baseline')
    parser.add_argument('--split_file', type=str, default=None,
                        help='Optional split file (lines of img_id). If None, use all labels/.')
    parser.add_argument('--label_folder', type=str, default='labels_box',
                        help='YOLO label folder name within dataset_dir (default: labels_box)')
    parser.add_argument('--image_folder', type=str, default='images')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Output folder name. Default: masks_bsnr_pstar or masks_geocenter')
    parser.add_argument('--expand_ratio', type=float, default=1.5)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--fg_threshold', type=float, default=0.5)
    parser.add_argument('--gauss_sigma_ratio', type=float, default=1.5)
    parser.add_argument('--soft_label', action='store_true', default=True)
    parser.add_argument('--max_value', type=float, default=1.0)
    args = parser.parse_args()

    # 确定输出目录
    if args.output_folder is None:
        args.output_folder = 'masks_bsnr_pstar' if args.mode == 'pstar' else 'masks_geocenter'
    output_dir = os.path.join(args.dataset_dir, args.output_folder)

    # 获取图像 ID 列表
    if args.split_file:
        split_path = args.split_file if os.path.exists(args.split_file) else \
                     os.path.join(args.dataset_dir, args.split_file)
        with open(split_path, 'r') as f:
            img_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(img_ids)} IDs from {split_path}")
    else:
        labels_dir = os.path.join(args.dataset_dir, args.label_folder)
        img_ids = [f[:-4] for f in sorted(os.listdir(labels_dir)) if f.endswith('.txt')]
        print(f"Found {len(img_ids)} label files in {labels_dir}")

    print(f"\nAblation Mask Generation")
    print(f"  Dataset:     {args.dataset_dir}")
    print(f"  Mode:        {args.mode} ({'HALO p* anchor' if args.mode == 'pstar' else 'Geometric center baseline'})")
    print(f"  Output:      {output_dir}")
    print(f"  expand_ratio:{args.expand_ratio}  temperature:{args.temperature}")
    print(f"  sigma_ratio: {args.gauss_sigma_ratio}  fg_thresh:{args.fg_threshold}")
    print()

    batch_generate(
        dataset_dir=args.dataset_dir,
        img_ids=img_ids,
        output_dir=output_dir,
        mode=args.mode,
        label_folder=args.label_folder,
        image_folder=args.image_folder,
        suffix=args.suffix,
        expand_ratio=args.expand_ratio,
        temperature=args.temperature,
        fg_threshold=args.fg_threshold,
        soft_label=args.soft_label,
        max_value=args.max_value,
        gauss_sigma_ratio=args.gauss_sigma_ratio,
    )
