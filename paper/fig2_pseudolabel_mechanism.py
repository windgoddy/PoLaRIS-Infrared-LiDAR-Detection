#!/usr/bin/env python3
"""
Figure 2: Pseudo-Label Generation Mechanism (Micro Close-up)
=============================================================
1行 × 3列 极近距离特写：

  Col 1: 原始 IR 局部 + YOLO 红框
  Col 2: B-SNR 热力图（inferno）+ argmax 十字准星 (p*)
  Col 3: PAG 高斯软标签（inferno）+ 同一 p* 准星

这张图可以 100% 由 Python 一键生成 PDF。

用法:
    python paper/fig2_pseudolabel_mechanism.py
    python paper/fig2_pseudolabel_mechanism.py --root /path/to/project --sample Misc_200 --dataset NUAA-SIRST
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ────────────────────────────────────────────────────────────────
# ★ 配置
# ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_CONFIGS = {
    'NUAA-SIRST': {
        'img_dir':   'dataset/NUAA-SIRST/images',
        'mask_dir':  'dataset/NUAA-SIRST/masks',
        'label_dir': 'dataset/NUAA-SIRST/labels_box',
        'split':     'dataset/NUAA-SIRST/50_50/test.txt',
    },
    'NUDT-SIRST': {
        'img_dir':   'dataset/NUDT-SIRST/images',
        'mask_dir':  'dataset/NUDT-SIRST/masks',
        'label_dir': 'dataset/NUDT-SIRST/labels_box',
        'split':     'dataset/NUDT-SIRST/50_50/test.txt',
    },
    'IRSTD-1k': {
        'img_dir':   'dataset/IRSTD-1k/IRSTD1k_Img',
        'mask_dir':  'dataset/IRSTD-1k/masks',
        'label_dir': 'dataset/IRSTD-1k/labels_box',
        'split':     'dataset/IRSTD-1k/50_50/test.txt',
    },
}

# ★ 裁剪尺寸：对于极小目标，较小的 patch 更有视觉冲击力
CROP_SIZE = 48  # px，若目标较大可改为 64 或 96
DPI = 300

# ────────────────────────────────────────────────────────────────


def load_bsnr_utils(root):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'bsnr_mask_utils',
        os.path.join(root, 'model_Mamba', 'dataset', 'bsnr_mask_utils.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_bsnr_weight


def load_yolo_boxes(label_path, img_h, img_w):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = (int(parts[0]), float(parts[1]),
                               float(parts[2]), float(parts[3]), float(parts[4]))
            x1 = max(0, int((cx - w / 2) * img_w))
            y1 = max(0, int((cy - h / 2) * img_h))
            x2 = min(img_w, int((cx + w / 2) * img_w))
            y2 = min(img_h, int((cy + h / 2) * img_h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def auto_select_sample(cfg, root):
    split_file = os.path.join(root, cfg['split'])
    with open(split_file, 'r') as f:
        names = [l.strip() for l in f if l.strip()]
    for name in names:
        label_path = os.path.join(root, cfg['label_dir'], name + '.txt')
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            return name
    raise RuntimeError("No valid sample found.")


def main(args):
    compute_bsnr_weight = load_bsnr_utils(args.root)
    cfg = DATASET_CONFIGS[args.dataset]

    sample = args.sample or auto_select_sample(cfg, args.root)
    print(f"Using sample: {sample} from {args.dataset}")

    img_path = os.path.join(args.root, cfg['img_dir'], sample + '.png')
    label_path = os.path.join(args.root, cfg['label_dir'], sample + '.txt')

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape
    img_float = img_gray.astype(np.float32) / 255.0

    boxes = load_yolo_boxes(label_path, H, W)
    if not boxes:
        raise RuntimeError(f"No boxes found in {label_path}")

    # 选择第一个（最大）目标
    box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    x1, y1, x2, y2 = box
    bx_center = (x1 + x2) // 2
    by_center = (y1 + y2) // 2

    # 计算 B-SNR（无高斯）
    bsnr_map = compute_bsnr_weight(
        img_float, box, expand_ratio=1.5, temperature=3.0,
        spatial_gaussian=False
    )

    # 计算 PAG（有高斯）
    pag_map = compute_bsnr_weight(
        img_float, box, expand_ratio=1.5, temperature=3.0,
        spatial_gaussian=True, gauss_sigma_ratio=1.5
    )

    # argmax 物理热点 (在 box 内的坐标)
    peak_flat = bsnr_map.argmax()
    box_h = y2 - y1
    box_w = x2 - x1
    peak_y_in_box = peak_flat // box_w
    peak_x_in_box = peak_flat % box_w
    # 转换为图像绝对坐标
    peak_abs_x = x1 + peak_x_in_box
    peak_abs_y = y1 + peak_y_in_box

    # ── 裁剪 patch ──
    half = CROP_SIZE // 2
    ox = max(0, min(bx_center - half, W - CROP_SIZE))
    oy = max(0, min(by_center - half, H - CROP_SIZE))

    def crop(arr):
        c = arr[oy:oy+CROP_SIZE, ox:ox+CROP_SIZE]
        if c.shape[0] < CROP_SIZE or c.shape[1] < CROP_SIZE:
            out = np.zeros((CROP_SIZE, CROP_SIZE), dtype=c.dtype)
            out[:c.shape[0], :c.shape[1]] = c
            return out
        return c

    img_patch = crop(img_gray)

    # B-SNR 和 PAG 是 box 内的 patch，需要还原到全图再裁
    bsnr_full = np.zeros((H, W), dtype=np.float32)
    bsnr_full[y1:y2, x1:x2] = bsnr_map
    pag_full = np.zeros((H, W), dtype=np.float32)
    pag_full[y1:y2, x1:x2] = pag_map

    bsnr_patch = crop(bsnr_full)
    pag_patch = crop(pag_full)

    # box 和 peak 在 patch 坐标系内
    bx_patch = x1 - ox
    by_patch = y1 - oy
    bw_patch = x2 - x1
    bh_patch = y2 - y1
    peak_px = peak_abs_x - ox
    peak_py = peak_abs_y - oy

    # ── 绘图 ──
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.patch.set_facecolor('black')

    CROSSHAIR_SIZE = max(3, CROP_SIZE // 8)
    CROSSHAIR_COLOR = '#00FF88'  # 亮绿色，在 inferno 背景上清晰可见

    def style_ax(ax, title):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color='white', fontsize=12, pad=5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    # Col 0: 原图 + YOLO 框
    ax = axes[0]
    ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    rect = patches.Rectangle((bx_patch, by_patch), bw_patch, bh_patch,
                              linewidth=1.5, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    style_ax(ax, 'Input IR + YOLO Box')

    # Col 1: B-SNR 热力图 + argmax 十字准星
    ax = axes[1]
    ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    ax.imshow(bsnr_patch, cmap='inferno', alpha=0.80,
              interpolation='nearest', vmin=0, vmax=1)
    # 十字准星
    ax.plot([peak_px - CROSSHAIR_SIZE, peak_px + CROSSHAIR_SIZE],
            [peak_py, peak_py], color=CROSSHAIR_COLOR, linewidth=1.5)
    ax.plot([peak_px, peak_px],
            [peak_py - CROSSHAIR_SIZE, peak_py + CROSSHAIR_SIZE],
            color=CROSSHAIR_COLOR, linewidth=1.5)
    ax.plot(peak_px, peak_py, 'o', color=CROSSHAIR_COLOR,
            markersize=4, markerfacecolor='none', markeredgewidth=1.5)
    # 标注文字（在矢量软件中可替换为更精美的 LaTeX 标注）
    ax.text(peak_px + CROSSHAIR_SIZE + 1, peak_py - CROSSHAIR_SIZE,
            r'$p^*$', color=CROSSHAIR_COLOR, fontsize=9, va='top')
    style_ax(ax, r'B-SNR  $W(i)$  +  Argmax $p^*$')

    # Col 2: PAG 高斯软标签 + 同一 p* 准星
    ax = axes[2]
    ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    ax.imshow(pag_patch, cmap='inferno', alpha=0.80,
              interpolation='nearest', vmin=0, vmax=1)
    ax.plot([peak_px - CROSSHAIR_SIZE, peak_px + CROSSHAIR_SIZE],
            [peak_py, peak_py], color=CROSSHAIR_COLOR, linewidth=1.5)
    ax.plot([peak_px, peak_px],
            [peak_py - CROSSHAIR_SIZE, peak_py + CROSSHAIR_SIZE],
            color=CROSSHAIR_COLOR, linewidth=1.5)
    ax.plot(peak_px, peak_py, 'o', color=CROSSHAIR_COLOR,
            markersize=4, markerfacecolor='none', markeredgewidth=1.5)
    style_ax(ax, r'PAG Soft Label  $P(i)$')

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace=0.04)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='black')
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=ROOT)
    parser.add_argument('--dataset', default='NUAA-SIRST',
                        choices=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k'])
    parser.add_argument('--sample', default='',
                        help='Sample name (without ext). Leave empty for auto-select.')
    parser.add_argument('--out', default=os.path.join(ROOT, 'paper', 'figures', 'fig2_mechanism.pdf'))
    args = parser.parse_args()
    main(args)
