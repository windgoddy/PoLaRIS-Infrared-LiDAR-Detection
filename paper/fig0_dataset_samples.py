#!/usr/bin/env python3
"""
Figure 0: Dataset Samples & Annotation Comparison
===================================================
3行（数据集） × 4列（对比项）展示图：
  Col 1: 原始 IR 灰度图（放大目标局部块）
  Col 2: YOLO 框标注叠加（红框）
  Col 3: GT 像素掩码
  Col 4: HALO PAG 软标签（magma 热力图叠加）

用法:
    python paper/fig0_dataset_samples.py
    python paper/fig0_dataset_samples.py --root /path/to/project --out paper/figures/fig0.pdf

制图铁律:
    - 所有 imshow 使用 interpolation='nearest'
    - 伪彩色统一使用 'inferno'
    - 最终输出 PDF（矢量文字在 AI/PPT 中后期添加）
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
# ★ 路径配置（在服务器上运行时修改此处）
# ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_CONFIGS = {
    'NUAA-SIRST': {
        'img_dir':   'dataset/NUAA-SIRST/images',
        'mask_dir':  'dataset/NUAA-SIRST/masks',
        'label_dir': 'dataset/NUAA-SIRST/labels_box',
        'split':     'dataset/NUAA-SIRST/50_50/test.txt',
        'img_ext':   '.png',
    },
    'NUDT-SIRST': {
        'img_dir':   'dataset/NUDT-SIRST/images',
        'mask_dir':  'dataset/NUDT-SIRST/masks',
        'label_dir': 'dataset/NUDT-SIRST/labels_box',
        'split':     'dataset/NUDT-SIRST/50_50/test.txt',
        'img_ext':   '.png',
    },
    'IRSTD-1k': {
        'img_dir':   'dataset/IRSTD-1k/IRSTD1k_Img',
        'mask_dir':  'dataset/IRSTD-1k/masks',
        'label_dir': 'dataset/IRSTD-1k/labels_box',
        'split':     'dataset/IRSTD-1k/50_50/test.txt',
        'img_ext':   '.png',
    },
}

# ★ 手动指定每个数据集要展示的样本文件名（不含扩展名）
# 建议策略：
#   NUAA: 选目标明显、背景干净的中等目标
#   NUDT: 选目标 <9px²（视觉上几乎是单个亮斑）
#   IRSTD: 选背景复杂（海天线 / 云层）的场景
# 如果留空 ''，脚本会自动选 test split 第一张有目标的图
SAMPLE_NAMES = {
    'NUAA-SIRST': '',   # e.g. 'Misc_200'
    'NUDT-SIRST': '',   # e.g. '000001'
    'IRSTD-1k':   '',   # e.g. '000001'
}

# 局部裁剪框大小（以目标中心为中心的正方形 patch，像素）
CROP_SIZE = 64

# 图像输出分辨率
DPI = 300

# ────────────────────────────────────────────────────────────────


def load_bsnr_utils(root):
    """直接加载 bsnr_mask_utils 避免触发 torch 导入"""
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
    """自动从 test split 中选第一张有 YOLO label 且目标非空的图"""
    split_file = os.path.join(root, cfg['split'])
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, 'r') as f:
        names = [l.strip() for l in f if l.strip()]
    for name in names:
        label_path = os.path.join(root, cfg['label_dir'], name + '.txt')
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            return name
    raise RuntimeError("No valid sample found in split.")


def get_target_center(boxes, img_h, img_w):
    """返回所有 boxes 的联合中心"""
    if not boxes:
        return img_w // 2, img_h // 2
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1 + x2) // 2, (y1 + y2) // 2


def crop_patch(arr, cx, cy, size, mode='gray'):
    """以 (cx, cy) 为中心裁剪 size×size 的 patch，nearest 填边"""
    h, w = arr.shape[:2]
    half = size // 2
    # 计算裁剪坐标（含边界 clamp）
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    patch = arr[y1:y2, x1:x2]
    # 如果贴边了，pad to size
    if patch.shape[0] < size or patch.shape[1] < size:
        if mode == 'gray':
            out = np.zeros((size, size), dtype=patch.dtype)
        else:
            out = np.zeros((size, size, patch.shape[2]), dtype=patch.dtype)
        out[:patch.shape[0], :patch.shape[1]] = patch
        patch = out
    # 同时返回 offset，用于在 patch 坐标系中重新绘制 box
    return patch, x1, y1


def prepare_row(dataset_name, cfg, root, sample_name, compute_bsnr_weight):
    """返回 (img_patch, img_box_patch, mask_patch, halo_patch, box_in_patch)"""
    img_path = os.path.join(root, cfg['img_dir'], sample_name + cfg['img_ext'])
    mask_path = os.path.join(root, cfg['mask_dir'], sample_name + cfg['img_ext'])
    label_path = os.path.join(root, cfg['label_dir'], sample_name + '.txt')

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros((H, W), dtype=np.uint8)
    mask = (mask > 127).astype(np.uint8) * 255

    boxes = load_yolo_boxes(label_path, H, W)
    cx, cy = get_target_center(boxes, H, W)

    # 裁剪 patch
    img_patch, ox, oy = crop_patch(img_gray, cx, cy, CROP_SIZE)
    mask_patch, _, _ = crop_patch(mask, cx, cy, CROP_SIZE)

    # boxes 在 patch 坐标系内的位置
    boxes_in_patch = []
    for (x1, y1, x2, y2) in boxes:
        px1 = x1 - ox
        py1 = y1 - oy
        px2 = x2 - ox
        py2 = y2 - oy
        boxes_in_patch.append((px1, py1, px2 - px1, py2 - py1))  # (x, y, w, h)

    # 生成 HALO PAG 伪标签（全图，然后裁剪）
    img_float = img_gray.astype(np.float32) / 255.0
    halo_full = np.zeros((H, W), dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = box
        w_map = compute_bsnr_weight(
            img_float, box,
            expand_ratio=1.5, temperature=3.0,
            spatial_gaussian=True, gauss_sigma_ratio=1.5
        )
        halo_full[y1:y2, x1:x2] = np.maximum(halo_full[y1:y2, x1:x2], w_map)
    halo_patch, _, _ = crop_patch(halo_full, cx, cy, CROP_SIZE)

    return img_patch, mask_patch, halo_patch, boxes_in_patch


def main(args):
    compute_bsnr_weight = load_bsnr_utils(args.root)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.patch.set_facecolor('black')

    col_titles = ['IR Image', 'YOLO Box', 'GT Mask', 'HALO (PAG)']
    row_labels = ['NUAA-SIRST\n(medium target)', 'NUDT-SIRST\n(tiny target <9px²)', 'IRSTD-1k\n(complex BG)']

    for row_idx, (ds_name, row_label) in enumerate(zip(
            ['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k'], row_labels)):
        cfg = DATASET_CONFIGS[ds_name]
        name = SAMPLE_NAMES[ds_name]
        if not name:
            name = auto_select_sample(cfg, args.root)
            print(f"[{ds_name}] auto-selected sample: {name}")
        else:
            print(f"[{ds_name}] using specified sample: {name}")

        img_patch, mask_patch, halo_patch, boxes_in_patch = prepare_row(
            ds_name, cfg, args.root, name, compute_bsnr_weight
        )

        # ── Col 0: 原始灰度图 ──
        ax = axes[row_idx][0]
        ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(col_titles[0], color='white', fontsize=11, pad=4)
        ax.set_ylabel(row_label, color='white', fontsize=8, labelpad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

        # ── Col 1: 原图 + YOLO 红框 ──
        ax = axes[row_idx][1]
        ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        for (bx, by, bw, bh) in boxes_in_patch:
            rect = patches.Rectangle((bx, by), bw, bh,
                                      linewidth=1.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(col_titles[1], color='white', fontsize=11, pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

        # ── Col 2: GT 掩码（白色轮廓叠加在原图上）──
        ax = axes[row_idx][2]
        ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        # 用青色轮廓叠加 GT mask
        gt_overlay = np.zeros((*img_patch.shape, 4), dtype=np.float32)
        gt_overlay[mask_patch > 127, :] = [0.0, 1.0, 1.0, 0.85]  # cyan
        ax.imshow(gt_overlay, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(col_titles[2], color='white', fontsize=11, pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

        # ── Col 3: HALO 软标签（inferno 热力图叠加）──
        ax = axes[row_idx][3]
        ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        im = ax.imshow(halo_patch, cmap='inferno', alpha=0.75,
                       interpolation='nearest', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(col_titles[3], color='white', fontsize=11, pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.12, hspace=0.08, wspace=0.05)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    plt.savefig(args.out, dpi=DPI, bbox_inches='tight', facecolor='black')
    print(f"Saved → {args.out}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=ROOT, help='Project root directory')
    parser.add_argument('--out', default=os.path.join(ROOT, 'paper', 'figures', 'fig0_samples.pdf'))
    args = parser.parse_args()
    main(args)
