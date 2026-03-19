#!/usr/bin/env python3
"""
Generate GrabCut Pseudo-Label Masks for DNANet Training (P1-F Group)
=====================================================================
为 NUAA-SIRST / NUDT-SIRST / IRSTD-1k 的所有图像生成 GrabCut 二值伪掩码，
保存到 dataset/<DS>/masks_grabcut/ 目录，供 train.py --mask_folder masks_grabcut 使用。

用法:
    python generate_grabcut_masks.py                    # 生成全部三个数据集
    python generate_grabcut_masks.py --dataset NUAA-SIRST
    python generate_grabcut_masks.py --root /path/to/dataset
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 数据集路径配置（与 grabcut_baseline.py / train.py 保持一致）
# ─────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    'NUAA-SIRST': {
        'img_dir':       'dataset/NUAA-SIRST/images',
        'label_dir':     'dataset/NUAA-SIRST/labels_box',
        'out_mask_dir':  'dataset/NUAA-SIRST/masks_grabcut',
        'img_suffix':    '.png',
    },
    'NUDT-SIRST': {
        'img_dir':       'dataset/NUDT-SIRST/images',
        'label_dir':     'dataset/NUDT-SIRST/labels_box',
        'out_mask_dir':  'dataset/NUDT-SIRST/masks_grabcut',
        'img_suffix':    '.png',
    },
    'IRSTD-1k': {
        'img_dir':       'dataset/IRSTD-1k/IRSTD1k_Img',
        'label_dir':     'dataset/IRSTD-1k/labels_box',
        'out_mask_dir':  'dataset/IRSTD-1k/masks_grabcut',
        'img_suffix':    '.png',
    },
    'SIRST3': {
        'img_dir':       'dataset/SIRST3/images',
        'label_dir':     'dataset/SIRST3/labels_box',
        'out_mask_dir':  'dataset/SIRST3/masks_grabcut',
        'img_suffix':    '.png',
    },
}


def load_yolo_boxes(label_path, img_h, img_w):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), \
                               float(parts[3]), float(parts[4])
            x1 = max(0, int((cx - w / 2) * img_w))
            y1 = max(0, int((cy - h / 2) * img_h))
            x2 = min(img_w - 1, int((cx + w / 2) * img_w))
            y2 = min(img_h - 1, int((cy + h / 2) * img_h))
            if x2 > x1 + 1 and y2 > y1 + 1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def grabcut_predict(img_gray, boxes):
    H, W = img_gray.shape
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    pred = np.zeros((H, W), dtype=np.uint8)

    for (x1, y1, x2, y2) in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw < 5 or bh < 5:
            # 极小目标 fallback：直接填充框（与 grabcut_baseline.py 一致）
            pred[y1:y2, x1:x2] = 255
            continue

        rect = (x1, y1, bw, bh)
        gc_mask = np.zeros((H, W), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_bgr, gc_mask, rect, bgd_model, fgd_model,
                        5, cv2.GC_INIT_WITH_RECT)
            fg = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255, 0
            ).astype(np.uint8)
            roi = np.zeros_like(fg)
            roi[y1:y2, x1:x2] = fg[y1:y2, x1:x2]
            pred = np.maximum(pred, roi)
        except cv2.error:
            pred[y1:y2, x1:x2] = 255  # fallback

    return pred  # uint8，值为 0 或 255


def generate_dataset(dataset_name, root='.'):
    cfg = DATASET_CONFIGS[dataset_name]

    img_dir      = os.path.join(root, cfg['img_dir'])
    label_dir    = os.path.join(root, cfg['label_dir'])
    out_mask_dir = os.path.join(root, cfg['out_mask_dir'])
    suffix       = cfg['img_suffix']

    if not os.path.isdir(img_dir):
        print(f"[SKIP] image dir not found: {img_dir}")
        return

    os.makedirs(out_mask_dir, exist_ok=True)

    # 枚举所有图像（不依赖 split 文件，生成全量伪标签供训练使用）
    img_names = sorted([
        f[:-len(suffix)] for f in os.listdir(img_dir)
        if f.endswith(suffix)
    ])
    # 同时尝试 .bmp / .jpg（IRSTD-1k 可能有不同后缀）
    if not img_names:
        for ext in ['.bmp', '.jpg', '.jpeg']:
            img_names = sorted([f[:-len(ext)] for f in os.listdir(img_dir) if f.endswith(ext)])
            if img_names:
                suffix = ext
                break

    print(f"\n[{dataset_name}] Found {len(img_names)} images → saving to {out_mask_dir}")

    done, skipped = 0, 0
    for name in img_names:
        img_path   = os.path.join(img_dir,   name + suffix)
        label_path = os.path.join(label_dir, name + '.txt')
        out_path   = os.path.join(out_mask_dir, name + '.png')

        # 已存在则跳过（断点续跑）
        if os.path.exists(out_path):
            done += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        H, W = img.shape
        boxes = load_yolo_boxes(label_path, H, W)

        if not boxes:
            # 无标注图（背景帧）→ 全黑掩码
            mask = np.zeros((H, W), dtype=np.uint8)
        else:
            mask = grabcut_predict(img, boxes)

        cv2.imwrite(out_path, mask)
        done += 1

        if done % 100 == 0:
            print(f"  [{dataset_name}] {done}/{len(img_names)} done ...")

    print(f"  [{dataset_name}] Finished: {done} saved, {skipped} skipped")
    print(f"  Output dir: {out_mask_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all',
                        choices=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k', 'SIRST3', 'all'])
    parser.add_argument('--root', default='.',
                        help='project root (default: current dir)')
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        generate_dataset(ds, root=args.root)

    print("\nDone. Now train with:")
    for ds in datasets:
        short = ds.lower().replace('-', '_').replace('sirst', '').replace('1k', '1k')
        print(f"  python train.py --dataset {ds} --mask_folder masks_grabcut "
              f"--net DNANet --epochs 1500 [other args...]")
