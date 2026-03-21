#!/usr/bin/env python3
"""
从 YOLO 框标注生成 HALO p* 点标注（单像素 PNG）
用于将 HALO 的物理热点作为 PAL 的点监督输入。

p* = argmax B-SNR = 框内 SNR 最大的像素（辐射热点）

用法：
  python paper/gen_pstar_point_labels.py \
      --dataset_dir dataset/NUDT-SIRST \
      --split_file dataset/NUDT-SIRST/50_50/train.txt \
      --output_dir dataset/NUDT-SIRST/masks_pstar_point

输出：每张训练图对应一个 PNG，与原图同分辨率，
      p* 位置像素值=255，其余=0（PAL masks_centroid 格式）
"""

import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_pstar(image_gray: np.ndarray,
                  box,
                  expand_ratio: float = 1.5,
                  temperature: float = 3.0,
                  epsilon: float = 1e-4):
    """
    对单个框计算 B-SNR 并返回 p* 在图像全局坐标中的 (y, x)。

    Args:
        image_gray: (H, W) float32, [0,1]
        box:        (x1, y1, x2, y2) 像素坐标（整数）
        expand_ratio: Context Box 膨胀倍率
        temperature:  sigmoid 温度参数 τ

    Returns:
        (py, px): p* 在全图中的 (row, col)
    """
    H, W = image_gray.shape
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return (y1 + box_h // 2, x1 + box_w // 2)

    # Context Box（统计域）
    pad_x = max(int((expand_ratio - 1.0) / 2.0 * box_w), 3)
    pad_y = max(int((expand_ratio - 1.0) / 2.0 * box_h), 3)
    ctx_x1 = max(0, x1 - pad_x)
    ctx_y1 = max(0, y1 - pad_y)
    ctx_x2 = min(W, x2 + pad_x)
    ctx_y2 = min(H, y2 + pad_y)

    # 背景统计（Window 方法）
    ctx_region = image_gray[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
    mu = ctx_region.mean()
    sigma = ctx_region.std()

    # 框内 B-SNR
    target_region = image_gray[y1:y2, x1:x2]
    snr = (target_region - mu) / (sigma + epsilon)

    # p* = argmax SNR（全局坐标）
    peak_flat = snr.argmax()
    peak_y_in_box = peak_flat // box_w
    peak_x_in_box = peak_flat % box_w

    py = y1 + peak_y_in_box
    px = x1 + peak_x_in_box
    py = max(0, min(H - 1, py))
    px = max(0, min(W - 1, px))
    return (py, px)


def process_dataset(dataset_dir, img_ids, output_dir,
                    label_folder='labels_box',
                    image_folder='images',
                    suffix='.png',
                    expand_ratio=1.5,
                    temperature=3.0):

    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(dataset_dir, label_folder)
    images_dir = os.path.join(dataset_dir, image_folder)

    success, empty, not_found = 0, 0, 0

    for img_id in img_ids:
        img_path = os.path.join(images_dir, img_id + suffix)
        if not os.path.exists(img_path):
            not_found += 1
            continue

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            not_found += 1
            continue

        H, W = image.shape[:2]

        # 灰度化 + 归一化
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.dtype == np.uint16:
            gray_f = gray.astype(np.float32) / 65535.0
        else:
            gray_f = gray.astype(np.float32) / 255.0

        # 读 YOLO 标签
        label_path = os.path.join(labels_dir, img_id + '.txt')
        point_mask = np.zeros((H, W), dtype=np.uint8)

        boxes_found = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cx_n, cy_n, bw_n, bh_n = map(float, parts[1:5])
                    cx = cx_n * W
                    cy = cy_n * H
                    bw = bw_n * W
                    bh = bh_n * H
                    x1 = int(max(0, cx - bw / 2))
                    y1 = int(max(0, cy - bh / 2))
                    x2 = int(min(W, cx + bw / 2))
                    y2 = int(min(H, cy + bh / 2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    py, px = compute_pstar(gray_f, (x1, y1, x2, y2),
                                           expand_ratio=expand_ratio,
                                           temperature=temperature)
                    point_mask[py, px] = 255
                    boxes_found += 1

        out_path = os.path.join(output_dir, img_id + '.png')
        cv2.imwrite(out_path, point_mask)

        if boxes_found > 0:
            success += 1
        else:
            empty += 1

    print(f"Done: {success} with points, {empty} empty (no labels), {not_found} images not found")
    print(f"Point labels saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--split_file', type=str, default=None,
                        help='train.txt 路径，None 则处理所有 labels_box/*.txt')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--label_folder', type=str, default='labels_box')
    parser.add_argument('--image_folder', type=str, default='images')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--expand_ratio', type=float, default=1.5)
    parser.add_argument('--temperature', type=float, default=3.0)
    args = parser.parse_args()

    if args.split_file:
        split_path = args.split_file if os.path.exists(args.split_file) else \
                     os.path.join(args.dataset_dir, args.split_file)
        with open(split_path) as f:
            img_ids = [l.strip() for l in f if l.strip()]
        print(f"Loaded {len(img_ids)} IDs from {split_path}")
    else:
        labels_dir = os.path.join(args.dataset_dir, args.label_folder)
        img_ids = [f[:-4] for f in sorted(os.listdir(labels_dir)) if f.endswith('.txt')]
        print(f"Found {len(img_ids)} label files")

    print(f"Dataset: {args.dataset_dir}")
    print(f"Output:  {args.output_dir}")
    print(f"expand_ratio={args.expand_ratio}, temperature={args.temperature}")

    process_dataset(
        dataset_dir=args.dataset_dir,
        img_ids=img_ids,
        output_dir=args.output_dir,
        label_folder=args.label_folder,
        image_folder=args.image_folder,
        suffix=args.suffix,
        expand_ratio=args.expand_ratio,
        temperature=args.temperature,
    )
