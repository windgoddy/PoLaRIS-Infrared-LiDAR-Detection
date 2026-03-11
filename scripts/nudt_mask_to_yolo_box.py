"""
从 NUDT-SIRST 像素掩码推导 YOLO Box 标注
用于模拟弱监督 Pilot 实验（B 组和 C 组的标注来源）

功能：
- 读取 masks/ 目录中的二值掩码 PNG
- 用 cv2.findContours 提取前景连通域的外接矩形
- 向外膨胀 pad_pixels 像素（模拟真实手工标注）
- 写成 YOLO 格式（归一化坐标）txt 到 labels/ 目录

用法：
    python scripts/nudt_mask_to_yolo_box.py \
        --dataset_dir dataset/NUDT-SIRST \
        --pad 2
"""

import os
import cv2
import numpy as np
import argparse


def mask_to_yolo_boxes(mask, pad=2):
    """
    从二值掩码提取 YOLO 格式 bounding box。

    Args:
        mask: (H, W) uint8 二值图，前景>0
        pad: 膨胀像素数（模拟手工标注的松散框）

    Returns:
        boxes: List of [class_id, cx_norm, cy_norm, w_norm, h_norm]
    """
    H, W = mask.shape

    # 二值化
    binary = (mask > 0).astype(np.uint8) * 255

    # 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for i in range(1, num_labels):  # 跳过背景 label=0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # 膨胀 pad 像素
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        w_pad = x2 - x1
        h_pad = y2 - y1

        # YOLO 归一化坐标
        cx_norm = (x1 + w_pad / 2) / W
        cy_norm = (y1 + h_pad / 2) / H
        w_norm = w_pad / W
        h_norm = h_pad / H

        boxes.append([0, cx_norm, cy_norm, w_norm, h_norm])

    return boxes


def convert_dataset(dataset_dir, mask_folder='masks', output_folder='labels_box', pad=2):
    """
    批量转换整个数据集的掩码为 YOLO box 标注。
    """
    masks_dir = os.path.join(dataset_dir, mask_folder)
    labels_dir = os.path.join(dataset_dir, output_folder)
    os.makedirs(labels_dir, exist_ok=True)

    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])

    total = len(mask_files)
    has_target = 0
    no_target = 0
    total_boxes = 0

    print(f"处理 {total} 个掩码文件 → {labels_dir}")
    print(f"膨胀像素: {pad}")
    print("=" * 50)

    for fname in mask_files:
        img_id = fname[:-4]  # 去掉 .png
        mask_path = os.path.join(masks_dir, fname)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  [WARN] 无法读取: {mask_path}")
            continue

        boxes = mask_to_yolo_boxes(mask, pad=pad)

        label_path = os.path.join(labels_dir, img_id + '.txt')
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

        if len(boxes) > 0:
            has_target += 1
            total_boxes += len(boxes)
        else:
            no_target += 1

    print(f"完成！")
    print(f"  有目标图像: {has_target} ({100*has_target/total:.1f}%)")
    print(f"  无目标图像: {no_target} ({100*no_target/total:.1f}%)")
    print(f"  总 Box 数:  {total_boxes}")
    print(f"  平均 Box/图: {total_boxes/max(has_target,1):.2f}")
    print(f"  输出目录:   {labels_dir}")

    return labels_dir


def verify_boxes(dataset_dir, output_folder='labels_box', n_samples=5):
    """
    验证生成的 box 标注（打印几个样例）。
    """
    labels_dir = os.path.join(dataset_dir, output_folder)
    masks_dir = os.path.join(dataset_dir, 'masks')
    images_dir = os.path.join(dataset_dir, 'images')

    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

    # 找有目标的样例
    samples_with_target = []
    for fname in label_files:
        with open(os.path.join(labels_dir, fname)) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            samples_with_target.append((fname[:-4], lines))
        if len(samples_with_target) >= n_samples:
            break

    print(f"\n验证样例（前 {n_samples} 个有目标）:")
    print("=" * 50)

    for img_id, lines in samples_with_target:
        # 读图像尺寸
        img_path = os.path.join(images_dir, img_id + '.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape

        # 读掩码
        mask_path = os.path.join(masks_dir, img_id + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        fg_pixels = (mask > 0).sum() if mask is not None else 0

        print(f"\n  [{img_id}] 图像: {W}×{H}, 前景像素: {fg_pixels}")
        for line in lines:
            parts = line.split()
            _, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            px_w, px_h = int(w * W), int(h * H)
            print(f"    Box: cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f} → {px_w}×{px_h} px")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从像素掩码生成 YOLO Box 标注')
    parser.add_argument('--dataset_dir', type=str, default='dataset/NUDT-SIRST',
                        help='数据集根目录')
    parser.add_argument('--mask_folder', type=str, default='masks',
                        help='掩码子目录名')
    parser.add_argument('--output_folder', type=str, default='labels_box',
                        help='输出 YOLO label 目录名')
    parser.add_argument('--pad', type=int, default=2,
                        help='边框膨胀像素数（模拟手工标注松散度）')
    parser.add_argument('--verify', action='store_true',
                        help='转换后打印验证样例')
    args = parser.parse_args()

    convert_dataset(
        dataset_dir=args.dataset_dir,
        mask_folder=args.mask_folder,
        output_folder=args.output_folder,
        pad=args.pad,
    )

    if args.verify:
        verify_boxes(args.dataset_dir, args.output_folder)
