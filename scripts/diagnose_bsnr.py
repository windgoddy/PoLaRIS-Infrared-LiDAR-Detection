"""
B-SNR 掩码生成诊断脚本
用于排查为什么大部分 masks_bsnr 是黑色图像
"""
import os
import sys
import cv2
import numpy as np

def diagnose(dataset_dir, split_file=None, image_folder='images-8bit', suffix='.png'):
    labels_dir = os.path.join(dataset_dir, 'labels')
    images_dir = os.path.join(dataset_dir, image_folder)

    # 1. 检查目录是否存在
    print("=" * 60)
    print("1. 目录检查")
    print("=" * 60)
    for d, name in [(labels_dir, 'labels'), (images_dir, 'images')]:
        exists = os.path.exists(d)
        count = len(os.listdir(d)) if exists else 0
        print(f"  {name}: {'✓' if exists else '✗'} {d} ({count} files)")

    # 2. 获取图像 ID 列表
    if split_file:
        split_path = os.path.join(dataset_dir, split_file)
        with open(split_path, 'r') as f:
            img_ids = [line.strip() for line in f if line.strip()]
        print(f"\n  Split file: {split_path} ({len(img_ids)} IDs)")
    else:
        img_ids = [f[:-4] for f in sorted(os.listdir(labels_dir)) if f.endswith('.txt')]
        print(f"\n  No split file, using all labels ({len(img_ids)} IDs)")

    # 3. 检查标签覆盖率
    print("\n" + "=" * 60)
    print("2. 标签覆盖率")
    print("=" * 60)
    has_label = 0
    no_label = 0
    empty_label = 0
    has_targets = 0
    sample_with_targets = []
    sample_without_labels = []

    for img_id in img_ids:
        label_path = os.path.join(labels_dir, img_id + '.txt')
        if not os.path.exists(label_path):
            no_label += 1
            if len(sample_without_labels) < 5:
                sample_without_labels.append(img_id)
            continue

        has_label += 1
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) == 0:
            empty_label += 1
        else:
            has_targets += 1
            if len(sample_with_targets) < 5:
                sample_with_targets.append((img_id, len(lines)))

    print(f"  总图像数:          {len(img_ids)}")
    print(f"  有标签文件:        {has_label} ({100*has_label/len(img_ids):.1f}%)")
    print(f"  无标签文件:        {no_label} ({100*no_label/len(img_ids):.1f}%)")
    print(f"  标签为空(无目标):  {empty_label}")
    print(f"  有目标的图像:      {has_targets} ({100*has_targets/len(img_ids):.1f}%)")

    if sample_without_labels:
        print(f"\n  无标签样例: {sample_without_labels}")
    if sample_with_targets:
        print(f"  有目标样例: {sample_with_targets}")

    # 4. 检查有目标的图像，测试 B-SNR 计算
    print("\n" + "=" * 60)
    print("3. B-SNR 计算测试（前5个有目标的图像）")
    print("=" * 60)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from model_Mamba.dataset.bsnr_mask_utils import generate_bsnr_mask, _load_yolo_labels

    for img_id, n_targets in sample_with_targets[:5]:
        img_path = os.path.join(images_dir, img_id + suffix)
        label_path = os.path.join(labels_dir, img_id + '.txt')

        # 加载
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"\n  [{img_id}] ✗ 图像加载失败: {img_path}")
            continue

        labels = _load_yolo_labels(label_path)
        H, W = image.shape[:2]

        print(f"\n  [{img_id}]")
        print(f"    图像: {image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")
        print(f"    标签: {len(labels)} 个目标")

        for i, lab in enumerate(labels):
            _, cx, cy, w, h = lab[:5]
            px_w, px_h = int(w * W), int(h * H)
            print(f"    目标 {i}: cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f} → {px_w}×{px_h} px")

        # 生成 B-SNR
        mask = generate_bsnr_mask(
            labels, (H, W), image,
            expand_ratio=1.5,
            temperature=3.0,
            fg_threshold=0.5,
        )

        fg_pixels = (mask > 0).sum()
        print(f"    B-SNR mask: range=[{mask.min():.4f}, {mask.max():.4f}], fg_pixels={fg_pixels}")

        # 也测试 fg_threshold=0 看是否有任何响应
        mask_nothresh = generate_bsnr_mask(
            labels, (H, W), image,
            expand_ratio=1.5,
            temperature=3.0,
            fg_threshold=0.0,
        )
        fg_nothresh = (mask_nothresh > 0.3).sum()
        print(f"    无阈值 mask: range=[{mask_nothresh.min():.4f}, {mask_nothresh.max():.4f}], fg>0.3={fg_nothresh}")

    # 5. 检查已生成的 masks_bsnr
    print("\n" + "=" * 60)
    print("4. 已生成的 masks_bsnr 统计")
    print("=" * 60)
    bsnr_dir = os.path.join(dataset_dir, 'masks_bsnr')
    if os.path.exists(bsnr_dir):
        bsnr_files = [f for f in os.listdir(bsnr_dir) if f.endswith('.png')]
        non_zero = 0
        for f in bsnr_files[:200]:
            m = cv2.imread(os.path.join(bsnr_dir, f), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.max() > 0:
                non_zero += 1
        print(f"  总文件数: {len(bsnr_files)}")
        print(f"  前200个中非零: {non_zero}")
    else:
        print(f"  masks_bsnr 目录不存在")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='dataset/Pohang-Canal-3k')
    parser.add_argument('--split_file', type=str, default='50_50_cat2/train.txt')
    parser.add_argument('--image_folder', type=str, default='images-8bit')
    parser.add_argument('--suffix', type=str, default='.png')
    args = parser.parse_args()

    diagnose(args.dataset_dir, args.split_file, args.image_folder, args.suffix)
