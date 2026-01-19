#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 dataset/Pohang-Canal-all 中的图像和掩码问题 (包含 16-bit 归一化)

问题诊断:
1. 图像几乎全黑 - 根本原因:
   - **原始图像是 16-bit** (像素值范围 0-65535，实际值可能在 2000-3000)
   - 直接软链接/复制了 16-bit 数据，没有归一化到 0-255
   - 导致图像在 8-bit 显示时几乎全黑
2. 掩码全黑 - YOLO 标签处理需要验证

解决方案:
1. 读取 16-bit 原始图像
2. 使用百分位数归一化 (2%-98%) 到 0-255
3. 保存为标准 8-bit PNG
4. 重新生成 YOLO 格式掩码
"""

import os
import numpy as np
import cv2
from tqdm import tqdm

# ================= 配置区域 =================
DATASET_ROOT = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all'
SOURCE_IMAGES_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/infrared/images'
YOLO_LABELS_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/00/all/00_inf_labels'
TARGET_CLASS_IDS = [0]  # YOLO class_id for boat
NORMALIZE_METHOD = 'percentile'  # 'minmax', 'percentile', 'clahe'
# ===========================================

def normalize_infrared_image(image_path, method='percentile'):
    """
    归一化红外图像（16-bit → 8-bit）

    Args:
        image_path: 图像路径
        method: 归一化方法 ('minmax', 'percentile', 'clahe')

    Returns:
        normalized: 8-bit 归一化图像，失败返回 None
    """
    try:
        # 尝试以 16-bit 读取
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            return None

        # 如果已经是 8-bit 且对比度正常，直接返回
        if img.dtype == np.uint8:
            if img.mean() >= 10:  # 正常的 8-bit 图像
                return img
            # 否则可能是错误的 8-bit 数据，继续尝试归一化

        # 转换为浮点数
        img_float = img.astype(np.float32)

        if method == 'minmax':
            # 方法 1: 简单 Min-Max 归一化
            img_min = img_float.min()
            img_max = img_float.max()

            if img_max == img_min:
                normalized = np.zeros_like(img_float, dtype=np.uint8)
            else:
                normalized = ((img_float - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

        elif method == 'percentile':
            # 方法 2: 百分位数归一化（推荐，避免异常值）
            vmin = np.percentile(img_float, 2)
            vmax = np.percentile(img_float, 98)

            if vmax == vmin:
                normalized = np.zeros_like(img_float, dtype=np.uint8)
            else:
                img_clipped = np.clip(img_float, vmin, vmax)
                normalized = ((img_clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)

        elif method == 'clahe':
            # 方法 3: CLAHE (对比度受限的自适应直方图均衡化)
            vmin = np.percentile(img_float, 2)
            vmax = np.percentile(img_float, 98)

            if vmax == vmin:
                normalized = np.zeros_like(img_float, dtype=np.uint8)
            else:
                img_clipped = np.clip(img_float, vmin, vmax)
                img_norm = ((img_clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                normalized = clahe.apply(img_norm)

        else:
            raise ValueError(f"Unknown method: {method}")

        return normalized

    except Exception as e:
        print(f"\n❌ 归一化错误 {image_path}: {e}")
        return None

def process_and_normalize_images():
    """处理并归一化图像（16-bit → 8-bit）"""
    print("=" * 80)
    print("📷 步骤 1/2: 处理并归一化图像 (16-bit → 8-bit)")
    print("=" * 80)
    print(f"方法: {NORMALIZE_METHOD}")

    images_dir = os.path.join(DATASET_ROOT, 'images')
    split_data_dir = os.path.join(DATASET_ROOT, 'split_data')

    # 加载数据列表
    train_file = os.path.join(split_data_dir, 'train.txt')
    test_file = os.path.join(split_data_dir, 'test.txt')

    all_frames = []
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            all_frames.extend([line.strip() for line in f if line.strip()])
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            all_frames.extend([line.strip() for line in f if line.strip()])

    if not all_frames:
        print("❌ 错误: 未找到数据列表")
        return

    print(f"\n需要处理 {len(all_frames)} 张图像")

    normalized_count = 0
    skip_count = 0
    error_count = 0
    first_stats = None

    for frame_name in tqdm(all_frames, desc="处理图像"):
        filename = frame_name + '.png'
        dst_path = os.path.join(images_dir, filename)
        src_path = os.path.join(SOURCE_IMAGES_DIR, filename)

        if not os.path.exists(src_path):
            error_count += 1
            continue

        # 归一化图像
        normalized = normalize_infrared_image(src_path, method=NORMALIZE_METHOD)

        if normalized is None:
            error_count += 1
            continue

        # 检查是否需要更新
        needs_update = True
        if os.path.exists(dst_path):
            existing = cv2.imread(dst_path, cv2.IMREAD_UNCHANGED)
            if existing is not None and existing.dtype == np.uint8 and existing.mean() >= 10:
                # 已经是正常的 8-bit 图像，跳过
                needs_update = False
                skip_count += 1

        if needs_update:
            # 删除旧文件（可能是软链接）
            if os.path.exists(dst_path):
                if os.path.islink(dst_path):
                    os.unlink(dst_path)
                else:
                    os.remove(dst_path)

            # 保存归一化后的图像
            cv2.imwrite(dst_path, normalized)
            normalized_count += 1

            # 记录第一张图像的统计
            if first_stats is None:
                original = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
                first_stats = {
                    'filename': filename,
                    'original_dtype': str(original.dtype),
                    'original_shape': original.shape,
                    'original_range': (int(original.min()), int(original.max())),
                    'original_mean': float(original.mean()),
                    'normalized_range': (int(normalized.min()), int(normalized.max())),
                    'normalized_mean': float(normalized.mean())
                }

    print(f"\n📊 图像处理统计:")
    print(f"  ✅ 归一化处理: {normalized_count} 张")
    print(f"  ⏭️  跳过（已正常）: {skip_count} 张")
    print(f"  ❌ 处理失败: {error_count} 张")

    if first_stats:
        print(f"\n📝 示例图像统计 ({first_stats['filename']}):")
        print(f"  原始图像:")
        print(f"    数据类型: {first_stats['original_dtype']}")
        print(f"    形状: {first_stats['original_shape']}")
        print(f"    像素范围: {first_stats['original_range']}")
        print(f"    像素均值: {first_stats['original_mean']:.2f}")
        print(f"  归一化后:")
        print(f"    像素范围: {first_stats['normalized_range']}")
        print(f"    像素均值: {first_stats['normalized_mean']:.2f}")

    if normalized_count > 0:
        print(f"\n✅ 成功将 {normalized_count} 张 16-bit 图像归一化为 8-bit！")

def yolo_to_mask(yolo_path, img_height, img_width, target_class_ids):
    """将 YOLO 格式标签转换为二值掩码"""
    try:
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if not os.path.exists(yolo_path):
            return mask

        with open(yolo_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            return mask

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            if class_id not in target_class_ids:
                continue

            # 转换为像素坐标
            center_x_px = int(center_x * img_width)
            center_y_px = int(center_y * img_height)
            width_px = int(width * img_width)
            height_px = int(height * img_height)

            # 计算边界框
            x1 = max(0, int(center_x_px - width_px / 2))
            y1 = max(0, int(center_y_px - height_px / 2))
            x2 = min(img_width - 1, int(center_x_px + width_px / 2))
            y2 = min(img_height - 1, int(center_y_px + height_px / 2))

            # 绘制填充矩形
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        return mask

    except Exception as e:
        print(f"Error processing {yolo_path}: {e}")
        return None

def regenerate_masks():
    """重新生成掩码"""
    print("\n" + "=" * 80)
    print("🎭 步骤 2/2: 重新生成掩码 (YOLO 格式)")
    print("=" * 80)

    masks_dir = os.path.join(DATASET_ROOT, 'masks')
    images_dir = os.path.join(DATASET_ROOT, 'images')
    split_data_dir = os.path.join(DATASET_ROOT, 'split_data')

    # 加载数据列表
    train_list = []
    test_list = []

    train_file = os.path.join(split_data_dir, 'train.txt')
    test_file = os.path.join(split_data_dir, 'test.txt')

    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            train_list = [line.strip() for line in f if line.strip()]

    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            test_list = [line.strip() for line in f if line.strip()]

    all_frames = train_list + test_list

    if not all_frames:
        print("❌ 错误: 未找到数据列表")
        return

    print(f"需要处理 {len(all_frames)} 个掩码")

    # 获取图像尺寸
    first_img_path = os.path.join(images_dir, all_frames[0] + '.png')
    img = cv2.imread(first_img_path)
    if img is None:
        print(f"❌ 无法读取图像: {first_img_path}")
        return

    img_height, img_width = img.shape[:2]
    print(f"图像尺寸: {img_height} x {img_width}")

    # 批处理生成掩码
    success_count = 0
    empty_count = 0
    non_empty_count = 0

    for frame_name in tqdm(all_frames, desc="生成掩码"):
        yolo_path = os.path.join(YOLO_LABELS_DIR, frame_name + '.txt')
        mask_path = os.path.join(masks_dir, frame_name + '.png')

        mask = yolo_to_mask(yolo_path, img_height, img_width, TARGET_CLASS_IDS)

        if mask is None:
            continue

        # 统计
        if np.sum(mask) == 0:
            empty_count += 1
        else:
            non_empty_count += 1

        cv2.imwrite(mask_path, mask)
        success_count += 1

    print(f"\n📊 掩码生成统计:")
    print(f"  成功生成: {success_count}")
    print(f"  空掩码（背景）: {empty_count}")
    print(f"  非空掩码（有目标）: {non_empty_count}")

    if non_empty_count == 0:
        print(f"\n⚠️  警告: 所有掩码都是空的！")
        print(f"   可能的原因:")
        print(f"   1. TARGET_CLASS_IDS = {TARGET_CLASS_IDS} 不正确")
        print(f"   2. YOLO 标签文件格式有问题")
        print(f"   3. 图像尺寸检测错误")
    else:
        print(f"\n✅ 掩码生成成功！非空掩码比例: {non_empty_count/success_count*100:.1f}%")

def verify_results():
    """验证修复结果"""
    print("\n" + "=" * 80)
    print("🔍 验证修复结果")
    print("=" * 80)

    images_dir = os.path.join(DATASET_ROOT, 'images')
    masks_dir = os.path.join(DATASET_ROOT, 'masks')

    # 检查几个示例图像
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')][:5]

    print("\n📷 检查示例图像:")
    for filename in image_files:
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is not None:
            is_symlink = os.path.islink(img_path)
            link_type = "软链接" if is_symlink else "实际文件"
            print(f"\n  {filename} ({link_type})")
            print(f"    数据类型: {img.dtype}")
            print(f"    尺寸: {img.shape}")
            print(f"    像素均值: {img.mean():.2f}")
            print(f"    像素范围: [{img.min()}, {img.max()}]")

            if img.dtype != np.uint8:
                print(f"    ⚠️  警告: 仍是 {img.dtype}，未归一化")
            elif img.mean() < 10:
                print(f"    ⚠️  警告: 图像仍然很暗")
            else:
                print(f"    ✅ 图像正常")

    # 检查几个示例掩码
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')][:10]

    print("\n\n🎭 检查示例掩码:")
    non_empty_found = False
    for filename in mask_files:
        mask_path = os.path.join(masks_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            foreground = np.sum(mask > 0)
            if foreground > 0:
                non_empty_found = True
                print(f"\n  {filename}")
                print(f"    尺寸: {mask.shape}")
                print(f"    前景像素: {foreground}")
                print(f"    前景比例: {foreground/mask.size*100:.2f}%")
                print(f"    ✅ 非空掩码")
                break

    if not non_empty_found:
        print("  ⚠️  前10个掩码都是空的")

def main():
    print("🔧 数据集修复工具 (包含 16-bit → 8-bit 归一化)")
    print("=" * 80)
    print(f"数据集路径: {DATASET_ROOT}")
    print(f"源图像路径: {SOURCE_IMAGES_DIR}")
    print(f"YOLO标签路径: {YOLO_LABELS_DIR}")
    print(f"归一化方法: {NORMALIZE_METHOD}")
    print("=" * 80)

    # 步骤 1: 处理并归一化图像
    process_and_normalize_images()

    # 步骤 2: 重新生成掩码
    regenerate_masks()

    # 步骤 3: 验证结果
    verify_results()

    print("\n" + "=" * 80)
    print("✅ 修复完成！")
    print("=" * 80)
    print("\n💡 关键要点:")
    print("  - 原始红外图像是 16-bit，需要归一化到 8-bit (0-255)")
    print("  - 使用百分位数归一化避免异常值影响")
    print("  - 图像现在应该具有正常的对比度和亮度")
    print("=" * 80)

if __name__ == "__main__":
    main()
