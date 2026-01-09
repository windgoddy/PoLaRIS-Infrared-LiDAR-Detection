#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 dataset/Pohang-Canal-all 中的图像和掩码问题

问题诊断:
1. 图像几乎全黑 - 原因: 使用软链接 (symlink) 而非实际复制
2. 掩码全黑 - 原因: YOLO 标签处理可能有问题

解决方案:
1. 将软链接的图像替换为实际复制的文件
2. 重新生成掩码（验证 YOLO 转换）
"""

import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm

# ================= 配置区域 =================
DATASET_ROOT = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all'
SOURCE_IMAGES_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/infrared/images'
YOLO_LABELS_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/00/all/00_inf_labels'
TARGET_CLASS_IDS = [0]  # YOLO class_id for boat
# ===========================================

def check_and_fix_images():
    """检查并修复图像（将软链接替换为实际文件）"""
    print("=" * 80)
    print("📷 步骤 1/2: 检查并修复图像")
    print("=" * 80)

    images_dir = os.path.join(DATASET_ROOT, 'images')

    if not os.path.exists(images_dir):
        print(f"❌ 错误: 目录不存在 {images_dir}")
        return

    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]

    print(f"\n发现 {len(image_files)} 个图像文件")
    print("检查软链接...")

    symlink_count = 0
    broken_count = 0
    fixed_count = 0

    for filename in tqdm(image_files, desc="处理图像"):
        file_path = os.path.join(images_dir, filename)

        # 检查是否是软链接
        if os.path.islink(file_path):
            symlink_count += 1

            # 检查链接是否有效
            if not os.path.exists(file_path):
                broken_count += 1
                print(f"\n❌ 发现损坏的软链接: {filename}")

                # 尝试从源目录复制
                source_path = os.path.join(SOURCE_IMAGES_DIR, filename)
                if os.path.exists(source_path):
                    # 删除损坏的软链接
                    os.unlink(file_path)
                    # 复制实际文件
                    shutil.copy2(source_path, file_path)
                    fixed_count += 1
                    print(f"   ✅ 已修复: 从源目录复制")
                else:
                    print(f"   ❌ 源文件也不存在: {source_path}")
            else:
                # 软链接有效，但我们仍然替换为实际文件（更可靠）
                try:
                    # 读取链接目标
                    target = os.readlink(file_path)

                    # 检查图像是否正常
                    img = cv2.imread(file_path)
                    if img is not None and img.mean() > 10:
                        # 图像正常，跳过
                        continue

                    # 图像有问题，替换为实际复制
                    source_path = os.path.join(SOURCE_IMAGES_DIR, filename)
                    if os.path.exists(source_path):
                        os.unlink(file_path)
                        shutil.copy2(source_path, file_path)
                        fixed_count += 1
                except Exception as e:
                    print(f"\n⚠️  处理 {filename} 时出错: {e}")

    print(f"\n📊 图像修复统计:")
    print(f"  软链接数量: {symlink_count}")
    print(f"  损坏的软链接: {broken_count}")
    print(f"  已修复: {fixed_count}")

    if fixed_count > 0:
        print(f"\n✅ 已将软链接替换为实际文件，更可靠！")

def yolo_to_mask(yolo_path, img_height, img_width, target_class_ids):
    """
    将 YOLO 格式标签转换为二值掩码
    """
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
    print("🎭 步骤 2/2: 重新生成掩码")
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
        img = cv2.imread(img_path)

        if img is not None:
            is_symlink = os.path.islink(img_path)
            link_type = "软链接" if is_symlink else "实际文件"
            print(f"\n  {filename} ({link_type})")
            print(f"    尺寸: {img.shape}")
            print(f"    像素均值: {img.mean():.2f}")
            print(f"    像素范围: [{img.min()}, {img.max()}]")

            if img.mean() < 10:
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
    print("🔧 数据集修复工具")
    print("=" * 80)
    print(f"数据集路径: {DATASET_ROOT}")
    print(f"源图像路径: {SOURCE_IMAGES_DIR}")
    print(f"YOLO标签路径: {YOLO_LABELS_DIR}")
    print("=" * 80)

    # 步骤 1: 修复图像
    check_and_fix_images()

    # 步骤 2: 重新生成掩码
    regenerate_masks()

    # 步骤 3: 验证结果
    verify_results()

    print("\n" + "=" * 80)
    print("✅ 修复完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
