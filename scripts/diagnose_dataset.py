#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集诊断脚本 - 检查图像和标签问题
"""

import os
import cv2
import numpy as np

def check_images():
    """检查图像问题"""
    print("=" * 80)
    print("🔍 检查图像 (images/)")
    print("=" * 80)

    images_dir = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all/images'

    if not os.path.exists(images_dir):
        print(f"❌ 目录不存在: {images_dir}")
        return

    # 获取几个示例文件
    all_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    files = sorted(all_files)[:5]

    print(f"图像总数: {len(all_files)}")
    print(f"检查前 5 个样本:\n")

    for fname in files:
        fpath = os.path.join(images_dir, fname)

        # 检查是否是软链接
        if os.path.islink(fpath):
            target = os.readlink(fpath)
            print(f"\n📄 {fname}")
            print(f"  类型: 软链接")
            print(f"  目标: {target}")

            # 检查目标是否存在
            if os.path.exists(fpath):
                print(f"  状态: ✅ 链接有效")

                # 读取图像
                img = cv2.imread(fpath)
                if img is not None:
                    print(f"  尺寸: {img.shape}")
                    print(f"  像素均值: {img.mean():.2f}")
                    print(f"  像素范围: [{img.min()}, {img.max()}]")

                    if img.mean() < 10:
                        print(f"  ⚠️  警告: 图像几乎全黑！")
                else:
                    print(f"  ❌ 无法读取图像")
            else:
                print(f"  ❌ 链接失效（目标不存在）")
        else:
            print(f"\n📄 {fname}")
            print(f"  类型: 实际文件")
            print(f"  大小: {os.path.getsize(fpath)} 字节")

            img = cv2.imread(fpath)
            if img is not None:
                print(f"  尺寸: {img.shape}")
                print(f"  像素均值: {img.mean():.2f}")
                print(f"  像素范围: [{img.min()}, {img.max()}]")

                if img.mean() < 10:
                    print(f"  ⚠️  警告: 图像几乎全黑！")
            else:
                print(f"  ❌ 无法读取图像")

def check_masks():
    """检查 mask 问题"""
    print("\n" + "=" * 80)
    print("🔍 检查掩码 (masks/)")
    print("=" * 80)

    masks_dir = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all/masks'

    if not os.path.exists(masks_dir):
        print(f"❌ 目录不存在: {masks_dir}")
        return

    all_masks = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
    files = sorted(all_masks)[:10]

    print(f"掩码总数: {len(all_masks)}")
    print(f"检查前 10 个样本:\n")

    total = 0
    empty = 0

    for fname in files:
        fpath = os.path.join(masks_dir, fname)
        mask = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            total += 1
            foreground = np.sum(mask > 0)

            if foreground == 0:
                empty += 1
                if empty <= 3:  # 只显示前3个
                    print(f"\n📄 {fname}")
                    print(f"  尺寸: {mask.shape}")
                    print(f"  前景像素: {foreground} (全黑！)")
            else:
                print(f"\n📄 {fname}")
                print(f"  尺寸: {mask.shape}")
                print(f"  前景像素: {foreground}")
                print(f"  前景比例: {foreground / mask.size * 100:.2f}%")

    print(f"\n📊 统计:")
    print(f"  检查的 mask 数量: {total}")
    print(f"  全黑的 mask 数量: {empty}")

    if empty == total:
        print(f"\n❌ 所有 mask 都是全黑的！")

def check_yolo_labels():
    """检查 YOLO 标签"""
    print("\n" + "=" * 80)
    print("🔍 检查 YOLO 标签")
    print("=" * 80)

    label_dir = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/00/all/00_inf_labels'

    if not os.path.exists(label_dir):
        print(f"❌ 目录不存在: {label_dir}")
        return

    # 检查第一个标签文件
    files = sorted(os.listdir(label_dir))[:5]

    for fname in files:
        if not fname.endswith('.txt'):
            continue

        fpath = os.path.join(label_dir, fname)
        print(f"\n📄 {fname}")

        try:
            with open(fpath, 'r') as f:
                lines = f.readlines()

            print(f"  行数: {len(lines)}")

            if lines:
                print(f"  示例内容:")
                for i, line in enumerate(lines[:3]):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        print(f"    行{i+1}: class={parts[0]}, cx={parts[1]}, cy={parts[2]}, w={parts[3]}, h={parts[4]}")
                    else:
                        print(f"    行{i+1}: {line.strip()} (格式可能有误)")
            else:
                print(f"  ⚠️  空文件")

        except Exception as e:
            print(f"  ❌ 读取错误: {e}")

def check_original_images():
    """检查原始图像"""
    print("\n" + "=" * 80)
    print("🔍 检查原始图像")
    print("=" * 80)

    orig_dir = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/infrared/images'

    if not os.path.exists(orig_dir):
        print(f"❌ 目录不存在: {orig_dir}")
        return

    files = sorted(os.listdir(orig_dir))[:3]

    for fname in files:
        if not fname.endswith('.png'):
            continue

        fpath = os.path.join(orig_dir, fname)
        print(f"\n📄 {fname}")

        img = cv2.imread(fpath)
        if img is not None:
            print(f"  尺寸: {img.shape}")
            print(f"  像素均值: {img.mean():.2f}")
            print(f"  像素范围: [{img.min()}, {img.max()}]")
            print(f"  ✅ 图像正常")
        else:
            print(f"  ❌ 无法读取")

def main():
    print("🔍 数据集诊断工具")
    print("=" * 80)

    check_original_images()
    check_images()
    check_masks()
    check_yolo_labels()

    print("\n" + "=" * 80)
    print("💡 建议:")
    print("=" * 80)
    print("1. 如果原始图像正常，但软链接的图像全黑:")
    print("   - 软链接可能失败")
    print("   - 建议改为复制实际文件")
    print("\n2. 如果所有 mask 都是全黑:")
    print("   - 检查 TARGET_CLASS_IDS 是否正确")
    print("   - 检查 YOLO 标签格式")
    print("   - 检查图像尺寸检测")
    print("=" * 80)

if __name__ == "__main__":
    main()
