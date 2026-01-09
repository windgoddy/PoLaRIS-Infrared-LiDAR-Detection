#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外图像归一化工具

功能：
- 读取 16-bit 红外原始图像
- 进行 Min-Max 归一化到 0-255 (8-bit)
- 保存为标准的 8-bit PNG 图像

红外图像特性：
- 原始数据通常是 16-bit (0-65535)
- 实际有效像素值集中在很窄的范围（例如 2000-3000）
- 需要拉伸对比度到 0-255 才能正常显示和处理
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def normalize_infrared_image(image_path, method='minmax', percentile_range=(2, 98)):
    """
    归一化红外图像（16-bit → 8-bit）

    Args:
        image_path: 图像路径
        method: 归一化方法
            - 'minmax': 简单的 Min-Max 归一化
            - 'percentile': 使用百分位数进行鲁棒归一化（推荐）
            - 'clahe': 对比度受限的自适应直方图均衡化
        percentile_range: 百分位数范围 (min, max)，用于 percentile 方法

    Returns:
        normalized_image: 8-bit 归一化图像，如果失败返回 None
    """
    try:
        # 读取图像（保留原始位深度）
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None

        # 检查图像位深度
        if img.dtype == np.uint8:
            # 已经是 8-bit，直接返回
            return img

        # 转换为浮点数
        img_float = img.astype(np.float32)

        if method == 'minmax':
            # 方法 1: 简单 Min-Max 归一化
            img_min = img_float.min()
            img_max = img_float.max()

            if img_max == img_min:
                # 图像全是相同值
                normalized = np.zeros_like(img_float, dtype=np.uint8)
            else:
                normalized = ((img_float - img_min) / (img_max - img_min) * 255.0)
                normalized = normalized.astype(np.uint8)

        elif method == 'percentile':
            # 方法 2: 百分位数归一化（更鲁棒，推荐）
            # 使用 2% 和 98% 百分位数来避免异常值影响
            p_low, p_high = percentile_range
            vmin = np.percentile(img_float, p_low)
            vmax = np.percentile(img_float, p_high)

            if vmax == vmin:
                normalized = np.zeros_like(img_float, dtype=np.uint8)
            else:
                # 裁剪并归一化
                img_clipped = np.clip(img_float, vmin, vmax)
                normalized = ((img_clipped - vmin) / (vmax - vmin) * 255.0)
                normalized = normalized.astype(np.uint8)

        elif method == 'clahe':
            # 方法 3: CLAHE (对比度受限的自适应直方图均衡化)
            # 先做基本归一化
            vmin = np.percentile(img_float, 2)
            vmax = np.percentile(img_float, 98)

            if vmax == vmin:
                normalized = np.zeros_like(img_float, dtype=np.uint8)
            else:
                img_clipped = np.clip(img_float, vmin, vmax)
                img_norm = ((img_clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)

                # 应用 CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                normalized = clahe.apply(img_norm)

        else:
            raise ValueError(f"Unknown method: {method}")

        return normalized

    except Exception as e:
        print(f"❌ 处理图像时出错 {image_path}: {e}")
        return None

def batch_normalize_images(source_dir, output_dir, method='percentile', overwrite=False):
    """
    批量归一化图像

    Args:
        source_dir: 源图像目录
        output_dir: 输出目录
        method: 归一化方法
        overwrite: 是否覆盖已存在的文件
    """
    print("=" * 80)
    print("🔥 红外图像归一化工具")
    print("=" * 80)
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"归一化方法: {method}")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"❌ 未找到图像文件在: {source_dir}")
        return

    print(f"\n发现 {len(image_files)} 个图像文件")
    print("开始处理...\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    # 记录第一张图像的统计信息
    first_stats = None

    for filename in tqdm(image_files, desc="处理进度"):
        input_path = os.path.join(source_dir, filename)

        # 输出文件名（统一为 .png）
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)

        # 检查是否需要跳过
        if os.path.exists(output_path) and not overwrite:
            skip_count += 1
            continue

        # 归一化
        normalized = normalize_infrared_image(input_path, method=method)

        if normalized is None:
            error_count += 1
            continue

        # 保存
        cv2.imwrite(output_path, normalized)
        success_count += 1

        # 记录第一张图像的统计信息
        if first_stats is None:
            original = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            first_stats = {
                'filename': filename,
                'original_dtype': original.dtype,
                'original_shape': original.shape,
                'original_range': (original.min(), original.max()),
                'original_mean': original.mean(),
                'normalized_range': (normalized.min(), normalized.max()),
                'normalized_mean': normalized.mean()
            }

    # 总结
    print(f"\n📊 处理完成:")
    print(f"  ✅ 成功: {success_count} 张")
    print(f"  ⏭️  跳过: {skip_count} 张")
    print(f"  ❌ 失败: {error_count} 张")

    # 显示示例统计
    if first_stats:
        print(f"\n📝 示例图像统计 ({first_stats['filename']}):")
        print(f"  原始:")
        print(f"    数据类型: {first_stats['original_dtype']}")
        print(f"    形状: {first_stats['original_shape']}")
        print(f"    像素范围: {first_stats['original_range']}")
        print(f"    像素均值: {first_stats['original_mean']:.2f}")
        print(f"  归一化后:")
        print(f"    像素范围: {first_stats['normalized_range']}")
        print(f"    像素均值: {first_stats['normalized_mean']:.2f}")

    print("\n" + "=" * 80)
    print("✅ 批量归一化完成！")
    print("=" * 80)

def main():
    """主函数 - 命令行使用"""
    import argparse

    parser = argparse.ArgumentParser(description='红外图像归一化工具 (16-bit → 8-bit)')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='源图像目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--method', type=str, default='percentile',
                        choices=['minmax', 'percentile', 'clahe'],
                        help='归一化方法 (default: percentile)')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的文件')

    args = parser.parse_args()

    batch_normalize_images(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        method=args.method,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()
