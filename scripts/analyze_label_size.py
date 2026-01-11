#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 标签尺寸分析工具

功能：
1. 分析所有 YOLO 标签的尺寸分布
2. 可视化极小目标（可能的标注噪声）
3. 生成统计报告和建议阈值
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== 配置 ====================
LABEL_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/00/all/00_inf_labels'
IMAGE_DIR = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all/images'
OUTPUT_DIR = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/debug_tiny_objects'

IMG_W, IMG_H = 640, 512

# 三维度过滤阈值（可调整）
AREA_THRESHOLD = 15          # 极小面积阈值（像素）
ASPECT_RATIO_THRESHOLD = 5.0  # 极端长宽比阈值
MIN_SIDE_THRESHOLD = 2        # 极窄边长阈值（像素）

# 对比分析的阈值
AREA_THRESHOLDS = [9, 15, 25, 50]  # 多个阈值对比

# 可视化样本数量
MAX_SAMPLES = 100
# ==============================================

def analyze_label_sizes():
    """分析标签尺寸分布"""
    print("=" * 80)
    print("🔍 YOLO 标签尺寸分析")
    print("=" * 80)
    print(f"标签目录: {LABEL_DIR}")
    print(f"图像目录: {IMAGE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有标签文件
    txt_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    print(f"\n发现 {len(txt_files)} 个标签文件")

    # 统计数据
    all_areas = []
    all_widths = []
    all_heights = []
    all_aspect_ratios = []

    # 三维度过滤统计
    filter_stats = {
        'tiny_area': 0,        # 面积 < 15
        'extreme_ratio': 0,     # 长宽比 > 5
        'thin_side': 0,         # 最短边 < 2
        'combined': 0,          # 满足任一条件（最终被过滤）
    }

    tiny_samples = {threshold: [] for threshold in AREA_THRESHOLDS}
    noise_samples = []  # 三维度判定的噪声样本
    total_objects = 0

    print("\n>>> 步骤 1/3: 分析标签尺寸...")

    for txt_file in tqdm(txt_files, desc="分析进度"):
        txt_path = os.path.join(LABEL_DIR, txt_file)

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            continue

        img_name = txt_file.replace('.txt', '.png')
        img_path = os.path.join(IMAGE_DIR, img_name)

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            total_objects += 1

            # YOLO 格式: class x_c y_c w h (归一化 0-1)
            w_norm = float(parts[3])
            h_norm = float(parts[4])

            # 转换为像素
            w_pix = w_norm * IMG_W
            h_pix = h_norm * IMG_H
            area = w_pix * h_pix

            all_areas.append(area)
            all_widths.append(w_pix)
            all_heights.append(h_pix)

            aspect_ratio = w_pix / h_pix if h_pix > 0 else 1.0
            all_aspect_ratios.append(aspect_ratio)

            # 三维度噪声判断
            is_tiny_area = area < AREA_THRESHOLD
            is_extreme_ratio = aspect_ratio > ASPECT_RATIO_THRESHOLD or aspect_ratio < (1.0 / ASPECT_RATIO_THRESHOLD)
            is_thin_side = min(w_pix, h_pix) < MIN_SIDE_THRESHOLD
            is_noise = is_tiny_area or is_extreme_ratio or is_thin_side

            # 统计各维度
            if is_tiny_area:
                filter_stats['tiny_area'] += 1
            if is_extreme_ratio:
                filter_stats['extreme_ratio'] += 1
            if is_thin_side:
                filter_stats['thin_side'] += 1
            if is_noise:
                filter_stats['combined'] += 1

                # 保存噪声样本
                if len(noise_samples) < MAX_SAMPLES:
                    noise_samples.append({
                        'txt_file': txt_file,
                        'img_path': img_path,
                        'area': area,
                        'width': w_pix,
                        'height': h_pix,
                        'aspect_ratio': aspect_ratio,
                        'x_c': float(parts[1]),
                        'y_c': float(parts[2]),
                        'reason': []
                    })
                    if is_tiny_area:
                        noise_samples[-1]['reason'].append(f'tiny_area<{AREA_THRESHOLD}')
                    if is_extreme_ratio:
                        noise_samples[-1]['reason'].append(f'ratio>{ASPECT_RATIO_THRESHOLD}')
                    if is_thin_side:
                        noise_samples[-1]['reason'].append(f'side<{MIN_SIDE_THRESHOLD}')

            # 检查是否为极小目标（旧的分析，保留对比）
            for threshold in AREA_THRESHOLDS:
                if area < threshold:
                    if len(tiny_samples[threshold]) < MAX_SAMPLES:
                        tiny_samples[threshold].append({
                            'txt_file': txt_file,
                            'img_path': img_path,
                            'area': area,
                            'width': w_pix,
                            'height': h_pix,
                            'x_c': float(parts[1]),
                            'y_c': float(parts[2])
                        })

    # 转换为 numpy 数组
    all_areas = np.array(all_areas)
    all_widths = np.array(all_widths)
    all_heights = np.array(all_heights)
    all_aspect_ratios = np.array(all_aspect_ratios)

    print(f"✅ 分析完成：共 {total_objects} 个目标")

    # 统计报告
    print("\n>>> 步骤 2/3: 生成统计报告...")
    print("\n" + "=" * 80)
    print("📊 统计报告")
    print("=" * 80)

    print(f"\n总目标数量: {total_objects}")
    print(f"\n面积统计 (像素):")
    print(f"  最小值: {all_areas.min():.2f}")
    print(f"  最大值: {all_areas.max():.2f}")
    print(f"  均值: {all_areas.mean():.2f}")
    print(f"  中位数: {np.median(all_areas):.2f}")
    print(f"  标准差: {all_areas.std():.2f}")

    print(f"\n尺寸统计 (像素):")
    print(f"  宽度 - 均值: {all_widths.mean():.2f}, 中位数: {np.median(all_widths):.2f}")
    print(f"  高度 - 均值: {all_heights.mean():.2f}, 中位数: {np.median(all_heights):.2f}")

    print(f"\n长宽比统计:")
    print(f"  均值: {all_aspect_ratios.mean():.2f}")
    print(f"  中位数: {np.median(all_aspect_ratios):.2f}")
    print(f"  最大值: {all_aspect_ratios.max():.2f}")
    print(f"  最小值: {all_aspect_ratios.min():.2f}")

    # 三维度过滤统计
    print("\n" + "=" * 80)
    print("🎯 三维度噪声过滤统计（推荐方案）")
    print("=" * 80)
    print(f"\n过滤规则:")
    print(f"  1. 极小面积: area < {AREA_THRESHOLD} 像素")
    print(f"  2. 极端长宽比: ratio > {ASPECT_RATIO_THRESHOLD} 或 < {1.0/ASPECT_RATIO_THRESHOLD:.2f}")
    print(f"  3. 极窄边长: min(w, h) < {MIN_SIDE_THRESHOLD} 像素")
    print(f"\n过滤统计 (满足任一条件即过滤):")
    print(f"  维度1 - 极小面积: {filter_stats['tiny_area']} 个 ({filter_stats['tiny_area']/total_objects*100:.2f}%)")
    print(f"  维度2 - 极端长宽比: {filter_stats['extreme_ratio']} 个 ({filter_stats['extreme_ratio']/total_objects*100:.2f}%)")
    print(f"  维度3 - 极窄边长: {filter_stats['thin_side']} 个 ({filter_stats['thin_side']/total_objects*100:.2f}%)")
    print(f"  ───────────────────────────────────")
    print(f"  📊 总计将过滤: {filter_stats['combined']} 个 ({filter_stats['combined']/total_objects*100:.2f}%)")
    print(f"  ✅ 保留: {total_objects - filter_stats['combined']} 个 ({(1-filter_stats['combined']/total_objects)*100:.2f}%)")

    # 不同阈值下的统计（旧方法对比）
    print("\n" + "=" * 80)
    print("📋 单维度面积阈值对比（仅供参考）")
    print("=" * 80)
    print(f"{'阈值 (像素)':<15} {'数量':<10} {'占比':<10} {'建议'}")
    print("-" * 60)

    for threshold in AREA_THRESHOLDS:
        count = np.sum(all_areas < threshold)
        ratio = count / total_objects * 100
        suggestion = "⚠️  考虑过滤" if ratio < 5 else "✅ 保守" if ratio < 15 else "❌ 过于激进"
        print(f"面积 < {threshold:<9} {count:<10} {ratio:>5.2f}%     {suggestion}")

    # 推荐阈值
    print("\n" + "=" * 80)
    print("💡 推荐阈值设置")
    print("=" * 80)

    # 使用 1% 和 5% 分位数作为参考
    p1 = np.percentile(all_areas, 1)
    p5 = np.percentile(all_areas, 5)

    print(f"\n基于分位数分析:")
    print(f"  1% 分位数: {p1:.2f} 像素")
    print(f"  5% 分位数: {p5:.2f} 像素")

    print(f"\n推荐策略:")
    print(f"  1. 保守策略（推荐）: 面积 < 15 像素 或 边长 < 3 像素")
    print(f"     - 过滤约 {np.sum(all_areas < 15) / total_objects * 100:.2f}% 的目标")
    print(f"     - 适用于初次训练，保留更多数据")
    print(f"")
    print(f"  2. 激进策略: 面积 < 25 像素 或 边长 < 5 像素")
    print(f"     - 过滤约 {np.sum(all_areas < 25) / total_objects * 100:.2f}% 的目标")
    print(f"     - 适用于已确认小目标为噪声的情况")

    # 可视化
    print("\n>>> 步骤 3/3: 生成可视化...")

    # 1. 面积分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 面积分布（全范围）
    axes[0, 0].hist(all_areas, bins=100, color='blue', alpha=0.7, edgecolor='black')
    for threshold in AREA_THRESHOLDS[:2]:
        axes[0, 0].axvline(x=threshold, color='red', linestyle='--',
                          label=f'Threshold={threshold}', linewidth=2)
    axes[0, 0].set_title('Area Distribution (Full Range)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Area (pixels)', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 面积分布（0-200 放大）
    axes[0, 1].hist(all_areas[all_areas < 200], bins=100, color='green', alpha=0.7, edgecolor='black')
    for threshold in AREA_THRESHOLDS[:2]:
        axes[0, 1].axvline(x=threshold, color='red', linestyle='--',
                          label=f'Threshold={threshold}', linewidth=2)
    axes[0, 1].set_title('Area Distribution (0-200 pixels, Zoomed)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Area (pixels)', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 宽度 vs 高度散点图
    axes[1, 0].scatter(all_widths, all_heights, alpha=0.3, s=10)
    axes[1, 0].axhline(y=MIN_SIDE_THRESHOLD, color='red', linestyle='--', label=f'Min side={MIN_SIDE_THRESHOLD}')
    axes[1, 0].axvline(x=MIN_SIDE_THRESHOLD, color='red', linestyle='--')
    axes[1, 0].set_title('Width vs Height Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Width (pixels)', fontsize=12)
    axes[1, 0].set_ylabel('Height (pixels)', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 长宽比分布
    axes[1, 1].hist(all_aspect_ratios[all_aspect_ratios < 10], bins=50,
                    color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=1.0, color='red', linestyle='--', label='Square (1:1)', linewidth=2)
    axes[1, 1].set_title('Aspect Ratio Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Aspect Ratio (width/height)', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'label_size_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"✅ 统计图已保存: {os.path.join(OUTPUT_DIR, 'label_size_analysis.png')}")

    # 2. 保存三维度噪声样本
    print("\n>>> 保存三维度噪声样本（用于人工检查）...")

    if noise_samples:
        noise_dir = os.path.join(OUTPUT_DIR, 'three_dimension_noise')
        os.makedirs(noise_dir, exist_ok=True)

        saved_count = 0
        for sample in tqdm(noise_samples[:100], desc="三维度噪声"):
            img_path = sample['img_path']
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            # 计算坐标
            x_c = sample['x_c'] * IMG_W
            y_c = sample['y_c'] * IMG_H
            w = sample['width']
            h = sample['height']

            x1 = int(x_c - w/2)
            y1 = int(y_c - h/2)
            x2 = int(x_c + w/2)
            y2 = int(y_c + h/2)

            # 外扩裁剪（方便观察上下文）
            pad = 30
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(IMG_W, x2 + pad)
            crop_y2 = min(IMG_H, y2 + pad)

            crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            # 检查裁剪是否有效
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            # 在裁剪图上画框
            box_x1_in_crop = max(0, x1 - crop_x1)
            box_y1_in_crop = max(0, y1 - crop_y1)
            box_x2_in_crop = min(crop.shape[1] - 1, x2 - crop_x1)
            box_y2_in_crop = min(crop.shape[0] - 1, y2 - crop_y1)

            # 确保坐标有效
            if box_x2_in_crop > box_x1_in_crop and box_y2_in_crop > box_y1_in_crop:
                cv2.rectangle(crop, (box_x1_in_crop, box_y1_in_crop),
                             (box_x2_in_crop, box_y2_in_crop), (0, 0, 255), 2)
            else:
                continue

            # 添加文本标注（显示过滤原因）
            reason_str = '+'.join(sample['reason'])
            text1 = f"Area={sample['area']:.1f} Ratio={sample['aspect_ratio']:.2f}"
            text2 = f"Reason: {reason_str}"
            cv2.putText(crop, text1, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(crop, text2, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (0, 255, 255), 1, cv2.LINE_AA)

            # 保存
            save_name = f"noise_{saved_count:03d}_{reason_str.replace('>', 'gt').replace('<', 'lt')}_{sample['txt_file'].replace('.txt', '.png')}"
            cv2.imwrite(os.path.join(noise_dir, save_name), crop)
            saved_count += 1

        print(f"  ✅ 三维度噪声: 已保存 {saved_count} 个样本到 {noise_dir}")

    # 3. 保存单维度面积样本（供对比）
    print("\n>>> 保存单维度面积样本（供对比）...")

    for threshold in AREA_THRESHOLDS[:2]:  # 只保存前两个阈值的样本
        samples = tiny_samples[threshold]
        if not samples:
            continue

        threshold_dir = os.path.join(OUTPUT_DIR, f'area_only_threshold_{threshold}')
        os.makedirs(threshold_dir, exist_ok=True)

        saved_count = 0
        for sample in tqdm(samples[:50], desc=f"面积<{threshold}"):
            img_path = sample['img_path']
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            # 计算坐标
            x_c = sample['x_c'] * IMG_W
            y_c = sample['y_c'] * IMG_H
            w = sample['width']
            h = sample['height']

            x1 = int(x_c - w/2)
            y1 = int(y_c - h/2)
            x2 = int(x_c + w/2)
            y2 = int(y_c + h/2)

            # 外扩裁剪（方便观察上下文）
            pad = 20
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(IMG_W, x2 + pad)
            crop_y2 = min(IMG_H, y2 + pad)

            crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            # 检查裁剪是否有效
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            # 在裁剪图上画框
            box_x1_in_crop = max(0, x1 - crop_x1)
            box_y1_in_crop = max(0, y1 - crop_y1)
            box_x2_in_crop = min(crop.shape[1] - 1, x2 - crop_x1)
            box_y2_in_crop = min(crop.shape[0] - 1, y2 - crop_y1)

            # 确保坐标有效
            if box_x2_in_crop > box_x1_in_crop and box_y2_in_crop > box_y1_in_crop:
                cv2.rectangle(crop, (box_x1_in_crop, box_y1_in_crop),
                             (box_x2_in_crop, box_y2_in_crop), (0, 0, 255), 2)
            else:
                continue

            # 添加文本标注
            text = f"Area={sample['area']:.1f} {int(w)}x{int(h)}"
            cv2.putText(crop, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # 保存
            save_name = f"tiny_{saved_count:03d}_area{int(sample['area']):.0f}_{sample['txt_file'].replace('.txt', '.png')}"
            cv2.imwrite(os.path.join(threshold_dir, save_name), crop)
            saved_count += 1

        print(f"  ✅ 阈值 {threshold}: 已保存 {saved_count} 个样本到 {threshold_dir}")

    # 最终报告
    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    print(f"\n📁 输出文件:")
    print(f"  - 统计图: {os.path.join(OUTPUT_DIR, 'label_size_analysis.png')}")
    print(f"  - 样本图像: {OUTPUT_DIR}/threshold_*/")
    print(f"\n📝 下一步:")
    print(f"  1. 打开 {OUTPUT_DIR} 查看极小目标样本")
    print(f"  2. 如果确认是噪声，运行过滤脚本:")
    print(f"     python scripts/filter_small_targets.py --area_threshold 15 --side_threshold 3")
    print("=" * 80)

if __name__ == "__main__":
    analyze_label_sizes()
