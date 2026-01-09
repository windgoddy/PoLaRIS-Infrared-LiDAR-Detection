#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 Pohang-Canal-all 数据集生成 masks

功能：
1. 读取 split_data/train.txt 和 test.txt（过滤后的数据）
2. 从原始 JSON 标签生成二值掩码
3. 保存到 dataset/Pohang-Canal-all/masks/

优势：
- 只处理有效数据（train.txt 和 test.txt 中列出的）
- 速度快，不浪费时间处理无标签图像
"""

import os
import json
import numpy as np
import cv2
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 数据集根目录
DATASET_ROOT = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all'

# 2. 原始 JSON 标签目录
RAW_LABEL_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/00/all/00_inf_labels'

# 3. LabelMe 中的目标类别名称
# 如果标签中使用的是 'boat'、'target'、'vessel' 等，请修改这里
TARGET_LABELS = ['boat', 'target', 'vessel', '0']  # 支持多个可能的标签名
# ===========================================

def json_to_mask(json_path, target_labels):
    """
    将 LabelMe JSON 转换为二值掩码

    Args:
        json_path: JSON 文件路径
        target_labels: 目标类别名称列表

    Returns:
        mask: 二值掩码 (numpy array)，如果失败返回 None
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取图像尺寸
        h = data.get('imageHeight')
        w = data.get('imageWidth')

        if h is None or w is None:
            print(f"Warning: Missing image dimensions in {json_path}")
            return None

        # 创建黑色背景
        mask = np.zeros((h, w), dtype=np.uint8)

        # 遍历所有形状
        if 'shapes' not in data:
            # 空标签（背景图像）
            return mask

        for shape in data['shapes']:
            label = shape.get('label', '')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', 'polygon')

            # 检查是否为目标类别
            if label in target_labels:
                points_array = np.array(points, dtype=np.int32)

                if shape_type == 'polygon':
                    cv2.fillPoly(mask, [points_array], 255)
                elif shape_type == 'rectangle':
                    if len(points_array) >= 2:
                        pt1 = tuple(points_array[0])
                        pt2 = tuple(points_array[1])
                        cv2.rectangle(mask, pt1, pt2, 255, -1)
                elif shape_type == 'circle':
                    if len(points_array) >= 2:
                        center = tuple(points_array[0].astype(int))
                        edge = tuple(points_array[1].astype(int))
                        radius = int(np.linalg.norm(np.array(center) - np.array(edge)))
                        cv2.circle(mask, center, radius, 255, -1)

        return mask

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None

def load_split_list(split_file):
    """
    加载 train.txt 或 test.txt

    Args:
        split_file: split 文件路径

    Returns:
        list: 文件名列表（不含扩展名）
    """
    if not os.path.exists(split_file):
        return []

    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    print("=" * 80)
    print("🏷️  标签准备脚本 - Pohang-Canal-all")
    print("=" * 80)
    print(f"数据集位置: {DATASET_ROOT}")
    print(f"标签源位置: {RAW_LABEL_DIR}")
    print(f"目标类别: {TARGET_LABELS}")
    print("=" * 80)

    # 1. 创建输出目录
    masks_dir = os.path.join(DATASET_ROOT, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    print(f"\n✅ 输出目录: {masks_dir}")

    # 2. 加载 split 列表（只处理有效数据）
    print("\n>>> 步骤 1/3: 加载数据集索引...")
    split_data_dir = os.path.join(DATASET_ROOT, 'split_data')

    train_list = load_split_list(os.path.join(split_data_dir, 'train.txt'))
    test_list = load_split_list(os.path.join(split_data_dir, 'test.txt'))

    if not train_list and not test_list:
        print("❌ 错误: 未找到 train.txt 或 test.txt")
        print("   请先运行 filter_dataset.py")
        return

    all_frames = train_list + test_list
    print(f"✅ 训练集: {len(train_list)} 张")
    print(f"✅ 测试集: {len(test_list)} 张")
    print(f"✅ 总计需要处理: {len(all_frames)} 张")

    # 3. 批处理生成 masks
    print("\n>>> 步骤 2/3: 生成二值掩码...")

    success_count = 0
    skip_count = 0
    error_count = 0

    for frame_name in tqdm(all_frames, desc="处理进度"):
        # 构建路径
        json_path = os.path.join(RAW_LABEL_DIR, frame_name + '.json')
        mask_path = os.path.join(masks_dir, frame_name + '.png')

        # 检查 JSON 是否存在
        if not os.path.exists(json_path):
            skip_count += 1
            continue

        # 转换为 mask
        mask = json_to_mask(json_path, TARGET_LABELS)

        if mask is None:
            error_count += 1
            continue

        # 保存
        cv2.imwrite(mask_path, mask)
        success_count += 1

    # 4. 总结
    print(f"\n>>> 步骤 3/3: 总结")
    print(f"✅ 成功生成: {success_count} 张")
    print(f"⚠️  跳过（未找到JSON）: {skip_count} 张")
    print(f"❌ 处理失败: {error_count} 张")

    if error_count > 0:
        print("\n💡 提示: 如果有处理失败的文件，请检查上面的错误信息")

    # 5. 验证
    print("\n>>> 验证...")
    mask_files = os.listdir(masks_dir)
    print(f"✅ masks 文件夹中共有 {len(mask_files)} 个文件")

    # 随机检查几个 mask 的统计信息
    if mask_files:
        sample_mask_path = os.path.join(masks_dir, mask_files[0])
        sample_mask = cv2.imread(sample_mask_path, cv2.IMREAD_GRAYSCALE)
        if sample_mask is not None:
            unique_values = np.unique(sample_mask)
            print(f"✅ 示例 mask ({mask_files[0]}):")
            print(f"   - 尺寸: {sample_mask.shape}")
            print(f"   - 像素值范围: {unique_values}")
            print(f"   - 前景像素数: {np.sum(sample_mask > 0)}")

    print("\n" + "=" * 80)
    print("✅ 标签准备完成！")
    print("=" * 80)
    print("\n📝 下一步:")
    print("  1. (可选) 生成 oracle masks: python scripts/generate_oracle_masks.py")
    print("  2. 开始训练: ./scripts/run_Phase3_improved_v3.sh")
    print("\n💡 提示:")
    print("  - masks 已保存到: dataset/Pohang-Canal-all/masks/")
    print("  - 每个 mask 是 0(背景) 和 255(前景) 的二值图像")
    print("=" * 80)

if __name__ == "__main__":
    main()
