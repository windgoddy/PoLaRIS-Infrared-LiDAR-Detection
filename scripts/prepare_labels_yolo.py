#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 Pohang-Canal-all 数据集生成 masks (YOLO 格式)

功能：
1. 读取 split_data/train.txt 和 test.txt（过滤后的数据）
2. 从 YOLO 格式的 TXT 标签生成二值掩码
3. 保存到 dataset/Pohang-Canal-all/masks/

YOLO 格式说明：
每行: class_id center_x center_y width height
坐标都是归一化的（0-1之间）
"""

import os
import numpy as np
import cv2
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 数据集根目录
DATASET_ROOT = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all'

# 2. 原始 YOLO 标签目录
RAW_LABEL_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/00/all/00_inf_labels'

# 3. 目标类别 ID（YOLO 中的 class_id）
# 如果只有一个类别（比如 boat），通常 class_id = 0
TARGET_CLASS_IDS = [0]  # 可以是 [0], [0, 1], 等

# 4. 图像尺寸（从实际图像中自动获取，这里是默认值）
DEFAULT_IMG_HEIGHT = 512  # 如果无法读取图像，使用默认值
DEFAULT_IMG_WIDTH = 640
# ===========================================

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

def yolo_to_mask(yolo_path, img_height, img_width, target_class_ids):
    """
    将 YOLO 格式标签转换为二值掩码

    Args:
        yolo_path: YOLO 标签文件路径
        img_height: 图像高度
        img_width: 图像宽度
        target_class_ids: 目标类别 ID 列表

    Returns:
        mask: 二值掩码 (numpy array)，如果失败返回 None
    """
    try:
        # 创建黑色背景
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if not os.path.exists(yolo_path):
            # 空标签（背景图像）
            return mask

        # 读取 YOLO 标签文件
        with open(yolo_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            # 空标签（背景图像）
            return mask

        # 解析每一行
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # 跳过格式错误的行

            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 检查是否为目标类别
            if class_id not in target_class_ids:
                continue

            # 将归一化坐标转换为像素坐标
            center_x_px = int(center_x * img_width)
            center_y_px = int(center_y * img_height)
            width_px = int(width * img_width)
            height_px = int(height * img_height)

            # 计算边界框的左上角和右下角
            x1 = int(center_x_px - width_px / 2)
            y1 = int(center_y_px - height_px / 2)
            x2 = int(center_x_px + width_px / 2)
            y2 = int(center_y_px + height_px / 2)

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            # 在 mask 上绘制填充的矩形
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        return mask

    except Exception as e:
        print(f"Error processing {yolo_path}: {e}")
        return None

def get_image_size(image_path):
    """
    获取图像尺寸

    Args:
        image_path: 图像文件路径

    Returns:
        (height, width): 图像尺寸，如果失败返回默认值
    """
    try:
        img = cv2.imread(image_path)
        if img is not None:
            return img.shape[0], img.shape[1]
    except:
        pass

    return DEFAULT_IMG_HEIGHT, DEFAULT_IMG_WIDTH

def main():
    print("=" * 80)
    print("🏷️  标签准备脚本 - Pohang-Canal-all (YOLO 格式)")
    print("=" * 80)
    print(f"数据集位置: {DATASET_ROOT}")
    print(f"标签源位置: {RAW_LABEL_DIR}")
    print(f"目标类别ID: {TARGET_CLASS_IDS}")
    print(f"默认图像尺寸: {DEFAULT_IMG_HEIGHT}x{DEFAULT_IMG_WIDTH}")
    print("=" * 80)

    # 1. 创建输出目录
    masks_dir = os.path.join(DATASET_ROOT, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    print(f"\n✅ 输出目录: {masks_dir}")

    # 2. 加载 split 列表（只处理有效数据）
    print("\n>>> 步骤 1/4: 加载数据集索引...")
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

    # 3. 获取图像尺寸（从第一张图像）
    print("\n>>> 步骤 2/4: 检测图像尺寸...")
    images_dir = os.path.join(DATASET_ROOT, 'images')
    first_frame = all_frames[0]
    first_image_path = os.path.join(images_dir, first_frame + '.png')

    img_height, img_width = get_image_size(first_image_path)
    print(f"✅ 图像尺寸: {img_height}x{img_width}")

    # 4. 批处理生成 masks
    print("\n>>> 步骤 3/4: 生成二值掩码...")

    success_count = 0
    skip_count = 0
    error_count = 0
    empty_count = 0  # 空标签（背景）

    for frame_name in tqdm(all_frames, desc="处理进度"):
        # 构建路径
        yolo_path = os.path.join(RAW_LABEL_DIR, frame_name + '.txt')
        mask_path = os.path.join(masks_dir, frame_name + '.png')

        # 检查 TXT 是否存在
        if not os.path.exists(yolo_path):
            skip_count += 1
            # 即使没有标签，也创建一个空 mask（全黑）
            empty_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.imwrite(mask_path, empty_mask)
            continue

        # 转换为 mask
        mask = yolo_to_mask(yolo_path, img_height, img_width, TARGET_CLASS_IDS)

        if mask is None:
            error_count += 1
            continue

        # 检查是否为空 mask
        if np.sum(mask) == 0:
            empty_count += 1

        # 保存
        cv2.imwrite(mask_path, mask)
        success_count += 1

    # 5. 总结
    print(f"\n>>> 步骤 4/4: 总结")
    print(f"✅ 成功生成: {success_count} 张")
    print(f"⚠️  跳过（未找到TXT）: {skip_count} 张")
    print(f"❌ 处理失败: {error_count} 张")
    print(f"ℹ️  空标签（背景图像）: {empty_count} 张")

    if error_count > 0:
        print("\n💡 提示: 如果有处理失败的文件，请检查上面的错误信息")

    # 6. 验证
    print("\n>>> 验证...")
    mask_files = os.listdir(masks_dir)
    print(f"✅ masks 文件夹中共有 {len(mask_files)} 个文件")

    # 随机检查几个 mask 的统计信息
    if mask_files:
        # 找一个非空的 mask
        sample_found = False
        for mask_file in mask_files[:10]:
            sample_mask_path = os.path.join(masks_dir, mask_file)
            sample_mask = cv2.imread(sample_mask_path, cv2.IMREAD_GRAYSCALE)
            if sample_mask is not None and np.sum(sample_mask) > 0:
                unique_values = np.unique(sample_mask)
                print(f"✅ 示例 mask ({mask_file}):")
                print(f"   - 尺寸: {sample_mask.shape}")
                print(f"   - 像素值范围: {unique_values}")
                print(f"   - 前景像素数: {np.sum(sample_mask > 0)}")
                print(f"   - 前景比例: {np.sum(sample_mask > 0) / sample_mask.size * 100:.2f}%")
                sample_found = True
                break

        if not sample_found:
            print("⚠️  警告: 前10个 mask 都是空的（全黑），这可能不正常")
            print("   请检查:")
            print("   1. TARGET_CLASS_IDS 是否正确")
            print("   2. YOLO 标签文件格式是否正确")

    print("\n" + "=" * 80)
    print("✅ 标签准备完成！")
    print("=" * 80)
    print("\n📝 下一步:")
    print("  1. (可选) 生成 oracle masks: python scripts/generate_oracle_masks.py")
    print("  2. 开始训练: ./scripts/run_Phase3_improved_v3.sh")
    print("\n💡 提示:")
    print("  - masks 已保存到: dataset/Pohang-Canal-all/masks/")
    print("  - 每个 mask 是 0(背景) 和 255(前景) 的二值图像")
    print("  - YOLO 边界框已转换为填充的矩形区域")
    print("=" * 80)

if __name__ == "__main__":
    main()
