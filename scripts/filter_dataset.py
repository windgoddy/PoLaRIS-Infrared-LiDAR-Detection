#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集清洗脚本 - 过滤没有标签的图像

功能：
1. 扫描已生成的图像文件
2. 扫描可用的标签文件（JSON格式）
3. 取交集，只保留同时有图像和标签的帧
4. 重新生成 train.txt 和 test.txt（按时间顺序 8:2 划分）

优势：
- 不需要重新运行 prepare_all_data.py（耗时）
- 不删除物理文件，只更新索引
- 执行速度快（几秒钟）
"""

import os
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 当前数据集目录 (Pohang-Canal-all)
TARGET_DATASET_ROOT = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all'

# 2. 原始标签所在的文件夹
RAW_LABEL_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/00/all/00_inf_labels'

# 3. 划分比例
TRAIN_RATIO = 0.8

# 4. 标签文件扩展名（根据实际情况修改）
LABEL_EXT = '.json'  # 如果是 .txt 格式，改为 '.txt'
# ===========================================

def get_frame_names_from_dir(directory, extension):
    """
    从目录中获取所有指定扩展名的文件，返回不带扩展名的文件名集合
    """
    if not os.path.exists(directory):
        return set()

    frame_names = set()
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            # 去掉扩展名
            frame_name = filename[:-len(extension)]
            frame_names.add(frame_name)

    return frame_names

def main():
    print("=" * 80)
    print("🚀 数据集清洗脚本")
    print("=" * 80)
    print(f"数据集位置: {TARGET_DATASET_ROOT}")
    print(f"标签源位置: {RAW_LABEL_DIR}")
    print(f"标签格式: {LABEL_EXT}")
    print(f"训练/测试比例: {TRAIN_RATIO:.0%}/{(1-TRAIN_RATIO):.0%}")
    print("=" * 80)

    # 1. 获取所有已准备好的图像
    images_dir = os.path.join(TARGET_DATASET_ROOT, 'images')
    if not os.path.exists(images_dir):
        print(f"❌ 错误: 找不到 images 文件夹: {images_dir}")
        print("   请先运行 prepare_all_data.py 生成数据集")
        return

    print("\n>>> 步骤 1/4: 扫描已存在的图像...")
    prepared_frames = get_frame_names_from_dir(images_dir, '.png')
    print(f"✅ 发现已准备图像: {len(prepared_frames)} 张")

    # 2. 获取所有可用的标签
    print("\n>>> 步骤 2/4: 扫描可用的标签文件...")
    if not os.path.exists(RAW_LABEL_DIR):
        print(f"❌ 错误: 找不到标签文件夹: {RAW_LABEL_DIR}")
        print("   请检查路径是否正确")
        return

    valid_labels = get_frame_names_from_dir(RAW_LABEL_DIR, LABEL_EXT)
    print(f"✅ 发现可用标签: {len(valid_labels)} 个")

    # 3. 取交集
    print("\n>>> 步骤 3/4: 计算交集...")
    valid_frames = list(prepared_frames & valid_labels)

    if len(valid_frames) == 0:
        print("❌ 错误: 交集为空！")
        print("\n可能的原因：")
        print("  1. 文件名不匹配（请检查图像和标签的文件名是否一致）")
        print("  2. 标签扩展名设置错误（当前设置为 {})".format(LABEL_EXT))
        print("\n调试信息：")
        print(f"  示例图像文件名: {list(prepared_frames)[:3] if prepared_frames else '无'}")
        print(f"  示例标签文件名: {list(valid_labels)[:3] if valid_labels else '无'}")
        return

    # 必须排序！保证时间顺序
    valid_frames.sort()

    final_count = len(valid_frames)
    removed_count = len(prepared_frames) - final_count

    print(f"✅ 交集计算完成")
    print(f"\n📊 清洗结果:")
    print(f"  ├─ 原始图像数: {len(prepared_frames)}")
    print(f"  ├─ 有效标签数: {len(valid_labels)}")
    print(f"  ├─ 最终有效数: {final_count}")
    print(f"  └─ 已过滤无标签图像: {removed_count} 张 ({removed_count/len(prepared_frames)*100:.1f}%)")

    # 4. 重新划分数据集
    print("\n>>> 步骤 4/4: 按时间顺序重新划分数据集...")

    split_idx = int(final_count * TRAIN_RATIO)
    train_list = valid_frames[:split_idx]
    test_list = valid_frames[split_idx:]

    # 保存新的 txt 文件
    split_dir = os.path.join(TARGET_DATASET_ROOT, 'split_data')
    os.makedirs(split_dir, exist_ok=True)

    train_txt_path = os.path.join(split_dir, 'train.txt')
    test_txt_path = os.path.join(split_dir, 'test.txt')

    with open(train_txt_path, 'w') as f:
        f.write('\n'.join(train_list))

    with open(test_txt_path, 'w') as f:
        f.write('\n'.join(test_list))

    print(f"✅ 训练集: {len(train_list)} 张 ({len(train_list)/final_count*100:.0f}%)")
    print(f"✅ 测试集: {len(test_list)} 张 ({len(test_list)/final_count*100:.0f}%)")
    print(f"\n📁 已更新索引文件:")
    print(f"  ├─ {train_txt_path}")
    print(f"  └─ {test_txt_path}")

    # 5. 验证
    print("\n>>> 验证...")
    print(f"✅ 示例训练样本 (前5个):")
    for name in train_list[:5]:
        print(f"  - {name}")

    print("\n" + "=" * 80)
    print("✅ 数据集清洗完成！")
    print("=" * 80)
    print("\n📝 下一步:")
    print("  1. 运行 prepare_labels_for_all.py 生成 masks")
    print("  2. 开始训练")
    print("\n💡 提示:")
    print("  - 已过滤的图像不会被删除，只是不会被训练使用")
    print("  - 如需恢复，删除 split_data/*.txt 重新运行即可")
    print("=" * 80)

if __name__ == "__main__":
    main()
