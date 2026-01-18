#!/usr/bin/env python3
"""
最终数据集生成脚本
根据分类结果筛选并划分训练集/测试集

策略：
- Label 0: 全部丢弃
- Label 1 (简单): 均匀时间降采样 1000 张
- Label 2 (困难极小): 均匀时间降采样 1000 张
- Label 3 (困难岸边): 随机采样 1000 张
- Block-wise 划分: 10个时间块，每块内前80%训练，后20%测试
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict


def natural_key(filename):
    """
    自然排序键函数
    将文件名中的数字转换为整数以实现正确的数字排序
    例如：00_9.png < 00_10.png
    """
    parts = filename.replace('.png', '').split('_')
    if len(parts) == 2:
        try:
            return (int(parts[0]), int(parts[1]))
        except:
            pass
    # 降级为字符串排序
    return (filename, 0)


def uniform_sampling(items, target_count):
    """
    均匀采样：从列表中均匀选取 target_count 个元素

    Args:
        items: 已排序的列表
        target_count: 目标采样数量

    Returns:
        采样后的列表
    """
    if len(items) <= target_count:
        return items

    # 使用 linspace 生成均匀分布的索引
    indices = np.linspace(0, len(items) - 1, target_count).astype(int)
    return [items[i] for i in indices]


def random_sampling(items, target_count, seed=42):
    """
    随机采样：从列表中随机选取 target_count 个元素

    Args:
        items: 列表
        target_count: 目标采样数量
        seed: 随机种子（保证可复现性）

    Returns:
        采样后的列表
    """
    if len(items) <= target_count:
        return items

    np.random.seed(seed)
    indices = np.random.choice(len(items), target_count, replace=False)
    # 对索引排序以保持相对时间顺序
    indices = np.sort(indices)
    return [items[i] for i in indices]


def load_classification_results(summary_file):
    """
    加载分类结果

    Args:
        summary_file: 分类结果文件路径

    Returns:
        按标签分组的字典 {label_id: [filenames]}
    """
    label_groups = defaultdict(list)

    with open(summary_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('|')
            if len(parts) != 2:
                continue

            filename = parts[0].strip()
            label_id = int(parts[1].strip())

            label_groups[label_id].append(filename)

    return label_groups


def sample_images(label_groups):
    """
    按照策略采样图像

    策略：
    - Label 0: 全部丢弃
    - Label 1 (简单): 按时间顺序均匀采样 1000 张
    - Label 2 (困难极小): 按时间顺序均匀采样 1000 张
    - Label 3 (困难岸边): 随机采样 1000 张

    Args:
        label_groups: 按标签分组的字典

    Returns:
        采样后的图像列表 [(filename, label_id), ...]
    """
    sampled_images = []

    # Label 0: 丢弃
    label0_count = len(label_groups.get(0, []))
    print(f"\n标签0 (其他): {label0_count} 张 -> 全部丢弃")

    # Label 1: 均匀时间采样 1000 张
    label1_images = sorted(label_groups.get(1, []), key=natural_key)
    label1_sampled = uniform_sampling(label1_images, 1000)
    sampled_images.extend([(img, 1) for img in label1_sampled])
    print(f"标签1 (简单): {len(label1_images)} 张 -> 均匀时间采样 {len(label1_sampled)} 张")

    # Label 2: 均匀时间采样 1000 张
    label2_images = sorted(label_groups.get(2, []), key=natural_key)
    label2_sampled = uniform_sampling(label2_images, 1000)
    sampled_images.extend([(img, 2) for img in label2_sampled])
    print(f"标签2 (困难极小): {len(label2_images)} 张 -> 均匀时间采样 {len(label2_sampled)} 张")

    # Label 3: 随机采样 1000 张
    label3_images = sorted(label_groups.get(3, []), key=natural_key)
    label3_sampled = random_sampling(label3_images, 1000)
    sampled_images.extend([(img, 3) for img in label3_sampled])
    print(f"标签3 (困难岸边): {len(label3_images)} 张 -> 随机采样 {len(label3_sampled)} 张")

    print(f"\n总采样数量: {len(sampled_images)} 张")

    return sampled_images


def block_split(images, num_blocks=10, train_ratio=0.8):
    """
    Block-wise 划分训练集和测试集
    防止视频数据的时间泄露 (Data Leakage)

    Args:
        images: 图像列表 [(filename, label_id), ...]
        num_blocks: 时间块数量
        train_ratio: 训练集比例

    Returns:
        train_set, test_set (每个元素为 (filename, label_id))
    """
    # 按文件名自然排序（恢复时间顺序）
    images_sorted = sorted(images, key=lambda x: natural_key(x[0]))

    total_count = len(images_sorted)
    block_size = total_count // num_blocks

    train_set = []
    test_set = []

    for i in range(num_blocks):
        # 计算当前块的范围
        start_idx = i * block_size
        if i == num_blocks - 1:
            # 最后一个块包含所有剩余图像
            end_idx = total_count
        else:
            end_idx = (i + 1) * block_size

        block = images_sorted[start_idx:end_idx]

        # 在块内按 8:2 划分
        train_count = int(len(block) * train_ratio)

        train_set.extend(block[:train_count])
        test_set.extend(block[train_count:])

        # 打印每个块的统计
        print(f"  块 {i+1:2d}: {len(block):4d} 张 -> Train: {train_count:4d}, Test: {len(block)-train_count:4d}")

    return train_set, test_set


def print_statistics(train_set, test_set):
    """
    打印详细的统计报告
    """
    print("\n" + "="*70)
    print("最终数据集统计报告")
    print("="*70)

    # 总数统计
    total_train = len(train_set)
    total_test = len(test_set)
    total = total_train + total_test

    print(f"\n总数量: {total} 张")
    print(f"  训练集: {total_train} 张 ({total_train/total*100:.1f}%)")
    print(f"  测试集: {total_test} 张 ({total_test/total*100:.1f}%)")

    # 各标签统计
    train_labels = defaultdict(int)
    test_labels = defaultdict(int)

    for _, label_id in train_set:
        train_labels[label_id] += 1

    for _, label_id in test_set:
        test_labels[label_id] += 1

    print("\n训练集标签分布:")
    for label_id in sorted(train_labels.keys()):
        count = train_labels[label_id]
        pct = count / total_train * 100
        label_name = {1: "简单", 2: "困难极小", 3: "困难岸边"}.get(label_id, "未知")
        print(f"  标签{label_id} ({label_name:8s}): {count:4d} 张 ({pct:5.1f}%)")

    print("\n测试集标签分布:")
    for label_id in sorted(test_labels.keys()):
        count = test_labels[label_id]
        pct = count / total_test * 100
        label_name = {1: "简单", 2: "困难极小", 3: "困难岸边"}.get(label_id, "未知")
        print(f"  标签{label_id} ({label_name:8s}): {count:4d} 张 ({pct:5.1f}%)")

    # 验证是否所有标签都被覆盖
    print("\n标签覆盖检查:")
    train_labels_set = set(train_labels.keys())
    test_labels_set = set(test_labels.keys())

    if train_labels_set == {1, 2, 3}:
        print("  ✓ 训练集覆盖所有标签 (1, 2, 3)")
    else:
        missing = {1, 2, 3} - train_labels_set
        print(f"  ✗ 训练集缺失标签: {missing}")

    if test_labels_set == {1, 2, 3}:
        print("  ✓ 测试集覆盖所有标签 (1, 2, 3)")
    else:
        missing = {1, 2, 3} - test_labels_set
        print(f"  ✗ 测试集缺失标签: {missing}")

    print("="*70)


def save_split_files(train_set, test_set, output_dir):
    """
    保存训练集和测试集文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存训练集
    train_file = os.path.join(output_dir, 'train.txt')
    with open(train_file, 'w') as f:
        for filename, _ in train_set:
            f.write(f"{filename}\n")

    # 保存测试集
    test_file = os.path.join(output_dir, 'test.txt')
    with open(test_file, 'w') as f:
        for filename, _ in test_set:
            f.write(f"{filename}\n")

    print(f"\n文件已保存:")
    print(f"  训练集: {train_file}")
    print(f"  测试集: {test_file}")


def main():
    # 获取项目根目录
    project_root = Path(__file__).parent.parent

    # 输入文件
    summary_file = project_root / 'dataset' / 'select-view' / 'selection_summary_new.txt'

    # 输出目录
    output_dir = project_root / 'dataset' / 'split_data'

    # 检查输入文件是否存在
    if not summary_file.exists():
        print(f"错误: 分类结果文件不存在: {summary_file}")
        print("请先运行 classify_images.py 生成分类结果")
        return

    print("="*70)
    print("最终数据集生成脚本")
    print("="*70)
    print(f"\n输入文件: {summary_file}")
    print(f"输出目录: {output_dir}")

    # 1. 加载分类结果
    print("\n[1/4] 加载分类结果...")
    label_groups = load_classification_results(summary_file)

    print(f"\n原始分类统计:")
    for label_id in sorted(label_groups.keys()):
        count = len(label_groups[label_id])
        label_name = {0: "其他", 1: "简单", 2: "困难极小", 3: "困难岸边"}.get(label_id, "未知")
        print(f"  标签{label_id} ({label_name:8s}): {count} 张")

    # 2. 采样图像
    print("\n[2/4] 按策略采样图像...")
    sampled_images = sample_images(label_groups)

    # 3. Block-wise 划分
    print("\n[3/4] Block-wise 划分训练集/测试集...")
    print("  策略: 切分为 10 个时间块，每块内前 80% 为训练集，后 20% 为测试集")
    print()

    train_set, test_set = block_split(sampled_images, num_blocks=10, train_ratio=0.8)

    # 4. 打印统计报告
    print_statistics(train_set, test_set)

    # 5. 保存文件
    print("\n[4/4] 保存文件...")
    save_split_files(train_set, test_set, output_dir)

    print("\n✓ 数据集生成完成！")


if __name__ == "__main__":
    main()
