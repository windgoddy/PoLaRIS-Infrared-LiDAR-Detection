#!/usr/bin/env python3
"""
分析50_50_2k和50_50_2k_new测试集的样本重叠情况
以及各自的Cat2/Cat3分布
"""
import os
import sys

sys.path.insert(0, 'model_Mamba')
from model.load_param_data import load_category_mapping


def load_split_samples(split_file):
    """加载split文件中的样本ID"""
    with open(split_file, 'r') as f:
        samples = [line.strip() for line in f if line.strip()]
    return set(samples)


def analyze_category_distribution(samples, category_map):
    """分析样本的类别分布"""
    cat_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for sample in samples:
        cat = category_map.get(sample, 0)
        cat_dist[cat] += 1
    return cat_dist


def main():
    dataset_root = 'dataset'
    dataset_name = 'Pohang-Canal-3k'

    # Load category mapping
    category_map, category_names = load_category_mapping(dataset_root, dataset_name)

    # Load test samples from both splits
    split1_file = f'{dataset_root}/{dataset_name}/50_50_2k/test.txt'
    split2_file = f'{dataset_root}/{dataset_name}/50_50_2k_new/test.txt'

    samples1 = load_split_samples(split1_file)
    samples2 = load_split_samples(split2_file)

    # Calculate overlap
    overlap = samples1 & samples2
    only_in_1 = samples1 - samples2
    only_in_2 = samples2 - samples1

    print("=" * 70)
    print("50_50_2k vs 50_50_2k_new 测试集对比分析")
    print("=" * 70)
    print()

    print(f"50_50_2k 测试样本数: {len(samples1)}")
    print(f"50_50_2k_new 测试样本数: {len(samples2)}")
    print(f"重叠样本数: {len(overlap)} ({len(overlap)/len(samples1)*100:.1f}%)")
    print(f"仅在50_50_2k中: {len(only_in_1)} ({len(only_in_1)/len(samples1)*100:.1f}%)")
    print(f"仅在50_50_2k_new中: {len(only_in_2)} ({len(only_in_2)/len(samples2)*100:.1f}%)")
    print()

    # Analyze category distribution
    dist1 = analyze_category_distribution(samples1, category_map)
    dist2 = analyze_category_distribution(samples2, category_map)
    dist_overlap = analyze_category_distribution(overlap, category_map)

    print("-" * 70)
    print("类别分布对比")
    print("-" * 70)
    print()

    print("50_50_2k 测试集:")
    for cat in [0, 1, 2, 3]:
        print(f"  Cat{cat} ({category_names[cat]}): {dist1[cat]:3d} ({dist1[cat]/len(samples1)*100:5.1f}%)")
    print()

    print("50_50_2k_new 测试集:")
    for cat in [0, 1, 2, 3]:
        print(f"  Cat{cat} ({category_names[cat]}): {dist2[cat]:3d} ({dist2[cat]/len(samples2)*100:5.1f}%)")
    print()

    if overlap:
        print("重叠部分 (两个数据集共有的样本):")
        for cat in [0, 1, 2, 3]:
            print(f"  Cat{cat} ({category_names[cat]}): {dist_overlap[cat]:3d} ({dist_overlap[cat]/len(overlap)*100:5.1f}%)")
        print()

    # Key insight
    print("=" * 70)
    print("关键发现")
    print("=" * 70)
    print()

    if len(overlap) / len(samples1) < 0.5:
        print("⚠️  两个数据集的测试样本重叠率 < 50%")
        print("   这意味着在这两个数据集上的性能**不能直接对比**！")
        print("   不同的测试样本可能有不同的难度分布。")
        print()

    cat2_diff = abs(dist1[2]/len(samples1) - dist2[2]/len(samples2)) * 100
    cat3_diff = abs(dist1[3]/len(samples1) - dist2[3]/len(samples2)) * 100

    print(f"Cat2占比差异: {cat2_diff:.1f}个百分点")
    print(f"Cat3占比差异: {cat3_diff:.1f}个百分点")
    print()

    if cat2_diff > 20 or cat3_diff > 20:
        print("⚠️  Cat2/Cat3占比差异 > 20%")
        print("   这会显著影响整体Box IoU，不适合用于对比不同训练策略。")
        print()

    print("=" * 70)
    print("建议")
    print("=" * 70)
    print()
    print("1. 创建一个**固定的测试集**用于公平对比：")
    print("   - 从全部数据中随机采样")
    print("   - 或使用重叠部分作为测试集")
    print()
    print("2. 分别在50_50_2k和50_50_2k_new上训练，然后在**同一个测试集**上评估")
    print()
    print("3. 使用渐进式测试 (progressive_test_4_all_cat3) 作为统一基准")
    print("   - 包含全部1015个Cat3样本")
    print("   - 更全面、更稳定的评估")
    print()


if __name__ == '__main__':
    main()
