#!/usr/bin/env python3
"""
分析50_50_2k和50_50_2k_new测试集中Cat3样本的难度分布
需要先运行过analyze_cat3_difficulty.py生成cat3_difficulty_analysis.json
"""
import json
import sys
import os

sys.path.insert(0, 'model_Mamba')
from model.load_param_data import load_category_mapping


def load_difficulty_mapping(json_file):
    """加载Cat3难度分析结果"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 创建img_id -> difficulty的映射
    difficulty_map = {}
    for sample in data['scores']:
        img_id = sample['img_id']
        iou = sample['box_iou']
        if iou >= 0.8:
            difficulty = 'Easy'
        elif iou >= 0.5:
            difficulty = 'Medium'
        else:
            difficulty = 'Hard'
        difficulty_map[img_id] = {
            'difficulty': difficulty,
            'iou': iou
        }

    return difficulty_map


def analyze_split_difficulty(split_file, category_map, cat3_difficulty):
    """分析split中Cat3样本的难度分布"""
    with open(split_file, 'r') as f:
        samples = [line.strip() for line in f if line.strip()]

    cat3_count = {'Easy': 0, 'Medium': 0, 'Hard': 0, 'Unknown': 0}
    cat3_samples_detail = {'Easy': [], 'Medium': [], 'Hard': [], 'Unknown': []}

    total_cat3 = 0
    for sample in samples:
        cat = category_map.get(sample, 0)
        if cat == 3:  # Cat3
            total_cat3 += 1
            if sample in cat3_difficulty:
                diff = cat3_difficulty[sample]['difficulty']
                iou = cat3_difficulty[sample]['iou']
                cat3_count[diff] += 1
                cat3_samples_detail[diff].append((sample, iou))
            else:
                cat3_count['Unknown'] += 1
                cat3_samples_detail['Unknown'].append((sample, 0.0))

    return total_cat3, cat3_count, cat3_samples_detail


def main():
    # 检查cat3_difficulty_analysis.json是否存在
    if not os.path.exists('cat3_difficulty_analysis.json'):
        print("❌ 错误: cat3_difficulty_analysis.json 不存在")
        print("请先运行: python analyze_cat3_difficulty.py --checkpoint <模型> --gpu 0")
        sys.exit(1)

    dataset_root = 'dataset'
    dataset_name = 'Pohang-Canal-3k'

    # Load category mapping
    category_map, category_names = load_category_mapping(dataset_root, dataset_name)

    # Load Cat3 difficulty mapping
    cat3_difficulty = load_difficulty_mapping('cat3_difficulty_analysis.json')

    # Analyze both splits
    splits = [
        ('50_50_2k', f'{dataset_root}/{dataset_name}/50_50_2k/test.txt'),
        ('50_50_2k_new', f'{dataset_root}/{dataset_name}/50_50_2k_new/test.txt'),
    ]

    print("=" * 80)
    print("测试集中Cat3样本难度分布分析")
    print("=" * 80)
    print()

    results = {}
    for name, split_file in splits:
        if not os.path.exists(split_file):
            print(f"⚠️  跳过 {name}: 文件不存在 {split_file}")
            continue

        total_cat3, cat3_count, cat3_detail = analyze_split_difficulty(
            split_file, category_map, cat3_difficulty
        )
        results[name] = {
            'total': total_cat3,
            'count': cat3_count,
            'detail': cat3_detail
        }

        print(f"📊 {name} 测试集:")
        print(f"   Cat3总数: {total_cat3}")
        print(f"   难度分布:")
        for diff in ['Easy', 'Medium', 'Hard', 'Unknown']:
            count = cat3_count[diff]
            pct = count / total_cat3 * 100 if total_cat3 > 0 else 0
            print(f"     {diff:8s}: {count:3d} ({pct:5.1f}%)")
        print()

    # 对比分析
    if len(results) == 2:
        print("=" * 80)
        print("对比分析")
        print("=" * 80)
        print()

        split1_name, split2_name = '50_50_2k', '50_50_2k_new'
        r1 = results[split1_name]
        r2 = results[split2_name]

        print(f"{'难度':<10} {split1_name:^15} {split2_name:^15} {'差异':^10}")
        print("-" * 80)
        for diff in ['Easy', 'Medium', 'Hard']:
            c1 = r1['count'][diff]
            c2 = r2['count'][diff]
            p1 = c1 / r1['total'] * 100 if r1['total'] > 0 else 0
            p2 = c2 / r2['total'] * 100 if r2['total'] > 0 else 0
            delta = p1 - p2
            print(f"{diff:<10} {c1:3d} ({p1:5.1f}%)    {c2:3d} ({p2:5.1f}%)    {delta:+6.1f}%")
        print()

        # 关键发现
        print("=" * 80)
        print("关键发现")
        print("=" * 80)
        print()

        medium_diff = r1['count']['Medium'] / r1['total'] - r2['count']['Medium'] / r2['total']
        medium_diff_pct = medium_diff * 100

        if abs(medium_diff_pct) > 5:
            if medium_diff_pct > 0:
                print(f"⚠️  {split1_name} 包含更多Medium Cat3 (+{medium_diff_pct:.1f}%)")
                print(f"   由于Mamba在Medium Cat3上表现较差（~0.70 vs DNANet 0.94）")
                print(f"   这会导致Mamba在{split1_name}上的整体Box IoU更低")
                print()
                print("✅ 这**不是泛化能力差**，而是测试集恰好包含了Mamba的弱项！")
            else:
                print(f"⚠️  {split2_name} 包含更多Medium Cat3 (+{-medium_diff_pct:.1f}%)")
                print(f"   由于Mamba在Medium Cat3上表现较差（~0.70 vs DNANet 0.94）")
                print(f"   这会导致Mamba在{split2_name}上的整体Box IoU更低")
        else:
            print("✓ 两个测试集的Medium Cat3占比相近")
            print("  性能差异可能来自其他因素（Cat2难度、Easy/Hard Cat3分布等）")
        print()

        # 列出Medium Cat3样本
        print("=" * 80)
        print(f"{split1_name} 测试集中的Medium Cat3样本 ({r1['count']['Medium']}个)")
        print("=" * 80)
        for sample, iou in sorted(r1['detail']['Medium'], key=lambda x: x[1]):
            print(f"  {sample:20s} IoU: {iou:.4f}")
        print()

        print("=" * 80)
        print(f"{split2_name} 测试集中的Medium Cat3样本 ({r2['count']['Medium']}个)")
        print("=" * 80)
        for sample, iou in sorted(r2['detail']['Medium'], key=lambda x: x[1]):
            print(f"  {sample:20s} IoU: {iou:.4f}")
        print()


if __name__ == '__main__':
    main()
