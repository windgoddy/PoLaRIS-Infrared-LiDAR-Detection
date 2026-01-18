#!/usr/bin/env python3
"""
快速调优分类阈值脚本
读取已保存的统计信息，快速测试不同阈值组合
"""

import os
import json
import argparse
from collections import defaultdict


def classify_with_thresholds(stats, thresholds):
    """
    使用给定的阈值对单张图像进行分类

    Args:
        stats: 图像统计信息 {'total_points', 'num_boxes', 'box_stats', ...}
        thresholds: 阈值字典

    Returns:
        label: 0/1/2/3
    """
    total_points = stats['total_points']
    num_boxes = stats['num_boxes']
    box_stats = stats['box_stats']
    total_points_in_boxes = sum(b['points'] for b in box_stats)

    # 优先级 3: 困难岸边样本
    if total_points >= thresholds['label3_min_points'] and num_boxes <= thresholds['label3_max_boxes']:
        has_high_fill = any(b['fill_ratio'] > thresholds['label3_min_fill_ratio'] for b in box_stats)
        if has_high_fill:
            return 3

    # 优先级 2: 困难极小样本
    if num_boxes > 0:
        has_empty_box = any(b['points'] == 0 for b in box_stats)
        small_boxes = [b for b in box_stats if b['area'] < thresholds['label2_small_box_area']]
        small_ratio = len(small_boxes) / num_boxes

        if has_empty_box and small_ratio > thresholds['label2_min_small_ratio']:
            return 2

    # 优先级 1: 简单样本
    if num_boxes <= thresholds['label1_max_boxes'] and total_points > 0:
        points_ratio = total_points_in_boxes / total_points if total_points > 0 else 0
        if points_ratio > thresholds['label1_min_points_ratio']:
            return 1

    # 优先级 0: 其他
    return 0


def load_stats_from_results(results_file):
    """
    从之前的结果文件中提取统计信息

    注意：这需要之前运行时保存了详细的统计信息
    如果没有，需要先运行一次完整处理并保存统计数据
    """
    # 这里我们假设有一个 stats.json 文件
    stats_file = results_file.replace('selection_summary_new.txt', 'image_stats.json')

    if not os.path.exists(stats_file):
        print(f"错误: 统计文件不存在: {stats_file}")
        print("请先运行一次完整处理并添加 --save-stats 参数")
        return None

    with open(stats_file, 'r') as f:
        return json.load(f)


def test_thresholds(stats_dict, thresholds):
    """
    测试给定的阈值组合

    Args:
        stats_dict: {image_name: stats, ...}
        thresholds: 阈值字典

    Returns:
        分类结果统计
    """
    label_counts = defaultdict(int)
    results = {}

    for image_name, stats in stats_dict.items():
        label = classify_with_thresholds(stats, thresholds)
        label_counts[label] += 1
        results[image_name] = label

    total = len(stats_dict)

    print("\n" + "="*60)
    print("阈值配置:")
    print("-" * 60)
    for key, value in thresholds.items():
        print(f"  {key}: {value}")

    print("\n分类结果:")
    print("-" * 60)
    for label in [0, 1, 2, 3]:
        count = label_counts[label]
        pct = count / total * 100 if total > 0 else 0
        print(f"  标签{label}: {count:5d} 张 ({pct:5.1f}%)")
    print("="*60)

    return results, label_counts


def interactive_tuning(stats_dict):
    """交互式调优模式"""
    # 默认阈值（优化后的橄榄型分布配置）
    thresholds = {
        # 标签3: 困难岸边样本 (目标8-10%)
        'label3_min_points': 650,
        'label3_max_boxes': 5,
        'label3_min_fill_ratio': 0.005,  # 0.5%

        # 标签2: 困难极小样本 (目标15-20%)
        'label2_small_box_area': 32 * 32,
        'label2_min_small_ratio': 0.83,

        # 标签1: 简单样本 (目标~20%)
        'label1_max_boxes': 7,
        'label1_min_points_ratio': 0.12,
    }

    print("="*60)
    print("交互式阈值调优工具")
    print("="*60)
    print("\n可调整的参数:")
    print("  1. label3_min_points      - 标签3最小点云数")
    print("  2. label3_min_fill_ratio  - 标签3最小填充率")
    print("  3. label1_min_points_ratio - 标签1最小框内点云占比")
    print("  4. test - 测试当前阈值")
    print("  5. preset - 使用预设方案")
    print("  6. quit - 退出")
    print()

    while True:
        cmd = input("请输入命令 (1-6): ").strip()

        if cmd == '1':
            val = float(input(f"  当前值: {thresholds['label3_min_points']}, 新值: "))
            thresholds['label3_min_points'] = val
        elif cmd == '2':
            val = float(input(f"  当前值: {thresholds['label3_min_fill_ratio']}, 新值: "))
            thresholds['label3_min_fill_ratio'] = val
        elif cmd == '3':
            val = float(input(f"  当前值: {thresholds['label1_min_points_ratio']}, 新值: "))
            thresholds['label1_min_points_ratio'] = val
        elif cmd == '4':
            test_thresholds(stats_dict, thresholds)
        elif cmd == '5':
            print("\n预设方案:")
            print("  a. 保守 (当前)")
            print("  b. 激进 (更多标签3)")
            print("  c. 平衡")
            preset = input("选择 (a/b/c): ").strip()

            if preset == 'a':
                thresholds.update({
                    'label3_min_points': 1500,
                    'label3_min_fill_ratio': 0.4,
                    'label1_min_points_ratio': 0.5,
                })
            elif preset == 'b':
                thresholds.update({
                    'label3_min_points': 1000,
                    'label3_min_fill_ratio': 0.3,
                    'label1_min_points_ratio': 0.4,
                })
            elif preset == 'c':
                thresholds.update({
                    'label3_min_points': 1200,
                    'label3_min_fill_ratio': 0.35,
                    'label1_min_points_ratio': 0.45,
                })
            print("已应用预设")
        elif cmd == '6' or cmd == 'quit':
            break
        else:
            print("无效命令")


def main():
    parser = argparse.ArgumentParser(description='快速调优分类阈值')
    parser.add_argument('--stats_file', type=str,
                        default='dataset/select-view/image_stats.json',
                        help='统计信息文件路径')
    parser.add_argument('--interactive', action='store_true',
                        help='交互式调优模式')
    parser.add_argument('--batch', action='store_true',
                        help='批量测试多组阈值')

    args = parser.parse_args()

    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stats_file = os.path.join(project_root, args.stats_file)

    # 检查统计文件是否存在
    if not os.path.exists(stats_file):
        print(f"错误: 统计文件不存在: {stats_file}")
        print("\n请先运行以下命令生成统计文件:")
        print("  python scripts/classify_images.py --save-stats")
        return

    # 加载统计信息
    print(f"加载统计信息: {stats_file}")
    with open(stats_file, 'r') as f:
        stats_dict = json.load(f)

    print(f"已加载 {len(stats_dict)} 张图像的统计信息")

    if args.interactive:
        interactive_tuning(stats_dict)
    elif args.batch:
        # 批量测试预设方案
        presets = [
            ("优化配置 (橄榄型分布)", {
                'label3_min_points': 650,
                'label3_max_boxes': 5,
                'label3_min_fill_ratio': 0.005,
                'label2_small_box_area': 32 * 32,
                'label2_min_small_ratio': 0.83,
                'label1_max_boxes': 7,
                'label1_min_points_ratio': 0.12,
            }),
            ("保守方案 (更少Label3)", {
                'label3_min_points': 700,
                'label3_max_boxes': 5,
                'label3_min_fill_ratio': 0.006,
                'label2_small_box_area': 32 * 32,
                'label2_min_small_ratio': 0.83,
                'label1_max_boxes': 7,
                'label1_min_points_ratio': 0.12,
            }),
            ("激进方案 (更多Label3)", {
                'label3_min_points': 600,
                'label3_max_boxes': 5,
                'label3_min_fill_ratio': 0.005,
                'label2_small_box_area': 32 * 32,
                'label2_min_small_ratio': 0.85,
                'label1_max_boxes': 7,
                'label1_min_points_ratio': 0.12,
            }),
        ]

        for name, thresholds in presets:
            print(f"\n{'='*60}")
            print(f"方案: {name}")
            test_thresholds(stats_dict, thresholds)
    else:
        # 默认：测试优化后的配置
        thresholds = {
            'label3_min_points': 500,
            'label3_max_boxes': 5,
            'label3_min_fill_ratio': 0.005,
            'label2_small_box_area': 32 * 32,
            'label2_min_small_ratio': 0.83,
            'label1_max_boxes': 7,
            'label1_min_points_ratio': 0.12,
        }
        test_thresholds(stats_dict, thresholds)


if __name__ == "__main__":
    main()
