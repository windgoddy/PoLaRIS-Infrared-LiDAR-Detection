#!/usr/bin/env python3
"""
创建渐进式测试集划分

根据Cat3难度分析结果，创建多个测试集划分：
1. baseline_nocat3: 仅Cat1+Cat2（当前最佳性能）
2. add_cat3_easy: Cat1+Cat2+Cat3_Easy（10-20个简单Cat3样本）
3. add_cat3_medium: Cat1+Cat2+Cat3_Easy+Cat3_Medium
4. add_cat3_all: 完整测试集

使用方法：
    python create_progressive_test_splits.py --difficulty_json cat3_difficulty_analysis.json
"""

import os
import json
import argparse
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Create progressive test splits')
    parser.add_argument('--difficulty_json', type=str, required=True,
                        help='Cat3 difficulty analysis JSON file')
    parser.add_argument('--source_split', type=str, default='dataset/Pohang-Canal-3k/50_50_2k_new_nocat3',
                        help='Source split (without Cat3)')
    parser.add_argument('--num_easy', type=int, default=20,
                        help='Number of easy Cat3 samples to add')
    parser.add_argument('--num_medium', type=int, default=50,
                        help='Number of medium Cat3 samples to add')
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*70)
    print("创建渐进式测试集划分")
    print("="*70)

    # 加载难度分析结果
    print("\n[1/4] 加载Cat3难度分析...")
    with open(args.difficulty_json, 'r') as f:
        difficulty_data = json.load(f)

    easy_samples = difficulty_data['easy']
    medium_samples = difficulty_data['medium']
    hard_samples = difficulty_data['hard']

    print(f"  ✓ Easy: {len(easy_samples)} 样本")
    print(f"  ✓ Medium: {len(medium_samples)} 样本")
    print(f"  ✓ Hard: {len(hard_samples)} 样本")

    # 读取baseline test.txt（Cat1+Cat2）
    baseline_test = os.path.join(args.source_split, 'test.txt')
    with open(baseline_test, 'r') as f:
        baseline_samples = [line.strip() for line in f if line.strip()]

    print(f"\n  ✓ Baseline测试集: {len(baseline_samples)} 样本 (Cat1+Cat2)")

    # 创建4个测试集
    print("\n[2/4] 创建测试集划分...")

    dataset_root = os.path.dirname(os.path.dirname(args.source_split))
    splits = []

    # Split 1: Baseline (无Cat3)
    split1_dir = os.path.join(dataset_root, 'progressive_test_1_baseline')
    os.makedirs(split1_dir, exist_ok=True)
    split1_samples = baseline_samples
    splits.append(('progressive_test_1_baseline', split1_samples, "Baseline (Cat1+Cat2)"))

    # Split 2: +Easy Cat3
    split2_dir = os.path.join(dataset_root, 'progressive_test_2_add_easy')
    os.makedirs(split2_dir, exist_ok=True)
    split2_samples = baseline_samples + easy_samples[:args.num_easy]
    splits.append(('progressive_test_2_add_easy', split2_samples, f"Baseline + {args.num_easy} Easy Cat3"))

    # Split 3: +Medium Cat3
    split3_dir = os.path.join(dataset_root, 'progressive_test_3_add_medium')
    os.makedirs(split3_dir, exist_ok=True)
    split3_samples = baseline_samples + easy_samples[:args.num_easy] + medium_samples[:args.num_medium]
    splits.append(('progressive_test_3_add_medium', split3_samples, f"Baseline + Easy + {args.num_medium} Medium Cat3"))

    # Split 4: 全部Cat3
    split4_dir = os.path.join(dataset_root, 'progressive_test_4_all_cat3')
    os.makedirs(split4_dir, exist_ok=True)
    split4_samples = baseline_samples + easy_samples + medium_samples + hard_samples
    splits.append(('progressive_test_4_all_cat3', split4_samples, "Baseline + All Cat3"))

    # 写入文件
    print("\n[3/4] 写入测试集文件...")
    for split_name, samples, description in splits:
        split_dir = os.path.join(dataset_root, split_name)

        # 复制train.txt（保持不变）
        train_src = os.path.join(args.source_split, 'train.txt')
        train_dst = os.path.join(split_dir, 'train.txt')
        shutil.copy2(train_src, train_dst)

        # 写入test.txt
        test_dst = os.path.join(split_dir, 'test.txt')
        with open(test_dst, 'w') as f:
            for sample_id in samples:
                f.write(f"{sample_id}\n")

        print(f"  ✓ {split_name}: {len(samples)} 测试样本")

    # 生成测试命令
    print("\n[4/4] 生成测试命令...")

    commands_file = 'progressive_test_commands.sh'
    with open(commands_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Progressive Cat3 Testing Commands\n")
        f.write("# Run these sequentially to see performance degradation with Cat3\n\n")

        f.write("CHECKPOINT=\"model_Mamba/result/tiny_LiDAR_16bit_PoLaRIS_d2.5p2.0_20260212_221812/best_model_epoch1148_IoU0.7071_BoxIoU0.7928.pth\"\n\n")

        for i, (split_name, samples, description) in enumerate(splits, 1):
            f.write(f"# Test {i}: {description}\n")
            f.write(f"echo \"Testing: {description}\"\n")
            f.write(f"python test_box_iou.py \\\n")
            f.write(f"  --checkpoint $CHECKPOINT \\\n")
            f.write(f"  --split_method {split_name}\n\n")

    os.chmod(commands_file, 0o755)

    # 打印汇总
    print("\n" + "="*70)
    print("✅ 完成！创建了4个渐进式测试集")
    print("="*70)

    for i, (split_name, samples, description) in enumerate(splits, 1):
        cat3_count = len(samples) - len(baseline_samples)
        print(f"\n{i}. {split_name}")
        print(f"   描述: {description}")
        print(f"   测试样本: {len(samples)} (Cat1+Cat2: {len(baseline_samples)}, Cat3: {cat3_count})")
        print(f"   路径: {os.path.join(dataset_root, split_name)}")

    print(f"\n测试命令脚本: {commands_file}")
    print("\n运行方式:")
    print(f"  bash {commands_file}")
    print("="*70)


if __name__ == '__main__':
    main()
