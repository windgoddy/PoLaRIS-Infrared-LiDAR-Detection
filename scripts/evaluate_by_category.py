#!/usr/bin/env python
"""
按类别评估模型性能

功能：
1. 加载训练好的模型检查点
2. 读取测试集类别标签（从 selection_summary_new.txt）
3. 对测试集进行推理
4. 按 Category 1、2、3 分别统计性能指标

用法:
    python scripts/evaluate_by_category.py \
        --checkpoint results/MS_CAFNet_DualGeo_Pohang-Canal-3k/mIoU__epoch_XXX.pth \
        --dataset Pohang-Canal-3k \
        --split_method 50_50_2k_new \
        --in_channels 2 \
        --threshold 0.5
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from tqdm import tqdm
from collections import defaultdict
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_Phase3 import MS_CAFNet_DualGeo
from model.utils_lidar import PoLaRISTestLoader


def load_category_mapping(dataset_dir):
    """
    从 selection_summary_new.txt 加载类别映射

    Returns:
        dict: {sample_id: category}
        例如: {'00_5596': 2, '01_595': 1, ...}
    """
    summary_file = os.path.join(dataset_dir, 'selection_summary_new.txt')

    if not os.path.exists(summary_file):
        print(f"⚠️  警告: 找不到类别映射文件 {summary_file}")
        print("   将无法按类别统计，只能统计整体性能")
        return {}

    category_map = {}
    with open(summary_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue

            parts = line.split('|')
            if len(parts) == 2:
                # 格式: "00_5596.png | 2"
                sample_id = parts[0].strip().replace('.png', '')  # 移除 .png 后缀
                category = int(parts[1].strip())
                category_map[sample_id] = category

    print(f"✅ 加载类别映射: {len(category_map)} 个样本")
    return category_map


def calculate_metrics(pred, target, threshold=0.5):
    """
    计算单张图的检测指标

    Args:
        pred: 预测图 [H, W], 0-1 范围
        target: 真值图 [H, W], 0/1
        threshold: 二值化阈值

    Returns:
        dict: {tp, fp, tn, fn, iou, precision, recall, f1}
    """
    # 二值化预测
    pred_binary = (pred >= threshold).astype(np.float32)
    target_binary = target.astype(np.float32)

    # 计算混淆矩阵
    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    tn = np.sum((pred_binary == 0) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))

    # 计算指标
    iou = tp / (tp + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class CategoryEvaluator:
    """按类别评估器"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置统计"""
        self.category_stats = defaultdict(lambda: {
            'tp': 0,
            'fp': 0,
            'tn': 0,
            'fn': 0,
            'iou_sum': 0.0,
            'count': 0
        })
        self.overall_stats = {
            'tp': 0,
            'fp': 0,
            'tn': 0,
            'fn': 0,
            'iou_sum': 0.0,
            'count': 0
        }

    def update(self, sample_id, pred, target, category_map, threshold=0.5):
        """
        更新统计

        Args:
            sample_id: 样本ID (例如 '00_5596')
            pred: 预测图 [H, W] numpy array
            target: 真值图 [H, W] numpy array
            category_map: 类别映射字典
            threshold: 二值化阈值
        """
        # 计算该样本的指标
        metrics = calculate_metrics(pred, target, threshold)

        # 获取类别
        category = category_map.get(sample_id, None)

        # 更新类别统计
        if category is not None:
            stats = self.category_stats[category]
            stats['tp'] += metrics['tp']
            stats['fp'] += metrics['fp']
            stats['tn'] += metrics['tn']
            stats['fn'] += metrics['fn']
            stats['iou_sum'] += metrics['iou']
            stats['count'] += 1

        # 更新整体统计
        self.overall_stats['tp'] += metrics['tp']
        self.overall_stats['fp'] += metrics['fp']
        self.overall_stats['tn'] += metrics['tn']
        self.overall_stats['fn'] += metrics['fn']
        self.overall_stats['iou_sum'] += metrics['iou']
        self.overall_stats['count'] += 1

    def get_results(self):
        """
        获取评估结果

        Returns:
            dict: {
                'overall': {...},
                'category_1': {...},
                'category_2': {...},
                'category_3': {...}
            }
        """
        results = {}

        # 整体结果
        results['overall'] = self._compute_final_metrics(self.overall_stats)

        # 各类别结果
        for cat in [1, 2, 3]:
            if cat in self.category_stats:
                results[f'category_{cat}'] = self._compute_final_metrics(
                    self.category_stats[cat]
                )
            else:
                results[f'category_{cat}'] = None

        return results

    def _compute_final_metrics(self, stats):
        """计算最终指标"""
        if stats['count'] == 0:
            return None

        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']

        iou = tp / (tp + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        mean_iou = stats['iou_sum'] / stats['count']

        # 计算 FP Rate (False Positive Rate)
        # FPR = FP / (FP + TN)
        fpr = fp / (fp + stats['tn'] + 1e-10)

        return {
            'count': stats['count'],
            'iou': iou,
            'mean_iou': mean_iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fp_rate': fpr,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }


def evaluate_model(args):
    """
    评估模型性能
    """
    print("=" * 80)
    print("按类别评估模型性能")
    print("=" * 80)

    # 1. 加载类别映射
    dataset_dir = os.path.join(args.root, args.dataset)
    category_map = load_category_mapping(dataset_dir)

    # 2. 加载测试集
    print(f"\n📂 加载测试集: {args.dataset} / {args.split_method}")

    # 读取测试集样本列表
    test_file = os.path.join(dataset_dir, args.split_method, 'test.txt')
    with open(test_file, 'r') as f:
        test_img_ids = [line.strip() for line in f if line.strip()]

    print(f"   测试集样本数: {len(test_img_ids)}")

    # 统计测试集类别分布
    cat_counts = defaultdict(int)
    for img_id in test_img_ids:
        cat = category_map.get(img_id, None)
        if cat is not None:
            cat_counts[cat] += 1

    print(f"   类别分布:")
    for cat in [1, 2, 3]:
        count = cat_counts.get(cat, 0)
        pct = 100.0 * count / len(test_img_ids) if len(test_img_ids) > 0 else 0
        cat_name = {1: '简单海面', 2: '一般海岸', 3: '复杂海岸'}[cat]
        print(f"     Category {cat} ({cat_name}): {count} ({pct:.1f}%)")

    # 创建测试 DataLoader
    testset = PoLaRISTestLoader(
        dataset_dir=dataset_dir,
        img_id=test_img_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        transform=None,
        suffix=args.suffix,
        normalize_16bit=args.normalize_16bit,
        in_channels=args.in_channels,
        image_folder=args.image_folder
    )

    test_loader = DataLoader(
        testset,
        batch_size=1,  # 按样本统计，使用 batch_size=1
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 3. 加载模型
    print(f"\n🤖 加载模型: {args.model}")
    model = MS_CAFNet_DualGeo(num_classes=1, input_channels=args.in_channels)

    # 加载检查点
    print(f"   检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # 处理不同的检查点格式
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    print(f"   ✅ 模型加载成功")

    # 4. 开始评估
    print(f"\n🔍 开始评估 (阈值: {args.threshold})")

    evaluator = CategoryEvaluator()

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="评估中")

        for batch_data in pbar:
            # 解析数据
            data = batch_data['image'].cuda()
            labels = batch_data['mask'].cuda()
            sample_id = batch_data['id'][0]  # batch_size=1，取第一个

            # 前向传播
            outputs, pred_conf = model(data)

            # 使用最终输出
            if isinstance(outputs, list):
                pred = outputs[-1]
            else:
                pred = outputs

            # 转换为 numpy (sigmoid 激活)
            pred_np = torch.sigmoid(pred).cpu().numpy()[0, 0]  # [H, W]
            target_np = labels.cpu().numpy()[0, 0]  # [H, W]

            # 更新评估器
            evaluator.update(sample_id, pred_np, target_np, category_map, args.threshold)

    # 5. 打印结果
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)

    results = evaluator.get_results()

    # 打印整体结果
    print("\n📊 整体性能:")
    print_metrics(results['overall'])

    # 打印各类别结果
    cat_names = {
        1: '简单海面 (Simple Sea)',
        2: '一般海岸 (General Shore)',
        3: '复杂海岸 (Complex Shore)'
    }

    for cat in [1, 2, 3]:
        key = f'category_{cat}'
        if results[key] is not None:
            print(f"\n📊 Category {cat} - {cat_names[cat]}:")
            print_metrics(results[key])
        else:
            print(f"\n📊 Category {cat} - {cat_names[cat]}: 无数据")

    # 6. 对比分析
    print("\n" + "=" * 80)
    print("类别对比分析")
    print("=" * 80)

    print(f"\n{'指标':<15} {'Cat1':>10} {'Cat2':>10} {'Cat3':>10} {'整体':>10}")
    print("-" * 60)

    metrics_to_compare = ['iou', 'precision', 'recall', 'f1', 'fp_rate']
    metric_names = {
        'iou': 'IoU',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'fp_rate': 'FP Rate'
    }

    for metric in metrics_to_compare:
        values = []
        for cat in [1, 2, 3]:
            key = f'category_{cat}'
            if results[key] is not None:
                val = results[key][metric]
                values.append(f"{val:>10.4f}")
            else:
                values.append(f"{'N/A':>10}")

        overall_val = results['overall'][metric]
        print(f"{metric_names[metric]:<15} {values[0]} {values[1]} {values[2]} {overall_val:>10.4f}")

    print("\n" + "=" * 80)
    print("✅ 评估完成!")
    print("=" * 80)

    return results


def print_metrics(metrics):
    """打印指标"""
    if metrics is None:
        print("  无数据")
        return

    print(f"  样本数量: {metrics['count']}")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  Mean IoU:  {metrics['mean_iou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  FP Rate:   {metrics['fp_rate']:.4f}")
    print(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='按类别评估模型性能')

    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径 (.pth 文件)')

    # 数据集参数
    parser.add_argument('--root', type=str, default='./dataset',
                        help='数据集根目录')
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k',
                        help='数据集名称')
    parser.add_argument('--split_method', type=str, default='50_50_2k_new',
                        help='数据集划分方法')
    parser.add_argument('--image_folder', type=str, default='images-ir-16bit',
                        help='图像文件夹名称')
    parser.add_argument('--suffix', type=str, default='.png',
                        help='图像文件后缀')

    # 模型参数
    parser.add_argument('--model', type=str, default='MS_CAFNet_DualGeo',
                        help='模型名称')
    parser.add_argument('--in_channels', type=int, default=2,
                        help='输入通道数 (1=仅IR, 2=IR+Depth)')

    # 评估参数
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值')
    parser.add_argument('--base_size', type=int, default=512,
                        help='基础尺寸')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='裁剪尺寸')
    parser.add_argument('--normalize_16bit', type=bool, default=True,
                        help='是否对 16-bit 图像归一化')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_model(args)
