#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析训练日志，找到最佳模型和性能
"""

import os
import re
import glob
from pathlib import Path

def analyze_iou_log(log_path):
    """解析 mIoU 日志文件"""
    if not os.path.exists(log_path):
        return None

    best_epoch = 0
    best_iou = 0.0
    train_loss = 0.0
    test_loss = 0.0

    with open(log_path, 'r') as f:
        lines = f.readlines()
        if lines:
            last_line = lines[-1]  # 最后一行是最佳结果
            # 格式：24/12/2025 21:36:46 - 0045:	 - train_loss: 0.1390:	 - test_loss: 0.3624:	 mIoU 0.6442
            match = re.search(r'(\d+):\s*-\s*train_loss:\s*([\d.]+):\s*-\s*test_loss:\s*([\d.]+):\s*mIoU\s*([\d.]+)', last_line)
            if match:
                best_epoch = int(match.group(1))
                train_loss = float(match.group(2))
                test_loss = float(match.group(3))
                best_iou = float(match.group(4))

    return {
        'epoch': best_epoch,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'mean_IoU': best_iou
    }

def analyze_other_metric_log(log_path):
    """解析其他指标日志文件"""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None

    # 最后几行包含 Recall 和 Precision
    recall_line = None
    precision_line = None

    for i in range(len(lines) - 1, -1, -1):
        if 'Recall' in lines[i]:
            recall_line = lines[i]
        if 'Precision' in lines[i]:
            precision_line = lines[i]
        if recall_line and precision_line:
            break

    recall = []
    precision = []

    if recall_line:
        recall = [float(x) for x in re.findall(r'[\d.]+', recall_line.split('Recall')[1])]

    if precision_line:
        precision = [float(x) for x in re.findall(r'[\d.]+', precision_line.split('Precision')[1])]

    return {
        'recall': recall,
        'precision': precision
    }

def find_result_dir(base_dir='result'):
    """找到最新的结果文件夹"""
    if not os.path.exists(base_dir):
        return None

    result_dirs = glob.glob(os.path.join(base_dir, '*MS_CAFNet*'))
    if not result_dirs:
        result_dirs = glob.glob(os.path.join(base_dir, 'Pohang*'))

    if not result_dirs:
        return None

    # 按修改时间排序，返回最新的
    result_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return result_dirs[0]

def main():
    print("=" * 60)
    print("训练结果分析工具")
    print("=" * 60)

    # 查找结果文件夹
    result_dir = find_result_dir()

    if not result_dir:
        print("❌ 未找到结果文件夹！")
        print("请确保在项目根目录运行此脚本。")
        return

    print(f"\n📁 结果文件夹: {result_dir}")
    print("-" * 60)

    # 查找日志文件
    iou_log = glob.glob(os.path.join(result_dir, '*_best_IoU_IoU.log'))
    other_log = glob.glob(os.path.join(result_dir, '*_best_IoU_other_metric.log'))

    if not iou_log:
        print("❌ 未找到 IoU 日志文件！")
        return

    # 分析 IoU 日志
    iou_data = analyze_iou_log(iou_log[0])

    if iou_data:
        print("\n📊 最佳模型性能:")
        print(f"  🏆 Epoch:       {iou_data['epoch']}")
        print(f"  📈 mean IoU:    {iou_data['mean_IoU']:.4f}")
        print(f"  📉 Train Loss:  {iou_data['train_loss']:.4f}")
        print(f"  📉 Test Loss:   {iou_data['test_loss']:.4f}")
    else:
        print("⚠️  无法解析 IoU 日志")

    # 分析其他指标
    if other_log:
        metric_data = analyze_other_metric_log(other_log[0])
        if metric_data and metric_data['recall']:
            print("\n📊 检测性能（阈值 0.5）:")
            # 通常索引 5 对应阈值 0.5
            idx = 5 if len(metric_data['recall']) > 5 else len(metric_data['recall']) // 2
            print(f"  🎯 Recall:      {metric_data['recall'][idx]:.4f}")
            print(f"  🎯 Precision:   {metric_data['precision'][idx]:.4f}")

    # 查找模型文件
    print("\n💾 保存的模型文件:")
    model_files = glob.glob(os.path.join(result_dir, '*.pth.tar'))

    if model_files:
        for model_file in sorted(model_files):
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"  📦 {os.path.basename(model_file)} ({file_size:.2f} MB)")
    else:
        print("  ⚠️  未找到模型文件")

    # 查看训练日志摘要
    train_log = os.path.join(result_dir, 'train_log.txt')
    if os.path.exists(train_log):
        print("\n📝 训练配置:")
        with open(train_log, 'r') as f:
            lines = f.readlines()[:15]  # 前15行
            for line in lines:
                if any(key in line for key in ['model', 'dataset', 'epochs', 'lr', 'in_channels']):
                    print(f"  {line.strip()}")

    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)

    # 提供下一步建议
    if iou_data:
        print("\n💡 下一步建议:")
        if iou_data['mean_IoU'] < 0.65:
            print("  1. IoU < 0.65，性能有提升空间")
            print("  2. 尝试调整学习率或训练更多 epoch")
            print("  3. 检查数据增强策略")
        elif iou_data['mean_IoU'] >= 0.65:
            print("  1. IoU >= 0.65，性能良好")
            print("  2. 在黄金集上进一步评估")
            print("  3. 与基线模型对比")

        if iou_data['test_loss'] > iou_data['train_loss'] * 1.5:
            print("  ⚠️  测试损失明显高于训练损失，可能存在过拟合")
            print("     建议：添加正则化或数据增强")

if __name__ == "__main__":
    main()
