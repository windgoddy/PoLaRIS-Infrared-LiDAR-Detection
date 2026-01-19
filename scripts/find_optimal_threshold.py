"""
最优阈值搜索脚本 (Optimal Threshold Search)

功能：
1. 加载训练好的模型
2. 在验证集/测试集上生成预测概率图
3. 搜索最优的二值化阈值（基于mIoU或F1-score）
4. 支持分组评估（有/无LiDAR、不同目标大小）
5. 输出详细的性能曲线和推荐阈值

支持的模型：
- MS_CAFNet_DualGeo: 返回 (output, pred_conf) 元组
- 其他单输出模型: 返回 output

用法：
    python scripts/find_optimal_threshold.py \
        --model_path results/Phase3_DualGeo/best_model.pth \
        --dataset_dir dataset/select \
        --split test \
        --output_dir results/threshold_search

作者：PoLaRIS Team
日期：2026-01-19
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# 添加项目根目录到 Python Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入模型和数据加载器
try:
    from model.model_Phase3 import MS_CAFNet_DualGeo
    from model.utils_lidar import PoLaRISTestLoader  # 支持16-bit和LiDAR的DataLoader
except ImportError as e:
    print(f"警告: 无法导入模型/DataLoader，请检查路径: {e}")
    # Fallback: 使用基础导入
    pass


# ============================================================================
# 评估指标计算
# ============================================================================

def compute_metrics(pred_mask, gt_mask, epsilon=1e-7):
    """
    计算二值分割的评估指标
    
    Args:
        pred_mask: (H, W) 预测掩码，值为0或1
        gt_mask: (H, W) Ground Truth掩码，值为0或1
        epsilon: 平滑因子，避免除零
    
    Returns:
        dict: 包含 IoU, Precision, Recall, F1
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    # True Positive, False Positive, False Negative
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    tn = np.sum(~pred_mask & ~gt_mask)
    
    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = (intersection + epsilon) / (union + epsilon) if union > 0 else 0.0
    
    # Precision = TP / (TP + FP)
    precision = (tp + epsilon) / (tp + fp + epsilon) if (tp + fp) > 0 else 0.0
    
    # Recall = TP / (TP + FN)
    recall = (tp + epsilon) / (tp + fn + epsilon) if (tp + fn) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon) if (precision + recall) > 0 else 0.0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def compute_mean_metrics(all_metrics):
    """
    计算多个样本的平均指标
    
    Args:
        all_metrics: List[dict] 每个样本的指标字典
    
    Returns:
        dict: 平均指标
    """
    if len(all_metrics) == 0:
        return {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    mean_metrics = {
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics]),
    }
    
    # 总体统计
    total_tp = sum([m['tp'] for m in all_metrics])
    total_fp = sum([m['fp'] for m in all_metrics])
    total_fn = sum([m['fn'] for m in all_metrics])
    
    # 全局 IoU (基于总体 TP/FP/FN)
    global_iou = total_tp / (total_tp + total_fp + total_fn + 1e-7)
    mean_metrics['global_iou'] = global_iou
    
    return mean_metrics


# ============================================================================
# 分组评估辅助函数
# ============================================================================

def check_has_lidar(lidar_data):
    """
    检查样本是否有LiDAR数据
    
    Args:
        lidar_data: LiDAR tensor 或 None
    
    Returns:
        bool: True if has valid LiDAR points
    """
    if lidar_data is None:
        return False
    if isinstance(lidar_data, torch.Tensor):
        return lidar_data.numel() > 0 and torch.any(lidar_data != 0)
    return False


def get_target_size(gt_mask):
    """
    计算GT掩码中的目标总面积（像素数）
    
    Args:
        gt_mask: (H, W) Ground Truth掩码
    
    Returns:
        int: 目标面积（像素数）
    """
    return int(np.sum(gt_mask > 0))


def categorize_target_size(area):
    """
    根据目标面积分类
    
    Args:
        area: 目标面积（像素数）
    
    Returns:
        str: 'tiny', 'small', 'medium', 'large'
    """
    if area <= 10:
        return 'tiny'
    elif area <= 50:
        return 'small'
    elif area <= 200:
        return 'medium'
    else:
        return 'large'


# ============================================================================
# 模型预测
# ============================================================================

def predict_probability_maps(model, dataloader, device='cuda'):
    """
    生成所有样本的预测概率图
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 计算设备
    
    Returns:
        predictions: List[dict] 包含 prob_map, gt_mask, img_id, has_lidar, target_size
    """
    model.eval()
    predictions = []
    
    print("🔮 生成预测概率图...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测进度"):
            images = batch['image'].to(device)
            gt_masks = batch['mask'].cpu().numpy()  # (B, 1, H, W)
            img_ids = batch['img_id']
            
            # 模型预测
            model_output = model(images)
            
            # 处理不同的模型输出格式
            # MS_CAFNet_DualGeo 返回 (output, pred_conf)
            # 其他模型可能只返回 output
            if isinstance(model_output, tuple):
                outputs = model_output[0]  # 取第一个元素（检测图）
                # pred_conf = model_output[1]  # 第二个元素是置信度图（可选使用）
            else:
                outputs = model_output
            
            # 转换为概率 (如果模型输出是logits)
            if outputs.min() < 0 or outputs.max() > 1:
                probs = torch.sigmoid(outputs)
            else:
                probs = outputs
            
            probs = probs.cpu().numpy()  # (B, 1, H, W)
            
            # 逐样本保存
            batch_size = images.size(0)
            for i in range(batch_size):
                prob_map = probs[i, 0]  # (H, W)
                gt_mask = gt_masks[i, 0]  # (H, W)
                img_id = img_ids[i]
                
                # 检查是否有LiDAR
                has_lidar = False
                if 'lidar' in batch:
                    has_lidar = check_has_lidar(batch['lidar'][i])
                
                # 计算目标大小
                target_size = get_target_size(gt_mask)
                size_category = categorize_target_size(target_size)
                
                predictions.append({
                    'prob_map': prob_map,
                    'gt_mask': gt_mask,
                    'img_id': img_id,
                    'has_lidar': has_lidar,
                    'target_size': target_size,
                    'size_category': size_category
                })
    
    print(f"✓ 完成 {len(predictions)} 个样本的预测")
    return predictions


# ============================================================================
# 阈值搜索
# ============================================================================

def search_optimal_threshold(predictions, thresholds, metric='iou', verbose=True):
    """
    搜索最优二值化阈值
    
    Args:
        predictions: List[dict] 预测结果列表
        thresholds: List[float] 阈值候选列表
        metric: str 优化目标 ('iou', 'f1', 'precision', 'recall')
        verbose: bool 是否打印详细信息
    
    Returns:
        results: dict 包含每个阈值的性能指标
        best_threshold: float 最优阈值
    """
    results = defaultdict(list)
    
    print(f"\n🔍 开始阈值搜索 (优化目标: {metric.upper()})")
    print(f"阈值范围: [{min(thresholds):.2f}, {max(thresholds):.2f}], 步长: {thresholds[1]-thresholds[0]:.3f}")
    print("-" * 80)
    
    for threshold in tqdm(thresholds, desc="阈值扫描"):
        all_metrics = []
        
        for pred_data in predictions:
            prob_map = pred_data['prob_map']
            gt_mask = pred_data['gt_mask']
            
            # 二值化
            pred_mask = (prob_map >= threshold).astype(np.uint8)
            gt_binary = (gt_mask > 0).astype(np.uint8)
            
            # 计算指标
            metrics = compute_metrics(pred_mask, gt_binary)
            all_metrics.append(metrics)
        
        # 计算平均指标
        mean_metrics = compute_mean_metrics(all_metrics)
        
        # 保存结果
        results['threshold'].append(threshold)
        results['iou'].append(mean_metrics['iou'])
        results['global_iou'].append(mean_metrics['global_iou'])
        results['precision'].append(mean_metrics['precision'])
        results['recall'].append(mean_metrics['recall'])
        results['f1'].append(mean_metrics['f1'])
    
    # 找出最优阈值
    best_idx = np.argmax(results[metric])
    best_threshold = results['threshold'][best_idx]
    best_value = results[metric][best_idx]
    
    if verbose:
        print(f"\n✨ 最优阈值: {best_threshold:.3f}")
        print(f"   {metric.upper()}: {best_value:.4f}")
        print(f"   mIoU: {results['iou'][best_idx]:.4f}")
        print(f"   Global IoU: {results['global_iou'][best_idx]:.4f}")
        print(f"   Precision: {results['precision'][best_idx]:.4f}")
        print(f"   Recall: {results['recall'][best_idx]:.4f}")
        print(f"   F1-Score: {results['f1'][best_idx]:.4f}")
    
    return dict(results), best_threshold


def search_optimal_threshold_by_group(predictions, thresholds, metric='iou'):
    """
    按分组搜索最优阈值（有/无LiDAR、不同目标大小）
    
    Args:
        predictions: List[dict] 预测结果列表
        thresholds: List[float] 阈值候选列表
        metric: str 优化目标
    
    Returns:
        group_results: dict 每个分组的最优阈值和性能
    """
    group_results = {}
    
    # 1. 按有无LiDAR分组
    print("\n" + "="*80)
    print("📊 按LiDAR可用性分组评估")
    print("="*80)
    
    lidar_groups = {
        'with_lidar': [p for p in predictions if p['has_lidar']],
        'without_lidar': [p for p in predictions if not p['has_lidar']]
    }
    
    for group_name, group_data in lidar_groups.items():
        if len(group_data) == 0:
            print(f"\n⚠️  {group_name}: 无数据，跳过")
            continue
        
        print(f"\n{'─'*80}")
        print(f"🔸 {group_name.upper()} (样本数: {len(group_data)})")
        print(f"{'─'*80}")
        
        results, best_th = search_optimal_threshold(
            group_data, thresholds, metric, verbose=True
        )
        
        group_results[group_name] = {
            'best_threshold': best_th,
            'results': results,
            'sample_count': len(group_data)
        }
    
    # 2. 按目标大小分组
    print("\n" + "="*80)
    print("📊 按目标大小分组评估")
    print("="*80)
    
    size_groups = defaultdict(list)
    for p in predictions:
        size_groups[p['size_category']].append(p)
    
    for size_cat in ['tiny', 'small', 'medium', 'large']:
        group_data = size_groups.get(size_cat, [])
        if len(group_data) == 0:
            continue
        
        print(f"\n{'─'*80}")
        print(f"🔸 {size_cat.upper()} (样本数: {len(group_data)})")
        print(f"{'─'*80}")
        
        results, best_th = search_optimal_threshold(
            group_data, thresholds, metric, verbose=True
        )
        
        group_results[f'size_{size_cat}'] = {
            'best_threshold': best_th,
            'results': results,
            'sample_count': len(group_data)
        }
    
    return group_results


# ============================================================================
# 可视化
# ============================================================================

def plot_threshold_curves(results, output_path, metric='iou'):
    """
    绘制阈值-性能曲线
    
    Args:
        results: dict 阈值搜索结果
        output_path: str 保存路径
        metric: str 主要优化指标
    """
    plt.figure(figsize=(12, 8))
    
    thresholds = results['threshold']
    
    # 绘制4条曲线
    plt.plot(thresholds, results['iou'], 'o-', label='mIoU', linewidth=2)
    plt.plot(thresholds, results['global_iou'], 's-', label='Global IoU', linewidth=2)
    plt.plot(thresholds, results['f1'], '^-', label='F1-Score', linewidth=2)
    plt.plot(thresholds, results['precision'], 'v-', label='Precision', linewidth=1.5, alpha=0.7)
    plt.plot(thresholds, results['recall'], 'd-', label='Recall', linewidth=1.5, alpha=0.7)
    
    # 标记最优阈值
    best_idx = np.argmax(results[metric])
    best_threshold = thresholds[best_idx]
    best_value = results[metric][best_idx]
    
    plt.axvline(best_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal Threshold={best_threshold:.3f}')
    plt.plot(best_threshold, best_value, 'r*', markersize=15)
    
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title(f'Threshold Search Curve (Optimized for {metric.upper()})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 曲线图已保存: {output_path}")
    plt.close()


def plot_group_comparison(group_results, output_path):
    """
    绘制不同分组的最优阈值对比
    
    Args:
        group_results: dict 分组结果
        output_path: str 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 有无LiDAR对比
    ax1 = axes[0]
    lidar_groups = ['with_lidar', 'without_lidar']
    lidar_thresholds = [group_results.get(g, {}).get('best_threshold', 0) 
                        for g in lidar_groups]
    lidar_samples = [group_results.get(g, {}).get('sample_count', 0) 
                     for g in lidar_groups]
    
    bars1 = ax1.bar(lidar_groups, lidar_thresholds, color=['#2ecc71', '#e74c3c'], alpha=0.7)
    ax1.set_ylabel('Optimal Threshold', fontsize=12)
    ax1.set_title('Best Threshold by LiDAR Availability', fontsize=14)
    ax1.set_ylim(0, 0.7)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # 添加样本数标注
    for i, (bar, count) in enumerate(zip(bars1, lidar_samples)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}\n(n={count})',
                ha='center', va='bottom', fontsize=10)
    
    # 2. 目标大小对比
    ax2 = axes[1]
    size_categories = ['tiny', 'small', 'medium', 'large']
    size_thresholds = [group_results.get(f'size_{s}', {}).get('best_threshold', 0) 
                       for s in size_categories]
    size_samples = [group_results.get(f'size_{s}', {}).get('sample_count', 0) 
                    for s in size_categories]
    
    colors = ['#9b59b6', '#3498db', '#f39c12', '#e67e22']
    bars2 = ax2.bar(size_categories, size_thresholds, color=colors, alpha=0.7)
    ax2.set_ylabel('Optimal Threshold', fontsize=12)
    ax2.set_title('Best Threshold by Target Size', fontsize=14)
    ax2.set_ylim(0, 0.7)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 添加样本数标注
    for i, (bar, count) in enumerate(zip(bars2, size_samples)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}\n(n={count})',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 分组对比图已保存: {output_path}")
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def load_model(model_path, device='cuda', input_channels=2):
    """
    加载训练好的模型
    
    Args:
        model_path: str 模型权重路径
        device: str 计算设备
        input_channels: int 输入通道数（1=仅IR，2=IR+Depth）
    
    Returns:
        model: 加载好的模型
        
    注意：
        MS_CAFNet_DualGeo 的 forward 返回元组 (output, pred_conf)
        - output: 检测分割图 (B, 1, H, W)
        - pred_conf: LiDAR置信度图 (B, 1, H, W)
    """
    print(f"📥 加载模型: {model_path}")
    
    # 根据模型路径选择模型架构
    # 默认使用 Phase3 双流模型
    model = MS_CAFNet_DualGeo(num_classes=1, input_channels=input_channels)
    
    print(f"   模型架构: MS_CAFNet_DualGeo (输入通道: {input_channels})")
    print(f"   输出格式: (检测图, 置信度图)")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理不同的checkpoint格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功")
    return model


def main():
    parser = argparse.ArgumentParser(description='最优阈值搜索脚本')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型权重路径')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='数据集根目录')
    
    # 可选参数
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test', 'val'],
                        help='使用哪个数据集划分')
    parser.add_argument('--output_dir', type=str, default='results/threshold_search',
                        help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批处理大小')
    
    # 阈值搜索参数
    parser.add_argument('--th_min', type=float, default=0.1,
                        help='最小阈值')
    parser.add_argument('--th_max', type=float, default=0.7,
                        help='最大阈值')
    parser.add_argument('--th_step', type=float, default=0.05,
                        help='阈值步长')
    parser.add_argument('--metric', type=str, default='iou',
                        choices=['iou', 'f1', 'precision', 'recall'],
                        help='优化目标指标')
    
    # 功能开关
    parser.add_argument('--group_analysis', action='store_true',
                        help='是否进行分组分析')
    parser.add_argument('--save_predictions', action='store_true',
                        help='是否保存预测概率图')
    
    # 模型参数
    parser.add_argument('--input_channels', type=int, default=2,
                        choices=[1, 2],
                        help='模型输入通道数 (1=仅IR, 2=IR+Depth)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 打印配置
    print("\n" + "="*80)
    print("🚀 最优阈值搜索")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"数据集: {args.dataset_dir}")
    print(f"数据划分: {args.split}")
    print(f"阈值范围: [{args.th_min}, {args.th_max}], 步长: {args.th_step}")
    print(f"优化目标: {args.metric.upper()}")
    print(f"输出目录: {args.output_dir}")
    print("="*80 + "\n")
    
    # 1. 加载模型
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device, input_channels=args.input_channels)
    
    # 2. 准备数据
    print(f"\n📂 准备数据集...")
    
    # 读取数据集划分文件
    split_file = os.path.join(args.dataset_dir, 'split_data', f'{args.split}.txt')
    if not os.path.exists(split_file):
        # Fallback: 尝试旧路径
        split_file = os.path.join(args.dataset_dir, f'{args.split}.txt')
    
    with open(split_file, 'r') as f:
        img_ids = [line.strip().replace('.png', '') for line in f.readlines()]
    
    print(f"✓ 加载 {len(img_ids)} 个样本")
    
    # 创建DataLoader
    dataset = PoLaRISTestLoader(
        dataset_dir=args.dataset_dir,
        img_id=img_ids,
        mode='test'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 3. 生成预测概率图
    predictions = predict_probability_maps(model, dataloader, device)
    
    # 可选：保存预测结果
    if args.save_predictions:
        pred_save_path = os.path.join(args.output_dir, 'predictions.npz')
        np.savez_compressed(
            pred_save_path,
            predictions=predictions
        )
        print(f"✓ 预测结果已保存: {pred_save_path}")
    
    # 4. 阈值搜索
    thresholds = np.arange(args.th_min, args.th_max + args.th_step/2, args.th_step)
    
    print("\n" + "="*80)
    print("🔍 全局阈值搜索")
    print("="*80)
    
    results, best_threshold = search_optimal_threshold(
        predictions, thresholds, metric=args.metric, verbose=True
    )
    
    # 5. 可视化
    curve_path = os.path.join(args.output_dir, 'threshold_curve.png')
    plot_threshold_curves(results, curve_path, metric=args.metric)
    
    # 6. 分组分析（可选）
    group_results = None
    if args.group_analysis:
        group_results = search_optimal_threshold_by_group(
            predictions, thresholds, metric=args.metric
        )
        
        # 绘制分组对比图
        group_plot_path = os.path.join(args.output_dir, 'group_comparison.png')
        plot_group_comparison(group_results, group_plot_path)
    
    # 7. 保存结果
    summary = {
        'best_threshold': float(best_threshold),
        'best_metric': args.metric,
        'best_value': float(results[args.metric][np.argmax(results[args.metric])]),
        'threshold_range': [float(args.th_min), float(args.th_max)],
        'threshold_step': float(args.th_step),
        'total_samples': len(predictions),
        'results': {k: [float(v) for v in vals] for k, vals in results.items()}
    }
    
    if group_results:
        summary['group_results'] = {
            k: {
                'best_threshold': float(v['best_threshold']),
                'sample_count': v['sample_count']
            } for k, v in group_results.items()
        }
    
    summary_path = os.path.join(args.output_dir, 'threshold_search_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✓ 结果摘要已保存: {summary_path}")
    
    # 8. 打印最终结论
    print("\n" + "="*80)
    print("📋 最终结论")
    print("="*80)
    print(f"\n✨ 对于当前模型，最佳推理阈值是: {best_threshold:.3f}")
    print(f"\n   在该阈值下:")
    best_idx = np.argmax(results[args.metric])
    print(f"   - mIoU:      {results['iou'][best_idx]:.4f}")
    print(f"   - Global IoU: {results['global_iou'][best_idx]:.4f}")
    print(f"   - F1-Score:   {results['f1'][best_idx]:.4f}")
    print(f"   - Precision:  {results['precision'][best_idx]:.4f}")
    print(f"   - Recall:     {results['recall'][best_idx]:.4f}")
    
    if group_results:
        print(f"\n   分组推荐阈值:")
        if 'with_lidar' in group_results:
            print(f"   - 有LiDAR目标:  {group_results['with_lidar']['best_threshold']:.3f}")
        if 'without_lidar' in group_results:
            print(f"   - 无LiDAR目标:  {group_results['without_lidar']['best_threshold']:.3f}")
    
    print("\n" + "="*80)
    print("✅ 阈值搜索完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
