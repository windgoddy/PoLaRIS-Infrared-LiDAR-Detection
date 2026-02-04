#!/usr/bin/env python3
"""
测试脚本：计算训练好的模型在测试集上的 Mask-to-Box IoU
==========================================

用法:
    python test_box_iou.py --checkpoint result/xxx/latest_best_model.pth.tar
    python test_box_iou.py --checkpoint result/xxx/best_model_epoch0100_mIoU0.5678.pth.tar --gpu 0

Author: PoLaRIS Team
Date: 2026-02-03
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 导入模型和工具
from model.model_DNANet import DNANet, Res_CBAM_block
from model.model_Phase3 import MS_CAFNet, MS_CAFNet_DualGeo
from model.utils import TestSetLoader
from model.utils_lidar import PoLaRISTestLoader, polaris_collate_fn
from model.metric import calculate_mask_to_box_iou, mIoU
from model.load_param_data import load_dataset, load_param

# Mamba model
from model_Mamba.core.polaris_mamba import PoLaRIS_Mamba


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test Mask-to-Box IoU for trained model')

    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth/.pth.tar file)')

    # 可选参数
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binary threshold for segmentation (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Test batch size (default: 4)')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k',
                        help='Dataset name')
    parser.add_argument('--root', type=str, default='dataset/',
                        help='Dataset root directory')
    parser.add_argument('--split_method', type=str, default='50_50_2k_new',
                        help='Train/test split method')
    parser.add_argument('--image_folder', type=str, default='images',
                        help='Image folder name')
    parser.add_argument('--suffix', type=str, default='.png',
                        help='Image file suffix')

    # 模型参数
    parser.add_argument('--model', type=str, default=None,
                        help='Model type (auto-detect from checkpoint if not specified)')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels (1=IR only, 2=IR+Depth)')
    parser.add_argument('--base_size', type=int, default=512,
                        help='Base image size')
    parser.add_argument('--crop_size', type=int, default=480,
                        help='Crop size for testing')

    # 高级参数
    parser.add_argument('--use_lidar_dataloader', type=str, default='False',
                        help='Use PoLaRIS LiDAR DataLoader (True/False)')
    parser.add_argument('--normalize_16bit', type=str, default='False',
                        help='Use Min-Max normalization for 16-bit images')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')

    # 评估策略参数 (Added 2026-02-04)
    parser.add_argument('--eval_strategy', type=str, default='auto',
                        choices=['auto', 'fixed', 'dynamic'],
                        help='Evaluation strategy: '
                             'auto (Mamba uses dynamic, others use fixed), '
                             'fixed (use --threshold value), '
                             'dynamic (sweep thresholds [0.1-0.9] per sample)')

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """从 checkpoint 加载模型"""
    print(f"\n📦 加载 checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 提取信息（兼容 mamba 保存字段）
    epoch = checkpoint.get('epoch', 'Unknown')
    mean_IOU = checkpoint.get('mean_IOU', checkpoint.get('mIoU', checkpoint.get('iou', 'Unknown')))
    box_IOU = checkpoint.get('box_IOU', checkpoint.get('box_iou', None))

    print(f"  ✓ Checkpoint 信息:")
    print(f"    - Epoch: {epoch}")
    print(f"    - Segmentation IoU: {mean_IOU}")
    if box_IOU is not None:
        print(f"    - Box IoU: {box_IOU:.4f}")

    checkpoint_info = {
        'epoch': epoch,
        'mean_IOU': mean_IOU,
        'box_IOU': box_IOU,
    }

    return checkpoint, checkpoint_info


def detect_model_params(state_dict, model_type):
    """从 checkpoint 的 state_dict 中自动检测模型参数"""
    params = {}

    # DNANet: deep supervision + in_channels
    if model_type == 'DNANet':
        has_deep_supervision = any('final1.' in key for key in state_dict.keys())
        params['deep_supervision'] = has_deep_supervision

        first_conv_key = None
        for key in state_dict.keys():
            if 'conv0_0' in key and 'weight' in key and 'conv0_0.0' in key:
                first_conv_key = key
                break

        if first_conv_key:
            conv_weight = state_dict[first_conv_key]
            if len(conv_weight.shape) == 4:
                in_channels = conv_weight.shape[1]
                params['in_channels'] = in_channels
                print(f"  ℹ️  从 {first_conv_key} 检测到 in_channels: {in_channels}")
        else:
            print(f"  ⚠️  未找到 conv0_0 层，无法检测 in_channels")

    # Mamba: infer in_channels, embed_dim, and use_lidar
    if 'patch_embed.proj.weight' in state_dict:
        w = state_dict['patch_embed.proj.weight']
        if len(w.shape) == 4:
            # patch_embed only processes IR, always in_channels=1
            params['in_channels'] = w.shape[1]
            params['embed_dim'] = w.shape[0]
            print(f"  ℹ️  Mamba patch_embed.proj.weight shape: {list(w.shape)} → IR_channels={params['in_channels']}, embed_dim={params['embed_dim']}")
            
            # CRITICAL FIX: Check if model uses LiDAR gating
            # LiDAR is injected via gate mechanism, not patch_embed
            has_lidar_gate = any('lidar_gate' in key for key in state_dict.keys())
            params['use_lidar'] = has_lidar_gate
            
            # For data loader: need 2 channels if LiDAR is used
            if has_lidar_gate:
                params['data_in_channels'] = 2  # IR + Depth
                print(f"  ℹ️  检测到 LiDAR gating → 数据加载器需要 2 通道 (IR + Depth)")
            else:
                params['data_in_channels'] = 1  # IR only
                print(f"  ℹ️  未检测到 LiDAR gating → 数据加载器只需 1 通道 (IR)")

    return params


def create_model(model_type, in_channels, checkpoint, device):
    """创建并加载模型"""
    print(f"\n🔧 创建模型: {model_type}")

    state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict')
    if state_dict is None:
        raise KeyError("Checkpoint missing state_dict/model_state_dict")

    detected_params = detect_model_params(state_dict, model_type)
    print(f"  ℹ️  从 checkpoint 检测到的参数:")
    if 'deep_supervision' in detected_params:
        print(f"    - deep_supervision: {detected_params['deep_supervision']}")
    if 'in_channels' in detected_params:
        print(f"    - in_channels: {detected_params['in_channels']}")
        # For Mamba, in_channels is only for IR (patch_embed)
        # Don't override the in_channels parameter here
        if model_type not in ['mamba', 'mamba_tiny', 'mamba_small', 'mamba_base']:
            in_channels = detected_params['in_channels']
    if 'embed_dim' in detected_params:
        print(f"    - embed_dim: {detected_params['embed_dim']}")
    if 'use_lidar' in detected_params:
        print(f"    - use_lidar: {detected_params['use_lidar']}")

    apply_sigmoid = True

    if model_type == 'DNANet':
        nb_filter, num_blocks = load_param('three', 'resnet_18')
        deep_supervision = detected_params.get('deep_supervision', False)
        model = DNANet(
            num_classes=1,
            input_channels=in_channels,
            block=Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=deep_supervision
        )
    elif model_type == 'MS_CAFNet':
        model = MS_CAFNet(num_classes=1, input_channels=in_channels)
    elif model_type == 'MS_CAFNet_DualGeo':
        model = MS_CAFNet_DualGeo(num_classes=1, input_channels=in_channels)
    elif model_type in ['mamba', 'mamba_tiny', 'mamba_small', 'mamba_base']:
        embed_dim = detected_params.get('embed_dim', 96)
        depths_map = {64: [2, 2, 4, 2], 96: [2, 2, 6, 2], 128: [2, 2, 12, 2]}
        depths = depths_map.get(embed_dim, [2, 2, 6, 2])
        use_lidar = detected_params.get('use_lidar', False)
        model = PoLaRIS_Mamba(
            in_channels=1,  # Always 1 for IR (patch_embed)
            embed_dim=embed_dim,
            depths=depths,
            use_lidar=use_lidar
        )
        apply_sigmoid = False  # GaussianHead 内部已做 sigmoid
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ 模型参数: {num_params / 1e6:.2f}M")

    return model, apply_sigmoid


def create_test_loader(args, force_lidar=False):
    """创建测试数据加载器"""
    print(f"\n📂 加载测试数据集...")

    dataset_dir = os.path.join(args.root, args.dataset)
    _, val_img_ids, _ = load_dataset(args.root, args.dataset, args.split_method)

    print(f"  ✓ 数据集: {args.dataset}")
    print(f"  ✓ 测试样本数: {len(val_img_ids)}")

    use_lidar_loader = force_lidar or (args.use_lidar_dataloader == 'True')

    if use_lidar_loader:
        print(f"  ✓ 使用 PoLaRIS LiDAR DataLoader")
        testset = PoLaRISTestLoader(
            dataset_dir=dataset_dir,
            img_id=val_img_ids,
            base_size=args.base_size,
            crop_size=args.crop_size,
            transform=None,
            suffix=args.suffix,
            normalize_16bit=(args.normalize_16bit == 'True'),
            in_channels=args.in_channels,
            image_folder=args.image_folder
        )
        test_loader = DataLoader(
            dataset=testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            drop_last=False,
            collate_fn=polaris_collate_fn
        )
    else:
        print(f"  ✓ 使用传统 DataLoader")
        if args.in_channels == 1:
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])
            ])

        testset = TestSetLoader(
            dataset_dir,
            img_id=val_img_ids,
            base_size=args.base_size,
            crop_size=args.crop_size,
            transform=input_transform,
            suffix=args.suffix,
            in_channels=args.in_channels,
            image_folder=args.image_folder
        )
        test_loader = DataLoader(
            dataset=testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            drop_last=False
        )

    return test_loader, use_lidar_loader


def test_model(model, test_loader, use_lidar_loader, threshold, device, apply_sigmoid=True, model_type='', eval_strategy='auto'):
    """测试模型并计算 Mask-to-Box IoU

    Args:
        model_type: 模型类型
        eval_strategy: 'auto' (Mamba用dynamic，其他用fixed), 'fixed', 'dynamic'
    """
    # 根据eval_strategy决定是否使用动态阈值扫描
    if eval_strategy == 'auto':
        # 默认行为：Mamba用dynamic，其他模型用fixed
        use_threshold_sweep = model_type.startswith('mamba')
    elif eval_strategy == 'dynamic':
        use_threshold_sweep = True
    else:  # 'fixed'
        use_threshold_sweep = False

    print(f"\n🧪 开始测试...")
    if use_threshold_sweep:
        print(f"  - 评估策略: 动态阈值扫描 [0.1-0.9]")
        print(f"  - 基准阈值: {threshold} (仅用于对比)")
    else:
        print(f"  - 评估策略: 固定阈值")
        print(f"  - 阈值: {threshold}")
    print(f"  - 设备: {device}")

    model.eval()

    # 对于非Mamba模型，使用传统的固定阈值评估
    if not use_threshold_sweep:
        miou_metric = mIoU(1, threshold=threshold)
    
    box_iou_sum = 0.0
    box_iou_count = 0
    iou_sum = 0.0  # For Mamba threshold sweep

    sample_ious = []
    sample_box_ious = []
    sample_best_thresholds = []  # Track best threshold per sample

    with torch.no_grad():
        tbar = tqdm(test_loader, desc='Testing')
        for batch_idx, batch_data in enumerate(tbar):
            if use_lidar_loader:
                data = batch_data['image'].to(device)
                labels = batch_data['mask'].to(device)
            else:
                data, labels = batch_data
                data = data.to(device)
                labels = labels.to(device)

            if batch_idx == 0:
                print(f"\n🔍 调试信息 (第一个 batch):")
                print(f"  - 数据形状: {data.shape}")
                print(f"  - 标签形状: {labels.shape}")
                print(f"  - 数据范围: [{data.min().item():.4f}, {data.max().item():.4f}]")
                print(f"  - 标签范围: [{labels.min().item():.4f}, {labels.max().item():.4f}]")
                print(f"  - 数据均值: {data.mean().item():.4f}")
                print(f"  - 标签正样本比例: {(labels > 0.5).float().mean().item():.4f}")

            if model_type.startswith('mamba'):
                if data.shape[1] >= 2:
                    ir = data[:, 0:1]
                    lidar = data[:, 1:2]
                else:
                    ir = data
                    lidar = torch.zeros_like(ir)
                pred = model(ir, lidar)
            else:
                pred = model(data)

            if isinstance(pred, list):
                pred = pred[-1]

            if apply_sigmoid:
                pred = torch.sigmoid(pred)

            if batch_idx == 0:
                print(f"  - 预测形状: {pred.shape}")
                print(f"  - 预测范围: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
                print(f"  - 预测均值: {pred.mean().item():.4f}")
                print(f"  - 预测正样本比例 (>0.5): {(pred > 0.5).float().mean().item():.4f}\n")

            # Mamba: 使用阈值扫描（与训练一致）
            if use_threshold_sweep:
                # Threshold sweep for each batch (like Mamba training)
                best_batch_iou = 0.0
                best_batch_threshold = 0.5
                
                for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    pred_bin = (pred > thresh).float()
                    gt_bin = (labels > 0.5).float()
                    
                    inter = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
                    union = (pred_bin + gt_bin).clamp(0, 1).sum(dim=(1, 2, 3))
                    iou_t = (inter / (union + 1e-7)).mean().item()
                    
                    if iou_t > best_batch_iou:
                        best_batch_iou = iou_t
                        best_batch_threshold = thresh
                
                # Use best threshold for this batch
                iou_sum += best_batch_iou
                batch_box_iou = calculate_mask_to_box_iou(pred, labels, threshold=best_batch_threshold)
                box_iou_sum += batch_box_iou
                box_iou_count += 1
                
                # Per-sample metrics using best threshold
                pred_binary = (pred > best_batch_threshold).float()
                labels_binary = (labels > 0.5).float()
                
                for i in range(pred.size(0)):
                    pred_i = pred_binary[i:i+1]
                    label_i = labels_binary[i:i+1]
                    inter = (pred_i * label_i).sum()
                    union = (pred_i + label_i).clamp(0, 1).sum()
                    sample_iou = (inter / (union + 1e-7)).item()
                    sample_ious.append(sample_iou)
                    sample_best_thresholds.append(best_batch_threshold)
                    
                    sample_box_iou = calculate_mask_to_box_iou(pred[i:i+1], labels[i:i+1], threshold=best_batch_threshold)
                    sample_box_ious.append(sample_box_iou)
                
                # Update progress bar
                current_mean_iou = iou_sum / box_iou_count
                current_box_iou = box_iou_sum / box_iou_count
                tbar.set_postfix({
                    'mIoU': f'{current_mean_iou:.4f}',
                    'Box_IoU': f'{current_box_iou:.4f}',
                    'BestThresh': f'{best_batch_threshold:.1f}'
                })
            else:
                # Traditional fixed threshold evaluation
                miou_metric.update(pred, labels)

                batch_box_iou = calculate_mask_to_box_iou(pred, labels, threshold=threshold)
                box_iou_sum += batch_box_iou
                box_iou_count += 1

                pred_binary = (pred > threshold).float()
                labels_binary = (labels > 0.5).float()

                for i in range(pred.size(0)):
                    pred_i = pred_binary[i:i+1]
                    label_i = labels_binary[i:i+1]
                    inter = (pred_i * label_i).sum()
                    union = (pred_i + label_i).clamp(0, 1).sum()
                    sample_iou = (inter / (union + 1e-7)).item()
                    sample_ious.append(sample_iou)

                    sample_box_iou = calculate_mask_to_box_iou(pred[i:i+1], labels[i:i+1], threshold=threshold)
                    sample_box_ious.append(sample_box_iou)

                _, current_mean_iou = miou_metric.get()
                current_box_iou = box_iou_sum / box_iou_count
                tbar.set_postfix({
                    'mIoU': f'{current_mean_iou:.4f}',
                    'Box_IoU': f'{current_box_iou:.4f}'
                })

    # Calculate final metrics
    if use_threshold_sweep:
        mean_iou = iou_sum / box_iou_count
        # Also calculate threshold distribution
        if sample_best_thresholds:
            threshold_counts = {}
            for t in sample_best_thresholds:
                threshold_counts[t] = threshold_counts.get(t, 0) + 1
            print(f"\n📊 Best threshold distribution:")
            for t in sorted(threshold_counts.keys()):
                count = threshold_counts[t]
                pct = count / len(sample_best_thresholds) * 100
                print(f"  {t:.1f}: {count:4d} samples ({pct:5.1f}%)")
    else:
        _, mean_iou = miou_metric.get()
    
    mean_box_iou = box_iou_sum / box_iou_count

    sample_ious = np.array(sample_ious)
    sample_box_ious = np.array(sample_box_ious)

    results = {
        'mean_iou': mean_iou,
        'mean_box_iou': mean_box_iou,
        'sample_ious': sample_ious,
        'sample_box_ious': sample_box_ious,
        'num_samples': len(sample_ious),
        'use_threshold_sweep': use_threshold_sweep,
    }

    return results


def print_results(results, checkpoint_info):
    """打印测试结果"""
    print(f"\n{'='*70}")
    print(f"测试结果")
    print(f"{'='*70}")

    print(f"\n📦 Checkpoint 信息:")
    print(f"  - Epoch: {checkpoint_info['epoch']}")
    if checkpoint_info['mean_IOU'] != 'Unknown':
        print(f"  - 训练时 Segmentation IoU: {checkpoint_info['mean_IOU']:.4f}")
    if checkpoint_info['box_IOU'] is not None:
        print(f"  - 训练时 Box IoU: {checkpoint_info['box_IOU']:.4f}")

    print(f"\n🎯 测试集结果 (共 {results['num_samples']} 个样本):")
    if results.get('use_threshold_sweep', False):
        print(f"  - 评估策略           : 动态阈值扫描 (与Mamba训练一致)")
    print(f"  - Segmentation IoU : {results['mean_iou']:.4f}")
    print(f"  - Mask-to-Box IoU  : {results['mean_box_iou']:.4f}")

    print(f"\n📊 统计分析:")
    print(f"  Segmentation IoU:")
    print(f"    - 最小值: {results['sample_ious'].min():.4f}")
    print(f"    - 最大值: {results['sample_ious'].max():.4f}")
    print(f"    - 标准差: {results['sample_ious'].std():.4f}")
    print(f"    - 中位数: {np.median(results['sample_ious']):.4f}")

    print(f"  Mask-to-Box IoU:")
    print(f"    - 最小值: {results['sample_box_ious'].min():.4f}")
    print(f"    - 最大值: {results['sample_box_ious'].max():.4f}")
    print(f"    - 标准差: {results['sample_box_ious'].std():.4f}")
    print(f"    - 中位数: {np.median(results['sample_box_ious']):.4f}")

    print(f"\n📈 IoU 分布:")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seg_hist, _ = np.histogram(results['sample_ious'], bins=bins)
    box_hist, _ = np.histogram(results['sample_box_ious'], bins=bins)

    print(f"  {'Range':<15} {'Seg IoU':<12} {'Box IoU':<12}")
    print(f"  {'-'*40}")
    for i in range(len(bins)-1):
        seg_count = seg_hist[i]
        box_count = box_hist[i]
        seg_pct = seg_count / results['num_samples'] * 100
        box_pct = box_count / results['num_samples'] * 100
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}):  {seg_count:4d} ({seg_pct:5.1f}%)  {box_count:4d} ({box_pct:5.1f}%)")

    print(f"\n{'='*70}\n")


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")

    checkpoint, checkpoint_info = load_model_from_checkpoint(args.checkpoint, device)

    if args.model is None:
        checkpoint_name = os.path.basename(args.checkpoint)
        checkpoint_dir = os.path.dirname(args.checkpoint)

        if 'mamba' in checkpoint_dir.lower() or 'mamba' in checkpoint_name.lower():
            if 'mamba_tiny' in checkpoint_name.lower() or 'mamba_tiny' in checkpoint_dir.lower():
                args.model = 'mamba_tiny'
            elif 'mamba_small' in checkpoint_name.lower() or 'mamba_small' in checkpoint_dir.lower():
                args.model = 'mamba_small'
            elif 'mamba_base' in checkpoint_name.lower() or 'mamba_base' in checkpoint_dir.lower():
                args.model = 'mamba_base'
            else:
                args.model = 'mamba'
        elif 'DNANet' in checkpoint_dir or 'DNANet' in checkpoint_name:
            args.model = 'DNANet'
        elif 'MS_CAFNet_DualGeo' in checkpoint_dir:
            args.model = 'MS_CAFNet_DualGeo'
        elif 'MS_CAFNet' in checkpoint_dir:
            args.model = 'MS_CAFNet'
        else:
            print("⚠️  无法自动推断模型类型，默认使用 DNANet")
            args.model = 'DNANet'

    print(f"  ✓ 模型类型: {args.model}")

    state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict')
    detected_params = detect_model_params(state_dict, args.model)

    # CRITICAL FIX: For Mamba, use data_in_channels (which includes LiDAR)
    # instead of model in_channels (which is only IR)
    if 'data_in_channels' in detected_params:
        print(f"  ✓ 从 checkpoint 检测到 data_in_channels={detected_params['data_in_channels']}，更新数据加载器配置")
        args.in_channels = detected_params['data_in_channels']
    elif 'in_channels' in detected_params:
        print(f"  ✓ 从 checkpoint 检测到 in_channels={detected_params['in_channels']}，更新数据加载器配置")
        args.in_channels = detected_params['in_channels']

    force_lidar = False
    if args.model.startswith('mamba'):
        args.use_lidar_dataloader = 'True'
        args.normalize_16bit = 'True'
        force_lidar = True

    model, apply_sigmoid = create_model(args.model, args.in_channels, checkpoint, device)

    test_loader, use_lidar_loader = create_test_loader(args, force_lidar=force_lidar)

    results = test_model(model, test_loader, use_lidar_loader, args.threshold, device,
                         apply_sigmoid=apply_sigmoid, model_type=args.model,
                         eval_strategy=args.eval_strategy)

    print_results(results, checkpoint_info)

    result_dir = os.path.dirname(args.checkpoint)
    result_file = os.path.join(result_dir, 'box_iou_test_results.txt')

    with open(result_file, 'w') as f:
        f.write(f"Mask-to-Box IoU Test Results\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Split: {args.split_method}\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        f.write(f"Results:\n")
        f.write(f"  - Segmentation IoU: {results['mean_iou']:.4f}\n")
        f.write(f"  - Mask-to-Box IoU:  {results['mean_box_iou']:.4f}\n")
        f.write(f"  - Num Samples:      {results['num_samples']}\n")

    print(f"✅ 结果已保存到: {result_file}")


if __name__ == '__main__':
    main()
