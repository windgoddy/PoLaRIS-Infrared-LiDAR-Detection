#!/usr/bin/env python3
"""
通用难度分析脚本 - 支持所有类别
完全参照test_box_iou.py实现，确保正确性
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model.utils import TestSetLoader
from model.utils_lidar import PoLaRISTestLoader, polaris_collate_fn
from model.metric import calculate_mask_to_box_iou
from model.model_DNANet import DNANet, Res_CBAM_block
from model.load_param_data import load_param
from model_Mamba.core.polaris_mamba_progressive import polaris_mamba_tiny_progressive


CATEGORY_NAMES = {
    0: "Category 0 (未分类场景)",
    1: "Category 1 (适中场景 - 点云适中)",
    2: "Category 2 (小目标场景 - 点云少)",
    3: "Category 3 (岸边场景 - 点云多)"
}


def parse_args():
    parser = argparse.ArgumentParser(description='分析所有类别样本的难度分级')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k', help='数据集名称')
    parser.add_argument('--category', type=str, default='all',
                       help='要分析的类别: 0/1/2/3/all (默认all分析所有类别)')
    parser.add_argument('--model_type', type=str, default='',
                       help='模型类型: DNANet/Mamba（默认从checkpoint路径自动推断）')
    parser.add_argument('--output_prefix', type=str, default='',
                       help='输出文件前缀（默认自动从checkpoint推断）')
    parser.add_argument('--image_folder', type=str, default='',
                       help='图像文件夹名称（默认自动检测：8bit模型用images-8bit，否则用images）')
    parser.add_argument('--gpu', type=str, default='0', help='GPU设备ID')
    return parser.parse_args()


def load_category_mapping(dataset_root, dataset_name):
    """加载类别映射"""
    category_file = os.path.join(dataset_root, dataset_name, 'selection_summary_new.txt')
    if not os.path.exists(category_file):
        category_file = os.path.join(dataset_root, dataset_name, 'select-view/selection_summary_new.txt')

    if not os.path.exists(category_file):
        return None

    category_map = {}
    with open(category_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '文件名' in line:
                continue
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 2:
                    img_id = parts[0].strip().replace('.png', '')
                    category = int(parts[1].strip())
                    category_map[img_id] = category

    return category_map


def analyze_single_category(category_id, model, device, dataset_root, dataset_name,
                            category_map, in_channels, output_file, image_folder, model_type):
    """分析单个类别的所有样本"""

    # 获取该类别的所有样本
    category_samples = sorted([img_id for img_id, cat in category_map.items() if cat == category_id])

    if len(category_samples) == 0:
        print(f"  ⚠️  Category {category_id} 没有样本，跳过")
        return None

    print(f"\n{'='*70}")
    print(f"分析 Category {category_id} - {CATEGORY_NAMES[category_id]}")
    print(f"{'='*70}")
    print(f"  样本数: {len(category_samples)}")

    # 加载数据
    dataset_dir = os.path.join(dataset_root, dataset_name)

    # 根据模型类型选择DataLoader
    if model_type == 'Mamba':
        # Mamba使用PoLaRISTestLoader
        test_loader_obj = PoLaRISTestLoader(
            dataset_dir=dataset_dir,
            img_id=category_samples,
            base_size=256,
            crop_size=256,
            transform=None,
            suffix='.png',
            normalize_16bit=True,
            in_channels=in_channels,
            image_folder=image_folder,
        )

        test_loader = DataLoader(
            test_loader_obj,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=polaris_collate_fn
        )
    else:
        # DNANet使用传统TestSetLoader
        if in_channels == 1:
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])
            ])

        test_loader_obj = TestSetLoader(
            dataset_dir,
            img_id=category_samples,
            base_size=256,
            crop_size=256,
            transform=input_transform,
            suffix='.png',
            in_channels=in_channels,
            image_folder=image_folder
        )

        test_loader = DataLoader(
            dataset=test_loader_obj,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )

    # 评估每个样本
    sample_scores = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc=f"  Cat{category_id}")):
            if model_type == 'Mamba':
                # Mamba使用字典格式
                data = batch_data['image'].to(device)
                labels = batch_data['mask'].to(device)
                img_id = batch_data['img_id'][0]

                # 分离IR和LiDAR
                if data.shape[1] == 2:
                    ir = data[:, 0:1]
                    lidar = data[:, 1:2]
                else:
                    ir = data
                    lidar = None

                # Mamba推理
                pred = model(ir, lidar)
            else:
                # DNANet使用元组格式
                data, labels = batch_data
                data = data.to(device)
                labels = labels.to(device)
                img_id = category_samples[batch_idx]

                # DNANet推理
                pred = model(data)

                # 处理深度监督返回list
                if isinstance(pred, list):
                    pred = pred[-1]

                # DNANet输出logits，需要sigmoid
                pred = torch.sigmoid(pred)

            # 动态阈值扫描
            best_iou = 0.0
            best_thresh = 0.5

            for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                iou = calculate_mask_to_box_iou(pred, labels, threshold=float(thresh))
                if iou > best_iou:
                    best_iou = iou
                    best_thresh = float(thresh)

            sample_scores.append({
                'img_id': img_id,
                'box_iou': best_iou,
                'best_thresh': best_thresh
            })

    # 分析难度分布
    sample_scores.sort(key=lambda x: x['box_iou'], reverse=True)

    easy = [s for s in sample_scores if s['box_iou'] >= 0.8]
    medium = [s for s in sample_scores if 0.5 <= s['box_iou'] < 0.8]
    hard = [s for s in sample_scores if s['box_iou'] < 0.5]

    print(f"\n  难度分布:")
    print(f"    Easy (≥0.8):   {len(easy):3d} ({len(easy)/len(sample_scores)*100:5.1f}%)")
    print(f"    Medium (0.5-0.8): {len(medium):3d} ({len(medium)/len(sample_scores)*100:5.1f}%)")
    print(f"    Hard (<0.5):   {len(hard):3d} ({len(hard)/len(sample_scores)*100:5.1f}%)")
    print(f"  平均IoU: {np.mean([s['box_iou'] for s in sample_scores]):.4f}")

    # 保存结果
    save_results(output_file, category_id, sample_scores, easy, medium, hard)

    return {
        'category': category_id,
        'total': len(sample_scores),
        'easy': len(easy),
        'medium': len(medium),
        'hard': len(hard),
        'mean_iou': np.mean([s['box_iou'] for s in sample_scores])
    }


def save_results(output_file, category_id, sample_scores, easy, medium, hard):
    """保存分析结果"""
    # 保存文本
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"Category {category_id} 样本难度分析\n")
        f.write(f"{CATEGORY_NAMES[category_id]}\n")
        f.write("="*70 + "\n\n")

        f.write(f"总样本: {len(sample_scores)}\n")
        f.write(f"平均IoU: {np.mean([s['box_iou'] for s in sample_scores]):.4f}\n")
        f.write(f"中位数IoU: {np.median([s['box_iou'] for s in sample_scores]):.4f}\n\n")

        f.write("难度分布:\n")
        f.write(f"  Easy (≥0.8):   {len(easy)} ({len(easy)/len(sample_scores)*100:.1f}%)\n")
        f.write(f"  Medium (0.5-0.8): {len(medium)} ({len(medium)/len(sample_scores)*100:.1f}%)\n")
        f.write(f"  Hard (<0.5):   {len(hard)} ({len(hard)/len(sample_scores)*100:.1f}%)\n\n")

        f.write("="*70 + "\n")
        f.write("详细样本列表\n")
        f.write("="*70 + "\n\n")

        for label, samples in [('Easy', easy), ('Medium', medium), ('Hard', hard)]:
            f.write(f"\n{label} ({len(samples)}):\n")
            f.write("-"*70 + "\n")
            for s in samples:
                f.write(f"  {s['img_id']:20s}  IoU: {s['box_iou']:.4f}  Thresh: {s['best_thresh']:.1f}\n")

    # 保存JSON
    json_out = output_file.replace('.txt', '.json')
    with open(json_out, 'w') as f:
        json.dump({
            'category': category_id,
            'category_name': CATEGORY_NAMES[category_id],
            'easy': [s['img_id'] for s in easy],
            'medium': [s['img_id'] for s in medium],
            'hard': [s['img_id'] for s in hard],
            'scores': sample_scores
        }, f, indent=2)

    print(f"  ✓ 保存结果: {output_file}")
    print(f"  ✓ 保存JSON: {json_out}")


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("通用难度分析脚本")
    print("="*70)
    print()

    # 1. 加载类别映射
    print("[1/3] 加载类别映射...")
    dataset_root = 'dataset'
    category_map = load_category_mapping(dataset_root, args.dataset)

    if category_map is None:
        print("❌ 无法加载类别映射")
        sys.exit(1)

    # 统计各类别样本数
    category_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for cat in category_map.values():
        category_counts[cat] += 1

    print("  ✓ 类别分布:")
    for cat in sorted(category_counts.keys()):
        print(f"    Cat{cat}: {category_counts[cat]} 样本")

    # 2. 加载模型（只加载一次）
    print("\n[2/3] 加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 自动检测模型类型
    if args.model_type == '':
        # 从checkpoint路径推断
        if 'mamba' in args.checkpoint.lower() or 'model_Mamba' in args.checkpoint:
            model_type = 'Mamba'
        elif 'dnanet' in args.checkpoint.lower():
            model_type = 'DNANet'
        else:
            # 从state_dict keys判断
            state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict')
            if any('aux_head' in key for key in state_dict.keys()):
                model_type = 'Mamba'
            elif any('conv0_0' in key for key in state_dict.keys()):
                model_type = 'DNANet'
            else:
                raise ValueError("无法自动检测模型类型，请使用 --model_type 指定")
    else:
        model_type = args.model_type

    print(f"  ✓ 检测到模型类型: {model_type}")

    # 加载模型
    if model_type == 'Mamba':
        state_dict = checkpoint['model_state_dict']

        # 检测Mamba配置
        has_deep_supervision = any('aux_head' in key for key in state_dict.keys())
        has_cbam_full = any('head.ca.' in key for key in state_dict.keys())
        use_cbam = 'full' if has_cbam_full else 'none'
        in_channels = 2  # Mamba固定使用2通道

        print(f"  ✓ 配置: deep_supervision={has_deep_supervision}, cbam={use_cbam}, in_channels={in_channels}")

        model = polaris_mamba_tiny_progressive(
            use_lidar=True,
            use_deep_supervision=has_deep_supervision,
            use_cbam=use_cbam
        )
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
    else:
        # DNANet
        state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict')
        if state_dict is None:
            raise KeyError("Checkpoint missing state_dict/model_state_dict")

        # 检测DNANet配置
        has_deep_supervision = any('final1.' in key for key in state_dict.keys())

        # 检测in_channels
        in_channels = 2
        first_conv_key = None
        for key in state_dict.keys():
            if 'conv0_0' in key and 'weight' in key and 'conv0_0.0' in key:
                first_conv_key = key
                break

        if first_conv_key:
            conv_weight = state_dict[first_conv_key]
            if len(conv_weight.shape) == 4:
                in_channels = conv_weight.shape[1]

        print(f"  ✓ 配置: deep_supervision={has_deep_supervision}, in_channels={in_channels}")

        # 加载DNANet
        nb_filter, num_blocks = load_param('three', 'resnet_18')
        model = DNANet(
            num_classes=1,
            input_channels=in_channels,
            block=Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=has_deep_supervision
        )
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

    print(f"  ✓ 模型加载完成")

    # 3. 确定要分析的类别
    if args.category == 'all':
        categories_to_analyze = [0, 1, 2, 3]
    else:
        categories_to_analyze = [int(args.category)]

    # 自动生成输出文件前缀和图像文件夹
    if args.output_prefix == '':
        # 从模型类型推断前缀
        prefix = model_type.lower()
    else:
        prefix = args.output_prefix

    # 自动检测图像文件夹
    if args.image_folder == '':
        # Mamba使用images，DNANet的8bit版本使用images-8bit
        if model_type == 'Mamba':
            image_folder = 'images'
        elif '8bit' in args.checkpoint.lower():
            image_folder = 'images-8bit'
        else:
            image_folder = 'images'
        print(f"  ✓ 自动检测图像文件夹: {image_folder}")
    else:
        image_folder = args.image_folder
        print(f"  ✓ 使用指定图像文件夹: {image_folder}")

    # 4. 分析每个类别
    print(f"\n[3/3] 分析类别...")
    results_summary = []

    for cat_id in categories_to_analyze:
        output_file = f"{prefix}_cat{cat_id}_difficulty_analysis.txt"
        result = analyze_single_category(
            cat_id, model, device, dataset_root, args.dataset,
            category_map, in_channels, output_file, image_folder, model_type
        )
        if result:
            results_summary.append(result)

    # 5. 打印总结
    print("\n" + "="*70)
    print("分析完成 - 总结")
    print("="*70)
    print()
    print(f"{'类别':<10} {'总数':<8} {'Easy':<8} {'Medium':<8} {'Hard':<8} {'平均IoU':<10}")
    print("-"*70)
    for r in results_summary:
        print(f"Cat{r['category']:<6} {r['total']:<8} {r['easy']:<8} {r['medium']:<8} {r['hard']:<8} {r['mean_iou']:<10.4f}")
    print()


if __name__ == '__main__':
    main()
