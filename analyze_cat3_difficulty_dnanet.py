#!/usr/bin/env python3
"""
分析Cat3样本的难度分级 - DNANet版本

完全参照test_box_iou.py实现，确保正确性
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model.utils_lidar import PoLaRISTestLoader, polaris_collate_fn
from model.metric import calculate_mask_to_box_iou
from model.model_DNANet import DNANet, Res_CBAM_block
from model.load_param_data import load_param


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k')
    parser.add_argument('--output', type=str, default='dnanet_cat3_difficulty_analysis.txt')
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()


def load_category_mapping(dataset_root, dataset_name):
    """加载类别映射"""
    category_file = os.path.join(dataset_root, dataset_name, 'selection_summary_new.txt')
    if not os.path.exists(category_file):
        category_file = os.path.join(dataset_root, dataset_name, 'select-view/selection_summary_new.txt')

    if not os.path.exists(category_file):
        return None, None

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

    return category_map, None


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("Cat3 样本难度分析 - DNANet")
    print("="*70)

    # 1. 加载类别映射
    print("\n[1/5] 加载类别映射...")
    dataset_root = 'dataset'
    category_map, _ = load_category_mapping(dataset_root, args.dataset)

    if category_map is None:
        print("❌ 无法加载类别映射")
        return

    # 获取Cat3样本
    cat3_samples = sorted([img_id for img_id, cat in category_map.items() if cat == 3])
    print(f"✓ 找到 {len(cat3_samples)} 个 Cat3 样本")

    # 2. 加载模型
    print("\n[2/5] 加载DNANet模型...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # 检测配置（参照test_box_iou.py）
    has_deep_supervision = any('final1.' in key for key in state_dict.keys())

    # 检测in_channels
    first_conv_key = None
    for key in state_dict.keys():
        if 'conv0_0' in key and 'weight' in key and 'conv0_0.0' in key:
            first_conv_key = key
            break

    in_channels = 2  # 默认值
    if first_conv_key:
        conv_weight = state_dict[first_conv_key]
        if len(conv_weight.shape) == 4:
            in_channels = conv_weight.shape[1]
            print(f"  ℹ️  从 {first_conv_key} 检测到 in_channels: {in_channels}")

    print(f"  检测配置: deep_supervision={has_deep_supervision}, in_channels={in_channels}")

    # 加载DNANet（参照test_box_iou.py第388-399行）
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

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ 模型加载完成 (参数: {num_params / 1e6:.2f}M)")

    # 3. 加载数据
    print("\n[3/5] 加载Cat3测试数据...")
    dataset_dir = os.path.join(dataset_root, args.dataset)
    test_loader_obj = PoLaRISTestLoader(
        dataset_dir=dataset_dir,
        img_id=cat3_samples,
        base_size=256,
        crop_size=256,
        transform=None,
        suffix='.png',
        normalize_16bit=True,
        in_channels=in_channels,
        image_folder='images',
    )

    test_loader = torch.utils.data.DataLoader(
        test_loader_obj,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=polaris_collate_fn
    )
    print(f"✓ 加载 {len(test_loader_obj)} 个样本")

    # 4. 评估
    print("\n[4/5] 评估每个样本...")
    sample_scores = []

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            data = batch_data['image'].to(device)
            labels = batch_data['mask'].to(device)
            img_id = batch_data['img_id'][0]

            # DNANet推理（参照test_box_iou.py）
            # DNANet直接接受完整输入，不需要分离IR和LiDAR
            pred = model(data)

            # 处理深度监督模型返回list的情况（test_box_iou.py第711-715行）
            # DNANet在deep_supervision=True时返回 [output1, output2, output3, output4]
            # 主输出在最后
            if isinstance(pred, list):
                pred = pred[-1]  # DNANet系列：主输出在最后

            # DNANet输出logits，需要sigmoid（test_box_iou.py第399行）
            pred = torch.sigmoid(pred)

            # 动态阈值扫描（参照test_box_iou.py第745行，用list而不是arange）
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

    # 5. 分析和保存
    print("\n[5/5] 分析难度分布...")
    sample_scores.sort(key=lambda x: x['box_iou'], reverse=True)

    easy = [s for s in sample_scores if s['box_iou'] >= 0.8]
    medium = [s for s in sample_scores if 0.5 <= s['box_iou'] < 0.8]
    hard = [s for s in sample_scores if s['box_iou'] < 0.5]

    print(f"\n难度分布:")
    print(f"  Easy (≥0.8):   {len(easy)} ({len(easy)/len(sample_scores)*100:.1f}%)")
    print(f"  Medium (0.5-0.8): {len(medium)} ({len(medium)/len(sample_scores)*100:.1f}%)")
    print(f"  Hard (<0.5):   {len(hard)} ({len(hard)/len(sample_scores)*100:.1f}%)")

    # 保存文本
    with open(args.output, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Cat3 样本难度分析 - DNANet\n")
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
    json_out = args.output.replace('.txt', '.json')
    with open(json_out, 'w') as f:
        json.dump({
            'easy': [s['img_id'] for s in easy],
            'medium': [s['img_id'] for s in medium],
            'hard': [s['img_id'] for s in hard],
            'scores': sample_scores
        }, f, indent=2)

    print(f"\n✅ 完成！")
    print(f"  文本: {args.output}")
    print(f"  JSON: {json_out}")
    print("="*70)


if __name__ == '__main__':
    main()
