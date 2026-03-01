#!/usr/bin/env python3
"""
分析Cat3样本的难度分级

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
from model_Mamba.core.polaris_mamba_progressive import polaris_mamba_tiny_progressive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k')
    parser.add_argument('--output', type=str, default='cat3_difficulty_analysis.txt')
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
    print("Cat3 样本难度分析")
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
    print("\n[2/5] 加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # 检测配置
    has_deep_supervision = any('aux_head' in key for key in state_dict.keys())
    has_cbam_full = any('head.ca.' in key for key in state_dict.keys())
    use_cbam = 'full' if has_cbam_full else 'none'

    print(f"  检测配置: deep_supervision={has_deep_supervision}, cbam={use_cbam}")

    model = polaris_mamba_tiny_progressive(
        use_lidar=True,
        use_deep_supervision=has_deep_supervision,
        use_cbam=use_cbam
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✓ 模型加载完成")

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
        in_channels=2,
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
            # 参照test_box_iou.py的正确方式
            data = batch_data['image'].to(device)
            labels = batch_data['mask'].to(device)  # 这里是'mask'不是'label'
            img_id = batch_data['img_id'][0]

            # Mamba模型推理（参照test_box_iou.py）
            if data.shape[1] == 2:
                ir = data[:, 0:1]
                lidar = data[:, 1:2]
            else:
                ir = data
                lidar = None

            pred = model(ir, lidar)

            # 动态阈值扫描（参照test_box_iou.py第757行）
            best_iou = 0.0
            best_thresh = 0.5

            for thresh in np.arange(0.1, 1.0, 0.1):
                # calculate_mask_to_box_iou接受tensor和threshold参数
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
        f.write("Cat3 样本难度分析\n")
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
