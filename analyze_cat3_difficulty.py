#!/usr/bin/env python3
"""
分析Cat3样本的难度分级

目标：
1. 在Cat3样本上运行模型预测
2. 根据Box IoU分为3个难度等级：Easy/Medium/Hard
3. 创建不同难度的测试集划分

使用方法：
    python analyze_cat3_difficulty.py \
      --checkpoint model_Mamba/result/xxx/best_model.pth \
      --output cat3_difficulty_analysis.txt
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from test_box_iou import (
    load_model_from_checkpoint,
    detect_model_params,
    create_model,
    load_category_mapping,
    calculate_mask_to_box_iou
)
from model.utils_lidar import PoLaRISTestLoader, polaris_collate_fn
from model.load_param_data import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Cat3 sample difficulty')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k',
                        help='Dataset name')
    parser.add_argument('--split_method', type=str, default='50_50_2k_new',
                        help='Dataset split')
    parser.add_argument('--output', type=str, default='cat3_difficulty_analysis.txt',
                        help='Output file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    return parser.parse_args()


def get_cat3_samples(category_map):
    """获取所有Cat3样本ID"""
    cat3_samples = []
    for img_id, cat in category_map.items():
        if cat == 3:
            cat3_samples.append(img_id)
    return sorted(cat3_samples)


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("Cat3 样本难度分析")
    print("="*70)

    # 加载类别映射
    print("\n[1/5] 加载类别映射...")
    dataset_dir = os.path.join('dataset', args.dataset)
    category_file = os.path.join(dataset_dir, 'select-view/selection_summary_new.txt')
    category_map, category_names = load_category_mapping(category_file)

    # 获取Cat3样本
    cat3_samples = get_cat3_samples(category_map)
    print(f"✓ 找到 {len(cat3_samples)} 个 Cat3 样本")

    # 加载模型
    print("\n[2/5] 加载模型...")
    checkpoint, checkpoint_info = load_model_from_checkpoint(args.checkpoint, device)

    # 简化：直接使用mamba_tiny_progressive
    from model_Mamba.core.polaris_mamba_progressive import polaris_mamba_tiny_progressive
    model = polaris_mamba_tiny_progressive(use_lidar=True, use_deep_supervision=False, use_cbam='none')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ 模型加载完成")

    # 加载测试集（仅Cat3样本）
    print("\n[3/5] 加载Cat3测试数据...")
    base_test_loader = PoLaRISTestLoader(
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
    print(f"✓ 加载了 {len(base_test_loader)} 个 Cat3 样本")

    # 评估每个样本
    print("\n[4/5] 评估每个Cat3样本...")
    sample_scores = []

    test_loader = torch.utils.data.DataLoader(
        base_test_loader,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=polaris_collate_fn
    )

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Cat3"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            img_ids = batch['img_id']

            # 预测
            outputs = model(images)

            # 计算Box IoU (使用动态阈值)
            best_iou = 0.0
            best_thresh = 0.5

            for thresh in np.arange(0.1, 1.0, 0.1):
                pred_binary = (outputs > thresh).float()
                iou = calculate_mask_to_box_iou(pred_binary[0], labels[0])
                if iou > best_iou:
                    best_iou = iou
                    best_thresh = thresh

            sample_scores.append({
                'img_id': img_ids[0],
                'box_iou': best_iou,
                'best_thresh': best_thresh
            })

    # 分析难度分布
    print("\n[5/5] 分析难度分布...")

    # 按Box IoU排序
    sample_scores.sort(key=lambda x: x['box_iou'], reverse=True)

    # 定义难度等级
    # Easy: Box IoU >= 0.8
    # Medium: 0.5 <= Box IoU < 0.8
    # Hard: Box IoU < 0.5

    easy_samples = [s for s in sample_scores if s['box_iou'] >= 0.8]
    medium_samples = [s for s in sample_scores if 0.5 <= s['box_iou'] < 0.8]
    hard_samples = [s for s in sample_scores if s['box_iou'] < 0.5]

    print(f"\n难度分布:")
    print(f"  Easy (IoU >= 0.8):   {len(easy_samples)} 样本 ({len(easy_samples)/len(sample_scores)*100:.1f}%)")
    print(f"  Medium (0.5-0.8):    {len(medium_samples)} 样本 ({len(medium_samples)/len(sample_scores)*100:.1f}%)")
    print(f"  Hard (IoU < 0.5):    {len(hard_samples)} 样本 ({len(hard_samples)/len(sample_scores)*100:.1f}%)")

    # 保存结果
    with open(args.output, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Cat3 样本难度分析\n")
        f.write("="*70 + "\n\n")

        f.write(f"总样本数: {len(sample_scores)}\n")
        f.write(f"平均Box IoU: {np.mean([s['box_iou'] for s in sample_scores]):.4f}\n")
        f.write(f"中位数Box IoU: {np.median([s['box_iou'] for s in sample_scores]):.4f}\n\n")

        f.write("难度分布:\n")
        f.write(f"  Easy (IoU >= 0.8):   {len(easy_samples)} 样本 ({len(easy_samples)/len(sample_scores)*100:.1f}%)\n")
        f.write(f"  Medium (0.5-0.8):    {len(medium_samples)} 样本 ({len(medium_samples)/len(sample_scores)*100:.1f}%)\n")
        f.write(f"  Hard (IoU < 0.5):    {len(hard_samples)} 样本 ({len(hard_samples)/len(sample_scores)*100:.1f}%)\n\n")

        f.write("="*70 + "\n")
        f.write("详细样本列表 (按难度排序)\n")
        f.write("="*70 + "\n\n")

        for difficulty, samples in [('Easy', easy_samples), ('Medium', medium_samples), ('Hard', hard_samples)]:
            f.write(f"\n{difficulty} 样本 ({len(samples)} 个):\n")
            f.write("-"*70 + "\n")
            for s in samples:
                f.write(f"  {s['img_id']:20s}  Box IoU: {s['box_iou']:.4f}  Thresh: {s['best_thresh']:.1f}\n")

    # 保存JSON格式（方便后续使用）
    json_output = args.output.replace('.txt', '.json')
    with open(json_output, 'w') as f:
        json.dump({
            'easy': [s['img_id'] for s in easy_samples],
            'medium': [s['img_id'] for s in medium_samples],
            'hard': [s['img_id'] for s in hard_samples],
            'scores': sample_scores
        }, f, indent=2)

    print(f"\n✅ 分析完成！")
    print(f"  - 文本报告: {args.output}")
    print(f"  - JSON数据: {json_output}")
    print("="*70)


if __name__ == '__main__':
    main()
