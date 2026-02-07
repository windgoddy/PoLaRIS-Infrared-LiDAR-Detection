#!/usr/bin/env python3
"""
DNANet vs Mamba 预测对比可视化脚本
========================================

功能：
1. 并排对比 DNANet 和 Mamba 的预测结果
2. 分析不同 IoU 区间的样本特征
3. 验证 "Mamba 学习物理形状 vs DNANet 拟合矩形框" 的假设

用法：
    python compare_predictions.py \
        --dnanet_checkpoint result/DNANet_baseline_8bit_Pohang-Canal-3k_DNANet_28_01_2026_17_37_58_wDS/latest_best_model.pth.tar \
        --mamba_checkpoint model_Mamba/result/MultiScale_Full_d2.5p2_20260206_162111/latest_best_model.pth \
        --num_samples 30 \
        --output_dir comparison_analysis/

输出：
    - 高 IoU 样本（>0.8）：10张
    - 中等 IoU 样本（0.5-0.8）：10张
    - 低 IoU 样本（<0.5）：10张
    - 统计报告：model_comparison_report.txt
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.utils_lidar import PoLaRISTestLoader, polaris_collate_fn
from model.utils import TestSetLoader
from model.load_param_data import load_dataset, load_param
import torchvision.transforms as transforms

# Mamba models
from model_Mamba.core.polaris_mamba import PoLaRIS_Mamba
from model_Mamba.core.polaris_mamba_multiscale import PoLaRIS_Mamba_MultiScale

# DNANet imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'model'))
from model.model_DNANet import DNANet, Res_CBAM_block


def parse_args():
    parser = argparse.ArgumentParser(description='DNANet vs Mamba Comparison')

    # Checkpoints
    parser.add_argument('--dnanet_checkpoint', type=str, required=True)
    parser.add_argument('--mamba_checkpoint', type=str, required=True)

    # Dataset
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--split_method', type=str, default='50_50_2k_new')

    # Visualization settings
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Total samples to visualize (10 per IoU range)')
    parser.add_argument('--output_dir', type=str, default='comparison_analysis/')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binary threshold for predictions')

    # Hardware
    parser.add_argument('--gpu', type=int, default=0)

    return parser.parse_args()


def load_dnanet_model(checkpoint_path, device):
    """加载 DNANet 模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 兼容不同的键名（参考 test_box_iou.py）
    state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict')
    if state_dict is None:
        raise KeyError("Checkpoint missing state_dict/model_state_dict")

    # 检测 in_channels
    first_conv_key = None
    for key in state_dict.keys():
        if 'conv0_0' in key and 'weight' in key and 'conv0_0.0' in key:
            first_conv_key = key
            break

    if first_conv_key:
        first_conv_weight = state_dict[first_conv_key]
        in_channels = first_conv_weight.shape[1]
    else:
        in_channels = 3  # 默认值

    # 检测 deep_supervision
    deep_supervision = any('final1.' in key for key in state_dict.keys())

    # 获取网络参数（参考 test_box_iou.py）
    nb_filter, num_blocks = load_param('three', 'resnet_18')

    model = DNANet(
        num_classes=1,
        input_channels=in_channels,
        block=Res_CBAM_block,
        num_blocks=num_blocks,
        nb_filter=nb_filter,
        deep_supervision=deep_supervision
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✅ DNANet loaded: in_channels={in_channels}, deep_supervision={deep_supervision}")
    print(f"   Checkpoint IoU: {checkpoint.get('mean_IOU', checkpoint.get('best_iou', 'N/A'))}")

    return model, in_channels


def load_mamba_model(checkpoint_path, device):
    """加载 Mamba 模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 兼容不同的键名
    state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict')
    if state_dict is None:
        raise KeyError("Checkpoint missing state_dict/model_state_dict")

    # 检测模型配置
    patch_embed_weight = state_dict['patch_embed.proj.weight']
    ir_channels = patch_embed_weight.shape[1]
    embed_dim = patch_embed_weight.shape[0]

    # 检测是否使用 LiDAR
    use_lidar = any('lidar_gate' in k for k in state_dict.keys())
    is_multiscale = any('skip_proj_s' in k for k in state_dict.keys())

    # 根据 embed_dim 推断 depths (参考 test_box_iou.py)
    depths_map = {64: [2, 2, 4, 2], 96: [2, 2, 6, 2], 128: [2, 2, 12, 2]}
    depths = depths_map.get(embed_dim, [2, 2, 6, 2])

    # 根据是否多尺度选择模型类
    if is_multiscale:
        model = PoLaRIS_Mamba_MultiScale(
            in_channels=1,  # Always 1 for IR (patch_embed)
            embed_dim=embed_dim,
            depths=depths,
            use_lidar=use_lidar
        )
        print(f"✅ Mamba (MultiScale) loaded: embed_dim={embed_dim}, depths={depths}, LiDAR={use_lidar}")
    else:
        model = PoLaRIS_Mamba(
            in_channels=1,  # Always 1 for IR (patch_embed)
            embed_dim=embed_dim,
            depths=depths,
            use_lidar=use_lidar
        )
        print(f"✅ Mamba (Standard) loaded: embed_dim={embed_dim}, depths={depths}, LiDAR={use_lidar}")

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"   Checkpoint IoU: {checkpoint.get('mean_IOU', checkpoint.get('best_iou', 'N/A'))}")

    return model, use_lidar


def calculate_iou(pred, gt, threshold=0.5):
    """计算 IoU"""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > 0.5).astype(np.float32)

    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection

    if union < 1e-6:
        return 0.0

    return intersection / union


def calculate_box_iou(pred, gt, threshold=0.5):
    """计算 Mask-to-Box IoU"""
    pred_binary = (pred > threshold).astype(np.uint8)
    gt_binary = (gt > 0.5).astype(np.uint8)

    # 获取 GT 的 bounding box
    contours_gt, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_gt) == 0:
        return 0.0

    # 取最大的 contour
    contour_gt = max(contours_gt, key=cv2.contourArea)
    x_gt, y_gt, w_gt, h_gt = cv2.boundingRect(contour_gt)

    # 创建 GT box mask
    gt_box_mask = np.zeros_like(gt_binary)
    gt_box_mask[y_gt:y_gt+h_gt, x_gt:x_gt+w_gt] = 1

    # 计算 IoU
    intersection = (pred_binary * gt_box_mask).sum()
    union = pred_binary.sum() + gt_box_mask.sum() - intersection

    if union < 1e-6:
        return 0.0

    return intersection / union


def visualize_comparison(
    ir_img,
    gt_mask,
    dnanet_pred,
    mamba_pred,
    dnanet_iou,
    mamba_iou,
    dnanet_box_iou,
    mamba_box_iou,
    img_id,
    output_path,
    threshold=0.5,
):
    """
    创建对比可视化图

    布局：
    [IR Image] [GT Mask] [DNANet Pred] [Mamba Pred] [Overlay Comparison]
    """
    H, W = ir_img.shape

    # 创建 5 列画布
    fig_width = W * 5
    canvas = np.zeros((H, fig_width, 3), dtype=np.uint8)

    # Column 1: IR Image
    ir_uint8 = (ir_img * 255).clip(0, 255).astype(np.uint8)
    ir_bgr = cv2.cvtColor(ir_uint8, cv2.COLOR_GRAY2BGR)
    canvas[:, 0:W, :] = ir_bgr

    # Column 2: GT Mask
    gt_uint8 = (gt_mask * 255).astype(np.uint8)
    gt_bgr = cv2.cvtColor(gt_uint8, cv2.COLOR_GRAY2BGR)
    # 画出 bounding box
    gt_binary = (gt_mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(gt_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    canvas[:, W:2*W, :] = gt_bgr

    # Column 3: DNANet Prediction
    dnanet_uint8 = (dnanet_pred * 255).clip(0, 255).astype(np.uint8)
    dnanet_bgr = cv2.applyColorMap(dnanet_uint8, cv2.COLORMAP_JET)
    canvas[:, 2*W:3*W, :] = dnanet_bgr

    # Column 4: Mamba Prediction
    mamba_uint8 = (mamba_pred * 255).clip(0, 255).astype(np.uint8)
    mamba_bgr = cv2.applyColorMap(mamba_uint8, cv2.COLORMAP_JET)
    canvas[:, 3*W:4*W, :] = mamba_bgr

    # Column 5: Overlay Comparison
    # 红色: DNANet 独有, 绿色: Mamba 独有, 黄色: 共同区域
    dnanet_binary = (dnanet_pred > threshold).astype(np.uint8)
    mamba_binary = (mamba_pred > threshold).astype(np.uint8)

    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[:, :, 2] = dnanet_binary * 255  # 红色通道: DNANet
    overlay[:, :, 1] = mamba_binary * 255   # 绿色通道: Mamba
    # 黄色区域会自动出现在重叠处

    canvas[:, 4*W:5*W, :] = overlay

    # 添加标签
    font_scale = 0.7
    font_thickness = 2
    cv2.putText(canvas, 'IR Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thickness)
    cv2.putText(canvas, f'GT (Box)', (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thickness)
    cv2.putText(canvas, f'DNANet', (2*W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thickness)
    cv2.putText(canvas, f'Mamba', (3*W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thickness)
    cv2.putText(canvas, f'Overlay', (4*W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thickness)

    # 添加 IoU 信息
    info_y = H - 80
    cv2.putText(canvas, f'DNANet:', (2*W + 10, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f'  Seg IoU: {dnanet_iou:.4f}', (2*W + 10, info_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(canvas, f'  Box IoU: {dnanet_box_iou:.4f}', (2*W + 10, info_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.putText(canvas, f'Mamba:', (3*W + 10, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f'  Seg IoU: {mamba_iou:.4f}', (3*W + 10, info_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(canvas, f'  Box IoU: {mamba_box_iou:.4f}', (3*W + 10, info_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Overlay 图例
    cv2.putText(canvas, 'Red: DNANet only', (4*W + 10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(canvas, 'Green: Mamba only', (4*W + 10, info_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(canvas, 'Yellow: Both', (4*W + 10, info_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 保存
    cv2.imwrite(output_path, canvas)

    return output_path


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'high_iou').mkdir(exist_ok=True)
    (output_dir / 'medium_iou').mkdir(exist_ok=True)
    (output_dir / 'low_iou').mkdir(exist_ok=True)

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}\n")

    # 加载模型
    print("=" * 60)
    print("Loading Models")
    print("=" * 60)
    dnanet_model, dnanet_channels = load_dnanet_model(args.dnanet_checkpoint, device)
    mamba_model, use_lidar = load_mamba_model(args.mamba_checkpoint, device)
    print()

    # 加载数据集
    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)
    dataset_dir = os.path.join(args.root, args.dataset)
    _, test_img_ids, _ = load_dataset(args.root, args.dataset, args.split_method)
    print(f"✅ Total test samples: {len(test_img_ids)}\n")

    # 创建 DNANet DataLoader (使用 TestSetLoader，与训练时一致)
    print(f"  ✓ 创建 DNANet DataLoader (TestSetLoader)")
    if dnanet_channels == 1:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    dnanet_dataset = TestSetLoader(
        dataset_dir,
        img_id=test_img_ids,
        base_size=256,
        crop_size=256,
        transform=input_transform,
        suffix='.png',
        in_channels=dnanet_channels,
        image_folder='images'
    )

    dnanet_loader = DataLoader(
        dataset=dnanet_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # 创建 Mamba DataLoader (使用 PoLaRISTestLoader，与训练时一致)
    print(f"  ✓ 创建 Mamba DataLoader (PoLaRISTestLoader)")
    mamba_dataset = PoLaRISTestLoader(
        dataset_dir=dataset_dir,
        img_id=test_img_ids,
        base_size=256,
        crop_size=256,
        in_channels=2,
        normalize_16bit=True,
        suffix='.png'
    )

    mamba_loader = DataLoader(
        dataset=mamba_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=polaris_collate_fn
    )

    print(f"✅ DNANet DataLoader: {len(dnanet_loader)} batches")
    print(f"✅ Mamba DataLoader: {len(mamba_loader)} batches\n")

    # 存储结果
    results = []

    print("=" * 60)
    print("Running Inference")
    print("=" * 60)

    # 推理
    with torch.no_grad():
        for idx, (dnanet_batch, mamba_batch) in enumerate(tqdm(
            zip(dnanet_loader, mamba_loader),
            total=len(dnanet_loader),
            desc="Testing"
        )):
            img_id = test_img_ids[idx]
            # ========== DNANet 数据 ==========
            # TestSetLoader 返回 (img, mask) tuple
            dnanet_img, dnanet_mask = dnanet_batch
            dnanet_img = dnanet_img.to(device)  # (1, C, H, W) - 已经过 transform 归一化
            gt_mask = dnanet_mask.cpu().numpy()[0, 0]  # (H, W) - 范围 [0, 1]

            # ========== Mamba 数据 ==========
            # PoLaRISTestLoader 返回 dict
            mamba_img = mamba_batch['image'].to(device)  # (1, 2, H, W) - [IR, Depth]

            # 调试：检查第一个 batch
            if idx == 0:
                print(f"\n🔍 Debug Info (first batch):")
                print(f"  DNANet img shape: {dnanet_img.shape}, range: [{dnanet_img.min():.4f}, {dnanet_img.max():.4f}]")
                print(f"  DNANet mask shape: {dnanet_mask.shape}, range: [{dnanet_mask.min():.4f}, {dnanet_mask.max():.4f}]")
                print(f"  Mamba img shape: {mamba_img.shape}, range: [{mamba_img.min():.4f}, {mamba_img.max():.4f}]")
                print(f"  GT mask shape: {gt_mask.shape}, sum: {gt_mask.sum():.4f}")
                print(f"  img_id: {img_id}")

            # ========== DNANet 推理 ==========
            # DNANet 输入已经过 TestSetLoader + transform 正确归一化
            dnanet_pred = dnanet_model(dnanet_img)
            if isinstance(dnanet_pred, tuple):
                dnanet_pred = dnanet_pred[0]
            if isinstance(dnanet_pred, list):
                dnanet_pred = dnanet_pred[-1]
            if idx == 0:
                print(f"  DNANet logits shape: {dnanet_pred.shape}, range: [{dnanet_pred.min().item():.4f}, {dnanet_pred.max().item():.4f}], mean: {dnanet_pred.mean().item():.4f}")
            dnanet_pred = torch.sigmoid(dnanet_pred)
            if idx == 0:
                print(f"  DNANet sigmoid range: [{dnanet_pred.min().item():.4f}, {dnanet_pred.max().item():.4f}], mean: {dnanet_pred.mean().item():.4f}")
                print(f"  DNANet >{args.threshold:.2f} ratio: {(dnanet_pred > args.threshold).float().mean().item():.6f}")
            dnanet_pred_np = dnanet_pred.cpu().numpy()[0, 0]  # (H, W)

            # ========== Mamba 推理 ==========
            # Mamba 需要分离 IR 和 Depth
            ir_channel = mamba_img[:, 0:1, :, :]  # (1, 1, H, W)
            depth_channel = mamba_img[:, 1:2, :, :]  # (1, 1, H, W)

            mamba_pred = mamba_model(ir_channel, depth_channel)
            if isinstance(mamba_pred, tuple):
                mamba_pred = mamba_pred[0]
            if isinstance(mamba_pred, list):
                mamba_pred = mamba_pred[-1]
            if idx == 0:
                print(f"  Mamba logits shape: {mamba_pred.shape}, range: [{mamba_pred.min().item():.4f}, {mamba_pred.max().item():.4f}], mean: {mamba_pred.mean().item():.4f}")
            mamba_pred = torch.sigmoid(mamba_pred)
            if idx == 0:
                print(f"  Mamba sigmoid range: [{mamba_pred.min().item():.4f}, {mamba_pred.max().item():.4f}], mean: {mamba_pred.mean().item():.4f}")
                print(f"  Mamba >{args.threshold:.2f} ratio: {(mamba_pred > args.threshold).float().mean().item():.6f}")
            mamba_pred_np = mamba_pred.cpu().numpy()[0, 0]  # (H, W)

            # ========== 计算指标 ==========
            dnanet_seg_iou = calculate_iou(dnanet_pred_np, gt_mask, args.threshold)
            mamba_seg_iou = calculate_iou(mamba_pred_np, gt_mask, args.threshold)

            dnanet_box_iou = calculate_box_iou(dnanet_pred_np, gt_mask, args.threshold)
            mamba_box_iou = calculate_box_iou(mamba_pred_np, gt_mask, args.threshold)

            # ========== 准备可视化图像 ==========
            # 使用 DNANet 的输入图像（需要反归一化）
            if dnanet_channels == 1:
                # 单通道：反归一化 [0.5, 0.5]
                ir_img = dnanet_img[0, 0].cpu().numpy() * 0.5 + 0.5
            elif dnanet_channels == 3:
                # RGB：反归一化 ImageNet，取第一个通道
                img_np = dnanet_img[0].cpu().numpy()  # (3, H, W)
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img_denorm = img_np * std + mean
                img_denorm = np.clip(img_denorm, 0, 1)
                ir_img = img_denorm[0]  # 取第一个通道
            else:
                # 2-channel
                ir_img = dnanet_img[0, 0].cpu().numpy() * 0.5 + 0.5

            # 存储结果
            results.append({
                'idx': idx,
                'img_id': img_id,
                'ir_img': ir_img,
                'gt_mask': gt_mask,
                'dnanet_pred': dnanet_pred_np,
                'mamba_pred': mamba_pred_np,
                'dnanet_seg_iou': dnanet_seg_iou,
                'mamba_seg_iou': mamba_seg_iou,
                'dnanet_box_iou': dnanet_box_iou,
                'mamba_box_iou': mamba_box_iou,
            })

    print(f"\n✅ Processed {len(results)} samples\n")

    # 分类样本
    print("=" * 60)
    print("Categorizing Samples")
    print("=" * 60)

    high_iou = [r for r in results if r['mamba_seg_iou'] > 0.8]
    medium_iou = [r for r in results if 0.5 <= r['mamba_seg_iou'] <= 0.8]
    low_iou = [r for r in results if r['mamba_seg_iou'] < 0.5]

    print(f"High IoU (>0.8):     {len(high_iou)} samples")
    print(f"Medium IoU (0.5-0.8): {len(medium_iou)} samples")
    print(f"Low IoU (<0.5):      {len(low_iou)} samples")
    print()

    # 选择代表性样本
    samples_per_category = args.num_samples // 3

    # 检查样本数量并给出警告
    if len(high_iou) < samples_per_category:
        print(f"⚠️  Warning: Only {len(high_iou)} high IoU samples available (requested {samples_per_category})")
    if len(medium_iou) < samples_per_category:
        print(f"⚠️  Warning: Only {len(medium_iou)} medium IoU samples available (requested {samples_per_category})")
    if len(low_iou) < samples_per_category:
        print(f"⚠️  Warning: Only {len(low_iou)} low IoU samples available (requested {samples_per_category})")

    selected_high = sorted(high_iou, key=lambda x: x['mamba_seg_iou'], reverse=True)[:samples_per_category]
    selected_medium = sorted(medium_iou, key=lambda x: abs(x['mamba_seg_iou'] - 0.65))[:samples_per_category]
    selected_low = sorted(low_iou, key=lambda x: x['mamba_seg_iou'])[:samples_per_category]

    print(f"\n✅ Selected samples: High={len(selected_high)}, Medium={len(selected_medium)}, Low={len(selected_low)}")
    print()

    # 生成可视化
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    for category, samples, subdir in [
        ('High IoU', selected_high, 'high_iou'),
        ('Medium IoU', selected_medium, 'medium_iou'),
        ('Low IoU', selected_low, 'low_iou'),
    ]:
        print(f"\n{category}:")
        for r in tqdm(samples, desc=f"  {category}"):
            output_path = output_dir / subdir / f"{r['img_id']}.png"
            visualize_comparison(
                r['ir_img'],
                r['gt_mask'],
                r['dnanet_pred'],
                r['mamba_pred'],
                r['dnanet_seg_iou'],
                r['mamba_seg_iou'],
                r['dnanet_box_iou'],
                r['mamba_box_iou'],
                r['img_id'],
                str(output_path),
                args.threshold,
            )

    # 生成统计报告
    print("\n" + "=" * 60)
    print("Generating Report")
    print("=" * 60)

    report_path = output_dir / 'model_comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DNANet vs Mamba Model Comparison Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Samples: {len(results)}\n\n")

        f.write("DNANet Performance:\n")
        f.write(f"  Avg Seg IoU:  {np.mean([r['dnanet_seg_iou'] for r in results]):.4f}\n")
        f.write(f"  Avg Box IoU:  {np.mean([r['dnanet_box_iou'] for r in results]):.4f}\n")
        f.write(f"  Median Seg IoU: {np.median([r['dnanet_seg_iou'] for r in results]):.4f}\n")
        f.write(f"  Median Box IoU: {np.median([r['dnanet_box_iou'] for r in results]):.4f}\n\n")

        f.write("Mamba Performance:\n")
        f.write(f"  Avg Seg IoU:  {np.mean([r['mamba_seg_iou'] for r in results]):.4f}\n")
        f.write(f"  Avg Box IoU:  {np.mean([r['mamba_box_iou'] for r in results]):.4f}\n")
        f.write(f"  Median Seg IoU: {np.median([r['mamba_seg_iou'] for r in results]):.4f}\n")
        f.write(f"  Median Box IoU: {np.median([r['mamba_box_iou'] for r in results]):.4f}\n\n")

        f.write("Gap Analysis (Box IoU - Seg IoU):\n")
        dnanet_gap = np.mean([r['dnanet_box_iou'] - r['dnanet_seg_iou'] for r in results])
        mamba_gap = np.mean([r['mamba_box_iou'] - r['mamba_seg_iou'] for r in results])
        f.write(f"  DNANet Gap: {dnanet_gap:.4f}\n")
        f.write(f"  Mamba Gap:  {mamba_gap:.4f}\n")
        f.write(f"  → Mamba gap is {'LARGER' if mamba_gap > dnanet_gap else 'SMALLER'} ")
        f.write(f"(diff: {abs(mamba_gap - dnanet_gap):.4f})\n")
        if mamba_gap > dnanet_gap:
            f.write(f"  → 说明: Mamba 边界定位好但内部填充弱（可能更符合物理形状）\n\n")
        else:
            f.write(f"  → 说明: Mamba 内部填充与 DNANet 相当或更好\n\n")

        f.write("Sample Distribution:\n")
        f.write(f"  High IoU (>0.8):     {len(high_iou)} ({len(high_iou)/len(results)*100:.1f}%)\n")
        f.write(f"  Medium IoU (0.5-0.8): {len(medium_iou)} ({len(medium_iou)/len(results)*100:.1f}%)\n")
        f.write(f"  Low IoU (<0.5):      {len(low_iou)} ({len(low_iou)/len(results)*100:.1f}%)\n\n")

        f.write("Critical Analysis Questions:\n")
        f.write("  1. 在 Overlay 图中，Mamba 的预测是否更符合物理形状（而非矩形）？\n")
        f.write("  2. DNANet 是否在角落区域有过多的假阳性（红色区域）？\n")
        f.write("  3. 低 IoU 样本的共同特征是什么（小目标/边缘/遮挡）？\n")
        f.write(f"  4. Mamba 的 Box-Seg Gap 是否比 DNANet 更大？\n")
        f.write(f"     → Answer: {'YES' if mamba_gap > dnanet_gap else 'NO'} ")
        f.write(f"(Mamba: {mamba_gap:.4f}, DNANet: {dnanet_gap:.4f})\n")

    print(f"\n✅ Report saved to: {report_path}")
    print(f"✅ Visualizations saved to: {output_dir}")
    print("\n" + "=" * 60)
    print("🎯 下一步：仔细检查可视化图，验证以下假设")
    print("=" * 60)
    print("1. Mamba 预测是否呈现 '物理形状'（斑点/椭圆）而非矩形？")
    print("2. DNANet 预测是否过度填充矩形框的四个角？")
    print("3. 低 IoU 样本是否有共同模式（目标尺寸、位置、海况）？")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
