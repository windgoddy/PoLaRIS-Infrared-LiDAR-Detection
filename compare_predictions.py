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
import numpy as np
import cv2
import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.utils_lidar import PoLaRISTestLoader
from model.load_param_data import load_dataset
from model_Mamba.core.polaris_mamba import create_multiscale_mamba

# DNANet imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'model'))
from model.DNANet import DNANet


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

    # 检测 in_channels
    first_conv_weight = checkpoint['model_state_dict']['conv0_0.0.conv1.weight']
    in_channels = first_conv_weight.shape[1]

    model = DNANet(in_channels=in_channels, num_classes=1)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    print(f"✅ DNANet loaded: in_channels={in_channels}")
    print(f"   Checkpoint IoU: {checkpoint.get('best_iou', 'N/A')}")

    return model, in_channels


def load_mamba_model(checkpoint_path, device):
    """加载 Mamba 模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 检测模型配置
    patch_embed_weight = checkpoint['model_state_dict']['patch_embed.proj.weight']
    ir_channels = patch_embed_weight.shape[1]
    embed_dim = patch_embed_weight.shape[0]

    # 检测是否使用 LiDAR
    use_lidar = any('lidar_gate' in k for k in checkpoint['model_state_dict'].keys())
    is_multiscale = any('skip_proj_s' in k for k in checkpoint['model_state_dict'].keys())

    model = create_multiscale_mamba(
        in_channels=ir_channels,
        embed_dim=embed_dim,
        use_lidar=use_lidar,
        num_classes=1,
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    print(f"✅ Mamba loaded: IR={ir_channels}, embed_dim={embed_dim}, LiDAR={use_lidar}, MultiScale={is_multiscale}")
    print(f"   Checkpoint IoU: {checkpoint.get('best_iou', 'N/A')}")

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

    # 创建 DataLoader
    # DNANet DataLoader (传统 3 通道)
    from model.load_param_data import test_dataloader
    dnanet_loader = test_dataloader(
        root=args.root,
        dataset=args.dataset,
        split_method=args.split_method,
        base_size=256,
        crop_size=256,
        batch_size=1,
    )

    # Mamba DataLoader (2 通道: IR + Depth)
    mamba_loader = PoLaRISTestLoader(
        dataset_dir=dataset_dir,
        img_id=test_img_ids,
        base_size=256,
        crop_size=256,
        in_channels=2,
        normalize_16bit=True,
    )

    # 存储结果
    results = []

    print("=" * 60)
    print("Running Inference")
    print("=" * 60)

    # 推理
    with torch.no_grad():
        for idx, (dnanet_sample, mamba_sample) in enumerate(tqdm(zip(dnanet_loader, mamba_loader),
                                                                   total=min(len(dnanet_loader), len(mamba_loader)))):
            # DNANet 推理
            dnanet_data = dnanet_sample['image'].to(device)  # (1, 3, H, W)
            dnanet_label = dnanet_sample['label'].cpu().numpy()[0, 0]  # (H, W)

            dnanet_pred = dnanet_model(dnanet_data)
            if isinstance(dnanet_pred, (list, tuple)):
                dnanet_pred = dnanet_pred[0]
            dnanet_pred = torch.sigmoid(dnanet_pred)
            dnanet_pred_np = dnanet_pred.cpu().numpy()[0, 0]  # (H, W)

            # Mamba 推理
            mamba_data = mamba_sample['image'].to(device)  # (1, 2, H, W)
            mamba_label = mamba_sample['label'].cpu().numpy()[0, 0]  # (H, W)

            # Split IR and Depth
            ir_input = mamba_data[:, 0:1, :, :]
            depth_input = mamba_data[:, 1:2, :, :]

            mamba_pred = mamba_model(ir_input, depth_input)
            if isinstance(mamba_pred, (list, tuple)):
                mamba_pred = mamba_pred[0]
            mamba_pred = torch.sigmoid(mamba_pred)
            mamba_pred_np = mamba_pred.cpu().numpy()[0, 0]  # (H, W)

            # 计算 IoU
            dnanet_seg_iou = calculate_iou(dnanet_pred_np, dnanet_label, args.threshold)
            mamba_seg_iou = calculate_iou(mamba_pred_np, mamba_label, args.threshold)

            dnanet_box_iou = calculate_box_iou(dnanet_pred_np, dnanet_label, args.threshold)
            mamba_box_iou = calculate_box_iou(mamba_pred_np, mamba_label, args.threshold)

            # IR 图像（从 DNANet 的输入中提取）
            ir_img = dnanet_data[0, 0].cpu().numpy()  # 取第一个通道

            # 存储结果
            results.append({
                'idx': idx,
                'img_id': mamba_sample['img_id'],
                'ir_img': ir_img,
                'gt_mask': dnanet_label,
                'dnanet_pred': dnanet_pred_np,
                'mamba_pred': mamba_pred_np,
                'dnanet_seg_iou': dnanet_seg_iou,
                'mamba_seg_iou': mamba_seg_iou,
                'dnanet_box_iou': dnanet_box_iou,
                'mamba_box_iou': mamba_box_iou,
            })

            # 限制样本数量
            if len(results) >= len(test_img_ids):
                break

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

    selected_high = sorted(high_iou, key=lambda x: x['mamba_seg_iou'], reverse=True)[:samples_per_category]
    selected_medium = sorted(medium_iou, key=lambda x: abs(x['mamba_seg_iou'] - 0.65))[:samples_per_category]
    selected_low = sorted(low_iou, key=lambda x: x['mamba_seg_iou'])[:samples_per_category]

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
        for r in tqdm(samples):
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

        f.write("Sample Distribution:\n")
        f.write(f"  High IoU (>0.8):     {len(high_iou)} ({len(high_iou)/len(results)*100:.1f}%)\n")
        f.write(f"  Medium IoU (0.5-0.8): {len(medium_iou)} ({len(medium_iou)/len(results)*100:.1f}%)\n")
        f.write(f"  Low IoU (<0.5):      {len(low_iou)} ({len(low_iou)/len(results)*100:.1f}%)\n\n")

        f.write("Critical Analysis Questions:\n")
        f.write("  1. 在 Overlay 图中，Mamba 的预测是否更符合物理形状（而非矩形）？\n")
        f.write("  2. DNANet 是否在角落区域有过多的假阳性（红色区域）？\n")
        f.write("  3. 低 IoU 样本的共同特征是什么（小目标/边缘/遮挡）？\n")
        f.write("  4. Mamba 的 Box IoU 与 Seg IoU 差距是否比 DNANet 更大（说明边界定位好但内部填充弱）？\n")

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
