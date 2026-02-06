#!/usr/bin/env python3
"""
测试 PoLaRIS Mamba MultiScale 模型的评估脚本

用法:
    python test_mamba_model.py \\
        --checkpoint model_Mamba/result/MultiScale_Full_d2.5p2_20260205_141846/latest_best_model.pth \\
        --dataset Pohang-Canal-3k \\
        --split_method 50_50_2k_new \\
        --image_folder images-8bit \\
        --in_channels 2
"""

import torch
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入 Mamba 模型和工具
from model_Mamba.core.polaris_mamba_multiscale import polaris_mamba_tiny_multiscale
from model_Mamba.core.metrics import compute_miou_with_sweep, calculate_mask_to_box_iou
from model.load_param_data import load_dataset
from model.utils import TestSetLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Test PoLaRIS Mamba Model')
    
    # 模型和数据
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k',
                        help='Dataset name')
    parser.add_argument('--root', type=str, default='dataset/',
                        help='Dataset root directory')
    parser.add_argument('--split_method', type=str, default='50_50_2k_new',
                        help='Train/test split method')
    parser.add_argument('--image_folder', type=str, default='images-8bit',
                        help='Image folder name (e.g., images-8bit, images-16bit)')
    parser.add_argument('--suffix', type=str, default='.png',
                        help='Image file suffix')
    
    # 输入配置
    parser.add_argument('--in_channels', type=int, default=2,
                        help='Number of input channels: 1 (IR only), 2 (IR + Depth)')
    parser.add_argument('--base_size', type=int, default=512,
                        help='Base image size')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='Crop size (test用全尺寸)')
    
    # 评估参数
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='Test batch size')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU IDs (e.g., 0 or 0,1)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("🚀 PoLaRIS Mamba MultiScale 模型测试")
    print("="*70)
    print(f"📂 数据集: {args.dataset}")
    print(f"📂 图像文件夹: {args.image_folder}")
    print(f"📂 分割方法: {args.split_method}")
    print(f"🔧 输入通道: {args.in_channels} ({'IR only' if args.in_channels == 1 else 'IR + Depth'})")
    print(f"📦 模型权重: {args.checkpoint}")
    print("="*70)
    
    # 加载数据集
    dataset_dir = os.path.join(args.root, args.dataset)
    _, val_img_ids, _ = load_dataset(args.root, args.dataset, args.split_method)
    
    print(f"\n📊 测试集大小: {len(val_img_ids)} 张图像")
    
    # 数据预处理（Mamba不需要normalize，已经在dataset内部处理）
    from torchvision import transforms
    input_transform = transforms.ToTensor()  # Mamba的normalize在TestSetLoader内部
    
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
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        drop_last=False,
        shuffle=False
    )
    
    # 加载模型
    print("\n🏗️  初始化模型...")
    use_lidar = (args.in_channels == 2)
    model = polaris_mamba_tiny_multiscale(
        use_lidar=use_lidar,
        use_deep_supervision=True  # Mamba训练时使用了deep supervision
    )
    model = model.to(device)
    
    # 加载权重
    print(f"📦 加载权重: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 检查checkpoint结构
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        best_iou = checkpoint.get('best_val_miou', 'unknown')
        print(f"   Epoch: {epoch}, Best Val mIoU: {best_iou}")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"   权重加载完成")
    else:
        model.load_state_dict(checkpoint)
        print(f"   权重加载完成（直接加载）")
    
    print("✅ 模型初始化完成")
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   总参数量: {total_params:,}")
    
    # 评估模式
    model.eval()
    
    # 收集预测和标签
    all_preds = []
    all_labels = []
    
    print("\n🔍 开始评估...")
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            data = data.to(device)
            labels = labels.to(device)
            
            # 前向传播（Mamba返回多个输出，取最后一个）
            outputs = model(data)
            if isinstance(outputs, (list, tuple)):
                pred = outputs[-1]  # 取deep supervision的最后一个输出
            else:
                pred = outputs
            
            # 转为概率
            prob = torch.sigmoid(pred)
            
            # 调试第一个batch
            if i == 0:
                print(f"\n🔍 [调试] 第一个 batch:")
                print(f"   输入形状: {data.shape}")
                print(f"   输入范围: [{data.min():.3f}, {data.max():.3f}]")
                print(f"   pred 范围: [{pred.min():.3f}, {pred.max():.3f}]")
                print(f"   prob 范围: [{prob.min():.3f}, {prob.max():.3f}]")
                print(f"   GT 非零像素: {(labels > 0.5).sum().item()}")
            
            # 收集数据
            all_preds.append(prob.cpu())
            all_labels.append(labels.cpu())
    
    # 合并所有预测
    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    
    # 动态阈值扫描
    print("\n" + "="*70)
    print("🔍 执行动态阈值扫描 (0.1~0.9)")
    print("="*70)
    
    best_iou, best_thresh, iou_curve = compute_miou_with_sweep(
        all_preds_tensor, all_labels_tensor,
        thresh_step=0.1, thresh_range=(0.1, 0.9)
    )
    
    # 计算 Box IoU
    box_iou = calculate_mask_to_box_iou(
        all_preds_tensor, all_labels_tensor, threshold=best_thresh
    )
    
    # 打印结果
    print(f"\n✅ 最佳 mIoU: {best_iou:.4f} @ 阈值 {best_thresh:.2f}")
    print(f"✅ 对应 Box IoU: {box_iou:.4f} (使用相同阈值 {best_thresh:.2f})")
    print(f"\n📊 阈值扫描曲线:")
    for thresh, iou in iou_curve:
        marker = " ← 最佳" if abs(thresh - best_thresh) < 0.01 else ""
        print(f"  阈值 {thresh:.1f}: IoU {iou:.4f}{marker}")
    
    print("\n" + "="*70)
    print("📊 测试结果汇总")
    print("="*70)
    print(f"mean_IOU:  {best_iou:.4f}")
    print(f"Box IoU:   {box_iou:.4f}")
    print(f"最佳阈值:   {best_thresh:.2f}")
    print("="*70)


if __name__ == "__main__":
    main()
