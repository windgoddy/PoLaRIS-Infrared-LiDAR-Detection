#!/usr/bin/env python3
"""
Mamba IR-Only Training Script (Fair Comparison with DNANet)
============================================================

目的：公平对比实验，使用与DNANet完全相同的输入：
- 8-bit IR图像（无LiDAR，无depth map）
- ImageNet归一化（与DNANet一致）
- Pixel-level mask（与DNANet一致，不使用oracle mask）
- TestSetLoader（与DNANet一致）

修改自：train.py
关键差异：
1. DataLoader: PoLaRISTrainLoader → TrainSetLoader
2. 输入: 2-channel (IR+Depth) → 1-channel (IR only)
3. 归一化: 自定义 → ImageNet标准
4. LiDAR: use_lidar=True → use_lidar=False
5. Mask: oracle_mask (soft) → hard mask (binary)

用法：
    python model_Mamba/train_ir_only.py \\
        --dataset Pohang-Canal-3k \\
        --model_type mamba_tiny_multiscale \\
        --mode train \\
        --batch_size 8 \\
        --epochs 500 \\
        --base_lr 0.001 \\
        --suffix .png
"""

import os
import sys
from datetime import datetime

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms  # 使用标准transforms
from tqdm import tqdm
import numpy as np
import random

# PoLaRIS utilities (使用传统DataLoader而非LiDAR版本)
from model.utils import TrainSetLoader, TestSetLoader  # ← 关键修改
from model.metric import ROCMetric, mIoU
from model.load_param_data import load_dataset

# Mamba models
from model_Mamba.core.polaris_mamba import PoLaRIS_Mamba
from model_Mamba.core.polaris_mamba_multiscale import PoLaRIS_Mamba_MultiScale

# Loss functions
from model_Mamba.core.loss import DiceLoss, BCEDiceLoss
from model_Mamba.core.loss_advanced import LossFactory


def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Mamba IR-Only Training (Fair Comparison)')

    # Dataset
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

    # Model
    parser.add_argument('--model_type', type=str, default='mamba_tiny_multiscale',
                        choices=['mamba_tiny', 'mamba_small', 'mamba_base', 'mamba_tiny_multiscale'],
                        help='Mamba model variant')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Input channels (1=IR only, 3=RGB-style)')

    # Training
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Training or testing mode')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='Test batch size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='Base learning rate')
    parser.add_argument('--base_size', type=int, default=256,
                        help='Base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Crop size for training')

    # Loss weights
    parser.add_argument('--bce_weight', type=float, default=1.0,
                        help='BCE loss weight')
    parser.add_argument('--dice_weight', type=float, default=2.5,
                        help='Dice loss weight')
    parser.add_argument('--focal_weight', type=float, default=0.0,
                        help='Focal loss weight')
    parser.add_argument('--projection_weight', type=float, default=2.0,
                        help='Projection loss weight')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'AdamW', 'SGD'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'StepLR', 'MultiStepLR'],
                        help='LR scheduler type')
    parser.add_argument('--lr_step', type=int, default=100,
                        help='LR decay step (for StepLR)')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='LR decay factor')

    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers')

    # Checkpointing
    parser.add_argument('--save_iter_step', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path to resume training')

    # Experiment naming
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')

    args = parser.parse_args()
    return args


def create_dataloaders(args):
    """创建数据加载器（使用传统TestSetLoader，与DNANet一致）"""
    print(f"\n📂 加载数据集...")

    dataset_dir = os.path.join(args.root, args.dataset)
    train_img_ids, val_img_ids, _ = load_dataset(args.root, args.dataset, args.split_method)

    print(f"  ✓ 数据集: {args.dataset}")
    print(f"  ✓ 训练样本: {len(train_img_ids)}")
    print(f"  ✓ 测试样本: {len(val_img_ids)}")
    print(f"  ✓ 输入通道: {args.in_channels} (IR only, no LiDAR)")

    # Transform: 与DNANet完全一致
    if args.in_channels == 1:
        # 单通道IR：使用简单归一化
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        print(f"  ✓ 归一化: Grayscale (mean=0.5, std=0.5)")
    else:
        # 3通道RGB：使用ImageNet归一化（与DNANet一致）
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        print(f"  ✓ 归一化: ImageNet (与DNANet一致)")

    # 使用传统DataLoader（与DNANet一致）
    trainset = TrainSetLoader(
        dataset_dir,
        img_id=train_img_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        transform=input_transform,
        suffix=args.suffix,
        in_channels=args.in_channels,
        image_folder=args.image_folder
    )

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

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True
    )

    return train_loader, test_loader


def create_model(args, device):
    """创建模型（use_lidar=False）"""
    print(f"\n🏗️  创建模型...")

    # Model config
    if args.model_type == 'mamba_tiny':
        embed_dim = 64
        depths = [2, 2, 4, 2]
        is_multiscale = False
    elif args.model_type == 'mamba_tiny_multiscale':
        embed_dim = 64
        depths = [2, 2, 4, 2]
        is_multiscale = True
    elif args.model_type == 'mamba_small':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        is_multiscale = False
    elif args.model_type == 'mamba_base':
        embed_dim = 128
        depths = [2, 2, 12, 2]
        is_multiscale = False
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    print(f"  ✓ 模型: {args.model_type}")
    print(f"  ✓ Embed dim: {embed_dim}")
    print(f"  ✓ Depths: {depths}")
    print(f"  ✓ MultiScale: {is_multiscale}")
    print(f"  ✓ use_lidar: False (IR-only mode)")

    if is_multiscale:
        model = PoLaRIS_Mamba_MultiScale(
            in_channels=1,  # Mamba内部总是使用1（patch_embed只处理IR）
            embed_dim=embed_dim,
            depths=depths,
            use_lidar=False  # ← 关键修改：不使用LiDAR
        )
    else:
        model = PoLaRIS_Mamba(
            in_channels=1,
            embed_dim=embed_dim,
            depths=depths,
            use_lidar=False  # ← 关键修改：不使用LiDAR
        )

    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total params: {total_params:,}")
    print(f"  ✓ Trainable params: {trainable_params:,}")

    return model


def create_optimizer(model, args):
    """创建优化器和学习率调度器"""
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        # T_0: 第一次restart的周期（epochs），T_mult: 每次restart后周期的倍数
        T_0 = 50  # 每50个epoch重启一次
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=1e-6)
    elif args.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == 'MultiStepLR':
        milestones = [args.epochs // 3, args.epochs * 2 // 3]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    print(f"\n🔧 优化器配置:")
    print(f"  ✓ Optimizer: {args.optimizer}")
    print(f"  ✓ Base LR: {args.base_lr}")
    print(f"  ✓ Scheduler: {args.scheduler}")

    return optimizer, scheduler


def create_loss_functions(args):
    """创建损失函数（使用LossFactory）"""
    # 使用LossFactory创建混合损失
    loss_config = {
        'dice_weight': args.dice_weight,
        'projection_weight': args.projection_weight,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'projection_mode': 'max'
    }

    # 根据权重设置选择损失类型
    if args.dice_weight > 0 and args.projection_weight > 0:
        loss_type = 'hybrid'
    elif args.projection_weight > 0:
        loss_type = 'projection'
    else:
        loss_type = 'bce_dice'

    loss_fn = LossFactory.create(loss_type, **loss_config)

    print(f"\n📉 损失函数配置:")
    print(f"  ✓ Type: {loss_type}")
    print(f"  ✓ Dice weight: {args.dice_weight:.2f}")
    print(f"  ✓ Projection weight: {args.projection_weight:.2f}")

    return loss_fn


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, args):
    """训练一个epoch"""
    model.train()

    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

    for batch_idx, (images, masks, oracle_masks) in enumerate(pbar):
        # TrainSetLoader返回 (img, mask, oracle_mask) 三个值
        # 注意：masks和oracle_masks已经是(B, 1, H, W)格式，不需要unsqueeze
        images = images.to(device)  # (B, C, H, W) - 已归一化
        masks = masks.to(device)  # (B, 1, H, W)
        oracle_masks = oracle_masks.to(device)  # (B, 1, H, W)

        # Forward
        if args.in_channels == 1:
            # 单通道输入：直接使用
            outputs = model(images, depth=None)  # depth=None for IR-only
        else:
            # 3通道输入：只使用第一个通道（IR）
            ir_channel = images[:, 0:1, :, :]  # (B, 1, H, W)
            outputs = model(ir_channel, depth=None)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]  # 取主输出

        # 计算损失
        optimizer.zero_grad()

        # LossFactory返回的loss可以直接调用
        # 对于hybrid loss，需要oracle_mask用于projection loss
        batch_loss = loss_fn(outputs, masks, oracle_mask=oracle_masks)

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{batch_loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)

    return avg_loss


def validate(model, test_loader, device, args):
    """验证模型"""
    model.eval()

    IoU = mIoU(2)
    ROC = ROCMetric(nclass=2, bins=10)

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)  # Already (B, 1, H, W) from TestSetLoader

            # Forward
            if args.in_channels == 1:
                outputs = model(images, depth=None)
            else:
                ir_channel = images[:, 0:1, :, :]
                outputs = model(ir_channel, depth=None)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs = torch.sigmoid(outputs)

            # 计算指标
            IoU.update(outputs, masks)
            ROC.update(outputs, masks)

    _, mean_iou = IoU.get()
    _, mean_pd, mean_fa = ROC.get()

    return mean_iou, mean_pd, mean_fa


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(42)

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    # 创建实验目录
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"Mamba_IR_Only_{args.model_type}_d{args.dice_weight}p{args.projection_weight}_{timestamp}"
    else:
        exp_name = args.exp_name

    result_dir = os.path.join(PROJECT_ROOT, 'model_Mamba', 'result', exp_name)
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n📁 实验目录: {result_dir}")

    # 保存配置
    import json
    with open(os.path.join(result_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(args)

    # 创建模型
    model = create_model(args, device)

    # 创建优化器
    optimizer, scheduler = create_optimizer(model, args)

    # 创建损失函数
    loss_fn = create_loss_functions(args)

    # 加载checkpoint（如果有）
    start_epoch = 1
    best_iou = 0.0

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n📥 加载checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"  ✓ 从 epoch {start_epoch} 继续训练")
        print(f"  ✓ 当前最佳 IoU: {best_iou:.4f}")

    # 训练循环
    print(f"\n🚀 开始训练...")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, args
        )

        # 验证
        mean_iou, mean_pd, mean_fa = validate(model, test_loader, device, args)

        # 更新学习率
        scheduler.step()

        # 打印结果
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val IoU: {mean_iou:.4f} | PD: {mean_pd:.4f} | FA: {mean_fa:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存checkpoint
        is_best = mean_iou > best_iou
        if is_best:
            best_iou = mean_iou

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'mean_IOU': mean_iou,
            'best_iou': best_iou,
            'args': vars(args)
        }

        # 保存latest
        torch.save(checkpoint, os.path.join(result_dir, 'latest_model.pth'))

        # 保存best
        if is_best:
            torch.save(checkpoint, os.path.join(result_dir, 'latest_best_model.pth'))
            print(f"  ✅ 新的最佳模型！IoU: {best_iou:.4f}")

        # 定期保存
        if epoch % args.save_iter_step == 0:
            torch.save(checkpoint, os.path.join(result_dir, f'epoch_{epoch}_model.pth'))

        print("=" * 80)

    print(f"\n🎉 训练完成！")
    print(f"  最佳 IoU: {best_iou:.4f}")
    print(f"  结果保存在: {result_dir}")


if __name__ == '__main__':
    main()
