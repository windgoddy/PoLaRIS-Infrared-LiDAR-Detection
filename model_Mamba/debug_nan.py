"""
NaN 诊断脚本 - 逐步检查训练管道中的 NaN 来源
============================================

这个脚本会详细检查训练管道的每一步，精确定位 NaN 的来源：
1. 数据加载（图像、LiDAR、标签）
2. Gaussian heatmap 生成
3. 模型前向传播（每一层）
4. Loss 计算
5. 梯度传播

使用方法：
    python model_Mamba/debug_nan.py --dataset Pohang-Canal-3k --split_method 50_50_2k_new

Author: PoLaRIS Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.utils_lidar import PoLaRISTrainLoader
from model.load_param_data import load_dataset
from model_Mamba.core.polaris_mamba import polaris_mamba_tiny
from model_Mamba.core.loss import GaussianFocalLoss
from model_Mamba.dataset.gaussian_utils import generate_gaussian_target, load_yolo_labels


def check_tensor(name, tensor, step=""):
    """
    检查张量中是否有 NaN/Inf，并打印统计信息

    Args:
        name: 张量名称
        tensor: 要检查的张量
        step: 当前步骤描述

    Returns:
        has_issue: 是否有 NaN/Inf
    """
    if tensor is None:
        print(f"  ⚠️  [{step}] {name}: None")
        return True

    if isinstance(tensor, (int, float)):
        if np.isnan(tensor) or np.isinf(tensor):
            print(f"  ❌ [{step}] {name}: {tensor} (NaN/Inf)")
            return True
        else:
            print(f"  ✅ [{step}] {name}: {tensor:.6f}")
            return False

    if not isinstance(tensor, (torch.Tensor, np.ndarray)):
        print(f"  ⚠️  [{step}] {name}: Unknown type {type(tensor)}")
        return False

    # Convert to numpy for consistent checking
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = tensor

    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()

    # Statistics
    min_val = np.nanmin(arr) if not has_nan else float('nan')
    max_val = np.nanmax(arr) if not has_nan else float('nan')
    mean_val = np.nanmean(arr) if not has_nan else float('nan')

    if has_nan or has_inf:
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        print(f"  ❌ [{step}] {name}: shape={arr.shape}, NaN={nan_count}, Inf={inf_count}")
        print(f"      min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
        return True
    else:
        print(f"  ✅ [{step}] {name}: shape={arr.shape}, min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
        return False


def check_data_loading(args):
    """
    Step 1: 检查数据加载
    """
    print("\n" + "="*80)
    print("Step 1: 检查数据加载")
    print("="*80)

    dataset_dir = os.path.join(args.root, args.dataset)

    # Load dataset split
    print(f"\n📂 数据集: {dataset_dir}")
    print(f"   划分方法: {args.split_method}")

    try:
        train_img_ids, val_img_ids, test_img_ids = load_dataset(args.root, args.dataset, args.split_method)
        print(f"  ✅ 训练集: {len(train_img_ids)} 样本")
        print(f"  ✅ 验证集: {len(val_img_ids)} 样本")
    except Exception as e:
        print(f"  ❌ 加载数据集划分失败: {e}")
        return False

    # Create dataloader
    print(f"\n📊 创建 DataLoader...")
    try:
        loader = PoLaRISTrainLoader(
            dataset_dir=dataset_dir,
            img_id=train_img_ids[:5],  # Only check first 5 samples
            base_size=512,
            crop_size=480,
            transform=None,
            suffix='.png',
            normalize_16bit=True,
            in_channels=args.in_channels,
            image_folder='images',
        )
        print(f"  ✅ DataLoader 创建成功，样本数: {len(loader)}")
    except Exception as e:
        print(f"  ❌ 创建 DataLoader 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check first sample
    print(f"\n🔍 检查第一个样本...")
    try:
        sample = loader[0]

        # Check image
        img = sample['image']
        check_tensor("image", img, "数据加载")

        # Check mask (if exists)
        if 'mask' in sample:
            mask = sample['mask']
            check_tensor("mask", mask, "数据加载")

        # Extract IR and LiDAR
        if img.shape[0] == 2:
            ir_img = img[0:1, :, :]
            lidar_img = img[1:2, :, :]
            print(f"\n  分离通道:")
            check_tensor("ir_img", ir_img, "通道分离")
            check_tensor("lidar_img", lidar_img, "通道分离")
        else:
            ir_img = img[0:1, :, :]
            check_tensor("ir_img", ir_img, "单通道")

        # Check label path
        img_id = sample['img_id']
        print(f"\n  图像 ID: {img_id}")

        # Try to find label file
        label_path = os.path.join(dataset_dir, 'labels', f'{img_id}.txt')
        if not os.path.exists(label_path):
            # Try alternative paths based on user's description
            # Pattern: "00_5596" -> "00/all/tir/005596.txt"
            if '_' in str(img_id):
                parts = str(img_id).split('_')
                group = parts[0]
                seq = parts[1].zfill(6)
                alt_path = f"/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/{group}/all/tir/{seq}.txt"
                if os.path.exists(alt_path):
                    label_path = alt_path
                    print(f"  ✅ 找到标签文件: {label_path}")
                else:
                    print(f"  ⚠️  标签文件不存在: {label_path}")
                    print(f"      也不存在: {alt_path}")
            else:
                print(f"  ⚠️  标签文件不存在: {label_path}")
        else:
            print(f"  ✅ 标签文件存在: {label_path}")

        return True

    except Exception as e:
        print(f"  ❌ 加载样本失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gaussian_generation(args):
    """
    Step 2: 检查 Gaussian heatmap 生成
    """
    print("\n" + "="*80)
    print("Step 2: 检查 Gaussian Heatmap 生成")
    print("="*80)

    # Create dummy labels (YOLO format)
    print("\n📊 测试 Gaussian 生成（使用虚拟标签）...")

    # Test case 1: Normal bbox
    labels = np.array([
        [0, 0.5, 0.5, 0.2, 0.3],  # center, reasonable size
        [0, 0.3, 0.7, 0.1, 0.15],  # smaller
    ])

    heatmap = generate_gaussian_target(
        labels,
        img_size=(480, 640),
        downscale=1,
        min_overlap=0.7,
        normalize=True,
    )

    has_issue = check_tensor("heatmap (normal)", heatmap, "Gaussian生成")

    # Test case 2: Very small bbox
    labels_small = np.array([
        [0, 0.5, 0.5, 0.01, 0.01],  # tiny box
    ])

    heatmap_small = generate_gaussian_target(
        labels_small,
        img_size=(480, 640),
        downscale=1,
        min_overlap=0.7,
        normalize=True,
    )

    has_issue_small = check_tensor("heatmap (tiny bbox)", heatmap_small, "Gaussian生成")

    # Test case 3: Empty labels
    labels_empty = np.array([])

    heatmap_empty = generate_gaussian_target(
        labels_empty,
        img_size=(480, 640),
        downscale=1,
        min_overlap=0.7,
        normalize=True,
    )

    has_issue_empty = check_tensor("heatmap (empty)", heatmap_empty, "Gaussian生成")

    # Test case 4: Edge case - very large bbox
    labels_large = np.array([
        [0, 0.5, 0.5, 0.9, 0.9],  # almost full image
    ])

    heatmap_large = generate_gaussian_target(
        labels_large,
        img_size=(480, 640),
        downscale=1,
        min_overlap=0.7,
        normalize=True,
    )

    has_issue_large = check_tensor("heatmap (large bbox)", heatmap_large, "Gaussian生成")

    return not (has_issue or has_issue_small or has_issue_empty or has_issue_large)


def check_model_forward(args):
    """
    Step 3: 检查模型前向传播
    """
    print("\n" + "="*80)
    print("Step 3: 检查模型前向传播")
    print("="*80)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n📊 设备: {device}")

    # Create model
    print(f"\n🔧 创建模型...")
    try:
        model = polaris_mamba_tiny(use_lidar=True)
        model = model.to(device)
        model.eval()  # Use eval mode to avoid randomness
        print(f"  ✅ 模型创建成功")
    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create dummy input
    print(f"\n📊 创建测试输入...")
    B, H, W = 2, 480, 640

    ir_img = torch.randn(B, 1, H, W).to(device) * 0.5 + 0.5  # [0, 1] range
    lidar_img = torch.randn(B, 1, H, W).to(device) * 0.5 + 0.5

    check_tensor("ir_img (input)", ir_img, "模型输入")
    check_tensor("lidar_img (input)", lidar_img, "模型输入")

    # Forward pass with hooks to check intermediate outputs
    print(f"\n🔍 执行前向传播（带中间层检查）...")

    intermediate_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                intermediate_outputs[name] = output.detach()
            elif isinstance(output, tuple):
                intermediate_outputs[name] = output[0].detach() if len(output) > 0 else None
        return hook

    # Register hooks on key layers
    hooks = []
    try:
        # Patch embedding
        if hasattr(model, 'patch_embed'):
            hooks.append(model.patch_embed.register_forward_hook(make_hook('patch_embed')))

        # Stages
        if hasattr(model, 'stages'):
            for i, stage in enumerate(model.stages):
                hooks.append(stage.register_forward_hook(make_hook(f'stage_{i}')))

        # Head
        if hasattr(model, 'head'):
            hooks.append(model.head.register_forward_hook(make_hook('head')))

        # Forward
        with torch.no_grad():
            try:
                output = model(ir_img, lidar_img)

                # Check output
                has_issue_output = check_tensor("output", output, "模型输出")

                # Check intermediate layers
                print(f"\n  中间层输出检查:")
                for name, tensor in intermediate_outputs.items():
                    check_tensor(name, tensor, "中间层")

                # Remove hooks
                for hook in hooks:
                    hook.remove()

                return not has_issue_output

            except Exception as e:
                print(f"  ❌ 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                return False

    except Exception as e:
        print(f"  ❌ Hook 注册失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_loss_computation(args):
    """
    Step 4: 检查 Loss 计算
    """
    print("\n" + "="*80)
    print("Step 4: 检查 Loss 计算")
    print("="*80)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create loss function
    print(f"\n🔧 创建 Loss 函数...")
    criterion = GaussianFocalLoss(alpha=2, beta=4, reduction='mean')

    # Test case 1: Normal case
    print(f"\n📊 测试 Case 1: 正常情况")
    pred = torch.rand(2, 1, 480, 640).to(device)  # [0, 1] random
    target = torch.rand(2, 1, 480, 640).to(device)  # [0, 1] random

    check_tensor("pred", pred, "Loss输入")
    check_tensor("target", target, "Loss输入")

    try:
        loss = criterion(pred, target)
        has_issue_1 = check_tensor("loss (normal)", loss.item(), "Loss计算")
    except Exception as e:
        print(f"  ❌ Loss 计算失败: {e}")
        has_issue_1 = True

    # Test case 2: Edge case - pred all zeros
    print(f"\n📊 测试 Case 2: pred 全零")
    pred_zero = torch.zeros(2, 1, 480, 640).to(device)

    try:
        loss_zero = criterion(pred_zero, target)
        has_issue_2 = check_tensor("loss (pred=0)", loss_zero.item(), "Loss计算")
    except Exception as e:
        print(f"  ❌ Loss 计算失败: {e}")
        has_issue_2 = True

    # Test case 3: Edge case - pred all ones
    print(f"\n📊 测试 Case 3: pred 全一")
    pred_one = torch.ones(2, 1, 480, 640).to(device)

    try:
        loss_one = criterion(pred_one, target)
        has_issue_3 = check_tensor("loss (pred=1)", loss_one.item(), "Loss计算")
    except Exception as e:
        print(f"  ❌ Loss 计算失败: {e}")
        has_issue_3 = True

    # Test case 4: Target all zeros (no objects)
    print(f"\n📊 测试 Case 4: target 全零（无目标）")
    target_zero = torch.zeros(2, 1, 480, 640).to(device)

    try:
        loss_target_zero = criterion(pred, target_zero)
        has_issue_4 = check_tensor("loss (target=0)", loss_target_zero.item(), "Loss计算")
    except Exception as e:
        print(f"  ❌ Loss 计算失败: {e}")
        has_issue_4 = True

    return not (has_issue_1 or has_issue_2 or has_issue_3 or has_issue_4)


def check_full_training_step(args):
    """
    Step 5: 检查完整的训练步骤（数据->模型->loss->backward）
    """
    print("\n" + "="*80)
    print("Step 5: 检查完整训练步骤")
    print("="*80)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load real data
    print(f"\n📂 加载真实数据...")
    dataset_dir = os.path.join(args.root, args.dataset)

    try:
        train_img_ids, _, _ = load_dataset(args.root, args.dataset, args.split_method)

        loader = PoLaRISTrainLoader(
            dataset_dir=dataset_dir,
            img_id=train_img_ids[:1],  # Only first sample
            base_size=512,
            crop_size=480,
            transform=None,
            suffix='.png',
            normalize_16bit=True,
            in_channels=args.in_channels,
            image_folder='images',
        )

        sample = loader[0]
        print(f"  ✅ 数据加载成功")

    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        return False

    # Process data
    print(f"\n🔧 处理数据...")
    img = sample['image']

    if img.shape[0] == 2:
        ir_img = img[0:1, :, :].unsqueeze(0).to(device)  # (1, 1, H, W)
        lidar_img = img[1:2, :, :].unsqueeze(0).to(device)
    else:
        ir_img = img[0:1, :, :].unsqueeze(0).to(device)
        lidar_img = torch.zeros_like(ir_img)

    check_tensor("ir_img (batch)", ir_img, "数据处理")
    check_tensor("lidar_img (batch)", lidar_img, "数据处理")

    # Generate Gaussian target
    print(f"\n📊 生成 Gaussian target...")
    img_id = sample['img_id']

    # Try to find label
    label_path = os.path.join(dataset_dir, 'labels', f'{img_id}.txt')
    if not os.path.exists(label_path) and '_' in str(img_id):
        parts = str(img_id).split('_')
        group = parts[0]
        seq = parts[1].zfill(6)
        label_path = f"/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/{group}/all/tir/{seq}.txt"

    if os.path.exists(label_path):
        labels = load_yolo_labels(label_path)
        print(f"  ✅ 加载标签: {len(labels)} 个目标")
    else:
        print(f"  ⚠️  标签文件不存在，使用空标签")
        labels = []

    H, W = ir_img.shape[2], ir_img.shape[3]
    heatmap = generate_gaussian_target(labels, img_size=(H, W), downscale=1, min_overlap=0.7)
    heatmap_gt = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, H, W)

    check_tensor("heatmap_gt", heatmap_gt, "Target生成")

    # Create model
    print(f"\n🔧 创建模型...")
    model = polaris_mamba_tiny(use_lidar=True)
    model = model.to(device)
    model.train()

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = GaussianFocalLoss(alpha=2, beta=4, reduction='mean')

    # Forward
    print(f"\n➡️  前向传播...")
    try:
        heatmap_pred = model(ir_img, lidar_img)
        has_issue_pred = check_tensor("heatmap_pred", heatmap_pred, "前向传播")
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Loss
    print(f"\n📉 计算 Loss...")
    try:
        loss = criterion(heatmap_pred, heatmap_gt)
        has_issue_loss = check_tensor("loss", loss.item(), "Loss计算")
    except Exception as e:
        print(f"  ❌ Loss 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Backward
    print(f"\n⬅️  反向传播...")
    try:
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        print(f"\n  梯度检查:")
        grad_has_issue = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_has_nan = torch.isnan(param.grad).any().item()
                grad_has_inf = torch.isinf(param.grad).any().item()

                if grad_has_nan or grad_has_inf:
                    print(f"    ❌ {name}: grad has NaN/Inf")
                    grad_has_issue = True
                else:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1000:
                        print(f"    ⚠️  {name}: grad_norm = {grad_norm:.2f} (very large!)")

        if not grad_has_issue:
            print(f"    ✅ 所有梯度正常")

        optimizer.step()
        print(f"  ✅ 参数更新完成")

        return not (has_issue_pred or has_issue_loss or grad_has_issue)

    except Exception as e:
        print(f"  ❌ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='PoLaRIS-Mamba NaN Debugging')

    # Dataset
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--split_method', type=str, default='50_50_2k_new')
    parser.add_argument('--in_channels', type=int, default=1)

    args = parser.parse_args()

    print("\n" + "="*80)
    print("PoLaRIS-Mamba NaN 诊断工具")
    print("="*80)
    print(f"\n配置:")
    print(f"  数据集: {args.dataset}")
    print(f"  划分方法: {args.split_method}")
    print(f"  输入通道: {args.in_channels}")

    # Run all checks
    results = {}

    results['data_loading'] = check_data_loading(args)
    results['gaussian_generation'] = check_gaussian_generation(args)
    results['model_forward'] = check_model_forward(args)
    results['loss_computation'] = check_loss_computation(args)
    results['full_training'] = check_full_training_step(args)

    # Summary
    print("\n" + "="*80)
    print("📊 诊断总结")
    print("="*80)

    for step, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {step:.<30} {status}")

    all_passed = all(results.values())

    if all_passed:
        print(f"\n🎉 所有检查通过！NaN 问题可能来自:")
        print(f"   1. 真实数据中的异常值（需要数据清洗）")
        print(f"   2. 学习率过高导致梯度爆炸")
        print(f"   3. 特定样本导致的数值问题")
        print(f"\n建议：")
        print(f"   - 运行 python model_Mamba/verify_dataset.py 检查所有数据")
        print(f"   - 降低学习率到 1e-5 或更低")
        print(f"   - 使用 gradient clipping")
    else:
        print(f"\n❌ 发现问题！请检查上述失败的步骤。")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
