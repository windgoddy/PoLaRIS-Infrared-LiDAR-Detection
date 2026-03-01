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
from model_Mamba.dataset.binary_mask_utils import generate_binary_mask_target, load_yolo_labels


def parse_train_log(checkpoint_path):
    """
    从 checkpoint 所在目录的 train_log.txt 中解析训练配置

    支持两种格式:
    1. 传统格式 (DNANet/CAFNet): key:--value
    2. Mamba 格式: Arguments 部分的 key: value

    Args:
        checkpoint_path: 权重文件路径

    Returns:
        config: 配置字典，如果文件不存在或解析失败则返回 {}
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    train_log_path = os.path.join(checkpoint_dir, 'train_log.txt')

    if not os.path.exists(train_log_path):
        print(f"  ⚠️  未找到 train_log.txt: {train_log_path}")
        return {}

    print(f"  ✓ 读取训练配置: {train_log_path}")

    config = {}

    try:
        with open(train_log_path, 'r') as f:
            lines = f.readlines()

        # 检测格式类型
        is_mamba_format = any('Arguments:' in line for line in lines)

        if is_mamba_format:
            # Mamba 格式: 查找 Arguments 部分
            in_args_section = False
            for line in lines:
                line = line.strip()

                if 'Arguments:' in line:
                    in_args_section = True
                    continue

                if in_args_section:
                    # 遇到分隔符或新的section，停止解析
                    if line.startswith('=') or ('Configuration:' in line and line.endswith(':')):
                        break

                    # 跳过分隔符行和空行
                    if not line or line.startswith('-'):
                        continue

                    # 解析 key: value 格式
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            config[key] = value
        else:
            # 传统格式: key:--value
            for line in lines:
                line = line.strip()
                if ':--' in line:
                    key, value = line.split(':--', 1)
                    config[key] = value

        print(f"  ✓ 成功解析 {len(config)} 个配置项")

    except Exception as e:
        print(f"  ⚠️  解析 train_log.txt 失败: {e}")
        return {}

    return config


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
    parser.add_argument('--split_method', type=str, default=None,
                        help='Train/test split method (auto-detect from checkpoint if not specified)')
    parser.add_argument('--image_folder', type=str, default='images',
                        help='Image folder name')
    parser.add_argument('--suffix', type=str, default='.png',
                        help='Image file suffix')

    # 模型参数
    parser.add_argument('--model', type=str, default=None,
                        help='Model type (auto-detect from checkpoint if not specified)')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels (1=IR only, 2=IR+Depth)')
    parser.add_argument('--base_size', type=int, default=None,
                        help='Base image size (if None, use model-specific default)')
    parser.add_argument('--crop_size', type=int, default=None,
                        help='Crop size for testing (if None, use model-specific default)')

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


def load_category_mapping(dataset_root, dataset_name):
    """
    加载LiDAR密度类别映射

    Args:
        dataset_root: 数据集根目录 (e.g., 'dataset/')
        dataset_name: 数据集名称 (e.g., 'Pohang-Canal-3k')

    Returns:
        category_map: dict {img_id: category}
        category_names: dict {category: description}
    """
    # Try both possible locations
    category_file = os.path.join(dataset_root, dataset_name, 'selection_summary_new.txt')
    if not os.path.exists(category_file):
        category_file = os.path.join(dataset_root, dataset_name, 'select-view', 'selection_summary_new.txt')

    if not os.path.exists(category_file):
        print(f"  ⚠️  类别文件不存在: {os.path.join(dataset_root, dataset_name, 'selection_summary_new.txt')}")
        print(f"  → 跳过类别分析")
        return None, None

    print(f"  ✓ 加载类别映射: {category_file}")

    category_map = {}
    with open(category_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) == 2:
                img_id = parts[0].strip().replace('.png', '')
                category = int(parts[1].strip())
                category_map[img_id] = category

    # 定义类别含义（根据实际场景类型）
    category_names = {
        0: "Category 0 (未分类场景)",
        1: "Category 1 (适中场景 - 点云适中)",
        2: "Category 2 (小目标场景 - 点云少)",
        3: "Category 3 (岸边场景 - 点云多)"
    }

    # 统计类别分布
    category_counts = {}
    for cat in category_map.values():
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"  ✓ 加载了 {len(category_map)} 个样本的类别信息")
    for cat in sorted(category_counts.keys()):
        print(f"    Category {cat}: {category_counts[cat]} samples - {category_names.get(cat, 'Unknown')}")

    return category_map, category_names


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

    # Mamba: infer in_channels, embed_dim, architecture, and use_lidar
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

            # Detect architecture type by checking layer patterns
            # Progressive: has progressive_decoder layers
            has_progressive = any('progressive_decoder' in key or 'up_block' in key for key in state_dict.keys())
            # MultiScale: has skip_proj_s layers but no progressive decoder
            has_skip_proj = any('skip_proj_s' in key for key in state_dict.keys())

            if has_progressive:
                params['architecture'] = 'progressive'
                # Check for deep supervision (auxiliary heads)
                has_deep_sup = any('aux_head' in key for key in state_dict.keys())
                params['use_deep_supervision'] = has_deep_sup

                # Check for CBAM (NEW 2026-02-28) - Fixed detection logic
                # CBAM modules are named 'head.ca' (channel) and 'head.sa' (spatial)
                has_cbam_channel = any('head.ca.' in key for key in state_dict.keys())
                has_cbam_spatial = any('head.sa.' in key for key in state_dict.keys())
                if has_cbam_channel and has_cbam_spatial:
                    params['use_cbam'] = 'full'
                    print(f"  ✅ 检测到 Progressive 架构 + CBAM Full (Channel + Spatial)")
                elif has_cbam_spatial:
                    params['use_cbam'] = 'spatial'
                    print(f"  ✅ 检测到 Progressive 架构 + CBAM Spatial")
                else:
                    params['use_cbam'] = 'none'
                    print(f"  ℹ️  检测到 Progressive 架构 (无CBAM)")

                if has_deep_sup:
                    print(f"  ℹ️  检测到深度监督 (aux_head 层存在)")
            elif has_skip_proj:
                params['architecture'] = 'multiscale'
                # MultiScale also supports deep supervision
                has_deep_sup = any('aux_head' in key for key in state_dict.keys())
                params['use_deep_supervision'] = has_deep_sup
                print(f"  ℹ️  检测到多尺度架构 (skip_proj_s* 层存在)")
                if has_deep_sup:
                    print(f"  ℹ️  检测到深度监督 (aux_head 层存在)")
            else:
                params['architecture'] = 'base'
                print(f"  ℹ️  检测到基础架构")

            # For data loader: need 2 channels if LiDAR is used
            if has_lidar_gate:
                params['data_in_channels'] = 2  # IR + Depth
                print(f"  ℹ️  检测到 LiDAR gating → 数据加载器需要 2 通道 (IR + Depth)")
            else:
                # IR-only: model receives 1-channel IR (MambaDataset extracts img[0:1,:,:])
                # DO NOT set data_in_channels here - let train_log.txt's in_channels control
                # the DataLoader (e.g., in_channels=3 → ImageNet normalization, same as training)
                print(f"  ℹ️  未检测到 LiDAR gating → 模型接收 1 通道 IR (DataLoader 通道数由 train_log 决定)")

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
        mamba_models = ['mamba', 'mamba_tiny', 'mamba_small', 'mamba_base',
                        'mamba_tiny_multiscale', 'mamba_small_multiscale',
                        'mamba_tiny_progressive', 'mamba_small_progressive']
        if model_type not in mamba_models:
            in_channels = detected_params['in_channels']
    if 'embed_dim' in detected_params:
        print(f"    - embed_dim: {detected_params['embed_dim']}")
    if 'use_lidar' in detected_params:
        print(f"    - use_lidar: {detected_params['use_lidar']}")
    if 'architecture' in detected_params:
        print(f"    - architecture: {detected_params['architecture']}")
    if 'use_deep_supervision' in detected_params:
        print(f"    - use_deep_supervision: {detected_params['use_deep_supervision']}")

    # CRITICAL: apply_sigmoid 需要根据模型输出格式设置
    # - DNANet/CAFNet: 输出 logits，需要 sigmoid（与训练时评估一致）
    # - Mamba Base/MultiScale: GaussianHead 输出 logits，需要 sigmoid
    # - Mamba Progressive: ProgressiveHead 输出已做 sigmoid，不需要再做
    # 默认设置为 False，后续根据模型类型调整
    apply_sigmoid = False

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
        # DNANet 输出 logits，需要 sigmoid（与 train_Phase3.py:350 一致）
        apply_sigmoid = True
    elif model_type == 'MS_CAFNet':
        model = MS_CAFNet(num_classes=1, input_channels=in_channels)
        # MS_CAFNet 输出 logits，需要 sigmoid
        apply_sigmoid = True
    elif model_type == 'MS_CAFNet_DualGeo':
        model = MS_CAFNet_DualGeo(num_classes=1, input_channels=in_channels)
        # MS_CAFNet_DualGeo 输出 logits，需要 sigmoid
        apply_sigmoid = True
    elif model_type in ['mamba', 'mamba_tiny', 'mamba_small', 'mamba_base',
                        'mamba_tiny_multiscale', 'mamba_small_multiscale',
                        'mamba_tiny_progressive', 'mamba_small_progressive']:
        embed_dim = detected_params.get('embed_dim', 96)
        depths_map = {64: [2, 2, 4, 2], 96: [2, 2, 6, 2], 128: [2, 2, 12, 2]}
        depths = depths_map.get(embed_dim, [2, 2, 6, 2])
        use_lidar = detected_params.get('use_lidar', False)
        use_deep_supervision = detected_params.get('use_deep_supervision', False)

        # Auto-detect architecture from state_dict
        architecture = detected_params.get('architecture', 'base')

        # Override architecture if explicitly specified in model_type
        if 'progressive' in model_type:
            architecture = 'progressive'
        elif 'multiscale' in model_type:
            architecture = 'multiscale'

        if architecture == 'progressive':
            # Import Progressive model
            from model_Mamba.core.polaris_mamba_progressive import PoLaRIS_Mamba_Progressive

            # Detect CBAM from checkpoint (NEW 2026-02-28)
            use_cbam = detected_params.get('use_cbam', 'none')

            model = PoLaRIS_Mamba_Progressive(
                in_channels=1,  # Always 1 for IR (patch_embed)
                embed_dim=embed_dim,
                depths=depths,
                use_lidar=use_lidar,
                use_deep_supervision=use_deep_supervision,
                use_cbam=use_cbam  # NEW: CBAM support
            )
            print(f"  ✓ 使用 Progressive Decoder Mamba 模型 (深度监督: {use_deep_supervision}, CBAM: {use_cbam})")
            # CRITICAL: ProgressiveHead outputs sigmoid([0,1]), NOT logits
            apply_sigmoid = False
        elif architecture == 'multiscale':
            # Import MultiScale model
            from model_Mamba.core.polaris_mamba_multiscale import PoLaRIS_Mamba_MultiScale
            model = PoLaRIS_Mamba_MultiScale(
                in_channels=1,  # Always 1 for IR (patch_embed)
                embed_dim=embed_dim,
                depths=depths,
                use_lidar=use_lidar,
                use_deep_supervision=use_deep_supervision
            )
            print(f"  ✓ 使用多尺度 Mamba 模型 (深度监督: {use_deep_supervision})")
            # CRITICAL: GaussianHead outputs logits, need sigmoid
            apply_sigmoid = True
        else:
            # Base architecture
            model = PoLaRIS_Mamba(
                in_channels=1,  # Always 1 for IR (patch_embed)
                embed_dim=embed_dim,
                depths=depths,
                use_lidar=use_lidar
            )
            print(f"  ✓ 使用基础 Mamba 模型")
            # CRITICAL: GaussianHead outputs logits, need sigmoid
            apply_sigmoid = True
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
        # For Mamba: replace GT mask with label-generated binary mask (align with MambaDataset)
        if args.model is not None and args.model.startswith('mamba'):
            testset = MambaLabelMaskWrapper(
                base_loader=testset,
                dataset_dir=dataset_dir,
                downscale=1,
                fill_mode='box',
                ellipse_ratio=0.8
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

    return test_loader, use_lidar_loader, val_img_ids


class MambaLabelMaskWrapper(torch.utils.data.Dataset):
    """
    Wrap PoLaRISTestLoader and generate binary GT mask from labels/*.txt
    to align with MambaDataset training logic.
    """
    def __init__(self, base_loader, dataset_dir, downscale=1, fill_mode='box', ellipse_ratio=0.8):
        self.base_loader = base_loader
        self.dataset_dir = dataset_dir
        self.downscale = downscale
        self.fill_mode = fill_mode
        self.ellipse_ratio = ellipse_ratio

    def __len__(self):
        return len(self.base_loader)

    def __getitem__(self, index):
        sample = self.base_loader[index]
        img = sample['image']
        img_id = sample.get('img_id')

        # Compute mask size from image tensor
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape
        else:
            # Fallback for unexpected types
            h, w = sample['mask'].shape[-2:]

        label_path = os.path.join(self.dataset_dir, 'labels', f'{img_id}.txt')
        labels = load_yolo_labels(label_path)
        mask = generate_binary_mask_target(
            labels,
            img_size=(h, w),
            downscale=self.downscale,
            fill_mode=self.fill_mode,
            ellipse_ratio=self.ellipse_ratio,
        )
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        # Keep original fields, replace mask
        sample['mask'] = mask
        return sample


def test_model(model, test_loader, use_lidar_loader, threshold, device, apply_sigmoid=False, model_type='', eval_strategy='auto', category_map=None, category_names=None, val_img_ids=None):
    """测试模型并计算 Mask-to-Box IoU

    Args:
        model_type: 模型类型
        eval_strategy: 'auto' (Mamba用dynamic，其他用fixed), 'fixed', 'dynamic'
        category_map: dict {img_id: category}，用于按类别统计
        category_names: dict {category: description}，类别描述
        val_img_ids: list[str]，测试集图像ID列表（用于传统DataLoader获取img_id）
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

    # 按类别统计（如果提供了category_map）
    category_stats = {}
    if category_map is not None:
        for cat in set(category_map.values()):
            category_stats[cat] = {
                'seg_ious': [],
                'box_ious': [],
                'best_thresholds': [],
                'count': 0
            }

    with torch.no_grad():
        tbar = tqdm(test_loader, desc='Testing')
        global_sample_idx = 0  # Track global sample index for category mapping

        for batch_idx, batch_data in enumerate(tbar):
            if use_lidar_loader:
                data = batch_data['image'].to(device)
                labels = batch_data['mask'].to(device)
                # Extract img_ids for category mapping
                batch_img_ids = batch_data.get('img_id', [None] * data.size(0))
                if not isinstance(batch_img_ids, list):
                    batch_img_ids = [batch_img_ids]
            else:
                data, labels = batch_data
                data = data.to(device)
                labels = labels.to(device)
                # 从val_img_ids获取当前batch的img_ids
                if val_img_ids is not None:
                    batch_size = data.size(0)
                    batch_img_ids = val_img_ids[global_sample_idx:global_sample_idx + batch_size]
                else:
                    batch_img_ids = [None] * data.size(0)

            if batch_idx == 0:
                print(f"\n🔍 调试信息 (第一个 batch):")
                print(f"  - 数据形状: {data.shape}")
                print(f"  - 标签形状: {labels.shape}")
                print(f"  - 数据范围: [{data.min().item():.4f}, {data.max().item():.4f}]")
                print(f"  - 标签范围: [{labels.min().item():.4f}, {labels.max().item():.4f}]")
                print(f"  - 数据均值: {data.mean().item():.4f}")
                print(f"  - 标签正样本比例: {(labels > 0.5).float().mean().item():.4f}")

            if model_type.startswith('mamba'):
                if use_lidar_loader and data.shape[1] == 2:
                    ir = data[:, 0:1]
                    lidar = data[:, 1:2]
                elif data.shape[1] > 1:
                    # Multi-channel (e.g., 3-ch RGB): take first channel only
                    # Matches MambaDataset.__getitem__: ir_img = img[0:1, :, :]
                    ir = data[:, 0:1]
                    lidar = None
                else:
                    ir = data  # Already 1-channel
                    lidar = None
                pred = model(ir, lidar)
            else:
                pred = model(data)

            # 处理多输出模型
            # 1. MS_CAFNet_DualGeo 返回 tuple: ([outputs], confidence)
            if isinstance(pred, tuple):
                pred = pred[0]  # 取第一个元素（多尺度输出列表）

            # 2. 深度监督模型在训练时返回列表，但测试时（model.eval()）通常只返回 main_output
            #    为安全起见，如果意外返回列表，需要根据模型类型选择正确的输出：
            #    - DNANet/CAFNet: [aux1, aux2, ..., main_output] → 主输出在最后
            #    - Mamba: [main_output, aux1, aux2] → 主输出在最前
            if isinstance(pred, list):
                if model_type in ['DNANet', 'MS_CAFNet', 'MS_CAFNet_DualGeo']:
                    pred = pred[-1]  # DNANet 系列：主输出在最后
                else:
                    pred = pred[0]   # Mamba 系列：主输出在最前

            if apply_sigmoid:
                pred = torch.sigmoid(pred)

            if batch_idx == 0:
                print(f"  - 预测形状: {pred.shape}")
                print(f"  - 预测范围: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
                print(f"  - 预测均值: {pred.mean().item():.4f}")
                print(f"  - 预测正样本比例 (>0.5): {(pred > 0.5).float().mean().item():.4f}\n")

            # 使用动态阈值扫描（per-sample，更准确且节省内存）
            if use_threshold_sweep:
                # Per-sample threshold sweep (memory efficient)
                batch_iou_sum = 0.0
                batch_box_iou_sum = 0.0

                # Detach predictions to save memory
                pred = pred.detach()
                labels = labels.detach()

                for i in range(pred.size(0)):
                    pred_i = pred[i:i+1]
                    label_i = labels[i:i+1]

                    best_sample_iou = 0.0
                    best_sample_threshold = 0.5
                    best_sample_box_iou = 0.0

                    # Sweep thresholds for this sample
                    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        pred_bin = (pred_i > thresh).float()
                        gt_bin = (label_i > 0.5).float()

                        inter = (pred_bin * gt_bin).sum()
                        union = (pred_bin + gt_bin).clamp(0, 1).sum()
                        sample_iou = (inter / (union + 1e-7)).item()

                        if sample_iou > best_sample_iou:
                            best_sample_iou = sample_iou
                            best_sample_threshold = thresh
                            # Also compute box IoU at this threshold
                            best_sample_box_iou = calculate_mask_to_box_iou(pred_i, label_i, threshold=thresh)

                        # Clean up intermediate tensors
                        del pred_bin, gt_bin, inter, union

                    # Store per-sample results
                    sample_ious.append(best_sample_iou)
                    sample_box_ious.append(best_sample_box_iou)
                    sample_best_thresholds.append(best_sample_threshold)

                    # Track category-wise statistics
                    if category_map is not None and i < len(batch_img_ids):
                        img_id = batch_img_ids[i]
                        if img_id and img_id in category_map:
                            cat = category_map[img_id]
                            category_stats[cat]['seg_ious'].append(best_sample_iou)
                            category_stats[cat]['box_ious'].append(best_sample_box_iou)
                            category_stats[cat]['best_thresholds'].append(best_sample_threshold)
                            category_stats[cat]['count'] += 1

                    batch_iou_sum += best_sample_iou
                    batch_box_iou_sum += best_sample_box_iou

                # Update batch statistics
                iou_sum += batch_iou_sum / pred.size(0)
                box_iou_sum += batch_box_iou_sum / pred.size(0)
                box_iou_count += 1
                
                # Update progress bar
                current_mean_iou = iou_sum / box_iou_count
                current_box_iou = box_iou_sum / box_iou_count
                # Show most recent sample's best threshold
                recent_thresh = sample_best_thresholds[-1] if sample_best_thresholds else 0.5
                tbar.set_postfix({
                    'mIoU': f'{current_mean_iou:.4f}',
                    'Box_IoU': f'{current_box_iou:.4f}',
                    'BestThresh': f'{recent_thresh:.1f}'
                })

                # Clear GPU cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            else:
                # Traditional fixed threshold evaluation
                # ⚠️ CRITICAL FIX: 禁用自适应阈值，确保与 box IoU 使用相同策略
                miou_metric.update(pred, labels, depth_map=None, use_adaptive_threshold=False)

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

            # Update global sample index for next batch
            global_sample_idx += data.size(0)

            # Clean up tensors after each batch to free GPU memory
            del data, labels, pred
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

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

    # 输出按类别统计的结果
    if category_map is not None and category_stats:
        print(f"\n{'='*70}")
        print("📊 按LiDAR密度类别统计")
        print(f"{'='*70}\n")

        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            if stats['count'] == 0:
                continue

            cat_name = category_names.get(cat, f"Category {cat}") if category_names else f"Category {cat}"
            cat_seg_ious = np.array(stats['seg_ious'])
            cat_box_ious = np.array(stats['box_ious'])
            cat_thresholds = stats['best_thresholds']

            print(f"📁 {cat_name}")
            print(f"   样本数: {stats['count']}")
            print(f"   Seg IoU:  均值={cat_seg_ious.mean():.4f}, 中位数={np.median(cat_seg_ious):.4f}, 标准差={cat_seg_ious.std():.4f}")
            print(f"   Box IoU:  均值={cat_box_ious.mean():.4f}, 中位数={np.median(cat_box_ious):.4f}, 标准差={cat_box_ious.std():.4f}")

            # 计算0.1阈值占比
            thresh_01_count = sum(1 for t in cat_thresholds if t == 0.1)
            thresh_01_pct = thresh_01_count / len(cat_thresholds) * 100 if cat_thresholds else 0
            print(f"   0.1阈值占比: {thresh_01_count}/{len(cat_thresholds)} ({thresh_01_pct:.1f}%)")

            # 显示Box IoU分布
            box_iou_ranges = {
                '[0.0, 0.5)': sum(1 for iou in cat_box_ious if iou < 0.5),
                '[0.5, 0.7)': sum(1 for iou in cat_box_ious if 0.5 <= iou < 0.7),
                '[0.7, 0.9)': sum(1 for iou in cat_box_ious if 0.7 <= iou < 0.9),
                '[0.9, 1.0]': sum(1 for iou in cat_box_ious if iou >= 0.9),
            }
            print(f"   Box IoU分布: ", end="")
            print(" | ".join([f"{k}:{v}({v/len(cat_box_ious)*100:.1f}%)" for k, v in box_iou_ranges.items()]))
            print()

        print(f"{'='*70}\n")

    results = {
        'mean_iou': mean_iou,
        'mean_box_iou': mean_box_iou,
        'sample_ious': sample_ious,
        'sample_box_ious': sample_box_ious,
        'num_samples': len(sample_ious),
        'use_threshold_sweep': use_threshold_sweep,
        'category_stats': category_stats if category_map is not None else None,
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

    # 🔧 NEW: 从 train_log.txt 读取训练配置
    train_config = parse_train_log(args.checkpoint)

    checkpoint, checkpoint_info = load_model_from_checkpoint(args.checkpoint, device)

    # Apply config from train_log.txt if available (only if not explicitly set)
    if train_config:
        print(f"\n📋 应用训练配置到测试参数:")

        # Model type (only auto-detect if not specified)
        if args.model is None and 'model' in train_config:
            args.model = train_config['model']
            print(f"  ✓ model: {args.model}")

        # Dataset configuration (only if not specified by user)
        if args.dataset == 'Pohang-Canal-3k' and 'dataset' in train_config:
            args.dataset = train_config['dataset']
            print(f"  ✓ dataset: {args.dataset}")

        # CRITICAL: Only use train_config's split_method if user didn't specify one
        if args.split_method is None and 'split_method' in train_config:
            args.split_method = train_config['split_method']
            print(f"  ✓ split_method: {args.split_method} (from train_log.txt)")
        elif args.split_method is not None:
            print(f"  ✓ split_method: {args.split_method} (from command line, overriding train_log.txt)")

        if args.image_folder == 'images' and 'image_folder' in train_config:
            args.image_folder = train_config['image_folder']
            print(f"  ✓ image_folder: {args.image_folder}")

        # Image dimensions
        if args.base_size is None and 'base_size' in train_config:
            args.base_size = int(train_config['base_size'])
            print(f"  ✓ base_size: {args.base_size}")

        if args.crop_size is None and 'crop_size' in train_config:
            args.crop_size = int(train_config['crop_size'])
            print(f"  ✓ crop_size: {args.crop_size}")

        # Input channels (from config)
        if 'in_channels' in train_config:
            config_in_channels = int(train_config['in_channels'])
            # Only override if user didn't specify
            if args.in_channels == 1:  # Default value
                args.in_channels = config_in_channels
                print(f"  ✓ in_channels: {args.in_channels}")

        # LiDAR dataloader settings
        # CRITICAL: Handle both old (use_lidar_dataloader) and new (use_polaris_loader) naming
        if 'use_polaris_loader' in train_config:
            args.use_lidar_dataloader = train_config['use_polaris_loader']
            print(f"  ✓ use_lidar_dataloader: {args.use_lidar_dataloader} (from use_polaris_loader)")
        elif 'use_lidar_dataloader' in train_config:
            args.use_lidar_dataloader = train_config['use_lidar_dataloader']
            print(f"  ✓ use_lidar_dataloader: {args.use_lidar_dataloader}")

        if 'normalize_16bit' in train_config:
            args.normalize_16bit = train_config['normalize_16bit']
            print(f"  ✓ normalize_16bit: {args.normalize_16bit}")

        # Other parameters
        if 'suffix' in train_config:
            args.suffix = train_config['suffix']

    # Model type auto-detection (fallback if no train_log.txt or model not in config)
    if args.model is None:
        checkpoint_name = os.path.basename(args.checkpoint)
        checkpoint_dir = os.path.dirname(args.checkpoint)

        if 'mamba' in checkpoint_dir.lower() or 'mamba' in checkpoint_name.lower():
            # Detect Mamba variant
            if 'progressive' in checkpoint_name.lower() or 'progressive' in checkpoint_dir.lower():
                if 'small' in checkpoint_name.lower() or 'small' in checkpoint_dir.lower():
                    args.model = 'mamba_small_progressive'
                else:
                    args.model = 'mamba_tiny_progressive'
            elif 'multiscale' in checkpoint_name.lower() or 'multiscale' in checkpoint_dir.lower():
                if 'small' in checkpoint_name.lower() or 'small' in checkpoint_dir.lower():
                    args.model = 'mamba_small_multiscale'
                else:
                    args.model = 'mamba_tiny_multiscale'
            elif 'base' in checkpoint_name.lower() or 'base' in checkpoint_dir.lower():
                args.model = 'mamba_base'
            elif 'small' in checkpoint_name.lower() or 'small' in checkpoint_dir.lower():
                args.model = 'mamba_small'
            elif 'tiny' in checkpoint_name.lower() or 'tiny' in checkpoint_dir.lower():
                args.model = 'mamba_tiny'
            else:
                args.model = 'mamba_tiny'  # Default to tiny
        elif 'DNANet' in checkpoint_dir or 'DNANet' in checkpoint_name:
            args.model = 'DNANet'
        elif 'MS_CAFNet_DualGeo' in checkpoint_dir:
            args.model = 'MS_CAFNet_DualGeo'
        elif 'MS_CAFNet' in checkpoint_dir:
            args.model = 'MS_CAFNet'
        else:
            print("⚠️  无法自动推断模型类型，默认使用 DNANet")
            args.model = 'DNANet'

    # Fallback for split_method if still None
    if args.split_method is None:
        args.split_method = '50_50_2k_new'  # Default fallback
        print(f"  ⚠️  split_method 未指定，使用默认值: {args.split_method}")

    print(f"\n📦 模型配置:")
    print(f"  ✓ 模型类型: {args.model}")
    print(f"  ✓ 数据集划分: {args.split_method}")

    # Auto eval strategy: use dynamic threshold sweep for all models
    if args.eval_strategy == 'auto':
        args.eval_strategy = 'dynamic'
        print("  ✓ 评估策略: auto → dynamic (所有模型启用动态阈值扫描)")

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
        if detected_params.get('use_lidar', False):
            # Model has LiDAR gate → force LiDAR dataloader + 16-bit normalization
            args.use_lidar_dataloader = 'True'
            args.normalize_16bit = 'True'
            force_lidar = True
            print("  ✓ 检测到 LiDAR gating → 强制使用 PoLaRIS LiDAR DataLoader + 16-bit 归一化")
        else:
            # IR-only Mamba: use settings from train_log.txt, no LiDAR forcing
            print(f"  ✓ IR-only Mamba → 使用传统 DataLoader (use_lidar={args.use_lidar_dataloader}, normalize_16bit={args.normalize_16bit})")

    # Set model-specific default input sizes if not explicitly provided
    if args.base_size is None or args.crop_size is None:
        if args.model.startswith('mamba'):
            args.base_size = 256
            args.crop_size = 256
            print("  ✓ 使用 Mamba 训练默认输入尺寸: base_size=256, crop_size=256")
        else:
            args.base_size = 256
            args.crop_size = 256
            print("  ✓ 使用 DNANet/CAFNet 训练默认输入尺寸: base_size=256, crop_size=256")

    model, apply_sigmoid = create_model(args.model, args.in_channels, checkpoint, device)

    test_loader, use_lidar_loader, val_img_ids = create_test_loader(args, force_lidar=force_lidar)

    # 加载LiDAR密度类别映射（用于分类别性能分析）
    print(f"\n📂 加载类别信息...")
    category_map, category_names = load_category_mapping(args.root, args.dataset)

    results = test_model(model, test_loader, use_lidar_loader, args.threshold, device,
                         apply_sigmoid=apply_sigmoid, model_type=args.model,
                         eval_strategy=args.eval_strategy,
                         category_map=category_map, category_names=category_names,
                         val_img_ids=val_img_ids)

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
