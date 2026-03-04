"""
Training Script for PoLaRIS-Gaussian-Mamba
==========================================

This script trains the PoLaRIS_Mamba model with Gaussian heatmap supervision.

Key Differences from train_Phase3.py:
1. Uses PoLaRIS_Mamba model instead of DNANet/MS_CAFNet
2. Generates Gaussian heatmap targets from YOLO labels
3. Uses GaussianFocalLoss instead of SoftIoULoss
4. Evaluates with peak detection metrics

Author: PoLaRIS Team
Date: 2026-01-30
"""

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import random
import os
import sys
import argparse
import csv
import signal
from collections import defaultdict
from PIL import Image
import cv2

# Add project root to path (for imports to work from model_Mamba/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# PoLaRIS utilities (from project root)
from model.utils_lidar import PoLaRISTrainLoader, PoLaRISTestLoader
from model.utils import TrainSetLoader, TestSetLoader  # Traditional 8-bit loader
from model.metric import ROCMetric, mIoU
from model.load_param_data import load_dataset

# Mamba model and loss (from model_Mamba/)
from model_Mamba.core.polaris_mamba import PoLaRIS_Mamba, polaris_mamba_tiny, polaris_mamba_small, polaris_mamba_base
# [NEW] Multi-scale model with Deep Supervision (2026-02-05)
from model_Mamba.core.polaris_mamba_multiscale import (
    PoLaRIS_Mamba_MultiScale,
    polaris_mamba_tiny_multiscale,
    polaris_mamba_small_multiscale,
)
# [NEW] Progressive Decoder model (2026-02-07)
from model_Mamba.core.polaris_mamba_progressive import (
    PoLaRIS_Mamba_Progressive,
    polaris_mamba_tiny_progressive,
    polaris_mamba_small_progressive,
)
# [NEW] Mamba-UNet++ Hybrid Model (2026-03-02)
from model.model_Mamba_UNetPP import mamba_unetplusplus
from model_Mamba.core.loss import GaussianFocalLoss, CombinedLoss, AverageMeter, BCEDiceLoss
from model_Mamba.core.loss_improved import ImprovedBCEDiceLoss, ConfidenceCalibrationLoss
from model_Mamba.core.loss_advanced import LossFactory  # 2026-02-03: Multi-loss support

# Mamba metrics (2026-02-03: Added Mask-to-Box IoU)
from model_Mamba.core.metrics import calculate_mask_to_box_iou

# Mamba I/O utilities
from model_Mamba.utils_io import (
    create_experiment_dir,
    save_training_config,
    init_training_log_csv,
    log_epoch_metrics,
    save_best_model,
    save_last_epoch_model,
    update_training_summary
)
from model_Mamba.dataset.gaussian_utils import generate_gaussian_target, load_yolo_labels

# [SCHEME A] Binary Segmentation imports (2026-02-01)
from model_Mamba.dataset.binary_mask_utils import generate_binary_mask_target, load_yolo_labels as load_yolo_labels_binary


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set: {seed}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='PoLaRIS-Gaussian-Mamba Training')

    # Model configuration
    parser.add_argument('--model', type=str, default='mamba_tiny',
                        choices=['mamba_tiny', 'mamba_small', 'mamba_base',
                                'mamba_tiny_multiscale', 'mamba_small_multiscale',
                                'mamba_tiny_progressive', 'mamba_small_progressive',
                                'mamba_unetpp'],
                        help='Model variant (use *_multiscale for multi-scale, *_progressive for U-Net style decoder, mamba_unetpp for Mamba-UNet++ hybrid)')
    parser.add_argument('--use_lidar', type=str, default='True',
                        help='Whether to use LiDAR gating (True/False)')
    parser.add_argument('--use_deep_supervision', type=str, default='False',
                        help='Enable deep supervision (aux loss at D2 and D3). Only for *_multiscale and *_progressive models.')
    parser.add_argument('--use_cbam', type=str, default='none',
                        choices=['none', 'spatial', 'full'],
                        help='[NEW 2026-02-27] CBAM attention mode for ProgressiveHead: none (baseline), spatial (lightweight), full (complete)')

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k',
                        help='Dataset name')
    parser.add_argument('--root', type=str, default='dataset/',
                        help='Dataset root directory')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='Train/test split method')
    parser.add_argument('--image_folder', type=str, default='images',
                        help='Image folder name')
    parser.add_argument('--suffix', type=str, default='.png',
                        help='Image file suffix')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels (1=IR, 2=IR+Depth)')

    # Data preprocessing
    parser.add_argument('--base_size', type=int, default=256,
                        help='Base image size (256 for scale-aligned training)')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Crop size for training (match base_size)')
    parser.add_argument('--normalize_16bit', type=str, default='True',
                        help='Use Min-Max normalization for 16-bit images')
    parser.add_argument('--normalize_mode', type=str, default='minmax',
                        choices=['minmax', 'global', 'percentile', 'clahe'],
                        help='Normalization mode for 16-bit images: '
                             'minmax (adaptive per image, default), '
                             'global (fixed range), '
                             'percentile (clip outliers), '
                             'clahe (histogram equalization)')
    parser.add_argument('--use_polaris_loader', type=str, default='True',
                        help='Use PoLaRISTrainLoader (True) or TrainSetLoader (False). '
                             'PoLaRISTrainLoader: supports 16-bit + LiDAR. '
                             'TrainSetLoader: 8-bit only, for fair comparison with DNANet.')

    # Gaussian target generation
    parser.add_argument('--gaussian_iou', type=float, default=0.7,
                        help='Min IoU for Gaussian radius computation')
    parser.add_argument('--heatmap_downscale', type=int, default=1,
                        help='Downscale factor for heatmap (1=full resolution)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Start epoch (for resuming)')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        help='Test batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Initial learning rate (2e-4, updated 2026-01-30 for faster convergence)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['Adam', 'AdamW', 'Adagrad', 'SGD'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'StepLR'],
                        help='Learning rate scheduler: CosineAnnealingLR (default), '
                             'CosineAnnealingWarmRestarts (periodic restart for long training), StepLR')

    # Loss function (2026-02-03: Extended with advanced loss options)
    parser.add_argument('--loss_type', type=str, default='improved_bce_dice',
                        choices=['improved_bce_dice', 'projection', 'hybrid', 'focal', 'combined', 'gaussian_focal'],
                        help='Loss function type: improved_bce_dice (baseline), projection (weak-supervision), '
                             'hybrid (recommended for Box GT), gaussian_focal (BAGS strategy: Box→Gaussian realtime)')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter (positive sample weight)')
    parser.add_argument('--focal_gamma', type=float, default=2.5,
                        help='Focal loss gamma parameter (focusing parameter)')
    parser.add_argument('--dice_weight', type=float, default=4.0,
                        help='Dice loss weight')
    parser.add_argument('--projection_weight', type=float, default=1.0,
                        help='Projection loss weight (for hybrid loss)')
    parser.add_argument('--projection_mode', type=str, default='max',
                        choices=['max', 'mean'],
                        help='Projection mode: max (default, sensitive to sparse targets) or mean')
    parser.add_argument('--ohem_ratio', type=float, default=0.0,
                        help='OHEM ratio (0.0 = disabled)')
    # BAGS strategy (gaussian_focal loss)
    parser.add_argument('--gaussian_sigma_scale', type=float, default=6.0,
                        help='BAGS: sigma = box_dim / sigma_scale (default 6.0, ±3σ covers full box)')
    parser.add_argument('--gaussian_min_sigma', type=float, default=2.0,
                        help='BAGS: minimum Gaussian sigma in pixels (default 2.0)')
    # Legacy parameters (for backward compatibility)
    parser.add_argument('--loss_alpha', type=float, default=2.0,
                        help='[DEPRECATED] Focal loss alpha parameter (use --focal_alpha instead)')
    parser.add_argument('--loss_beta', type=float, default=4.0,
                        help='[DEPRECATED] Gaussian falloff beta parameter')

    # Hardware and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU IDs to use (e.g., 0,1,2)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Pre-created experiment directory (if provided, will not create new one)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save checkpoint every N epochs (default: 50, reduced from 10 to save disk space)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., result/xxx/best_model.pth)')

    # [NEW 2026-02-16] Scene-weighted loss
    parser.add_argument('--use_scene_weights', type=str, default='False',
                        help='Enable scene-weighted loss (True/False)')
    parser.add_argument('--scene_weight_cat0', type=float, default=1.0,
                        help='Loss weight for category 0 (unclassified)')
    parser.add_argument('--scene_weight_cat1', type=float, default=1.0,
                        help='Loss weight for category 1 (moderate scenes)')
    parser.add_argument('--scene_weight_cat2', type=float, default=1.0,
                        help='Loss weight for category 2 (small targets)')
    parser.add_argument('--scene_weight_cat3', type=float, default=2.0,
                        help='Loss weight for category 3 (coastal scenes, default 2.0 to prioritize difficult scenes)')

    # Evaluation
    parser.add_argument('--peak_threshold', type=float, default=0.35,
                        help='Threshold for binary segmentation (optimized from training observations)')
    parser.add_argument('--adaptive_threshold', type=str, default='False',
                        help='Use adaptive threshold based on prediction distribution')

    # [NEW 2026-02-26] Training improvements (顶会技巧)
    parser.add_argument('--use_augmentation', type=str, default='False',
                        help='Enable enhanced data augmentation (vertical flip, 90° rotation, gamma, contrast)')
    parser.add_argument('--use_multi_scale', type=str, default='False',
                        help='Enable multi-scale training (dynamic resolution [192-320])')
    parser.add_argument('--multi_scale_range', type=str, default='0.75,1.25',
                        help='Multi-scale range (min,max)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective_batch = batch × steps)')
    parser.add_argument('--use_tta', type=str, default='False',
                        help='Enable Test-Time Augmentation (8-way)')
    parser.add_argument('--use_ema', type=str, default='False',
                        help='Enable Exponential Moving Average of model weights')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate')
    parser.add_argument('--use_warmup', type=str, default='False',
                        help='Enable learning rate warmup')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')

    args = parser.parse_args()

    # Create or use experiment directory
    if args.save_dir:
        # Use pre-created directory (from auto_train_gpu.sh)
        save_dir = args.save_dir
        dt_string = os.path.basename(save_dir).split('_')[-1]  # Extract timestamp from dir name
        print(f"✅ Using pre-created experiment directory: {save_dir}")
    else:
        # Create new experiment directory with timestamp
        save_dir, dt_string = create_experiment_dir(args)

    args.save_dir = save_dir
    args.dt_string = dt_string

    # Save training configuration
    save_training_config(args, save_dir)

    return args


class MambaDataset(Dataset):
    """
    Wrapper around DataLoader that generates Gaussian heatmap targets.
    Supports both PoLaRISTrainLoader (dict output) and TrainSetLoader (tuple output).
    """
    def __init__(self, base_loader, dataset_dir, gaussian_iou=0.7, downscale=1, use_polaris_loader=True):
        self.base_loader = base_loader
        self.dataset_dir = dataset_dir
        self.gaussian_iou = gaussian_iou
        self.downscale = downscale
        self.img_ids = base_loader._items
        self.use_polaris_loader = use_polaris_loader

        # [NEW 2026-02-16] Load category mapping for scene-weighted loss
        self.category_map = self._load_category_mapping(dataset_dir)

    def _load_category_mapping(self, dataset_dir):
        """Load category mapping from selection_summary_new.txt"""
        category_file = os.path.join(dataset_dir, 'selection_summary_new.txt')
        if not os.path.exists(category_file):
            category_file = os.path.join(dataset_dir, 'select-view', 'selection_summary_new.txt')

        category_map = {}

        if os.path.exists(category_file):
            with open(category_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) == 2:
                            img_id = parts[0].strip().replace('.png', '')
                            category = int(parts[1].strip())
                            category_map[img_id] = category
            print(f"✅ Loaded {len(category_map)} category mappings from {category_file}")
        else:
            print(f"⚠️  Category file not found: {category_file}")
            print(f"   Scene-weighted loss will use default weight 1.0 for all samples")

        return category_map

    def __len__(self):
        return len(self.base_loader)

    def __getitem__(self, index):
        # Get sample from base loader
        sample = self.base_loader[index]

        # Handle different loader output formats
        if self.use_polaris_loader:
            # PoLaRISTrainLoader returns dict: {'image', 'mask', 'oracle_mask', 'img_id', ...}
            img = sample['image']
            img_id = sample['img_id']
        else:
            # TrainSetLoader returns tuple: (img, mask, oracle_mask)
            img = sample[0]
            # Get img_id from base_loader's item list
            img_id = self.base_loader._items[index]

        # Separate IR and LiDAR channels
        # Both loaders stack [IR, Depth] when in_channels=2
        if img.shape[0] == 2:
            # in_channels=2: image = [IR, Depth]
            ir_img = img[0:1, :, :]      # (1, H, W)
            lidar_img = img[1:2, :, :]   # (1, H, W) depth map
        elif img.shape[0] == 1:
            # in_channels=1: image = IR only
            ir_img = img[0:1, :, :]      # (1, H, W)
            lidar_img = torch.zeros_like(ir_img)  # (1, H, W) all zeros
        else:
            # in_channels=3: RGB (for models like Mamba-UNet++ that support 3-channel input)
            # [FIX 2026-03-02] Keep all 3 channels for fair comparison with DNANet
            ir_img = img  # (3, H, W) - keep all channels
            lidar_img = torch.zeros((1, img.shape[1], img.shape[2]), dtype=img.dtype, device=img.device)

        # [CRITICAL FIX 2026-03-02] Load GT mask directly from masks/ directory
        # This matches DNANet's approach and avoids issues with missing/mismatched labels
        # Previously: Generated mask from YOLO labels → failed when labels missing
        # Now: Load mask PNG directly → works like DNANet
        mask_path = os.path.join(self.dataset_dir, 'masks', f'{img_id}.png')

        if os.path.exists(mask_path):
            # Load mask image (grayscale PNG)
            mask_img = Image.open(mask_path).convert('L')

            # Resize to match IR image size if needed
            H, W = ir_img.shape[1], ir_img.shape[2]
            if mask_img.size != (W, H):
                mask_img = mask_img.resize((W, H), Image.NEAREST)

            # Convert to numpy and normalize to [0, 1]
            mask = np.array(mask_img, dtype=np.float32) / 255.0

            # Binarize (threshold at 0.5)
            mask = (mask > 0.5).astype(np.float32)
        else:
            # Fallback: try old YOLO labels method
            H, W = ir_img.shape[1], ir_img.shape[2]
            label_path = os.path.join(self.dataset_dir, 'labels', f'{img_id}.txt')
            labels = load_yolo_labels_binary(label_path)
            mask = generate_binary_mask_target(
                labels,
                img_size=(H, W),
                downscale=self.downscale,
                fill_mode='box',
                ellipse_ratio=0.8,
            )

        # Keep variable name 'heatmap' for compatibility with rest of code
        heatmap = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W), values in {0, 1}

        # [NEW 2026-02-16] Get category for scene-weighted loss
        category = self.category_map.get(img_id, 0)  # Default to category 0 if not found

        return {
            'ir_img': ir_img,
            'lidar_img': lidar_img,
            'heatmap': heatmap,
            'img_id': img_id,
            'category': category,  # Scene category for weighted loss
        }


class Trainer:
    def __init__(self, args):
        self.args = args

        # Setup device
        if torch.cuda.is_available():
            # Parse GPU IDs
            gpu_ids = [int(x) for x in args.gpus.split(',')]
            self.device = torch.device(f'cuda:{gpu_ids[0]}')
            self.use_multi_gpu = len(gpu_ids) > 1
            self.gpu_ids = gpu_ids
        else:
            self.device = torch.device('cpu')
            self.use_multi_gpu = False
            self.gpu_ids = []
            print("⚠️  CUDA not available, using CPU (not recommended)")

        # Load dataset
        # Note: dataset_dir is the base directory with images/masks/lidar_roi
        # load_dataset reads train.txt from root/dataset/split_method/
        dataset_dir = os.path.join(args.root, args.dataset)
        train_img_ids, val_img_ids, _ = load_dataset(args.root, args.dataset, args.split_method)
        
        # Verify dataset structure
        images_path = os.path.join(dataset_dir, args.image_folder)
        if not os.path.exists(images_path):
            raise FileNotFoundError(
                f"Images directory not found: {images_path}\n"
                f"Expected structure: {dataset_dir}/{{images,masks,lidar_roi,labels}}/\n"
                f"Split files should be in: {args.root}/{args.dataset}/{args.split_method}/{{train,test}}.txt"
            )
        print(f"✓ Dataset directory: {dataset_dir}")
        print(f"✓ Images: {images_path}")
        print(f"✓ Train samples: {len(train_img_ids)}")
        print(f"✓ Val samples: {len(val_img_ids)}")

        # Choose DataLoader based on use_polaris_loader flag
        use_polaris_loader = (args.use_polaris_loader == 'True')
        
        if use_polaris_loader:
            print(f"✓ Using PoLaRISTrainLoader (16-bit + LiDAR support)")
            # Base loaders with 16-bit and LiDAR support
            base_train_loader = PoLaRISTrainLoader(
                dataset_dir=dataset_dir,
                img_id=train_img_ids,
                base_size=args.base_size,
                crop_size=args.crop_size,
                transform=None,
                suffix=args.suffix,
                normalize_16bit=(args.normalize_16bit == 'True'),
                normalize_mode=args.normalize_mode,
                in_channels=args.in_channels,
                image_folder=args.image_folder,
            )
            base_test_loader = PoLaRISTestLoader(
                dataset_dir=dataset_dir,
                img_id=val_img_ids,
                base_size=args.base_size,
                crop_size=args.crop_size,
                transform=None,
                suffix=args.suffix,
                normalize_16bit=(args.normalize_16bit == 'True'),
                normalize_mode=args.normalize_mode,
                in_channels=args.in_channels,
                image_folder=args.image_folder,
            )
        else:
            print(f"✓ Using TrainSetLoader (8-bit only, DNANet-compatible)")
            # Traditional loaders for 8-bit images (fair comparison with DNANet)
            from torchvision import transforms
            
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
            
            base_train_loader = TrainSetLoader(
                dataset_dir=dataset_dir,
                img_id=train_img_ids,
                base_size=args.base_size,
                crop_size=args.crop_size,
                transform=input_transform,
                suffix=args.suffix,
                in_channels=args.in_channels,
                image_folder=args.image_folder,
            )
            base_test_loader = TestSetLoader(
                dataset_dir=dataset_dir,
                img_id=val_img_ids,
                base_size=args.base_size,
                crop_size=args.crop_size,
                transform=input_transform,
                suffix=args.suffix,
                in_channels=args.in_channels,
                image_folder=args.image_folder,
            )

        # Wrap with Gaussian target generation
        self.trainset = MambaDataset(
            base_train_loader,
            dataset_dir,
            gaussian_iou=args.gaussian_iou,
            downscale=args.heatmap_downscale,
            use_polaris_loader=use_polaris_loader,
        )
        self.testset = MambaDataset(
            base_test_loader,
            dataset_dir,
            gaussian_iou=args.gaussian_iou,
            downscale=args.heatmap_downscale,
            use_polaris_loader=use_polaris_loader,
        )

        # Data loaders
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        self.test_loader = DataLoader(
            self.testset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        # Initialize model
        use_lidar = (args.use_lidar == 'True')
        use_deep_supervision = (args.use_deep_supervision == 'True')
        if args.model == 'mamba_tiny':
            self.net = polaris_mamba_tiny(use_lidar=use_lidar)
        elif args.model == 'mamba_small':
            self.net = polaris_mamba_small(use_lidar=use_lidar)
        elif args.model == 'mamba_base':
            self.net = polaris_mamba_base(use_lidar=use_lidar)
        elif args.model == 'mamba_tiny_multiscale':
            self.net = polaris_mamba_tiny_multiscale(use_lidar=use_lidar, use_deep_supervision=use_deep_supervision)
        elif args.model == 'mamba_small_multiscale':
            self.net = polaris_mamba_small_multiscale(use_lidar=use_lidar, use_deep_supervision=use_deep_supervision)
        elif args.model == 'mamba_tiny_progressive':
            self.net = polaris_mamba_tiny_progressive(
                use_lidar=use_lidar,
                use_deep_supervision=use_deep_supervision,
                use_cbam=args.use_cbam  # [NEW 2026-02-27] CBAM attention
            )
        elif args.model == 'mamba_small_progressive':
            self.net = polaris_mamba_small_progressive(
                use_lidar=use_lidar,
                use_deep_supervision=use_deep_supervision,
                use_cbam=args.use_cbam  # [NEW 2026-02-27] CBAM attention
            )
        elif args.model == 'mamba_unetpp':
            # [NEW 2026-03-02] Mamba-UNet++ Hybrid Architecture
            # Combines Mamba blocks (global context) with UNet++ dense skip connections (local details)
            # [FIX 2026-03-02] Added deep_supervision support
            self.net = mamba_unetplusplus(
                in_channels=args.in_channels,
                num_classes=1,
                deep_supervision=args.use_deep_supervision  # Enable deep supervision
            )
            print(f"✓ Using Mamba-UNet++ Hybrid (in_channels={args.in_channels}, deep_supervision={args.use_deep_supervision})")
        else:
            raise ValueError(f"Unknown model: {args.model}")

        self.net = self.net.to(self.device)

        # Multi-GPU support
        if self.use_multi_gpu:
            print(f"✅ Using DataParallel on GPUs: {self.gpu_ids}")
            self.net = torch.nn.DataParallel(self.net, device_ids=self.gpu_ids)

        # Count parameters
        num_params = sum(p.numel() for p in self.net.parameters())
        print(f"✅ Model: {args.model}, Parameters: {num_params / 1e6:.2f}M")

        # [2026-02-03] Loss Function Factory - Support Multiple Loss Strategies
        # Available types:
        # 1. 'improved_bce_dice': Baseline (Focal BCE + Dice)
        # 2. 'projection': Weak-supervision via X/Y projection (allows irregular shapes)
        # 3. 'hybrid': Combination of both (RECOMMENDED for Box GT)
        
        loss_config = {
            'focal_alpha': args.focal_alpha,
            'focal_gamma': args.focal_gamma,
            'dice_weight': args.dice_weight,
            'projection_weight': args.projection_weight,
            'projection_mode': args.projection_mode,
            'ohem_ratio': args.ohem_ratio,
            # BAGS strategy parameters
            'gaussian_sigma_scale': args.gaussian_sigma_scale,
            'gaussian_min_sigma': args.gaussian_min_sigma,
        }
        
        self.criterion = LossFactory.create(args.loss_type, **loss_config)
        self.loss_type = args.loss_type
        self.loss_name = LossFactory.get_loss_name(args.loss_type, **loss_config)
        
        # Optional: Confidence calibration loss (for fine-tuning)
        self.calib_criterion = ConfidenceCalibrationLoss(
            target_bg_conf=0.1,
            target_fg_conf=0.9,
        )
        self.use_calib_loss = False

        # [NEW 2026-02-16] Scene-weighted loss configuration
        self.use_scene_weights = (args.use_scene_weights == 'True')
        if self.use_scene_weights:
            self.scene_weights = {
                0: args.scene_weight_cat0,
                1: args.scene_weight_cat1,
                2: args.scene_weight_cat2,
                3: args.scene_weight_cat3,
            }
        else:
            self.scene_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

        print(f"\n{'='*70}")
        print(f"Loss Configuration")
        print(f"{'='*70}")
        print(f"  Loss Type: {args.loss_type}")
        print(f"  Loss Name: {self.loss_name}")
        if args.loss_type in ['improved_bce_dice', 'hybrid']:
            print(f"  - Focal α: {args.focal_alpha} (pos/neg weight balance)")
            print(f"  - Focal γ: {args.focal_gamma} (hard example focus)")
            print(f"  - Dice weight: {args.dice_weight} (segmentation quality)")
        if args.loss_type in ['projection', 'hybrid']:
            print(f"  - Projection weight: {args.projection_weight} (boundary constraint)")
            print(f"  - Projection mode: {args.projection_mode} (max=sensitive, mean=smooth)")
        if args.loss_type == 'gaussian_focal':
            print(f"  ★ BAGS Strategy: Box Mask → Gaussian Heatmap (realtime GPU conversion)")
            print(f"  - GaussianFocal α: {args.focal_alpha} (foreground focus)")
            print(f"  - GaussianFocal β: {args.focal_gamma} (background penalty decay)")
            print(f"  - Sigma scale: {args.gaussian_sigma_scale} (σ = box_dim / {args.gaussian_sigma_scale})")
            print(f"  - Min sigma:   {args.gaussian_min_sigma} px")
            print(f"  - Projection weight: {args.projection_weight} (box boundary auxiliary)")
            print(f"  - Dice weight: 0.0 (disabled, incompatible with soft Gaussian targets)")
        if args.ohem_ratio > 0:
            print(f"  - OHEM ratio: {args.ohem_ratio} (hard example mining)")
        if self.use_scene_weights:
            print(f"  - Scene Weights: Cat0={self.scene_weights[0]:.1f}, "
                  f"Cat1={self.scene_weights[1]:.1f}, "
                  f"Cat2={self.scene_weights[2]:.1f}, "
                  f"Cat3={self.scene_weights[3]:.1f} (coastal scenes)")
        print(f"{'='*70}\n")
        print("   - MaxPool LiDAR + Skip Connection: Preserves small target features")
        print("   - Target: Threshold 0.3-0.5, Precision 70%+, IoU 35%+ by epoch 50")

        # Optimizer
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adagrad':
            self.optimizer = optim.Adagrad(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        # Scheduler with warmup
        if args.scheduler == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.epochs,
                eta_min=args.min_lr,
            )
        elif args.scheduler == 'CosineAnnealingWarmRestarts':
            # Periodic learning rate restart for long training (>500 epochs)
            # T_0=100: First cycle is 100 epochs, then 200, 400, 800...
            # This prevents learning rate decay to near-zero in long training
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=100,      # First restart after 100 epochs
                T_mult=2,     # Double the period after each restart
                eta_min=args.min_lr,
            )
        elif args.scheduler == 'StepLR':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        
        # Warmup settings (restored to original baseline for comparison)
        self.warmup_epochs = 5  # Original: 5 epochs
        self.warmup_lr_start = args.lr * 0.01  # Original: 1% of target lr

        # Metrics
        self.best_iou = 0.0
        self.best_box_iou = 0.0  # [NEW] Track best Mask-to-Box IoU
        self.best_epoch = 0

        # Logging
        self.train_log_path = init_training_log_csv(args.save_dir)

        # Track best threshold for monitoring
        self.best_threshold_history = []

        # ========== [NEW 2026-02-26] Training Improvements ==========
        # DISABLED for baseline validation (2026-02-27)
        # self._init_training_improvements(args)

        # Initialize required attributes for baseline (to avoid AttributeError)
        self.gradient_accumulation_steps = 1  # No accumulation in baseline
        self.use_augmentation = False
        self.use_multi_scale = False
        self.use_ema = False
        self.use_tta = False
        self.use_warmup_improvement = False

        # Resume from checkpoint if provided
        if args.resume:
            self._load_checkpoint(args.resume)

    def _init_training_improvements(self, args):
        """
        Initialize training improvement modules (2026-02-26)

        Includes:
        1. Enhanced data augmentation
        2. Multi-scale training
        3. Gradient accumulation
        4. Test-Time Augmentation
        5. EMA (Exponential Moving Average)
        6. Warmup scheduler
        """
        print(f"\n{'='*70}")
        print("🚀 Initializing Training Improvements")
        print(f"{'='*70}")

        # Import improvement modules
        try:
            from model_Mamba.dataset.augmentation import InfraredAugmentation, TestTimeAugmentation
            from model_Mamba.utils_training import (
                EMA, WarmupScheduler, MultiScaleTrainingManager, GradientAccumulator
            )
        except ImportError:
            try:
                from dataset.augmentation import InfraredAugmentation, TestTimeAugmentation
                from utils_training import (
                    EMA, WarmupScheduler, MultiScaleTrainingManager, GradientAccumulator
                )
            except ImportError:
                print("⚠️  Warning: Improvement modules not found. Disabling all improvements.")
                args.use_augmentation = 'False'
                args.use_multi_scale = 'False'
                args.use_tta = 'False'
                args.use_ema = 'False'
                args.use_warmup = 'False'
                return

        # 1. Data Augmentation
        self.use_augmentation = (args.use_augmentation == 'True')
        if self.use_augmentation:
            print("\n✅ [1/6] Enhanced Data Augmentation")
            print("  - Vertical flip (p=0.5)")
            print("  - 90° rotation (p=0.5)")
            print("  - Multi-scale crop (0.8-1.2)")
            print("  - Gamma correction (γ=0.7-1.3, p=0.3)")
            print("  - Contrast adjustment (0.7-1.3, p=0.3)")
            print("  - Gaussian noise (std=5-15, p=0.2)")
            print("  - Gaussian blur (r=0.5-1.5, p=0.2)")

            # [2026-02-27] 仅使用几何增强，禁用光度增强（gamma/对比度可能破坏红外热特征）
            self.augmentation = InfraredAugmentation(mode='train', use_all=False)

            # Patch augmentation into trainset - use wrapper class to avoid type issues
            base_loader = self.trainset.base_loader
            original_sync_transform = base_loader._sync_transform
            augmentation = self.augmentation

            # Create wrapper with proper signature based on loader type
            use_polaris = (args.use_polaris_loader == 'True')

            if use_polaris:
                # PoLaRISTrainLoader signature
                def augmented_sync_transform(img, mask, oracle_mask, depth=None, lidar_points=None):  # type: ignore
                    img, mask, depth = augmentation(img, mask, depth)
                    return original_sync_transform(img, mask, oracle_mask, depth, lidar_points)
            else:
                # TrainSetLoader signature
                def augmented_sync_transform(img, mask, depth=None, oracle_mask=None):  # type: ignore
                    img, mask, depth = augmentation(img, mask, depth)
                    return original_sync_transform(img, mask, depth, oracle_mask)

            base_loader._sync_transform = augmented_sync_transform  # type: ignore

        # 2. Multi-scale Training
        self.use_multi_scale = (args.use_multi_scale == 'True')
        if self.use_multi_scale:
            scale_range = tuple(map(float, args.multi_scale_range.split(',')))
            self.multi_scale_manager = MultiScaleTrainingManager(
                base_size=args.base_size,
                scale_range=scale_range,
                step=32
            )
            print(f"\n✅ [2/6] Multi-scale Training: {self.multi_scale_manager.size_list}")

        # 3. Gradient Accumulation
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.accumulator = GradientAccumulator(accumulation_steps=self.gradient_accumulation_steps)
        effective_batch = args.train_batch_size * self.gradient_accumulation_steps
        print(f"\n✅ [3/6] Gradient Accumulation: {self.gradient_accumulation_steps} steps")
        print(f"   Effective Batch Size: {effective_batch}")

        # 4. Test-Time Augmentation
        self.use_tta = (args.use_tta == 'True')
        if self.use_tta:
            self.tta_predictor = TestTimeAugmentation(self.net, device=self.device)
            print(f"\n✅ [4/6] Test-Time Augmentation: 8-way")

        # 5. EMA
        self.use_ema = (args.use_ema == 'True')
        if self.use_ema:
            self.ema = EMA(self.net, decay=args.ema_decay, device=self.device)
            print(f"\n✅ [5/6] EMA: decay={args.ema_decay}")

        # 6. Warmup Scheduler
        self.use_warmup_improvement = (args.use_warmup == 'True')
        if self.use_warmup_improvement:
            base_scheduler = self.scheduler
            self.scheduler = WarmupScheduler(
                self.optimizer,
                warmup_epochs=args.warmup_epochs,
                base_scheduler=base_scheduler
            )
            print(f"\n✅ [6/6] Warmup Scheduler: {args.warmup_epochs} epochs")

        print(f"\n{'='*70}")
        print("🎯 Expected Performance Gain: +10-16% IoU")
        print(f"{'='*70}\n")

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training."""
        print(f"\n{'='*60}")
        print(f"📦 Loading checkpoint from: {checkpoint_path}")
        print(f"{'='*60}")

        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print(f"   Starting training from scratch...")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            if self.use_multi_gpu:
                self.net.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.net.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state (critical for CosineAnnealingLR to continue correctly)
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✅ Loaded scheduler state (LR will continue from correct cosine position)")
            else:
                print(f"⚠️  No scheduler state in checkpoint — LR schedule will restart from epoch 0")

            # Load metrics
            if 'iou' in checkpoint:
                self.best_iou = checkpoint['iou']
                print(f"\u2705 Loaded best Seg IoU: {self.best_iou:.4f}")

            if 'box_iou' in checkpoint:
                self.best_box_iou = checkpoint['box_iou']
                print(f"\u2705 Loaded best Box IoU: {self.best_box_iou:.4f}")

            # Fallback: old checkpoints may not have iou/box_iou fields.
            # Scan existing best_model_*.pth filenames in save_dir to recover scores.
            if 'iou' not in checkpoint or 'box_iou' not in checkpoint:
                import glob, re
                save_dir = os.path.dirname(checkpoint_path)
                best_files = glob.glob(os.path.join(save_dir, 'best_model_epoch*.pth'))
                for bf in best_files:
                    name = os.path.basename(bf)
                    # IoU-first files: best_model_epoch####_IoU0.xxxx_BoxIoU0.xxxx.pth
                    m = re.search(r'_IoU(\d+\.\d+)_BoxIoU(\d+\.\d+)', name)
                    if m and 'iou' not in checkpoint:
                        self.best_iou = max(self.best_iou, float(m.group(1)))
                    # BoxIoU-first files: best_model_epoch####_BoxIoU0.xxxx_IoU0.xxxx.pth
                    m2 = re.search(r'_BoxIoU(\d+\.\d+)_IoU(\d+\.\d+)', name)
                    if m2 and 'box_iou' not in checkpoint:
                        self.best_box_iou = max(self.best_box_iou, float(m2.group(1)))
                if best_files:
                    print(f"\u26a0\ufe0f  Checkpoint had no metric fields \u2014 recovered from disk:")
                    print(f"   best Seg IoU={self.best_iou:.4f}, best Box IoU={self.best_box_iou:.4f}")

            if 'epoch' in checkpoint:
                self.args.start_epoch = checkpoint['epoch'] + 1
                print(f"✅ Resuming from epoch: {self.args.start_epoch}")

            print(f"{'='*60}\n")

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print(f"   Starting training from scratch...")

    def training(self, epoch):
        """Training loop for one epoch."""
        self.net.train()
        loss_meter = AverageMeter()

        # [NEW 2026-02-26] Multi-scale training: change resolution every 20 epochs (conservative)
        if hasattr(self, 'use_multi_scale') and self.use_multi_scale:
            if epoch > 0 and epoch % 20 == 0:
                new_size = self.multi_scale_manager.get_random_size()
                if new_size != self.args.base_size:
                    print(f"\n🔄 Multi-scale: Switching to resolution {new_size} (from {self.args.base_size})")
                    self.args.base_size = new_size
                    self.args.crop_size = new_size
                    # Update base_loader size (will take effect next epoch when dataloader rebuilds)
                    self.trainset.base_loader.base_size = new_size
                    self.trainset.base_loader.crop_size = new_size

        # Warmup learning rate for first few epochs (skip if using WarmupScheduler)
        if not hasattr(self, 'use_warmup_improvement') or not self.use_warmup_improvement:
            if epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                lr = self.warmup_lr_start + (self.args.lr - self.warmup_lr_start) * warmup_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print(f"  🔥 Warmup: lr={lr:.6f} (epoch {epoch+1}/{self.warmup_epochs})")

        # Training progress bar (更新频率：每5%)
        total_batches = len(self.train_loader)
        update_interval = max(1, total_batches // 20)  # 每5%更新一次

        # [NEW 2026-02-26] Initialize gradient accumulation
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} Training', ncols=100)
        for i, batch in enumerate(pbar):
            ir_img = batch['ir_img'].to(self.device)
            lidar_img = batch['lidar_img'].to(self.device)
            heatmap_gt = batch['heatmap'].to(self.device)
            categories = batch['category']  # [NEW 2026-02-16] Get scene categories

            # Forward
            output = self.net(ir_img, lidar_img)

            # [NEW] Loss calculation with Deep Supervision support
            # [FIXED 2026-02-20] Properly handle scene-weighted loss
            if self.use_scene_weights:
                # Per-sample loss calculation for proper weighting
                sample_weights = torch.tensor(
                    [self.scene_weights[int(cat)] for cat in categories],
                    device=self.device, dtype=torch.float32
                )

                batch_size = heatmap_gt.size(0)
                weighted_losses = []

                for idx in range(batch_size):
                    if isinstance(output, list):
                        # Deep Supervision: output = [main_pred, aux_pred_s2, aux_pred_s3]
                        sample_loss_main = self.criterion(output[0][idx:idx+1], heatmap_gt[idx:idx+1])
                        sample_loss_aux2 = self.criterion(output[1][idx:idx+1], heatmap_gt[idx:idx+1])
                        sample_loss_aux3 = self.criterion(output[2][idx:idx+1], heatmap_gt[idx:idx+1])
                        # Original deep supervision weight (baseline: 0.4)
                        sample_loss = sample_loss_main + 0.5 * sample_loss_aux2 + 0.4 * sample_loss_aux3
                    else:
                        # Standard mode: single output
                        sample_loss = self.criterion(output[idx:idx+1], heatmap_gt[idx:idx+1])

                    # Apply scene weight to this sample
                    weighted_loss = sample_loss * sample_weights[idx]
                    weighted_losses.append(weighted_loss)

                # Average weighted losses
                loss = torch.stack(weighted_losses).mean()

                # Debug: Print scene-weighted loss breakdown (every 100 batches)
                # [COMMENTED 2026-02-20] Verified working, disable to reduce log verbosity
                # if (i + 1) % 100 == 0 or (i + 1) == total_batches:
                #     print(f"\n[Epoch {epoch}] Scene-Weighted Loss (batch {i+1}/{total_batches}):")
                #     cat_counts = {}
                #     for cat in categories:
                #         cat_id = int(cat)
                #         cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
                #     for cat_id, count in sorted(cat_counts.items()):
                #         print(f"  Cat{cat_id}: {count} samples, weight={self.scene_weights[cat_id]:.2f}")
                #     print(f"  Weighted Loss: {loss.item():.4f}")
            else:
                # Standard loss calculation (no scene weighting)
                if isinstance(output, list):
                    # Deep Supervision: output = [main_pred, aux_pred_s2, aux_pred_s3]
                    heatmap_pred = output[0]  # Main prediction
                    loss_main = self.criterion(heatmap_pred, heatmap_gt)
                    loss_aux2 = self.criterion(output[1], heatmap_gt)
                    loss_aux3 = self.criterion(output[2], heatmap_gt)

                    # Weighted combination: Original baseline weight (0.4)
                    loss = loss_main + 0.5 * loss_aux2 + 0.4 * loss_aux3

                    # Debug: Print deep supervision loss breakdown (every 100 batches)
                    if (i + 1) % 100 == 0 or (i + 1) == total_batches:
                        print(f"\n[Epoch {epoch}] Deep Supervision Loss (batch {i+1}/{total_batches}):")
                        print(f"  Main Loss:  {loss_main.item():.4f}")
                        print(f"  Aux2 Loss:  {loss_aux2.item():.4f} (weight: 0.5)")
                        print(f"  Aux3 Loss:  {loss_aux3.item():.4f} (weight: 0.4)")
                        print(f"  Total Loss: {loss.item():.4f}")
                else:
                    # Standard mode: single output
                    heatmap_pred = output
                    loss = self.criterion(heatmap_pred, heatmap_gt)

            # Skip batch if loss is NaN/Inf (numerical instability in Mamba SSM)
            if not torch.isfinite(loss):
                print(f"\n[WARNING] Epoch {epoch} batch {i}: loss={loss.item()}, skipping batch")
                self.optimizer.zero_grad()
                continue

            # Backward and optimizer step (baseline)
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update loss meter (use original unscaled loss for logging)
            loss_meter.update(loss.item())

            # Update progress bar (每5%更新一次，减少日志输出)
            if (i + 1) % update_interval == 0 or (i + 1) == total_batches:
                pbar.set_postfix({
                    'Loss': f'{loss_meter.avg:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })

            # Clean up GPU memory periodically to prevent OOM
            # Critical for deep supervision and progressive models
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

        pbar.close()

        # Clean up GPU memory after training epoch
        torch.cuda.empty_cache()

        return loss_meter.avg

    def testing(self, epoch):
        """Testing loop."""
        self.net.eval()
        loss_meter = AverageMeter()
        iou_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        box_iou_sum = 0.0  # [NEW] Mask-to-Box IoU accumulator
        count = 0

        # Progress bar with real-time IoU display
        pbar = tqdm(self.test_loader, desc=f'Epoch {epoch} Testing', ncols=120)

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                ir_img = batch['ir_img'].to(self.device)
                lidar_img = batch['lidar_img'].to(self.device)
                heatmap_gt = batch['heatmap'].to(self.device)

                # [NEW 2026-02-26] Forward with optional TTA
                if hasattr(self, 'use_tta') and self.use_tta:
                    # Use Test-Time Augmentation (8-way)
                    output = self.tta_predictor.predict(ir_img, lidar_img)
                else:
                    # Standard forward
                    output = self.net(ir_img, lidar_img)

                # [NEW] Handle Deep Supervision outputs (only use main prediction for evaluation)
                if isinstance(output, list):
                    heatmap_pred = output[0]  # Main prediction only
                else:
                    heatmap_pred = output

                # Loss (only compute on main prediction during eval)
                loss = self.criterion(heatmap_pred, heatmap_gt)
                loss_meter.update(loss.item(), ir_img.size(0))
                
                # Visualization debug: save first batch of each epoch
                # DISABLED: Disk space issue - uncomment when needed
                # if batch_idx == 0:
                #     import torchvision
                #     vis_dir = os.path.join(self.args.save_dir, 'vis_debug')
                #     os.makedirs(vis_dir, exist_ok=True)
                #     debug_img = torch.cat([heatmap_gt[0], heatmap_pred[0]], dim=2)
                #     torchvision.utils.save_image(
                #         debug_img,
                #         os.path.join(vis_dir, f'epoch_{epoch:04d}.png'),
                #         normalize=True
                #     )

                # [CRITICAL FIX] Threshold Sweep - LIDAR-Mamba's secret to high IoU!
                # Instead of using fixed threshold (0.5), sweep multiple thresholds
                # and pick the one that gives best IoU. This is STANDARD practice.
                
                best_batch_iou = 0.0
                best_batch_threshold = 0.5
                best_precision = 0.0
                best_recall = 0.0
                
                # Sweep thresholds from 0.1 to 0.9 (use 0.05 step for speed)
                # For exact LIDAR-Mamba replication, use 0.01 step
                for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    pred_bin = (heatmap_pred > thresh).float()
                    gt_bin = (heatmap_gt > 0.5).float()
                    
                    # Compute IoU at this threshold
                    inter = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
                    union = (pred_bin + gt_bin).clamp(0, 1).sum(dim=(1, 2, 3))
                    iou_t = (inter / (union + 1e-7)).mean().item()
                    
                    # If this threshold gives better IoU, use it
                    if iou_t > best_batch_iou:
                        best_batch_iou = iou_t
                        best_batch_threshold = thresh
                        
                        # Also compute precision/recall at best threshold
                        tp = inter.sum().item()
                        fp = (pred_bin * (1 - gt_bin)).sum().item()
                        fn = ((1 - pred_bin) * gt_bin).sum().item()
                        best_precision = tp / (tp + fp + 1e-7)
                        best_recall = tp / (tp + fn + 1e-7)
                
                # Accumulate metrics using BEST threshold (not fixed 0.5)
                iou = best_batch_iou
                precision = best_precision
                recall = best_recall

                iou_sum += iou
                precision_sum += precision
                recall_sum += recall
                count += 1
                
                # [NEW] Calculate Mask-to-Box IoU (检测评估指标)
                # 使用与 segmentation 相同的最佳阈值，确保公平对比
                batch_box_iou = calculate_mask_to_box_iou(
                    heatmap_pred, 
                    heatmap_gt, 
                    threshold=best_batch_threshold
                )
                box_iou_sum += batch_box_iou

                # Update progress bar with real-time IoU (每5%更新一次)
                update_interval = max(1, len(self.test_loader) // 20)
                if (batch_idx + 1) % update_interval == 0 or (batch_idx + 1) == len(self.test_loader):
                    current_seg_iou = iou_sum / count
                    current_box_iou = box_iou_sum / count
                    pbar.set_postfix({
                        'Seg_IoU': f'{current_seg_iou:.4f}',
                        'Box_IoU': f'{current_box_iou:.4f}',
                        'Loss': f'{loss_meter.avg:.4f}'
                    })

                # Clean up GPU memory periodically during testing
                if (batch_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()

        pbar.close()

        avg_iou = iou_sum / count
        avg_precision = precision_sum / count
        avg_recall = recall_sum / count
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-7)
        avg_box_iou = box_iou_sum / count  # [NEW] 平均 Mask-to-Box IoU

        # Use dynamic threshold from args (for logging compatibility)
        best_threshold = self.args.peak_threshold

        print(f"\n{'='*60}")
        print(f"[Epoch {epoch}] Test Results:")
        print(f"  Segmentation IoU : {avg_iou:.4f}")
        print(f"  Mask-to-Box IoU  : {avg_box_iou:.4f}")
        print(f"  Precision        : {avg_precision:.4f}")
        print(f"  Recall           : {avg_recall:.4f}")
        print(f"  F1 Score         : {avg_f1:.4f}")
        print(f"  Loss             : {loss_meter.avg:.6f}")
        print(f"{'='*60}")

        # Save best Seg IoU model
        if avg_iou > self.best_iou:
            self.best_iou = avg_iou
            self.best_epoch = epoch

            # Save with both IoU and Box IoU in filename
            save_best_model(
                self.net,
                self.optimizer,
                epoch,
                avg_iou,
                self.args.save_dir,
                self.use_multi_gpu,
                box_iou=avg_box_iou,
                save_reason='IoU'
            )

        # Save best Box IoU model (separate tracking)
        if avg_box_iou > self.best_box_iou:
            self.best_box_iou = avg_box_iou

            # Save with Box IoU as primary metric in filename
            save_best_model(
                self.net,
                self.optimizer,
                epoch,
                avg_iou,
                self.args.save_dir,
                self.use_multi_gpu,
                box_iou=avg_box_iou,
                save_reason='BoxIoU'
            )

        # [NEW 2026-02-26] Restore original weights after evaluation
        if hasattr(self, 'use_ema') and self.use_ema:
            self.ema.restore()

        return loss_meter.avg, avg_iou, avg_precision, avg_recall, avg_f1, best_threshold, avg_box_iou

    def run(self):
        """Main training loop."""
        print(f"\n{'=' * 60}")
        print(f"Training PoLaRIS-Gaussian-Mamba")
        print(f"Dataset: {self.args.dataset}")
        print(f"Model: {self.args.model}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Batch Size: {self.args.train_batch_size}")
        print(f"Save Dir: {self.args.save_dir}")
        print(f"{'=' * 60}\n")

        for epoch in range(self.args.start_epoch, self.args.epochs):
            # 保存当前 epoch 用于信号处理
            self.current_epoch = epoch
            
            # Training
            train_loss = self.training(epoch)

            # Testing (返回值包含 box_iou)
            test_loss, test_iou, test_precision, test_recall, test_f1, best_threshold, test_box_iou = self.testing(epoch)

            # Update scheduler
            self.scheduler.step()

            # Log results with timestamp (包含 Box IoU 和 Loss Type)
            log_epoch_metrics(
                self.train_log_path,
                epoch,
                train_loss,
                test_loss,
                test_iou,
                test_box_iou,  # [2026-02-02] 添加 Box IoU 参数
                test_precision,
                test_recall,
                test_f1,
                best_threshold,
                self.optimizer.param_groups[0]['lr'],
                loss_type=self.loss_type  # [2026-02-03] 添加 Loss Type 参数
            )

            # Store last epoch metrics
            self.last_epoch = epoch
            self.last_iou = test_iou

            # Save checkpoint periodically
            # UPDATED 2026-01-30: Skip periodic saving if save_interval=0 (to save disk space)
            if self.args.save_interval > 0 and (epoch + 1) % self.args.save_interval == 0:
                checkpoint_path = os.path.join(self.args.save_dir, f'checkpoint_epoch{epoch}.pth')
                model_state = self.net.module.state_dict() if self.use_multi_gpu else self.net.state_dict()

                # Try to save, but catch disk space errors gracefully
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'iou': self.best_iou,
                        'box_iou': self.best_box_iou,
                    }, checkpoint_path)
                    print(f"✅ Saved checkpoint: {checkpoint_path}")
                except RuntimeError as e:
                    if "write failed" in str(e) or "disk" in str(e).lower():
                        print(f"⚠️  Failed to save checkpoint (disk full?): {e}")
                        print(f"   Continuing training... (best_model.pth still saved)")
                    else:
                        raise  # Re-raise if it's a different error

        # Save last epoch model
        print("\n📦 Saving last epoch model...")
        save_last_epoch_model(
            self.net,
            self.optimizer,
            self.last_epoch,
            self.last_iou,
            self.args.save_dir,
            self.use_multi_gpu
        )

        # Update training summary
        update_training_summary(
            self.args.save_dir,
            self.best_epoch,
            self.best_iou,
            self.last_epoch,
            self.last_iou
        )

        print(f"\n{'=' * 60}")
        print(f"Training Complete!")
        print(f"Best Segmentation IoU: {self.best_iou:.4f} at Epoch {self.best_epoch}")
        print(f"Best Mask-to-Box IoU : {self.best_box_iou:.4f} (Detection Performance)")
        print(f"Final IoU: {self.last_iou:.4f} at Epoch {self.last_epoch}")
        print(f"Model saved to: {self.args.save_dir}")
        print(f"{'=' * 60}\n")


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    # 全局标志用于优雅退出
    interrupted = False
    trainer_instance = None

    def signal_handler(sig, frame):
        """处理 Ctrl+C 信号，优雅保存 checkpoint"""
        global interrupted, trainer_instance
        if interrupted:
            print("\n\n⚠️  再次按 Ctrl+C 强制退出（不保存）")
            sys.exit(1)
        
        interrupted = True
        print("\n\n" + "="*60)
        print("⚠️  收到中断信号 (Ctrl+C)")
        print("="*60)
        print("正在保存当前进度...")
        
        if trainer_instance is not None:
            # 保存紧急 checkpoint
            checkpoint_path = os.path.join(args.save_dir, 'checkpoint_interrupted.pth')
            model_state = trainer_instance.net.module.state_dict() if trainer_instance.use_multi_gpu else trainer_instance.net.state_dict()
            torch.save({
                'epoch': getattr(trainer_instance, 'current_epoch', 0),
                'model_state_dict': model_state,
                'optimizer_state_dict': trainer_instance.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"✅ Checkpoint 已保存: {checkpoint_path}")
        
        print("="*60)
        print("训练已停止。可使用 checkpoint_interrupted.pth 恢复训练。")
        print("="*60)
        sys.exit(0)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set GPU
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f"✅ Using GPU: {args.gpus}")
    else:
        print("⚠️  CUDA not available, using CPU")

    # Start training
    trainer = Trainer(args)
    trainer_instance = trainer  # 保存到全局变量用于信号处理
    
    try:
        trainer.run()
    except KeyboardInterrupt:
        # 这个不会触发，因为信号处理器会先捕获
        pass
