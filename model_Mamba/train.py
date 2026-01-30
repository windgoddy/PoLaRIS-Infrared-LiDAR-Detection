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
from model.metric import ROCMetric, mIoU
from model.load_param_data import load_dataset

# Mamba model and loss (from model_Mamba/)
from model_Mamba.core.polaris_mamba import PoLaRIS_Mamba, polaris_mamba_tiny, polaris_mamba_small, polaris_mamba_base
from model_Mamba.core.loss import GaussianFocalLoss, AverageMeter
from model_Mamba.dataset.gaussian_utils import generate_gaussian_target, load_yolo_labels


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
                        choices=['mamba_tiny', 'mamba_small', 'mamba_base'],
                        help='Model variant')
    parser.add_argument('--use_lidar', type=str, default='True',
                        help='Whether to use LiDAR gating (True/False)')

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
    parser.add_argument('--base_size', type=int, default=512,
                        help='Base image size')
    parser.add_argument('--crop_size', type=int, default=480,
                        help='Crop size for training')
    parser.add_argument('--normalize_16bit', type=str, default='True',
                        help='Use Min-Max normalization for 16-bit images')

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
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['Adam', 'AdamW', 'SGD'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'StepLR'],
                        help='Learning rate scheduler')

    # Loss function
    parser.add_argument('--loss_alpha', type=float, default=2.0,
                        help='Focal loss alpha parameter')
    parser.add_argument('--loss_beta', type=float, default=4.0,
                        help='Gaussian falloff beta parameter')

    # Hardware and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU IDs to use (e.g., 0,1,2)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Evaluation
    parser.add_argument('--peak_threshold', type=float, default=0.5,
                        help='Threshold for peak detection')

    args = parser.parse_args()

    # Create save directory
    if args.experiment_name:
        save_dir = f'result/{args.dataset}/{args.model}_{args.experiment_name}'
    else:
        save_dir = f'result/{args.dataset}/{args.model}'
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    return args


class MambaDataset(Dataset):
    """
    Wrapper around PoLaRISTrainLoader that generates Gaussian heatmap targets.
    """
    def __init__(self, base_loader, dataset_dir, gaussian_iou=0.7, downscale=1):
        self.base_loader = base_loader
        self.dataset_dir = dataset_dir
        self.gaussian_iou = gaussian_iou
        self.downscale = downscale
        self.img_ids = base_loader._items

    def __len__(self):
        return len(self.base_loader)

    def __getitem__(self, index):
        # Get sample from base loader
        sample = self.base_loader[index]

        # Extract image (C, H, W) where C=1 or 2
        img = sample['image']
        img_id = sample['img_id']

        # Separate IR and LiDAR channels
        # PoLaRISTrainLoader stacks [IR, Depth] when in_channels=2
        if img.shape[0] == 2:
            # in_channels=2: image = [IR, Depth]
            ir_img = img[0:1, :, :]      # (1, H, W)
            lidar_img = img[1:2, :, :]   # (1, H, W) depth map
        elif img.shape[0] == 1:
            # in_channels=1: image = IR only
            ir_img = img[0:1, :, :]      # (1, H, W)
            lidar_img = torch.zeros_like(ir_img)  # (1, H, W) all zeros
        else:
            # in_channels=3: RGB (not supported for Mamba, use first channel)
            ir_img = img[0:1, :, :]      # (1, H, W)
            lidar_img = torch.zeros_like(ir_img)

        # Load YOLO labels
        label_path = os.path.join(self.dataset_dir, 'labels', f'{img_id}.txt')
        labels = load_yolo_labels(label_path)

        # Generate Gaussian heatmap target
        H, W = ir_img.shape[1], ir_img.shape[2]
        heatmap = generate_gaussian_target(
            labels,
            img_size=(H, W),
            downscale=self.downscale,
            min_overlap=self.gaussian_iou,
        )
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).float()  # (1, H, W)

        return {
            'ir_img': ir_img,
            'lidar_img': lidar_img,
            'heatmap': heatmap,
            'img_id': img_id,
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

        # Base loaders
        base_train_loader = PoLaRISTrainLoader(
            dataset_dir=dataset_dir,
            img_id=train_img_ids,
            base_size=args.base_size,
            crop_size=args.crop_size,
            transform=None,
            suffix=args.suffix,
            normalize_16bit=(args.normalize_16bit == 'True'),
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
            in_channels=args.in_channels,
            image_folder=args.image_folder,
        )

        # Wrap with Gaussian target generation
        self.trainset = MambaDataset(
            base_train_loader,
            dataset_dir,
            gaussian_iou=args.gaussian_iou,
            downscale=args.heatmap_downscale,
        )
        self.testset = MambaDataset(
            base_test_loader,
            dataset_dir,
            gaussian_iou=args.gaussian_iou,
            downscale=args.heatmap_downscale,
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
        if args.model == 'mamba_tiny':
            self.net = polaris_mamba_tiny(use_lidar=use_lidar)
        elif args.model == 'mamba_small':
            self.net = polaris_mamba_small(use_lidar=use_lidar)
        elif args.model == 'mamba_base':
            self.net = polaris_mamba_base(use_lidar=use_lidar)
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

        # Loss function
        self.criterion = GaussianFocalLoss(
            alpha=args.loss_alpha,
            beta=args.loss_beta,
            reduction='mean',
        )

        # Optimizer
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        # Scheduler
        if args.scheduler == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.epochs,
                eta_min=args.min_lr,
            )
        elif args.scheduler == 'StepLR':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

        # Metrics
        self.best_iou = 0.0
        self.best_epoch = 0

        # Logging
        self.train_log_path = os.path.join(args.save_dir, 'train_log.csv')
        with open(self.train_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Test_Loss', 'IoU', 'Precision', 'Recall', 'LR'])

    def training(self, epoch):
        """Training loop for one epoch."""
        self.net.train()
        loss_meter = AverageMeter()

        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for i, batch in enumerate(tbar):
            ir_img = batch['ir_img'].to(self.device)
            lidar_img = batch['lidar_img'].to(self.device)
            heatmap_gt = batch['heatmap'].to(self.device)

            # Forward
            heatmap_pred = self.net(ir_img, lidar_img)

            # Loss
            loss = self.criterion(heatmap_pred, heatmap_gt)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update meter
            loss_meter.update(loss.item(), ir_img.size(0))

            # Update tqdm
            tbar.set_postfix(loss=f'{loss_meter.avg:.6f}', lr=f'{self.optimizer.param_groups[0]["lr"]:.6f}')

        return loss_meter.avg

    def testing(self, epoch):
        """Testing loop."""
        self.net.eval()
        loss_meter = AverageMeter()
        iou_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        count = 0
        
        # Debug: track prediction statistics
        pred_stats = {'min': [], 'max': [], 'mean': []}
        gt_stats = {'min': [], 'max': [], 'mean': [], 'num_pos': []}

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='Testing')):
                ir_img = batch['ir_img'].to(self.device)
                lidar_img = batch['lidar_img'].to(self.device)
                heatmap_gt = batch['heatmap'].to(self.device)

                # Forward
                heatmap_pred = self.net(ir_img, lidar_img)

                # Collect statistics (first epoch only)
                if epoch == 0 and batch_idx < 5:
                    pred_stats['min'].append(heatmap_pred.min().item())
                    pred_stats['max'].append(heatmap_pred.max().item())
                    pred_stats['mean'].append(heatmap_pred.mean().item())
                    gt_stats['min'].append(heatmap_gt.min().item())
                    gt_stats['max'].append(heatmap_gt.max().item())
                    gt_stats['mean'].append(heatmap_gt.mean().item())
                    gt_stats['num_pos'].append((heatmap_gt > 0.5).sum().item())

                # Loss
                loss = self.criterion(heatmap_pred, heatmap_gt)
                loss_meter.update(loss.item(), ir_img.size(0))

                # Compute metrics (simple binary IoU)
                pred_binary = (heatmap_pred > self.args.peak_threshold).float()
                gt_binary = (heatmap_gt > 0.5).float()

                intersection = (pred_binary * gt_binary).sum(dim=(1, 2, 3))
                union = (pred_binary + gt_binary).clamp(0, 1).sum(dim=(1, 2, 3))
                iou = (intersection / (union + 1e-7)).mean().item()

                # Precision and Recall
                tp = intersection.sum().item()
                fp = (pred_binary * (1 - gt_binary)).sum().item()
                fn = ((1 - pred_binary) * gt_binary).sum().item()

                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)

                iou_sum += iou
                precision_sum += precision
                recall_sum += recall
                count += 1

        avg_iou = iou_sum / count
        avg_precision = precision_sum / count
        avg_recall = recall_sum / count

        print(f"\n[Epoch {epoch}] Test Loss: {loss_meter.avg:.6f}, IoU: {avg_iou:.4f}, "
              f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")
        
        # Print debug stats for first epoch
        if epoch == 0 and pred_stats['min']:
            print(f"\n  📊 Debug Statistics (first 5 batches):")
            print(f"     Pred: min={min(pred_stats['min']):.6f}, max={max(pred_stats['max']):.6f}, mean={sum(pred_stats['mean'])/len(pred_stats['mean']):.6f}")
            print(f"     GT:   min={min(gt_stats['min']):.6f}, max={max(gt_stats['max']):.6f}, mean={sum(gt_stats['mean'])/len(gt_stats['mean']):.6f}")
            print(f"     GT positive pixels: {sum(gt_stats['num_pos'])/len(gt_stats['num_pos']):.1f} per batch")
            print(f"     Threshold: {self.args.peak_threshold}")
        avg_recall = recall_sum / count

        print(f"\n[Epoch {epoch}] Test Loss: {loss_meter.avg:.6f}, IoU: {avg_iou:.4f}, "
              f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

        # Save best model
        if avg_iou > self.best_iou:
            self.best_iou = avg_iou
            self.best_epoch = epoch
            checkpoint_path = os.path.join(self.args.save_dir, 'best_model.pth')

            # Handle DataParallel wrapper
            model_state = self.net.module.state_dict() if self.use_multi_gpu else self.net.state_dict()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_iou': self.best_iou,
            }, checkpoint_path)
            print(f"✅ Saved best model (IoU: {self.best_iou:.4f})")

        return loss_meter.avg, avg_iou, avg_precision, avg_recall

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

            # Testing
            test_loss, test_iou, test_precision, test_recall = self.testing(epoch)

            # Update scheduler
            self.scheduler.step()

            # Log results
            with open(self.train_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    train_loss,
                    test_loss,
                    test_iou,
                    test_precision,
                    test_recall,
                    self.optimizer.param_groups[0]['lr']
                ])

            # Save checkpoint periodically
            if (epoch + 1) % self.args.save_interval == 0:
                checkpoint_path = os.path.join(self.args.save_dir, f'checkpoint_epoch{epoch}.pth')
                model_state = self.net.module.state_dict() if self.use_multi_gpu else self.net.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_path)

        print(f"\n{'=' * 60}")
        print(f"Training Complete!")
        print(f"Best IoU: {self.best_iou:.4f} at Epoch {self.best_epoch}")
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
