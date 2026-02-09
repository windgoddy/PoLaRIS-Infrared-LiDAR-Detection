"""
I/O Utilities for PoLaRIS-Mamba Training
========================================

Handles experiment directory structure, logging, and model saving.
Consistent with Phase3 training output format.

Author: PoLaRIS Team
Date: 2026-02-02
"""

import os
import sys
import csv
import torch
from datetime import datetime


def create_experiment_dir(args):
    """
    Create experiment directory with timestamp.

    Format: {PROJECT_ROOT}/model_Mamba/result/{experiment_name}_{dataset}_{model}_{timestamp}/

    Args:
        args: Parsed arguments from argparse

    Returns:
        save_dir: Absolute path to the created directory
        dt_string: Timestamp string (YYYYMMDD_HHMMSS)
    """
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")

    # Build directory name
    if args.experiment_name:
        dir_name = f"{args.experiment_name}_{args.dataset}_{args.model}_{dt_string}"
    else:
        dir_name = f"{args.dataset}_{args.model}_{dt_string}"

    # Get absolute paths (CRITICAL FIX: Avoid relative path issues)
    # utils_io.py is in model_Mamba/, so __file__'s dir is model_Mamba/
    MODEL_MAMBA_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(MODEL_MAMBA_DIR)

    # Create directory using absolute path: {PROJECT_ROOT}/model_Mamba/result/dir_name/
    save_dir = os.path.join(PROJECT_ROOT, 'model_Mamba', 'result', dir_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"✅ Experiment directory created: {save_dir}")
    return save_dir, dt_string


def save_training_config(args, save_dir):
    """
    Save training configuration to train_log.txt.

    Args:
        args: Parsed arguments
        save_dir: Directory to save the config file
    """
    config_file = os.path.join(save_dir, 'train_log.txt')

    with open(config_file, 'w') as f:
        # Write command line at the very top
        f.write("=" * 80 + "\n")
        f.write("Command Line\n")
        f.write("=" * 80 + "\n")
        # Reconstruct command from sys.argv
        command = ' '.join(sys.argv)
        f.write(f"{command}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write header
        now = datetime.now()
        f.write("=" * 80 + "\n")
        f.write("PoLaRIS-Mamba Training Configuration\n")
        f.write("=" * 80 + "\n")
        f.write(f"Start Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Write all arguments
        f.write("Arguments:\n")
        f.write("-" * 80 + "\n")
        dict_args = vars(args)
        for key, value in sorted(dict_args.items()):
            f.write(f"  {key:25s}: {value}\n")

        # Write Loss configuration (2026-02-03)
        f.write("\n" + "=" * 80 + "\n")
        f.write("Loss Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Loss Type: {getattr(args, 'loss_type', 'improved_bce_dice')}\n")
        if hasattr(args, 'focal_alpha'):
            f.write(f"  Focal Alpha: {args.focal_alpha}\n")
        if hasattr(args, 'focal_gamma'):
            f.write(f"  Focal Gamma: {args.focal_gamma}\n")
        if hasattr(args, 'dice_weight'):
            f.write(f"  Dice Weight: {args.dice_weight}\n")
        if hasattr(args, 'projection_weight'):
            f.write(f"  Projection Weight: {args.projection_weight}\n")
        if hasattr(args, 'projection_mode'):
            f.write(f"  Projection Mode: {args.projection_mode}\n")
        if hasattr(args, 'ohem_ratio'):
            f.write(f"  OHEM Ratio: {args.ohem_ratio}\n")

        # Write model info
        f.write("\n" + "=" * 80 + "\n")
        f.write("Model Information:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Architecture        : PoLaRIS-Mamba ({args.model})\n")
        f.write(f"  Loss Function       : Improved BCE+Dice (Focal mechanism)\n")
        f.write(f"  Input Channels      : {args.in_channels} ({'IR+Depth' if args.in_channels == 2 else 'IR only'})\n")
        f.write(f"  LiDAR Gating        : {args.use_lidar}\n")
        f.write(f"  16-bit Normalization: {args.normalize_16bit}\n")
        f.write("=" * 80 + "\n")

    print(f"✅ Training config saved: {config_file}")


def init_training_log_csv(save_dir):
    """
    Initialize train_log.csv with headers.

    Args:
        save_dir: Directory to save the CSV file

    Returns:
        csv_path: Path to the CSV file
    """
    csv_path = os.path.join(save_dir, 'train_log.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Timestamp',
            'Epoch',
            'Train_Loss',
            'Test_Loss',
            'IoU',
            'Box_IoU',
            'Precision',
            'Recall',
            'F1',
            'Best_Threshold',
            'LR',
            'Loss_Type'  # 2026-02-03: Track loss function type
        ])

    print(f"✅ Training log CSV initialized: {csv_path}")
    return csv_path


def log_epoch_metrics(csv_path, epoch, train_loss, test_loss, iou, box_iou, precision, recall, f1, best_threshold, lr, loss_type='improved_bce_dice'):
    """
    Log epoch metrics to CSV file with timestamp.

    Args:
        csv_path: Path to the CSV file
        epoch: Current epoch number
        train_loss: Training loss
        test_loss: Testing loss
        iou: IoU metric (pixel-wise segmentation)
        box_iou: Box IoU metric (detection performance)
        precision: Precision metric
        recall: Recall metric
        f1: F1 score
        best_threshold: Best threshold found
        lr: Current learning rate
        loss_type: Loss function type (2026-02-03)
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            epoch,
            f'{train_loss:.6f}',
            f'{test_loss:.6f}',
            f'{iou:.4f}',
            f'{box_iou:.4f}',
            f'{precision:.4f}',
            f'{recall:.4f}',
            f'{f1:.4f}',
            f'{best_threshold:.2f}',
            f'{lr:.6f}',
            loss_type  # 2026-02-03: Record loss function type
        ])


def save_best_model(model, optimizer, epoch, iou, save_dir, use_multi_gpu=False, box_iou=None, save_reason='IoU'):
    """
    Save best model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        iou: Current IoU score (Seg IoU)
        save_dir: Directory to save the model (absolute path)
        use_multi_gpu: Whether using DataParallel
        box_iou: Mask-to-Box IoU score (optional)
        save_reason: Reason for saving ('IoU' or 'BoxIoU'), affects filename format
    """
    # Verify save directory exists
    abs_save_dir = os.path.abspath(save_dir)
    if not os.path.exists(abs_save_dir):
        print(f"⚠️  Save directory does not exist: {abs_save_dir}")
        os.makedirs(abs_save_dir, exist_ok=True)
        print(f"✅ Created directory: {abs_save_dir}")

    # [NEW] Delete previous best_model_epoch*.pth files to save disk space
    import glob
    old_best_models = glob.glob(os.path.join(abs_save_dir, 'best_model_epoch*.pth'))
    for old_model in old_best_models:
        try:
            os.remove(old_model)
            print(f"🗑️  Deleted old best model: {os.path.basename(old_model)}")
        except Exception as e:
            print(f"⚠️  Failed to delete {old_model}: {e}")

    # Extract model state (handle DataParallel)
    model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()

    # Generate filename based on save reason
    if save_reason == 'BoxIoU':
        # Saving for best Box IoU: BoxIoU first, then Seg IoU
        if box_iou is not None:
            checkpoint_path = os.path.join(abs_save_dir, f'best_model_epoch{epoch:04d}_BoxIoU{box_iou:.4f}_IoU{iou:.4f}.pth')
        else:
            checkpoint_path = os.path.join(abs_save_dir, f'best_model_epoch{epoch:04d}_BoxIoU_IoU{iou:.4f}.pth')
    else:
        # Saving for best Seg IoU: IoU first, then BoxIoU
        if box_iou is not None:
            checkpoint_path = os.path.join(abs_save_dir, f'best_model_epoch{epoch:04d}_IoU{iou:.4f}_BoxIoU{box_iou:.4f}.pth')
        else:
            checkpoint_path = os.path.join(abs_save_dir, f'best_model_epoch{epoch:04d}_IoU{iou:.4f}.pth')

    try:
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'iou': iou,
        }

        # Add box_iou if provided
        if box_iou is not None:
            checkpoint_data['box_iou'] = box_iou

        torch.save(checkpoint_data, checkpoint_path)

        # Verify file was actually saved
        if os.path.exists(checkpoint_path):
            file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"✅ Best model saved ({save_reason}): {checkpoint_path}")
            print(f"   📊 File size: {file_size_mb:.2f} MB")
        else:
            print(f"❌ ERROR: File not found after save: {checkpoint_path}")
    except Exception as e:
        print(f"❌ ERROR saving best model: {e}")
        raise

    return checkpoint_path


def save_last_epoch_model(model, optimizer, epoch, iou, save_dir, use_multi_gpu=False):
    """
    Save last epoch model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        iou: Current IoU score
        save_dir: Directory to save the model (absolute path)
        use_multi_gpu: Whether using DataParallel
    """
    # Verify save directory exists
    abs_save_dir = os.path.abspath(save_dir)
    if not os.path.exists(abs_save_dir):
        print(f"⚠️  Save directory does not exist: {abs_save_dir}")
        os.makedirs(abs_save_dir, exist_ok=True)
        print(f"✅ Created directory: {abs_save_dir}")

    # Extract model state (handle DataParallel)
    model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()

    # Save last epoch model
    checkpoint_path = os.path.join(abs_save_dir, f'last_epoch_model_epoch{epoch:04d}_IoU{iou:.4f}.pth')

    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'iou': iou,
        }, checkpoint_path)

        # Verify file was actually saved
        if os.path.exists(checkpoint_path):
            file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"✅ Last epoch model saved: {checkpoint_path}")
            print(f"   📊 File size: {file_size_mb:.2f} MB")
        else:
            print(f"❌ ERROR: File not found after save: {checkpoint_path}")
    except Exception as e:
        print(f"❌ ERROR saving last epoch model: {e}")
        raise

    return checkpoint_path


def update_training_summary(save_dir, best_epoch, best_iou, last_epoch, last_iou):
    """
    Update training summary at the end of training_log.txt.

    Args:
        save_dir: Directory containing train_log.txt
        best_epoch: Epoch with best IoU
        best_iou: Best IoU achieved
        last_epoch: Final epoch
        last_iou: Final IoU
    """
    config_file = os.path.join(save_dir, 'train_log.txt')

    with open(config_file, 'a') as f:
        now = datetime.now()
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Training Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"End Time        : {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs    : {last_epoch + 1}\n")
        f.write(f"Best IoU        : {best_iou:.4f} (Epoch {best_epoch})\n")
        f.write(f"Final IoU       : {last_iou:.4f} (Epoch {last_epoch})\n")
        f.write("=" * 80 + "\n")

    print(f"✅ Training summary updated in: {config_file}")
