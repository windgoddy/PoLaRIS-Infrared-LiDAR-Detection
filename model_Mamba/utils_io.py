"""
I/O Utilities for PoLaRIS-Mamba Training
========================================

Handles experiment directory structure, logging, and model saving.
Consistent with Phase3 training output format.

Author: PoLaRIS Team
Date: 2026-02-02
"""

import os
import csv
from datetime import datetime


def create_experiment_dir(args):
    """
    Create experiment directory with timestamp.

    Format: model_Mamba/result/{experiment_name}_{dataset}_{model}_{timestamp}/

    Args:
        args: Parsed arguments from argparse

    Returns:
        save_dir: Path to the created directory
        dt_string: Timestamp string (YYYYMMDD_HHMMSS)
    """
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")

    # Build directory name
    if args.experiment_name:
        dir_name = f"{args.experiment_name}_{args.dataset}_{args.model}_{dt_string}"
    else:
        dir_name = f"{args.dataset}_{args.model}_{dt_string}"

    # Create directory: model_Mamba/result/dir_name/
    save_dir = os.path.join('model_Mamba', 'result', dir_name)
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
            'Precision',
            'Recall',
            'F1',
            'Best_Threshold',
            'LR'
        ])

    print(f"✅ Training log CSV initialized: {csv_path}")
    return csv_path


def log_epoch_metrics(csv_path, epoch, train_loss, test_loss, iou, precision, recall, f1, best_threshold, lr):
    """
    Log epoch metrics to CSV file with timestamp.

    Args:
        csv_path: Path to the CSV file
        epoch: Current epoch number
        train_loss: Training loss
        test_loss: Testing loss
        iou: IoU metric
        precision: Precision metric
        recall: Recall metric
        f1: F1 score
        best_threshold: Best threshold found
        lr: Current learning rate
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
            f'{precision:.4f}',
            f'{recall:.4f}',
            f'{f1:.4f}',
            f'{best_threshold:.2f}',
            f'{lr:.6f}'
        ])


def save_best_model(model, optimizer, epoch, iou, save_dir, use_multi_gpu=False):
    """
    Save best model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        iou: Current IoU score
        save_dir: Directory to save the model
        use_multi_gpu: Whether using DataParallel
    """
    # Extract model state (handle DataParallel)
    model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()

    # Save with IoU in filename
    checkpoint_path = os.path.join(save_dir, f'best_model_epoch{epoch:04d}_IoU{iou:.4f}.pth')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'iou': iou,
    }, checkpoint_path)

    # Also save as latest_best_model.pth for easy loading
    latest_path = os.path.join(save_dir, 'latest_best_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'iou': iou,
    }, latest_path)

    print(f"✅ Best model saved: {checkpoint_path}")
    return checkpoint_path


def save_last_epoch_model(model, optimizer, epoch, iou, save_dir, use_multi_gpu=False):
    """
    Save last epoch model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        iou: Current IoU score
        save_dir: Directory to save the model
        use_multi_gpu: Whether using DataParallel
    """
    # Extract model state (handle DataParallel)
    model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()

    # Save last epoch model
    checkpoint_path = os.path.join(save_dir, f'last_epoch_model_epoch{epoch:04d}_IoU{iou:.4f}.pth')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'iou': iou,
    }, checkpoint_path)

    print(f"✅ Last epoch model saved: {checkpoint_path}")
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


# Import torch here (at module level) for save functions
import torch
