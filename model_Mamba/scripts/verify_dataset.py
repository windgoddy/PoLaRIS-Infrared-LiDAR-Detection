#!/usr/bin/env python3
"""
Dataset Verification Script for PoLaRIS-Mamba
==============================================

Verifies that the dataset is correctly structured and can be loaded
without errors (no NaN/Inf values).

Usage:
    python model_Mamba/scripts/verify_dataset.py --dataset Pohang-Canal-3k --split_method 50_50_3k
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.utils_lidar import PoLaRISTrainLoader, get_img_ids_from_dir
from model.load_param_data import load_dataset
from model_Mamba.dataset.gaussian_utils import load_yolo_labels, generate_gaussian_target


def verify_dataset(args):
    """Verify dataset structure and data quality."""
    
    print("=" * 60)
    print("PoLaRIS Dataset Verification")
    print("=" * 60)
    
    # 1. Check directory structure
    dataset_dir = os.path.join(args.root, args.dataset)
    print(f"\n1️⃣  Checking directory structure...")
    print(f"   Dataset dir: {dataset_dir}")
    
    required_dirs = ['images', 'masks', 'lidar_roi', 'labels']
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_dir, dir_name)
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if not f.startswith('.')])
            print(f"   ✓ {dir_name}/  ({file_count} files)")
        else:
            print(f"   ✗ {dir_name}/  (MISSING)")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n   ⚠️  Missing directories: {missing_dirs}")
        print(f"   Please ensure dataset is properly set up.")
        return False
    
    # 2. Check split files
    print(f"\n2️⃣  Checking split files...")
    split_dir = os.path.join(args.root, args.dataset, args.split_method)
    train_txt = os.path.join(split_dir, 'train.txt')
    test_txt = os.path.join(split_dir, 'test.txt')
    
    if not os.path.exists(train_txt):
        print(f"   ✗ train.txt not found: {train_txt}")
        return False
    if not os.path.exists(test_txt):
        print(f"   ✗ test.txt not found: {test_txt}")
        return False
    
    train_img_ids, val_img_ids, _ = load_dataset(args.root, args.dataset, args.split_method)
    print(f"   ✓ train.txt ({len(train_img_ids)} samples)")
    print(f"   ✓ test.txt ({len(val_img_ids)} samples)")
    
    # 3. Verify sample files exist
    print(f"\n3️⃣  Verifying sample files...")
    sample_ids = train_img_ids[:min(10, len(train_img_ids))]
    
    for img_id in sample_ids:
        img_path = os.path.join(dataset_dir, 'images', f'{img_id}.png')
        mask_path = os.path.join(dataset_dir, 'masks', f'{img_id}.png')
        label_path = os.path.join(dataset_dir, 'labels', f'{img_id}.txt')
        
        if not os.path.exists(img_path):
            print(f"   ✗ Image not found: {img_path}")
            return False
        if not os.path.exists(mask_path):
            print(f"   ✗ Mask not found: {mask_path}")
            return False
        if not os.path.exists(label_path):
            print(f"   ⚠️  Label not found: {label_path} (will use empty)")
    
    print(f"   ✓ All sample files exist")
    
    # 4. Test dataloader
    print(f"\n4️⃣  Testing dataloader...")
    try:
        train_loader = PoLaRISTrainLoader(
            dataset_dir=dataset_dir,
            img_id=sample_ids,
            base_size=512,
            crop_size=480,
            normalize_16bit=True,
            in_channels=1,
            image_folder='images',
        )
        print(f"   ✓ PoLaRISTrainLoader initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize dataloader: {e}")
        return False
    
    # 5. Check data quality
    print(f"\n5️⃣  Checking data quality (NaN/Inf/all-zero)...")
    issues = []
    
    for i in range(min(10, len(train_loader))):
        try:
            sample = train_loader[i]
            img = sample['image']
            mask = sample['mask']
            img_id = sample['img_id']
            
            # Check for NaN/Inf
            if torch.isnan(img).any() or torch.isinf(img).any():
                issues.append(f"   ✗ [{img_id}] Image has NaN/Inf")
            if torch.isnan(mask).any() or torch.isinf(mask).any():
                issues.append(f"   ✗ [{img_id}] Mask has NaN/Inf")
            
            # Check for all-zero masks
            if mask.max() == 0:
                issues.append(f"   ⚠️  [{img_id}] Mask is all zeros (no targets)")
            
            # Check Gaussian heatmap generation
            label_path = os.path.join(dataset_dir, 'labels', f'{img_id}.txt')
            if os.path.exists(label_path):
                labels = load_yolo_labels(label_path)
                H, W = img.shape[1], img.shape[2]
                heatmap = generate_gaussian_target(labels, (H, W), downscale=1, min_overlap=0.7)
                
                if np.isnan(heatmap).any() or np.isinf(heatmap).any():
                    issues.append(f"   ✗ [{img_id}] Heatmap has NaN/Inf")
                if heatmap.max() > 1.0:
                    issues.append(f"   ⚠️  [{img_id}] Heatmap max > 1.0: {heatmap.max()}")
        
        except Exception as e:
            issues.append(f"   ✗ [{i}] Failed to load sample: {e}")
    
    if issues:
        print("\n   Issues found:")
        for issue in issues[:20]:  # Show first 20 issues
            print(issue)
        if len(issues) > 20:
            print(f"   ... and {len(issues) - 20} more issues")
        return False
    else:
        print(f"   ✓ All samples pass quality checks")
    
    # 6. Summary
    print(f"\n{'=' * 60}")
    print(f"✅ Dataset verification PASSED")
    print(f"{'=' * 60}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split_method}")
    print(f"Train samples: {len(train_img_ids)}")
    print(f"Val samples: {len(val_img_ids)}")
    print(f"\nYou can now run training with:")
    print(f"  bash model_Mamba/scripts/auto_train_gpu.sh")
    print(f"{'=' * 60}\n")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify PoLaRIS dataset')
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k',
                        help='Dataset name')
    parser.add_argument('--root', type=str, default='dataset/',
                        help='Dataset root directory')
    parser.add_argument('--split_method', type=str, default='50_50_2k_new',
                        help='Train/test split method')
    
    args = parser.parse_args()
    
    success = verify_dataset(args)
    sys.exit(0 if success else 1)
