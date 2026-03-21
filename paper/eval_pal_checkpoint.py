#!/usr/bin/env python3
"""
用我们自己的评估协议（阈值扫描 0.05~0.95，取 best mIoU）
评估 PAL 训练出的最佳 checkpoint，使结果与 HALO 实验数字口径一致。

用法（服务器项目根目录下）：
  python paper/eval_pal_checkpoint.py \
      --ckpt  PAL/result/DNA__NUDT_SIRST_HALO_point__masks_centroid__*/best_mIoU_checkpoint_*.pth.tar \
      --dataset NUDT-SIRST \
      --root   dataset/ \
      --split  50_50/test.txt \
      --model_dir PAL

输出：
  mIoU at each threshold (0.05 step)
  Best mIoU = XX.XX%  @  threshold = X.XX
  （与 HALO P1_D 实验口径完全一致）
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

# 引入 PAL 的 DNA 模型
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'PAL'))
from model.DNA.DNA_no_sigmoid import DNANet_No_Sigmoid


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class IRTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_ids):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.img_ids  = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + '.png'), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_id}.png")
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=2)
        elif img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=2)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)

        mask = cv2.imread(os.path.join(self.mask_dir, img_id + '.png'), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {img_id}.png")
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return img, mask, img_id


# ──────────────────────────────────────────────
# mIoU 计算（与 model/metric.py 一致）
# ──────────────────────────────────────────────
def compute_miou_at_threshold(preds_list, masks_list, threshold):
    """
    preds_list: list of (H, W) numpy sigmoid output
    masks_list: list of (H, W) numpy binary GT
    """
    total_inter = 0
    total_union = 0
    for pred, gt in zip(preds_list, masks_list):
        pred_bin = (pred > threshold).astype(np.float32)
        inter = (pred_bin * gt).sum()
        union = pred_bin.sum() + gt.sum() - inter
        if union > 0:
            total_inter += inter
            total_union += union
    if total_union == 0:
        return 0.0
    return total_inter / total_union


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',     type=str, required=True,  help='PAL checkpoint 路径（*.pth.tar）')
    parser.add_argument('--dataset',  type=str, default='NUDT-SIRST')
    parser.add_argument('--root',     type=str, default='dataset/')
    parser.add_argument('--split',    type=str, default='50_50/test.txt', help='测试集 split 文件')
    parser.add_argument('--img_folder',  type=str, default='images')
    parser.add_argument('--mask_folder', type=str, default='masks')
    parser.add_argument('--device',   type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── 数据 ──
    dataset_dir = os.path.join(args.root, args.dataset)
    split_path  = os.path.join(dataset_dir, args.split)
    with open(split_path) as f:
        img_ids = [l.strip() for l in f if l.strip()]
    print(f"Test set: {len(img_ids)} images from {split_path}")

    img_dir  = os.path.join(dataset_dir, args.img_folder)
    mask_dir = os.path.join(dataset_dir, args.mask_folder)
    ds = IRTestDataset(img_dir, mask_dir, img_ids)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    # ── 模型 ──
    model = DNANet_No_Sigmoid(mode='test').to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    # PAL 保存格式兼容
    state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    # 去掉可能的 'module.' 前缀
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # ── 推理 ──
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            # DNANet 返回列表，取最后一个输出
            if isinstance(outputs, (list, tuple)):
                out = outputs[-1]
            else:
                out = outputs
            prob = torch.sigmoid(out).squeeze().cpu().numpy()  # (H, W)
            gt   = masks.squeeze().numpy()                     # (H, W)
            all_preds.append(prob)
            all_masks.append(gt)

    # ── 阈值扫描（与 HALO 口径一致）──
    thresholds = np.arange(0.05, 1.0, 0.05)
    best_miou  = 0.0
    best_thresh = 0.0
    results = []
    for t in thresholds:
        miou = compute_miou_at_threshold(all_preds, all_masks, t)
        results.append((t, miou))
        if miou > best_miou:
            best_miou   = miou
            best_thresh = t

    print("\n── mIoU @ each threshold ──")
    for t, m in results:
        marker = " ← best" if abs(t - best_thresh) < 1e-6 else ""
        print(f"  thresh={t:.2f}  mIoU={m*100:.2f}%{marker}")

    print(f"\n{'='*40}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Best mIoU  = {best_miou*100:.2f}%  @  threshold = {best_thresh:.2f}")
    print(f"{'='*40}")
    print("\n（PAL 论文原始报告：NUDT=73.29%，我们的 HALO P1_D=62.97%）")


if __name__ == '__main__':
    main()
