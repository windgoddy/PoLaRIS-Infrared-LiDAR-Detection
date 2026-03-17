#!/usr/bin/env python3
"""
GrabCut Baseline for Box-Supervised IRSTD
==========================================
用途：作为"通用框监督"的负面对比，展示缺乏物理先验时框监督的失效
输出：mIoU（与训练实验一致，扫描阈值取最佳）

用法：
    python grabcut_baseline.py                        # 跑全部三个数据集
    python grabcut_baseline.py --dataset NUAA-SIRST   # 只跑一个
    python grabcut_baseline.py --root dataset         # 指定数据集根目录
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


# ════════════════════════════════════════════════════════════
# 数据集配置
# ════════════════════════════════════════════════════════════
DATASET_CONFIGS = {
    'NUAA-SIRST': {
        'img_dir':    'dataset/NUAA-SIRST/images',
        'mask_dir':   'dataset/NUAA-SIRST/masks',
        'label_dir':  'dataset/NUAA-SIRST/labels_box',
        'split_file': 'dataset/NUAA-SIRST/50_50/test.txt',
    },
    'NUDT-SIRST': {
        'img_dir':    'dataset/NUDT-SIRST/images',
        'mask_dir':   'dataset/NUDT-SIRST/masks',
        'label_dir':  'dataset/NUDT-SIRST/labels_box',
        'split_file': 'dataset/NUDT-SIRST/50_50/test.txt',
    },
    'IRSTD-1k': {
        'img_dir':    'dataset/IRSTD-1k/IRSTD1k_Img',
        'mask_dir':   'dataset/IRSTD-1k/masks',
        'label_dir':  'dataset/IRSTD-1k/labels_box',
        'split_file': 'dataset/IRSTD-1k/50_50/test.txt',
    },
}


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════
def load_yolo_boxes(label_path, img_h, img_w):
    """读取 YOLO 格式标签，返回像素坐标列表 [(x1,y1,x2,y2), ...]"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = max(0, int((cx - w / 2) * img_w))
            y1 = max(0, int((cy - h / 2) * img_h))
            x2 = min(img_w - 1, int((cx + w / 2) * img_w))
            y2 = min(img_h - 1, int((cy + h / 2) * img_h))
            if x2 > x1 + 1 and y2 > y1 + 1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def grabcut_predict(img_gray, boxes):
    """
    对每个 box 运行 GrabCut，合并结果返回二值掩码。
    img_gray: (H, W) uint8 单通道灰度图
    boxes: [(x1,y1,x2,y2), ...]
    返回: (H, W) uint8, 值为 0 或 1
    """
    H, W = img_gray.shape
    # GrabCut 需要 3 通道 BGR
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    pred = np.zeros((H, W), dtype=np.uint8)

    for (x1, y1, x2, y2) in boxes:
        bw, bh = x2 - x1, y2 - y1

        # 极小 box（<5px）：直接填充，GrabCut 无法处理
        if bw < 5 or bh < 5:
            pred[y1:y2, x1:x2] = 1
            continue

        rect = (x1, y1, bw, bh)
        gc_mask = np.zeros((H, W), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(img_bgr, gc_mask, rect, bgd_model, fgd_model,
                        5, cv2.GC_INIT_WITH_RECT)
            fg = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                1, 0
            ).astype(np.uint8)
            # 只保留 box 内部的预测（box 外不应有预测）
            roi = np.zeros_like(fg)
            roi[y1:y2, x1:x2] = fg[y1:y2, x1:x2]
            pred = np.maximum(pred, roi)
        except cv2.error as e:
            # GrabCut 失败时 fallback：填充整个 box
            pred[y1:y2, x1:x2] = 1

    return pred


def compute_iou(pred_bin, gt_bin):
    """计算单张图片的 IoU"""
    intersection = int((pred_bin & gt_bin).sum())
    union = int((pred_bin | gt_bin).sum())
    if union == 0:
        return 1.0  # 都是空，IoU=1
    return intersection / union


def compute_pd_fa(pred_bin, gt_bin, img_area):
    """计算 Pd (检测率) 和 Fa (虚警率)"""
    # 简单版本：基于连通域
    # gt 中有目标 → 检查预测是否覆盖
    gt_has_target = gt_bin.sum() > 0
    pred_has_target = pred_bin.sum() > 0

    if gt_has_target:
        overlap = (pred_bin & gt_bin).sum()
        pd = 1.0 if overlap > 0 else 0.0
    else:
        pd = None  # 无目标图不计 Pd

    fa = pred_bin.sum() / img_area  # FA = 预测前景 / 总像素

    return pd, fa


# ════════════════════════════════════════════════════════════
# 主评估函数
# ════════════════════════════════════════════════════════════
def evaluate_dataset(dataset_name, root='.'):
    cfg = DATASET_CONFIGS[dataset_name]

    # 路径拼上 root
    img_dir    = os.path.join(root, cfg['img_dir'])
    mask_dir   = os.path.join(root, cfg['mask_dir'])
    label_dir  = os.path.join(root, cfg['label_dir'])
    split_file = os.path.join(root, cfg['split_file'])

    if not os.path.exists(split_file):
        print(f"[SKIP] split file not found: {split_file}")
        return None

    with open(split_file, 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    # 阈值扫描（与 train.py 的 mIoU 评估保持一致）
    thresholds = np.arange(0.1, 0.95, 0.05)
    iou_records = {t: [] for t in thresholds}
    pd_list, fa_list = [], []
    skipped = 0

    for name in names:
        img_path   = os.path.join(img_dir,   name + '.png')
        mask_path  = os.path.join(mask_dir,  name + '.png')
        label_path = os.path.join(label_dir, name + '.txt')

        # 容错：尝试其他后缀
        for suffix in ['.png', '.bmp', '.jpg']:
            if os.path.exists(os.path.join(img_dir, name + suffix)):
                img_path  = os.path.join(img_dir,  name + suffix)
                mask_path = os.path.join(mask_dir, name + suffix)
                break

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            skipped += 1
            continue

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        gt   = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or gt is None:
            skipped += 1
            continue

        H, W = img.shape
        # 如果 mask 尺寸与图像不一致，resize mask 到图像尺寸
        if gt.shape != img.shape:
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt > 0).astype(np.uint8)

        boxes = load_yolo_boxes(label_path, H, W)

        if not boxes:
            # 无标注 → 全背景预测
            pred = np.zeros((H, W), dtype=np.uint8)
        else:
            pred = grabcut_predict(img, boxes)

        # 记录各阈值 IoU（GrabCut 输出是 0/1，阈值 >0 均等价于 0.5）
        for t in thresholds:
            pred_bin = (pred >= 1).astype(np.uint8)  # binary output
            iou = compute_iou(pred_bin, gt_bin)
            iou_records[t].append(iou)

        # Pd / Fa（threshold=0.5 等价）
        pred_bin = (pred >= 1).astype(np.uint8)
        pd, fa = compute_pd_fa(pred_bin, gt_bin, H * W)
        if pd is not None:
            pd_list.append(pd)
        fa_list.append(fa)

    # 取最佳阈值下的 mIoU（与 train.py 一致）
    mean_ious = {t: float(np.mean(v)) for t, v in iou_records.items() if v}
    if not mean_ious:
        print(f"[{dataset_name}] No valid samples!")
        return None

    best_thresh = max(mean_ious, key=mean_ious.get)
    best_miou   = mean_ious[best_thresh]
    mean_pd = float(np.mean(pd_list)) if pd_list else 0.0
    mean_fa = float(np.mean(fa_list)) if fa_list else 0.0

    print(f"\n{'='*50}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Samples : {len(names) - skipped} (skipped {skipped})")
    print(f"  mIoU    : {best_miou * 100:.2f}%  (best thresh={best_thresh:.2f})")
    print(f"  Pd      : {mean_pd * 100:.2f}%")
    print(f"  Fa      : {mean_fa:.4e}")
    print(f"{'='*50}")

    return {'mIoU': best_miou, 'Pd': mean_pd, 'Fa': mean_fa}


# ════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GrabCut Baseline for IRSTD')
    parser.add_argument('--dataset', default='all',
                        choices=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k', 'all'],
                        help='Dataset to evaluate')
    parser.add_argument('--root', default='.',
                        help='Project root directory (default: current dir)')
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == 'all' else [args.dataset]

    all_results = {}
    for ds in datasets:
        result = evaluate_dataset(ds, root=args.root)
        if result:
            all_results[ds] = result

    if len(all_results) > 1:
        print("\n\n====== FINAL SUMMARY (GrabCut Baseline) ======")
        print(f"{'Dataset':<20} {'mIoU':>8} {'Pd':>8} {'Fa':>12}")
        print('-' * 52)
        for ds, r in all_results.items():
            print(f"{ds:<20} {r['mIoU']*100:>7.2f}%  {r['Pd']*100:>7.2f}%  {r['Fa']:>12.4e}")
        print('=' * 52)
