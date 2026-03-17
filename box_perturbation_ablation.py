#!/usr/bin/env python3
"""
Box Perturbation Ablation Study for HALO
==========================================
测量 B-SNR 和 PAG 伪标签对 Bounding Box 扰动的鲁棒性。

实验设计：
  不重新训练，直接计算"伪标签与 GT 掩码的 IoU"（Pseudo-Label Quality IoU）
  随扰动幅度的变化曲线，反映方法对标注噪声的鲁棒性。

扰动方式：
  四个边角各自独立随机偏移 ±δ 像素（模拟真实标注抖动），
  每张图重复 N_TRIALS 次取均值，消除随机性影响。

对比方法：
  - Box Fill  : 直接填充整个框（无物理先验）
  - B-SNR     : 物理 SNR 后验掩码（C 组方法）
  - PAG       : 物理锚定高斯精炼（D 组方法，本文核心）

用法：
    python box_perturbation_ablation.py              # 全部三个数据集
    python box_perturbation_ablation.py --dataset NUAA-SIRST
    python box_perturbation_ablation.py --root /path/to/project
"""

import cv2
import numpy as np
import os
import sys
import argparse

# 将项目根目录加入 Python 路径，以便导入 model_Mamba 子包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_Mamba.dataset.bsnr_mask_utils import compute_bsnr_weight


# ════════════════════════════════════════════════════════════
# 数据集配置（与 grabcut_baseline.py 保持一致）
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

PERTURBATION_LEVELS = [0, 2, 4, 6, 8]  # 扰动幅度（像素）
N_TRIALS = 10                            # 每张图的随机试验次数（δ>0 时）


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
            _, cx, cy, w, h = (int(parts[0]), float(parts[1]),
                               float(parts[2]), float(parts[3]), float(parts[4]))
            x1 = max(0, int((cx - w / 2) * img_w))
            y1 = max(0, int((cy - h / 2) * img_h))
            x2 = min(img_w - 1, int((cx + w / 2) * img_w))
            y2 = min(img_h - 1, int((cy + h / 2) * img_h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def perturb_boxes(boxes, delta, img_h, img_w, rng):
    """
    对每个 box 的四个边分别施加独立随机偏移 ±delta 像素。
    模拟真实标注中的边界抖动误差（比整体平移更贴近实际）。
    """
    perturbed = []
    for (x1, y1, x2, y2) in boxes:
        x1p = int(np.clip(x1 + rng.randint(-delta, delta + 1), 0, img_w - 2))
        y1p = int(np.clip(y1 + rng.randint(-delta, delta + 1), 0, img_h - 2))
        x2p = int(np.clip(x2 + rng.randint(-delta, delta + 1), x1p + 1, img_w - 1))
        y2p = int(np.clip(y2 + rng.randint(-delta, delta + 1), y1p + 1, img_h - 1))
        perturbed.append((x1p, y1p, x2p, y2p))
    return perturbed


def generate_box_fill(boxes, H, W):
    """Box Fill 伪标签：直接将整个框填为前景"""
    mask = np.zeros((H, W), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        mask[y1:y2, x1:x2] = 1.0
    return mask


def generate_bsnr_mask(img_gray, boxes, H, W, spatial_gaussian=False):
    """
    生成 B-SNR（spatial_gaussian=False）或 PAG（spatial_gaussian=True）伪标签。
    调用 bsnr_mask_utils.compute_bsnr_weight，参数与训练时完全一致。
    """
    gray_f = img_gray.astype(np.float32) / 255.0
    mask = np.zeros((H, W), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        weight = compute_bsnr_weight(
            gray_f, (x1, y1, x2, y2),
            expand_ratio=1.5,
            temperature=3.0,
            spatial_gaussian=spatial_gaussian,
            gauss_sigma_ratio=1.5,
        )
        # weight.shape == (box_h, box_w)，写入全局掩码
        bh = min(y2, H) - y1
        bw = min(x2, W) - x1
        if bh > 0 and bw > 0:
            mask[y1:y1 + bh, x1:x1 + bw] = np.maximum(
                mask[y1:y1 + bh, x1:x1 + bw],
                weight[:bh, :bw]
            )
    return mask


def compute_pseudo_iou(pseudo_mask, gt_bin, threshold=0.5):
    """计算伪标签（阈值化后）与 GT 掩码的 IoU"""
    pred_bin = (pseudo_mask >= threshold).astype(np.uint8)
    intersection = int((pred_bin & gt_bin).sum())
    union = int((pred_bin | gt_bin).sum())
    if union == 0:
        return 1.0
    return intersection / union


# ════════════════════════════════════════════════════════════
# 主评估函数
# ════════════════════════════════════════════════════════════
def evaluate_perturbation(dataset_name, root='.'):
    cfg = DATASET_CONFIGS[dataset_name]
    img_dir    = os.path.join(root, cfg['img_dir'])
    mask_dir   = os.path.join(root, cfg['mask_dir'])
    label_dir  = os.path.join(root, cfg['label_dir'])
    split_file = os.path.join(root, cfg['split_file'])

    if not os.path.exists(split_file):
        print(f"[SKIP] split file not found: {split_file}")
        return None

    with open(split_file, 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    # 结果容器：{delta: {method: [per-image IoU]}}
    results = {d: {'box_fill': [], 'bsnr': [], 'pag': []}
               for d in PERTURBATION_LEVELS}
    rng = np.random.RandomState(42)
    skipped = 0

    for name in names:
        # 容错：尝试多种图像后缀
        img_path = None
        for suffix in ['.png', '.bmp', '.jpg']:
            p = os.path.join(img_dir, name + suffix)
            if os.path.exists(p):
                img_path = p
                mask_path = os.path.join(mask_dir, name + suffix)
                break
        if img_path is None or not os.path.exists(mask_path):
            skipped += 1
            continue

        img = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        gt  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            skipped += 1
            continue

        H, W = img.shape
        if gt.shape != img.shape:
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt > 0).astype(np.uint8)

        label_path = os.path.join(label_dir, name + '.txt')
        orig_boxes = load_yolo_boxes(label_path, H, W)
        if not orig_boxes:
            continue  # 无标注的图跳过（不影响扰动实验）

        for delta in PERTURBATION_LEVELS:
            n_trials = 1 if delta == 0 else N_TRIALS
            trial_bf, trial_bsnr, trial_pag = [], [], []

            for _ in range(n_trials):
                boxes = orig_boxes if delta == 0 else perturb_boxes(
                    orig_boxes, delta, H, W, rng)

                bf_mask   = generate_box_fill(boxes, H, W)
                bsnr_mask = generate_bsnr_mask(img, boxes, H, W, spatial_gaussian=False)
                pag_mask  = generate_bsnr_mask(img, boxes, H, W, spatial_gaussian=True)

                trial_bf.append(compute_pseudo_iou(bf_mask,   gt_bin))
                trial_bsnr.append(compute_pseudo_iou(bsnr_mask, gt_bin))
                trial_pag.append(compute_pseudo_iou(pag_mask,  gt_bin))

            results[delta]['box_fill'].append(float(np.mean(trial_bf)))
            results[delta]['bsnr'].append(float(np.mean(trial_bsnr)))
            results[delta]['pag'].append(float(np.mean(trial_pag)))

    # ——— 汇总并打印 ———
    n_valid = len(results[0]['box_fill'])
    print(f"\n{'='*64}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Samples : {n_valid} (skipped {skipped}), Trials/img : {N_TRIALS}")
    print(f"{'='*64}")
    print(f"  {'δ(px)':<6} {'Box Fill':>10} {'B-SNR':>10} {'PAG':>10}"
          f"  {'PAG vs BF':>10}  {'PAG vs BSNR':>12}")
    print(f"  {'-'*60}")

    summary = {}
    for delta in PERTURBATION_LEVELS:
        bf_   = float(np.mean(results[delta]['box_fill'])) * 100
        bsnr_ = float(np.mean(results[delta]['bsnr']))    * 100
        pag_  = float(np.mean(results[delta]['pag']))     * 100
        print(f"  {delta:<6} {bf_:>9.2f}%  {bsnr_:>9.2f}%  {pag_:>9.2f}%"
              f"  {pag_-bf_:>+9.2f}%  {pag_-bsnr_:>+11.2f}%")
        summary[delta] = {'box_fill': bf_, 'bsnr': bsnr_, 'pag': pag_}

    # 相对保留率（以 δ=0 为 100%）
    base = {m: summary[0][m] for m in ('box_fill', 'bsnr', 'pag')}
    print(f"\n  相对保留率（δ=0 基准 = 100%）：")
    print(f"  {'δ(px)':<6} {'Box Fill':>10} {'B-SNR':>10} {'PAG':>10}")
    print(f"  {'-'*38}")
    for delta in PERTURBATION_LEVELS:
        def retain(m):
            return summary[delta][m] / base[m] * 100 if base[m] > 0 else 0.0
        print(f"  {delta:<6} {retain('box_fill'):>9.1f}%  "
              f"{retain('bsnr'):>9.1f}%  {retain('pag'):>9.1f}%")

    print(f"{'='*64}")
    return summary


# ════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Box Perturbation Ablation Study for HALO')
    parser.add_argument('--dataset', default='all',
                        choices=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k', 'all'],
                        help='Dataset to evaluate (default: all)')
    parser.add_argument('--root', default='.',
                        help='Project root directory (default: current dir)')
    args = parser.parse_args()

    datasets = (list(DATASET_CONFIGS.keys())
                if args.dataset == 'all' else [args.dataset])

    all_results = {}
    for ds in datasets:
        r = evaluate_perturbation(ds, root=args.root)
        if r:
            all_results[ds] = r

    # 跨数据集 PAG 鲁棒性汇总
    if len(all_results) > 1:
        print("\n\n====== PAG 跨数据集鲁棒性汇总（Pseudo-Label IoU %）======")
        ds_names = list(all_results.keys())
        header = f"{'δ(px)':<6}" + "".join(f"  {d[:12]:>13}" for d in ds_names) + f"  {'平均':>8}"
        print(header)
        print('-' * len(header))
        for delta in PERTURBATION_LEVELS:
            vals = [all_results[ds][delta]['pag'] for ds in ds_names]
            row = f"{delta:<6}" + "".join(f"  {v:>12.2f}%" for v in vals)
            row += f"  {np.mean(vals):>7.2f}%"
            print(row)
        print('=' * len(header))

        print("\n====== 三方法对比（δ=8px 时的绝对值，越高越鲁棒）======")
        print(f"{'Dataset':<15} {'Box Fill':>10} {'B-SNR':>10} {'PAG':>10}")
        print('-' * 48)
        for ds in ds_names:
            r = all_results[ds][8]
            print(f"{ds:<15} {r['box_fill']:>9.2f}%  {r['bsnr']:>9.2f}%  {r['pag']:>9.2f}%")
        print('=' * 48)
