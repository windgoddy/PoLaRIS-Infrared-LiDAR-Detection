#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Ablation for HALO
==============================================
验证 expand_ratio 和 gauss_sigma_ratio 两个物理设计常数的鲁棒性。

实验设计：
  不重新训练，直接计算"PAG 伪标签与 GT 掩码的 IoU"（Pseudo-Label Quality IoU）
  随常数取值变化的曲线，证明性能对这两个常数不敏感（即它们是物理设计常数
  而非数据拟合超参数）。

扫描参数：
  1. expand_ratio sweep（固定 gauss_sigma_ratio=1.5）:
       [1.1, 1.3, 1.5, 2.0, 3.0]
       物理含义：Context Box 相对原始框的膨胀倍率，类似 CA-CFAR 的背景窗比

  2. gauss_sigma_ratio sweep（固定 expand_ratio=1.5）:
       [0.8, 1.0, 1.274, 1.5, 2.0, 2.5]
       其中 1.274 ≈ 1/√0.616 是 2D 高斯 FWHM 加权方差的物理校准值

对比方法：
  - B-SNR：仅受 expand_ratio 影响（无 spatial_gaussian）
  - PAG：同时受 expand_ratio 和 gauss_sigma_ratio 影响

用法：
    python hyperparameter_sensitivity_ablation.py                  # 全部三个数据集
    python hyperparameter_sensitivity_ablation.py --dataset NUAA-SIRST
    python hyperparameter_sensitivity_ablation.py --sweep expand    # 只跑 expand_ratio
    python hyperparameter_sensitivity_ablation.py --sweep sigma     # 只跑 gauss_sigma_ratio
    python hyperparameter_sensitivity_ablation.py --root /path/to/project
"""

import cv2
import numpy as np
import os
import sys
import argparse

import importlib.util

# 直接加载 bsnr_mask_utils.py，不经过 model_Mamba/__init__.py（避免 torch 依赖）
_root = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "bsnr_mask_utils",
    os.path.join(_root, "model_Mamba", "dataset", "bsnr_mask_utils.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_bsnr_weight = _mod.compute_bsnr_weight


# ════════════════════════════════════════════════════════════
# 数据集配置（与 box_perturbation_ablation.py 完全一致）
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

# 默认值（论文中使用的值）
DEFAULT_EXPAND_RATIO   = 1.5
DEFAULT_SIGMA_RATIO    = 1.5
DEFAULT_TEMPERATURE    = 3.0

# 扫描范围
EXPAND_RATIO_VALUES  = [1.0, 1.1, 1.3, 1.5, 2.0, 3.0]
# 1.274 ≈ 1/sqrt(0.616)，是 2D 高斯 FWHM 加权方差的精确物理校准值
SIGMA_RATIO_VALUES   = [0.8, 1.0, 1.274, 1.5, 2.0, 2.5]
TEMPERATURE_VALUES   = [1.5, 3.0, 5.0]


# ════════════════════════════════════════════════════════════
# 工具函数（与 box_perturbation_ablation.py 保持一致）
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


def generate_pseudo_mask(img_gray, boxes, H, W,
                         expand_ratio, gauss_sigma_ratio,
                         spatial_gaussian, temperature=DEFAULT_TEMPERATURE):
    """
    生成 B-SNR 或 PAG 伪标签。

    Args:
        spatial_gaussian: False → B-SNR（C组），True → PAG（D组）
        expand_ratio:     Context Box 膨胀倍率
        gauss_sigma_ratio: 高斯 σ 比例系数（仅 PAG 使用）
        temperature:      B-SNR sigmoid 温度系数 τ
    """
    gray_f = img_gray.astype(np.float32) / 255.0
    mask = np.zeros((H, W), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        weight = compute_bsnr_weight(
            gray_f, (x1, y1, x2, y2),
            expand_ratio=expand_ratio,
            temperature=temperature,
            spatial_gaussian=spatial_gaussian,
            gauss_sigma_ratio=gauss_sigma_ratio,
        )
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


def load_dataset_images(dataset_name, root):
    """加载数据集的图像、GT掩码和标注框，返回列表。"""
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

    samples = []
    skipped = 0
    for name in names:
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
        boxes = load_yolo_boxes(label_path, H, W)
        if not boxes:
            continue  # 无标注跳过

        samples.append({
            'name': name,
            'img': img,
            'gt_bin': gt_bin,
            'boxes': boxes,
            'H': H, 'W': W,
        })

    print(f"  Loaded {len(samples)} samples ({skipped} skipped)")
    return samples


# ════════════════════════════════════════════════════════════
# 扫描 1：expand_ratio
# ════════════════════════════════════════════════════════════
def sweep_expand_ratio(dataset_name, root='.'):
    """
    固定 gauss_sigma_ratio=1.5，扫描 expand_ratio。
    同时测 B-SNR 和 PAG，因为两者都受 expand_ratio 影响。
    """
    print(f"\n{'='*68}")
    print(f"  [expand_ratio sweep]  Dataset: {dataset_name}")
    print(f"  Fixed: gauss_sigma_ratio = {DEFAULT_SIGMA_RATIO}")
    print(f"{'='*68}")

    samples = load_dataset_images(dataset_name, root)
    if samples is None:
        return None

    # results[ratio] = {'bsnr': [iou...], 'pag': [iou...]}
    results = {r: {'bsnr': [], 'pag': []} for r in EXPAND_RATIO_VALUES}

    for s in samples:
        img, gt_bin, boxes = s['img'], s['gt_bin'], s['boxes']
        H, W = s['H'], s['W']
        for ratio in EXPAND_RATIO_VALUES:
            bsnr_mask = generate_pseudo_mask(img, boxes, H, W,
                                             expand_ratio=ratio,
                                             gauss_sigma_ratio=DEFAULT_SIGMA_RATIO,
                                             spatial_gaussian=False)
            pag_mask  = generate_pseudo_mask(img, boxes, H, W,
                                             expand_ratio=ratio,
                                             gauss_sigma_ratio=DEFAULT_SIGMA_RATIO,
                                             spatial_gaussian=True)
            results[ratio]['bsnr'].append(compute_pseudo_iou(bsnr_mask, gt_bin))
            results[ratio]['pag'].append(compute_pseudo_iou(pag_mask,  gt_bin))

    # 汇总
    summary = {}
    for ratio in EXPAND_RATIO_VALUES:
        bsnr_iou = float(np.mean(results[ratio]['bsnr'])) * 100
        pag_iou  = float(np.mean(results[ratio]['pag']))  * 100
        summary[ratio] = {'bsnr': bsnr_iou, 'pag': pag_iou}

    default_bsnr = summary[DEFAULT_EXPAND_RATIO]['bsnr']
    default_pag  = summary[DEFAULT_EXPAND_RATIO]['pag']

    print(f"\n  {'expand_ratio':<14} {'B-SNR IoU':>12}  {'vs default':>10}"
          f"  {'PAG IoU':>10}  {'vs default':>10}")
    print(f"  {'-'*60}")
    for ratio in EXPAND_RATIO_VALUES:
        tag = ' ← default' if ratio == DEFAULT_EXPAND_RATIO else ''
        bsnr_diff = summary[ratio]['bsnr'] - default_bsnr
        pag_diff  = summary[ratio]['pag']  - default_pag
        print(f"  {ratio:<14.3f} {summary[ratio]['bsnr']:>11.2f}%  "
              f"{bsnr_diff:>+9.2f}%  "
              f"{summary[ratio]['pag']:>9.2f}%  "
              f"{pag_diff:>+9.2f}%{tag}")
    print(f"{'='*68}")

    return summary


# ════════════════════════════════════════════════════════════
# 扫描 2：gauss_sigma_ratio
# ════════════════════════════════════════════════════════════
def sweep_sigma_ratio(dataset_name, root='.'):
    """
    固定 expand_ratio=1.5，扫描 gauss_sigma_ratio。
    仅测 PAG（B-SNR 不使用此参数）。
    1.274 ≈ 1/√0.616 是 2D 高斯 FWHM 加权方差的物理校准值。
    """
    print(f"\n{'='*68}")
    print(f"  [gauss_sigma_ratio sweep]  Dataset: {dataset_name}")
    print(f"  Fixed: expand_ratio = {DEFAULT_EXPAND_RATIO}")
    print(f"  Note: 1.274 ≈ 1/√0.616 is the theoretical PSF calibration value")
    print(f"{'='*68}")

    samples = load_dataset_images(dataset_name, root)
    if samples is None:
        return None

    results = {r: {'pag': []} for r in SIGMA_RATIO_VALUES}

    for s in samples:
        img, gt_bin, boxes = s['img'], s['gt_bin'], s['boxes']
        H, W = s['H'], s['W']
        for ratio in SIGMA_RATIO_VALUES:
            pag_mask = generate_pseudo_mask(img, boxes, H, W,
                                            expand_ratio=DEFAULT_EXPAND_RATIO,
                                            gauss_sigma_ratio=ratio,
                                            spatial_gaussian=True)
            results[ratio]['pag'].append(compute_pseudo_iou(pag_mask, gt_bin))

    summary = {}
    for ratio in SIGMA_RATIO_VALUES:
        pag_iou = float(np.mean(results[ratio]['pag'])) * 100
        summary[ratio] = {'pag': pag_iou}

    default_pag = summary[DEFAULT_SIGMA_RATIO]['pag']

    print(f"\n  {'sigma_ratio':<14} {'PAG IoU':>10}  {'vs default':>10}  Notes")
    print(f"  {'-'*58}")
    for ratio in SIGMA_RATIO_VALUES:
        diff = summary[ratio]['pag'] - default_pag
        if ratio == DEFAULT_SIGMA_RATIO:
            note = ' ← default (conservative)'
        elif abs(ratio - 1.274) < 0.001:
            note = ' ← physical calibration (1/√0.616)'
        elif ratio == 1.0:
            note = ' ← exact FWHM-weighted σ'
        else:
            note = ''
        print(f"  {ratio:<14.3f} {summary[ratio]['pag']:>9.2f}%  "
              f"{diff:>+9.2f}%{note}")
    print(f"{'='*68}")

    return summary


# ════════════════════════════════════════════════════════════
# 扫描 3：temperature (τ)
# ════════════════════════════════════════════════════════════
def sweep_temperature(dataset_name, root='.'):
    """
    固定 expand_ratio=1.5, gauss_sigma_ratio=1.5，扫描 B-SNR 温度系数 τ。
    同时测 B-SNR 和 PAG（两者均受 temperature 影响）。
    """
    print(f"\n{'='*68}")
    print(f"  [temperature sweep]  Dataset: {dataset_name}")
    print(f"  Fixed: expand_ratio={DEFAULT_EXPAND_RATIO}, sigma_ratio={DEFAULT_SIGMA_RATIO}")
    print(f"{'='*68}")

    samples = load_dataset_images(dataset_name, root)
    if samples is None:
        return None

    results = {t: {'bsnr': [], 'pag': []} for t in TEMPERATURE_VALUES}

    for s in samples:
        img, gt_bin, boxes = s['img'], s['gt_bin'], s['boxes']
        H, W = s['H'], s['W']
        for tau in TEMPERATURE_VALUES:
            bsnr_mask = generate_pseudo_mask(img, boxes, H, W,
                                             expand_ratio=DEFAULT_EXPAND_RATIO,
                                             gauss_sigma_ratio=DEFAULT_SIGMA_RATIO,
                                             spatial_gaussian=False,
                                             temperature=tau)
            pag_mask  = generate_pseudo_mask(img, boxes, H, W,
                                             expand_ratio=DEFAULT_EXPAND_RATIO,
                                             gauss_sigma_ratio=DEFAULT_SIGMA_RATIO,
                                             spatial_gaussian=True,
                                             temperature=tau)
            results[tau]['bsnr'].append(compute_pseudo_iou(bsnr_mask, gt_bin))
            results[tau]['pag'].append(compute_pseudo_iou(pag_mask,  gt_bin))

    summary = {}
    for tau in TEMPERATURE_VALUES:
        bsnr_iou = float(np.mean(results[tau]['bsnr'])) * 100
        pag_iou  = float(np.mean(results[tau]['pag']))  * 100
        summary[tau] = {'bsnr': bsnr_iou, 'pag': pag_iou}

    default_bsnr = summary[DEFAULT_TEMPERATURE]['bsnr']
    default_pag  = summary[DEFAULT_TEMPERATURE]['pag']

    print(f"\n  {'temperature (τ)':<16} {'B-SNR IoU':>12}  {'vs default':>10}"
          f"  {'PAG IoU':>10}  {'vs default':>10}")
    print(f"  {'-'*64}")
    for tau in TEMPERATURE_VALUES:
        tag = ' ← default' if tau == DEFAULT_TEMPERATURE else ''
        bsnr_diff = summary[tau]['bsnr'] - default_bsnr
        pag_diff  = summary[tau]['pag']  - default_pag
        print(f"  {tau:<16.1f} {summary[tau]['bsnr']:>11.2f}%  "
              f"{bsnr_diff:>+9.2f}%  "
              f"{summary[tau]['pag']:>9.2f}%  "
              f"{pag_diff:>+9.2f}%{tag}")
    print(f"{'='*68}")

    return summary


# ════════════════════════════════════════════════════════════
# 跨数据集汇总
# ════════════════════════════════════════════════════════════
def print_cross_dataset_summary(all_results_expand, all_results_sigma):
    """打印跨三个数据集的均值汇总表，方便直接写入论文。"""
    ds_names = list(all_results_expand.keys())
    if not ds_names:
        return

    print("\n\n" + "="*72)
    print("  【论文用汇总表 1】expand_ratio sensitivity — PAG IoU (三数据集均值)")
    print("="*72)
    print(f"  {'expand_ratio':<14}" + "".join(f"  {d[:12]:>13}" for d in ds_names)
          + f"  {'平均':>8}")
    print(f"  {'-'*60}")
    for ratio in EXPAND_RATIO_VALUES:
        vals = [all_results_expand[ds][ratio]['pag'] for ds in ds_names
                if ds in all_results_expand]
        tag = ' ←' if ratio == DEFAULT_EXPAND_RATIO else ''
        row = (f"  {ratio:<14.3f}"
               + "".join(f"  {v:>12.2f}%" for v in vals)
               + f"  {np.mean(vals):>7.2f}%{tag}")
        print(row)

    print("\n\n" + "="*72)
    print("  【论文用汇总表 2】gauss_sigma_ratio sensitivity — PAG IoU (三数据集均值)")
    print("="*72)
    print(f"  {'sigma_ratio':<14}" + "".join(f"  {d[:12]:>13}" for d in ds_names)
          + f"  {'平均':>8}  Notes")
    print(f"  {'-'*72}")
    for ratio in SIGMA_RATIO_VALUES:
        vals = [all_results_sigma[ds][ratio]['pag'] for ds in ds_names
                if ds in all_results_sigma]
        if ratio == DEFAULT_SIGMA_RATIO:
            note = '  ← default'
        elif abs(ratio - 1.274) < 0.001:
            note = '  ← physical calibration'
        else:
            note = ''
        row = (f"  {ratio:<14.3f}"
               + "".join(f"  {v:>12.2f}%" for v in vals)
               + f"  {np.mean(vals):>7.2f}%{note}")
        print(row)

    # 鲁棒性统计：最大波动幅度
    print("\n\n  【鲁棒性统计】PAG IoU 在合理范围内的最大波动（三数据集均值）")
    print(f"  {'-'*50}")
    expand_pag_avgs = []
    for ratio in EXPAND_RATIO_VALUES:
        vals = [all_results_expand[ds][ratio]['pag'] for ds in ds_names
                if ds in all_results_expand]
        expand_pag_avgs.append(np.mean(vals))
    sigma_pag_avgs = []
    for ratio in SIGMA_RATIO_VALUES:
        vals = [all_results_sigma[ds][ratio]['pag'] for ds in ds_names
                if ds in all_results_sigma]
        sigma_pag_avgs.append(np.mean(vals))

    print(f"  expand_ratio [{EXPAND_RATIO_VALUES[0]}–{EXPAND_RATIO_VALUES[-1]}]:"
          f"  最大波动 {max(expand_pag_avgs) - min(expand_pag_avgs):.2f}%")
    print(f"  sigma_ratio  [{SIGMA_RATIO_VALUES[0]}–{SIGMA_RATIO_VALUES[-1]}]:"
          f"  最大波动 {max(sigma_pag_avgs) - min(sigma_pag_avgs):.2f}%")

    # 1.274 vs 1.5 差距
    if all(ds in all_results_sigma for ds in ds_names):
        vals_1274 = [all_results_sigma[ds][1.274]['pag'] for ds in ds_names]
        vals_15   = [all_results_sigma[ds][1.5 ]['pag'] for ds in ds_names]
        diff = np.mean(vals_15) - np.mean(vals_1274)
        print(f"  sigma 1.274 vs 1.5 差距:  {diff:+.2f}% (|Δ| = {abs(diff):.2f}%)")
    print("="*72)


# ════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hyperparameter Sensitivity Ablation for HALO')
    parser.add_argument('--dataset', default='all',
                        choices=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k', 'all'],
                        help='Dataset to evaluate (default: all)')
    parser.add_argument('--sweep', default='all',
                        choices=['expand', 'sigma', 'temperature', 'all'],
                        help='Which sweep to run (default: all)')
    parser.add_argument('--root', default='.',
                        help='Project root directory (default: current dir)')
    args = parser.parse_args()

    datasets = (list(DATASET_CONFIGS.keys())
                if args.dataset == 'all' else [args.dataset])

    all_results_expand      = {}
    all_results_sigma       = {}
    all_results_temperature = {}

    for ds in datasets:
        print(f"\n{'#'*68}")
        print(f"#  Processing: {ds}")
        print(f"{'#'*68}")

        if args.sweep in ('expand', 'all'):
            r = sweep_expand_ratio(ds, root=args.root)
            if r:
                all_results_expand[ds] = r

        if args.sweep in ('sigma', 'all'):
            r = sweep_sigma_ratio(ds, root=args.root)
            if r:
                all_results_sigma[ds] = r

        if args.sweep in ('temperature', 'all'):
            r = sweep_temperature(ds, root=args.root)
            if r:
                all_results_temperature[ds] = r

    # 跨数据集汇总（仅三个数据集都跑完时）
    if (len(all_results_expand) > 1 or len(all_results_sigma) > 1):
        print_cross_dataset_summary(all_results_expand, all_results_sigma)

    if len(all_results_temperature) > 1:
        ds_names = list(all_results_temperature.keys())
        print("\n\n" + "="*72)
        print("  【论文用汇总表 3】temperature (τ) sensitivity — PAG IoU (三数据集均值)")
        print("="*72)
        print(f"  {'temperature':<14}" + "".join(f"  {d[:12]:>13}" for d in ds_names)
              + f"  {'平均':>8}")
        print(f"  {'-'*64}")
        for tau in TEMPERATURE_VALUES:
            vals = [all_results_temperature[ds][tau]['pag'] for ds in ds_names]
            tag = ' ←' if tau == DEFAULT_TEMPERATURE else ''
            row = (f"  {tau:<14.1f}"
                   + "".join(f"  {v:>12.2f}%" for v in vals)
                   + f"  {np.mean(vals):>7.2f}%{tag}")
            print(row)
        tau_avgs = [np.mean([all_results_temperature[ds][t]['pag'] for ds in ds_names])
                    for t in TEMPERATURE_VALUES]
        print(f"\n  temperature [{TEMPERATURE_VALUES[0]}–{TEMPERATURE_VALUES[-1]}]:"
              f"  最大波动 {max(tau_avgs) - min(tau_avgs):.2f}%")
        print("="*72)
