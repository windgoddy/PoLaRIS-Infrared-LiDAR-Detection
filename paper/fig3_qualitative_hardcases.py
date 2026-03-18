#!/usr/bin/env python3
"""
Figure 3: Qualitative Results – Hard Case Mining & Visualization
================================================================
分两步运行：

  Step 1 (挖掘):  --mode mine
    遍历测试集，计算 Box Fill vs HALO 的伪标签 IoU，
    筛选 BoxFill_IoU < BOX_FAIL_THRESH 且 HALO_IoU > HALO_PASS_THRESH 的样本，
    按场景分类输出候选列表（供你手工选图）。

  Step 2 (绘图):  --mode plot  （需要先在 HARD_CASES 中填入手工选定的样本）
    展示 3行 × 4列 对比图：
      Col 1: 原始 IR 局部图
      Col 2: GT 掩码（青色轮廓叠加）
      Col 3: Box Fill 预测（失败案例）
      Col 4: HALO 预测（成功案例）

  Step 1+2: --mode both  （先挖掘并打印结果，再用 HARD_CASES 绘图）

网络预测图路径说明：
    需要先将 DNANet+BoxFill（B组）和 DNANet+HALO（D组）的测试预测图导出。
    推荐通过 test.py 的 --save_pred 选项保存，或使用 test_and_visulization.py。
    预测图格式：灰度 PNG，值域 [0, 255]，与 GT 同名同路径结构。

用法:
    # 第一步：挖掘硬难样本
    python paper/fig3_qualitative_hardcases.py --mode mine --dataset NUAA-SIRST

    # 第二步：填入 HARD_CASES 后绘图（见下方配置）
    python paper/fig3_qualitative_hardcases.py --mode plot

    # 全流程（挖掘 + 用当前 HARD_CASES 绘图）
    python paper/fig3_qualitative_hardcases.py --mode both
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'DejaVu Sans'
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ────────────────────────────────────────────────────────────────
# ★ 路径配置
# ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_CONFIGS = {
    'NUAA-SIRST': {
        'img_dir':   'dataset/NUAA-SIRST/images',
        'mask_dir':  'dataset/NUAA-SIRST/masks',
        'label_dir': 'dataset/NUAA-SIRST/labels_box',
        'split':     'dataset/NUAA-SIRST/50_50/test.txt',
    },
    'NUDT-SIRST': {
        'img_dir':   'dataset/NUDT-SIRST/images',
        'mask_dir':  'dataset/NUDT-SIRST/masks',
        'label_dir': 'dataset/NUDT-SIRST/labels_box',
        'split':     'dataset/NUDT-SIRST/50_50/test.txt',
    },
    'IRSTD-1k': {
        'img_dir':   'dataset/IRSTD-1k/IRSTD1k_Img',
        'mask_dir':  'dataset/IRSTD-1k/masks',
        'label_dir': 'dataset/IRSTD-1k/labels_box',
        'split':     'dataset/IRSTD-1k/50_50/test.txt',
    },
}

# ★ 网络预测结果目录
# 运行 python test.py 后，预测图保存在 result/ 的子目录下。
# 请根据你的实际训练输出目录填写（只填到包含图像文件的目录）。
# 例：'result/DNANet_bsnr_gauss_NUAA-SIRST_xxx/img/'
PRED_DIRS = {
    'boxfill': {
        'NUAA-SIRST': '',  # ← 填入 Box Fill（B组）NUAA 预测图目录
        'NUDT-SIRST': '',  # ← 填入 Box Fill（B组）NUDT 预测图目录
        'IRSTD-1k':   '',  # ← 填入 Box Fill（B组）IRSTD 预测图目录
    },
    'halo': {
        'NUAA-SIRST': '',  # ← 填入 HALO（D组）NUAA 预测图目录
        'NUDT-SIRST': '',  # ← 填入 HALO（D组）NUDT 预测图目录
        'IRSTD-1k':   '',  # ← 填入 HALO（D组）IRSTD 预测图目录
    },
}

# ★ 手工选定的硬难样本（运行 --mode mine 后填入）
# 格式：(dataset_name, sample_name, row_description)
# row_description 用于 ylabel（可用中英文）
HARD_CASES = [
    # ★ row_desc 只用英文，中文在服务器上会乱码
    # ('NUDT-SIRST', '000001', 'Tiny target\n(NUDT <9px^2)'),
    # ('NUAA-SIRST', 'Misc_200', 'Strong horizon\n(NUAA)'),
    # ('IRSTD-1k',   '000300', 'Cloud clutter\n(IRSTD)'),
]

# 挖掘阈值
BOX_FAIL_THRESH  = 0.25   # Box Fill IoU 低于此阈值算"失败"
HALO_PASS_THRESH = 0.60   # HALO IoU 高于此阈值算"成功"

CROP_SIZE = 64
DPI = 300

# ────────────────────────────────────────────────────────────────


def load_bsnr_utils(root):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'bsnr_mask_utils',
        os.path.join(root, 'model_Mamba', 'dataset', 'bsnr_mask_utils.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_bsnr_weight


def load_yolo_boxes(label_path, img_h, img_w):
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
            x2 = min(img_w, int((cx + w / 2) * img_w))
            y2 = min(img_h, int((cy + h / 2) * img_h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def compute_iou(pred_mask, gt_mask):
    """两个二值 mask 的 IoU"""
    inter = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
    union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
    return float(inter) / (float(union) + 1e-8)


def make_box_fill(boxes, H, W):
    """Box Fill 伪标签：直接填充整个框"""
    mask = np.zeros((H, W), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        mask[y1:y2, x1:x2] = 1.0
    return mask


def crop_patch(arr, cx, cy, size):
    H, W = arr.shape[:2]
    half = size // 2
    ox = max(0, min(cx - half, W - size))
    oy = max(0, min(cy - half, H - size))
    patch = arr[oy:oy+size, ox:ox+size]
    if patch.shape[0] < size or patch.shape[1] < size:
        if arr.ndim == 2:
            out = np.zeros((size, size), dtype=arr.dtype)
        else:
            out = np.zeros((size, size, arr.shape[2]), dtype=arr.dtype)
        out[:patch.shape[0], :patch.shape[1]] = patch
        patch = out
    return patch, ox, oy


def mine_hard_cases(args, compute_bsnr_weight):
    """遍历测试集，打印 Box Fill 失败 + HALO 成功的候选样本"""
    datasets = ['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        split_file = os.path.join(args.root, cfg['split'])
        with open(split_file, 'r') as f:
            names = [l.strip() for l in f if l.strip()]

        print(f"\n{'='*60}")
        print(f"Dataset: {ds}  (BOX_FAIL<{BOX_FAIL_THRESH:.2f}, HALO_PASS>{HALO_PASS_THRESH:.2f})")
        print(f"{'='*60}")
        print(f"{'Sample':<20} {'BoxFill IoU':>12} {'HALO IoU':>10} {'Gain':>8}")
        print('-' * 55)

        candidates = []
        for name in names:
            img_path   = os.path.join(args.root, cfg['img_dir'], name + '.png')
            mask_path  = os.path.join(args.root, cfg['mask_dir'], name + '.png')
            label_path = os.path.join(args.root, cfg['label_dir'], name + '.txt')

            if not (os.path.exists(img_path) and os.path.exists(label_path)):
                continue

            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            gt_mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
            if img_gray is None or gt_mask is None:
                continue

            H, W = img_gray.shape
            boxes = load_yolo_boxes(label_path, H, W)
            if not boxes:
                continue

            gt_bin = (gt_mask > 127).astype(np.uint8)
            if gt_bin.sum() == 0:
                continue

            img_float = img_gray.astype(np.float32) / 255.0

            # Box Fill
            bf_mask = make_box_fill(boxes, H, W)
            bf_iou  = compute_iou(bf_mask > 0.5, gt_bin)

            # HALO PAG
            halo_full = np.zeros((H, W), dtype=np.float32)
            for box in boxes:
                x1, y1, x2, y2 = box
                w_map = compute_bsnr_weight(
                    img_float, box, expand_ratio=1.5, temperature=3.0,
                    spatial_gaussian=True, gauss_sigma_ratio=1.5
                )
                halo_full[y1:y2, x1:x2] = np.maximum(halo_full[y1:y2, x1:x2], w_map)
            halo_iou = compute_iou(halo_full > 0.5, gt_bin)

            if bf_iou < BOX_FAIL_THRESH and halo_iou > HALO_PASS_THRESH:
                gain = halo_iou - bf_iou
                candidates.append((name, bf_iou, halo_iou, gain))

        candidates.sort(key=lambda x: -x[3])  # 按 gain 降序
        for name, bf, halo, gain in candidates[:20]:
            print(f"{name:<20} {bf:>12.4f} {halo:>10.4f} {gain:>+8.4f}")

        print(f"\nTotal candidates: {len(candidates)}")


def load_prediction(pred_dir, name):
    """从预测目录加载与 GT 同名的预测图（灰度）"""
    if not pred_dir:
        return None
    for ext in ['.png', '.jpg', '.bmp']:
        p = os.path.join(pred_dir, name + ext)
        if os.path.exists(p):
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return None


def plot_hard_cases(args, compute_bsnr_weight):
    if not HARD_CASES:
        print("[WARNING] HARD_CASES is empty. Please run --mode mine first,")
        print("          then fill in HARD_CASES at the top of this script.")
        return

    n_rows = len(HARD_CASES)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.patch.set_facecolor('black')

    col_titles = ['IR Image', 'GT Mask', 'Box Fill (Fail)', 'HALO (Ours)']

    for row_idx, (ds_name, sample_name, row_desc) in enumerate(HARD_CASES):
        cfg = DATASET_CONFIGS[ds_name]
        img_path   = os.path.join(args.root, cfg['img_dir'], sample_name + '.png')
        mask_path  = os.path.join(args.root, cfg['mask_dir'], sample_name + '.png')
        label_path = os.path.join(args.root, cfg['label_dir'], sample_name + '.txt')

        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gt_mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else np.zeros_like(img_gray)
        H, W = img_gray.shape
        boxes = load_yolo_boxes(label_path, H, W)

        # 计算目标中心
        if boxes:
            cx = int(np.mean([(b[0]+b[2])//2 for b in boxes]))
            cy = int(np.mean([(b[1]+b[3])//2 for b in boxes]))
        else:
            cx, cy = W//2, H//2

        img_patch, ox, oy = crop_patch(img_gray, cx, cy, CROP_SIZE)
        gt_patch,  _,  _  = crop_patch(gt_mask,  cx, cy, CROP_SIZE)

        # Box Fill 伪标签（用作对比；若有网络预测图则优先用网络预测）
        img_float = img_gray.astype(np.float32) / 255.0
        bf_full   = make_box_fill(boxes, H, W) * 255
        bf_pred   = load_prediction(PRED_DIRS['boxfill'].get(ds_name, ''), sample_name)
        if bf_pred is None:
            bf_pred = bf_full.astype(np.uint8)  # fallback: 伪标签本身
        bf_patch, _, _ = crop_patch(bf_pred, cx, cy, CROP_SIZE)

        # HALO 网络预测（若无则用伪标签 fallback）
        halo_pred = load_prediction(PRED_DIRS['halo'].get(ds_name, ''), sample_name)
        if halo_pred is None:
            halo_full = np.zeros((H, W), dtype=np.float32)
            for box in boxes:
                x1, y1, x2, y2 = box
                w_map = compute_bsnr_weight(img_float, box, expand_ratio=1.5,
                                            temperature=3.0, spatial_gaussian=True)
                halo_full[y1:y2, x1:x2] = np.maximum(halo_full[y1:y2, x1:x2], w_map)
            halo_pred = (halo_full * 255).astype(np.uint8)
        halo_patch, _, _ = crop_patch(halo_pred, cx, cy, CROP_SIZE)

        def style_ax(ax, title=None):
            ax.set_xticks([])
            ax.set_yticks([])
            if title and row_idx == 0:
                ax.set_title(title, color='white', fontsize=11, pad=4)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444444')

        # Col 0: IR
        ax = axes[row_idx][0]
        ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        ax.set_ylabel(row_desc, color='white', fontsize=8, labelpad=4)
        style_ax(ax, col_titles[0])

        # Col 1: GT（青色轮廓叠加）
        ax = axes[row_idx][1]
        ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        gt_overlay = np.zeros((*img_patch.shape, 4), dtype=np.float32)
        gt_overlay[gt_patch > 127] = [0.0, 1.0, 1.0, 0.9]  # cyan
        ax.imshow(gt_overlay, interpolation='nearest')
        style_ax(ax, col_titles[1])

        # Col 2: Box Fill（橙色热力图）
        ax = axes[row_idx][2]
        ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        ax.imshow(bf_patch.astype(np.float32) / 255.0, cmap='Oranges',
                  alpha=0.70, interpolation='nearest', vmin=0, vmax=1)
        style_ax(ax, col_titles[2])

        # Col 3: HALO（inferno 热力图）
        ax = axes[row_idx][3]
        ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        ax.imshow(halo_patch.astype(np.float32) / 255.0, cmap='inferno',
                  alpha=0.75, interpolation='nearest', vmin=0, vmax=1)
        style_ax(ax, col_titles[3])

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.12, hspace=0.08, wspace=0.04)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='black')
    print(f"Saved → {out_path}")
    plt.close()


def main(args):
    compute_bsnr_weight = load_bsnr_utils(args.root)

    if args.mode in ('mine', 'both'):
        mine_hard_cases(args, compute_bsnr_weight)

    if args.mode in ('plot', 'both'):
        plot_hard_cases(args, compute_bsnr_weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=ROOT)
    parser.add_argument('--mode', default='mine', choices=['mine', 'plot', 'both'],
                        help='mine=挖掘候选 / plot=绘图 / both=两步都跑')
    parser.add_argument('--dataset', default='all',
                        choices=['all', 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k'],
                        help='mine 时用哪个数据集（默认全部）')
    parser.add_argument('--out', default=os.path.join(ROOT, 'paper', 'figures', 'fig3_qualitative.pdf'))
    args = parser.parse_args()
    main(args)
