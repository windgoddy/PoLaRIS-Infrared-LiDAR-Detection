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
        'NUAA-SIRST': 'paper/figures/preds/boxfill_nuaa',
        'NUDT-SIRST': 'paper/figures/preds/boxfill_nudt',
        'IRSTD-1k':   'paper/figures/preds/boxfill_irstd',
    },
    'bsnr': {
        'NUAA-SIRST': 'paper/figures/preds/bsnr_nuaa',
        'NUDT-SIRST': 'paper/figures/preds/bsnr_nudt',
        'IRSTD-1k':   'paper/figures/preds/bsnr_irstd',
    },
    'halo': {
        'NUAA-SIRST': 'paper/figures/preds/halo_nuaa',
        'NUDT-SIRST': 'paper/figures/preds/halo_nudt',
        'IRSTD-1k':   'paper/figures/preds/halo_irstd',
    },
    'gt_sup': {
        'NUAA-SIRST': 'paper/figures/preds/gt_nuaa',
        'NUDT-SIRST': 'paper/figures/preds/gt_nudt',
        'IRSTD-1k':   'paper/figures/preds/gt_irstd',
    },
}

# ★ 手工选定的硬难样本（运行 --mode mine 后填入）
# 格式：(dataset_name, sample_name, row_description)
# row_description 用于 ylabel（可用中英文）
HARD_CASES = [
    # NUAA：单目标 + 多目标
    ('NUAA-SIRST', 'Misc_399', 'NUAA\nSingle'),
    ('NUAA-SIRST', 'Misc_187', 'NUAA\nMulti'),
    # NUDT：单目标 + 多目标
    ('NUDT-SIRST', '000314',   'NUDT\nSingle'),
    ('NUDT-SIRST', '000135',   'NUDT\nMulti'),
    # IRSTD：单目标 + 多目标
    ('IRSTD-1k',   'XDU369',   'IRSTD\nSingle'),
    ('IRSTD-1k',   'XDU244',   'IRSTD\nMulti'),
]


# 挖掘阈值 —— Hard cases
BOX_FAIL_THRESH  = 0.25   # Box Fill IoU 低于此阈值算"失败"
HALO_PASS_THRESH = 0.60   # HALO IoU 高于此阈值算"成功"
# 挖掘阈值 —— Easy cases
EASY_BF_THRESH   = 0.50   # Box Fill IoU 高于此阈值算"容易"（BF 也能过）
EASY_HALO_MARGIN = 0.05   # HALO 至少比 BF 高出此边距才算"HALO 仍有提升"

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


def _compute_ious(args, ds, names, compute_bsnr_weight):
    """返回 {name: (bf_iou, halo_iou)} 字典"""
    cfg = DATASET_CONFIGS[ds]
    result = {}
    for name in names:
        img_path   = os.path.join(args.root, cfg['img_dir'],   name + '.png')
        mask_path  = os.path.join(args.root, cfg['mask_dir'],  name + '.png')
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
        if gt_mask.shape != (H, W):
            gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt_mask > 127).astype(np.uint8)
        if gt_bin.sum() == 0:
            continue
        img_float = img_gray.astype(np.float32) / 255.0

        bf_pred_img = load_prediction(PRED_DIRS['boxfill'].get(ds, ''), name)
        if bf_pred_img is not None:
            if bf_pred_img.shape != (H, W):
                bf_pred_img = cv2.resize(bf_pred_img, (W, H), interpolation=cv2.INTER_NEAREST)
            bf_mask = bf_pred_img.astype(np.float32) / 255.0
        else:
            bf_mask = make_box_fill(boxes, H, W)

        halo_pred_img = load_prediction(PRED_DIRS['halo'].get(ds, ''), name)
        if halo_pred_img is not None:
            if halo_pred_img.shape != (H, W):
                halo_pred_img = cv2.resize(halo_pred_img, (W, H), interpolation=cv2.INTER_NEAREST)
            halo_mask = halo_pred_img.astype(np.float32) / 255.0
        else:
            halo_mask = np.zeros((H, W), dtype=np.float32)
            for box in boxes:
                x1, y1, x2, y2 = box
                w_map = compute_bsnr_weight(
                    img_float, box, expand_ratio=1.5, temperature=3.0,
                    spatial_gaussian=True, gauss_sigma_ratio=1.5)
                halo_mask[y1:y2, x1:x2] = np.maximum(halo_mask[y1:y2, x1:x2], w_map)

        result[name] = (compute_iou(bf_mask > 0.5, gt_bin),
                        compute_iou(halo_mask > 0.5, gt_bin))
    return result


def mine_hard_cases(args, compute_bsnr_weight):
    """遍历测试集，分别打印 Hard（BF失败+HALO成功）和 Easy（BF也过+HALO更好）候选"""
    datasets = ['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k'] if args.dataset == 'all' else [args.dataset]
    case_type = getattr(args, 'case_type', 'both')

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        with open(os.path.join(args.root, cfg['split']), 'r') as f:
            names = [l.strip() for l in f if l.strip()]

        ious = _compute_ious(args, ds, names, compute_bsnr_weight)

        hard, easy = [], []
        for name, (bf_iou, halo_iou) in ious.items():
            gain = halo_iou - bf_iou
            if bf_iou < BOX_FAIL_THRESH and halo_iou > HALO_PASS_THRESH:
                hard.append((name, bf_iou, halo_iou, gain))
            elif bf_iou > EASY_BF_THRESH and gain > EASY_HALO_MARGIN:
                easy.append((name, bf_iou, halo_iou, gain))

        hard.sort(key=lambda x: -x[3])
        easy.sort(key=lambda x: -x[3])

        if case_type in ('hard', 'both'):
            print(f"\n{'='*60}")
            print(f"[HARD] {ds}  BF<{BOX_FAIL_THRESH:.2f}, HALO>{HALO_PASS_THRESH:.2f}")
            print(f"{'='*60}")
            print(f"{'Sample':<20} {'BF IoU':>10} {'HALO IoU':>10} {'Gain':>8}")
            print('-' * 52)
            for name, bf, halo, gain in hard[:20]:
                print(f"{name:<20} {bf:>10.4f} {halo:>10.4f} {gain:>+8.4f}")
            print(f"Total: {len(hard)}")

        if case_type in ('easy', 'both'):
            print(f"\n{'='*60}")
            print(f"[EASY] {ds}  BF>{EASY_BF_THRESH:.2f}, HALO-BF>{EASY_HALO_MARGIN:.2f}")
            print(f"{'='*60}")
            print(f"{'Sample':<20} {'BF IoU':>10} {'HALO IoU':>10} {'Gain':>8}")
            print('-' * 52)
            for name, bf, halo, gain in easy[:20]:
                print(f"{name:<20} {bf:>10.4f} {halo:>10.4f} {gain:>+8.4f}")
            print(f"Total: {len(easy)}")


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
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.patch.set_facecolor('black')

    col_titles = ['IR Image', 'GT Mask', 'Box Fill', 'B-SNR', 'HALO (Ours)', 'GT Supervised']

    for row_idx, (ds_name, sample_name, row_desc) in enumerate(HARD_CASES):
        cfg = DATASET_CONFIGS[ds_name]
        img_path   = os.path.join(args.root, cfg['img_dir'], sample_name + '.png')
        mask_path  = os.path.join(args.root, cfg['mask_dir'], sample_name + '.png')
        label_path = os.path.join(args.root, cfg['label_dir'], sample_name + '.txt')

        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        H, W = img_gray.shape
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else np.zeros((H, W), dtype=np.uint8)
        if gt_mask.shape != (H, W):
            gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        boxes = load_yolo_boxes(label_path, H, W)
        img_float = img_gray.astype(np.float32) / 255.0

        # 裁剪区域
        if len(boxes) > 1:
            ax1 = min(b[0] for b in boxes); ay1 = min(b[1] for b in boxes)
            ax2 = max(b[2] for b in boxes); ay2 = max(b[3] for b in boxes)
            pad = max(ax2 - ax1, ay2 - ay1) // 2
            rx1 = max(0, ax1 - pad); ry1 = max(0, ay1 - pad)
            rx2 = min(W, ax2 + pad); ry2 = min(H, ay2 + pad)
            side = max(rx2 - rx1, ry2 - ry1)
            rcx = (rx1 + rx2) // 2; rcy = (ry1 + ry2) // 2
            rx1 = max(0, rcx - side // 2); ry1 = max(0, rcy - side // 2)
            rx2 = min(W, rx1 + side);      ry2 = min(H, ry1 + side)
            def get_patch(arr):
                return cv2.resize(arr[ry1:ry2, rx1:rx2], (CROP_SIZE, CROP_SIZE),
                                  interpolation=cv2.INTER_NEAREST)
            img_patch = get_patch(img_gray)
            gt_patch  = get_patch(gt_mask)
        else:
            cx = (boxes[0][0] + boxes[0][2]) // 2 if boxes else W // 2
            cy = (boxes[0][1] + boxes[0][3]) // 2 if boxes else H // 2
            img_patch, _, _ = crop_patch(img_gray, cx, cy, CROP_SIZE)
            gt_patch,  _, _ = crop_patch(gt_mask,  cx, cy, CROP_SIZE)
            def get_patch(arr):
                p, _, _ = crop_patch(arr, cx, cy, CROP_SIZE)
                return p

        def load_resize(key):
            pred = load_prediction(PRED_DIRS[key].get(ds_name, ''), sample_name)
            if pred is not None and pred.shape != (H, W):
                pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
            return pred

        _bf = load_resize('boxfill')
        bf_pred   = _bf if _bf is not None else (make_box_fill(boxes, H, W) * 255).astype(np.uint8)
        bsnr_pred = load_resize('bsnr')
        halo_pred = load_resize('halo')
        gt_sup    = load_resize('gt_sup')

        if halo_pred is None:
            halo_full = np.zeros((H, W), dtype=np.float32)
            for box in boxes:
                x1, y1, x2, y2 = box
                w_map = compute_bsnr_weight(img_float, box, expand_ratio=1.5,
                                            temperature=3.0, spatial_gaussian=True)
                halo_full[y1:y2, x1:x2] = np.maximum(halo_full[y1:y2, x1:x2], w_map)
            halo_pred = (halo_full * 255).astype(np.uint8)

        bf_patch   = get_patch(bf_pred)
        bsnr_patch = get_patch(bsnr_pred) if bsnr_pred is not None else np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        halo_patch = get_patch(halo_pred)
        gtsup_patch = get_patch(gt_sup) if gt_sup is not None else np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)

        def style_ax(ax, title=None):
            ax.set_xticks([]); ax.set_yticks([])
            if title and row_idx == 0:
                ax.set_title(title, color='white', fontsize=9, pad=4)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444444')

        patches_data = [
            (img_patch,   None,      None,      col_titles[0]),
            (img_patch,   gt_patch,  'cyan',    col_titles[1]),
            (img_patch,   bf_patch,  'Oranges', col_titles[2]),
            (img_patch,   bsnr_patch,'YlOrRd',  col_titles[3]),
            (img_patch,   halo_patch,'inferno', col_titles[4]),
            (img_patch,   gtsup_patch,'viridis',col_titles[5]),
        ]

        for col_idx, (base, overlay, cmap, title) in enumerate(patches_data):
            ax = axes[row_idx][col_idx]
            ax.imshow(base, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
            if overlay is not None and cmap == 'cyan':
                ov = np.zeros((*base.shape, 4), dtype=np.float32)
                ov[overlay > 127] = [0.0, 1.0, 1.0, 0.9]
                ax.imshow(ov, interpolation='nearest')
            elif overlay is not None and overlay.max() > 0:
                ax.imshow(overlay.astype(np.float32) / 255.0, cmap=cmap,
                          alpha=0.75, interpolation='nearest', vmin=0, vmax=1)
            if col_idx == 0:
                ax.set_ylabel(row_desc, color='white', fontsize=7, labelpad=4)
            style_ax(ax, title)

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(left=0.10, hspace=0.06, wspace=0.03)

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
    parser.add_argument('--case_type', default='both', choices=['hard', 'easy', 'both'],
                        help='mine 模式：hard=BF失败HALO成功 / easy=BF也过HALO更好 / both=两类都打印')
    parser.add_argument('--out', default=os.path.join(ROOT, 'paper', 'figures', 'fig3_qualitative.png'))
    args = parser.parse_args()
    main(args)
