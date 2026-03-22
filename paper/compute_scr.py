#!/usr/bin/env python3
"""
Compute SCR (Signal-to-Clutter Ratio) Statistics for IRSTD Datasets
=====================================================================
统计三个基准数据集的内禀 SCR 分布，用于解释 HALO 在 NUDT 上的性能边界。

SCR 定义（CA-CFAR 口径）:
    SCR = (μ_target - μ_ctx) / σ_ctx

其中：
    μ_target = 目标像素的均值强度（从 GT 掩码提取连通域）
    μ_ctx    = Context Box 的均值（expand_ratio 倍膨胀框，含目标+背景）
    σ_ctx    = Context Box 的标准差

Context Box 与 HALO B-SNR 保持一致: expand_ratio=3.0（比 HALO 默认 1.5 更大，
确保 μ_ctx 更接近纯背景）。

用法:
    python paper/compute_scr.py
    python paper/compute_scr.py --expand_ratio 3.0 --split all
    python paper/compute_scr.py --dataset NUDT-SIRST

输出:
    终端打印 Markdown 表格（可直接贴入 ADVISOR_REPORT.md）
    paper/figures/scr_distribution.png（分布直方图，可选）

结果参考（expand_ratio=3.0，2026-03-20）:
    NUAA-SIRST: mean=8.163  median=7.542  std=3.792  SCR<1=0.0%   SCR<3=4.5%
    IRSTD-1k  : mean=7.931  median=7.255  std=4.157  SCR<1=1.1%   SCR<3=8.0%
    NUDT-SIRST: mean=5.506  median=5.488  std=2.897  SCR<1=2.7%   SCR<3=21.8%
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────────────
# 数据集配置
# ──────────────────────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    'NUAA-SIRST': {
        'img_dir':   'dataset/NUAA-SIRST/images',
        'mask_dir':  'dataset/NUAA-SIRST/masks',
        'split_dir': 'dataset/NUAA-SIRST/50_50',
        'img_suffix': '.png',
    },
    'NUDT-SIRST': {
        'img_dir':   'dataset/NUDT-SIRST/images',
        'mask_dir':  'dataset/NUDT-SIRST/masks',
        'split_dir': 'dataset/NUDT-SIRST/50_50',
        'img_suffix': '.png',
    },
    'IRSTD-1k': {
        'img_dir':   'dataset/IRSTD-1k/IRSTD1k_Img',
        'mask_dir':  'dataset/IRSTD-1k/masks',
        'split_dir': 'dataset/IRSTD-1k',
        'img_suffix': '.png',
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# SCR 计算
# ──────────────────────────────────────────────────────────────────────────────
def compute_scr_single(
    image_gray: np.ndarray,
    mask: np.ndarray,
    expand_ratio: float = 3.0,
    epsilon: float = 1e-4,
) -> List[float]:
    """
    从一张图像中提取所有目标的 SCR。

    Args:
        image_gray: (H, W) uint8 灰度图
        mask:       (H, W) uint8 二值掩码（目标=255，背景=0）
        expand_ratio: Context Box 膨胀倍率（默认 3.0）

    Returns:
        scr_values: 该图中每个连通目标的 SCR 列表
    """
    H, W = image_gray.shape
    img = image_gray.astype(np.float32)

    # 连通域分析，提取各目标 BBox
    binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    scr_values = []
    for i in range(1, num_labels):  # 跳过背景 (label=0)
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        x2 = x1 + bw
        y2 = y1 + bh

        if bw <= 0 or bh <= 0:
            continue

        # ── 目标区域均值 ──
        target_mask = labels[y1:y2, x1:x2] == i
        target_pixels = img[y1:y2, x1:x2][target_mask]
        if len(target_pixels) == 0:
            continue
        mu_target = target_pixels.mean()

        # ── Context Box（膨胀 expand_ratio 倍，与 HALO B-SNR 一致）──
        pad_x = max(int((expand_ratio - 1.0) / 2.0 * bw), 3)
        pad_y = max(int((expand_ratio - 1.0) / 2.0 * bh), 3)
        ctx_x1 = max(0, x1 - pad_x)
        ctx_y1 = max(0, y1 - pad_y)
        ctx_x2 = min(W, x2 + pad_x)
        ctx_y2 = min(H, y2 + pad_y)

        ctx_region = img[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
        if ctx_region.size == 0:
            continue
        mu_ctx  = ctx_region.mean()
        sig_ctx = ctx_region.std()

        # ── SCR = (μ_target - μ_ctx) / σ_ctx ──
        scr = (mu_target - mu_ctx) / (sig_ctx + epsilon)
        scr_values.append(float(scr))

    return scr_values


def load_split_ids(cfg: dict, split: str = 'all') -> List[str]:
    """读取数据集 split（train/val/test/all）的图像 ID。"""
    split_dir = os.path.join(ROOT, cfg['split_dir'])

    if split == 'all':
        # 尝试合并 train.txt 和 val.txt / test.txt
        candidates = []
        for fname in ['train.txt', 'val.txt', 'test.txt']:
            fpath = os.path.join(split_dir, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    candidates += [l.strip() for l in f if l.strip()]
        if not candidates:
            # 直接列举 mask 目录
            mask_dir = os.path.join(ROOT, cfg['mask_dir'])
            candidates = [f[:-4] for f in sorted(os.listdir(mask_dir))
                          if f.endswith(('.png', '.bmp', '.jpg'))]
        # 去重
        seen = set()
        ids = []
        for x in candidates:
            if x not in seen:
                seen.add(x)
                ids.append(x)
        return ids
    else:
        fpath = os.path.join(split_dir, f'{split}.txt')
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Split file not found: {fpath}")
        with open(fpath) as f:
            return [l.strip() for l in f if l.strip()]


def compute_dataset_scr(
    dataset_name: str,
    cfg: dict,
    expand_ratio: float = 3.0,
    split: str = 'all',
    verbose: bool = False,
) -> Dict:
    """计算单个数据集的 SCR 统计量。"""
    img_dir  = os.path.join(ROOT, cfg['img_dir'])
    mask_dir = os.path.join(ROOT, cfg['mask_dir'])
    ids      = load_split_ids(cfg, split)

    all_scr = []
    n_images = 0
    n_missing = 0

    for img_id in ids:
        # 读图像
        img_path = None
        for ext in ['.png', '.bmp', '.jpg']:
            cand = os.path.join(img_dir, img_id + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            n_missing += 1
            continue

        # 读掩码
        mask_path = None
        for ext in ['.png', '.bmp', '.jpg']:
            cand = os.path.join(mask_dir, img_id + ext)
            if os.path.exists(cand):
                mask_path = cand
                break
        if mask_path is None:
            n_missing += 1
            continue

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            n_missing += 1
            continue

        scr_vals = compute_scr_single(image, mask, expand_ratio=expand_ratio)
        all_scr.extend(scr_vals)
        n_images += 1
        if verbose and n_images % 100 == 0:
            print(f"  {dataset_name}: processed {n_images}/{len(ids)} images, "
                  f"{len(all_scr)} targets so far...")

    if not all_scr:
        print(f"  [WARN] {dataset_name}: no targets found!")
        return {}

    arr = np.array(all_scr)
    # 将 SCR 归一化到 uint8 范围前先除以 255（原始图像为 uint8）
    # 注意：img 已是 float32，所以 SCR 单位已是 [0, 255] 范围的强度差/std
    # 如果需要与文献中 "归一化 SCR" 对比，除以 255；但通常直接报告原始值
    stats = {
        'n_images':  n_images,
        'n_targets': len(arr),
        'mean':      float(arr.mean()),
        'median':    float(np.median(arr)),
        'std':       float(arr.std()),
        'min':       float(arr.min()),
        'max':       float(arr.max()),
        'scr_lt1':   float((arr < 1.0).mean() * 100),   # SCR<1 占比 (%)
        'scr_lt3':   float((arr < 3.0).mean() * 100),   # SCR<3 占比 (%)
        'scr_lt5':   float((arr < 5.0).mean() * 100),   # SCR<5 占比 (%)
        'raw':       arr,
    }
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 输出
# ──────────────────────────────────────────────────────────────────────────────
def print_markdown_table(results: Dict[str, Dict]):
    """打印 Markdown 格式汇总表。"""
    print(f"\n{'='*75}")
    print(f"  SCR Statistics (expand_ratio for context box as configured)")
    print(f"{'='*75}")
    print(f"| {'数据集':<12} | {'SCR 均值':>8} | {'SCR 中位数':>10} | {'SCR std':>7} | "
          f"{'SCR<1 占比':>10} | {'SCR<3 占比':>10} | {'目标数':>6} |")
    print(f"|{'-'*14}|{'-'*10}|{'-'*12}|{'-'*9}|{'-'*12}|{'-'*12}|{'-'*8}|")
    for ds, s in results.items():
        if not s:
            continue
        print(f"| {ds:<12} | {s['mean']:>8.3f} | {s['median']:>10.3f} | "
              f"{s['std']:>7.3f} | {s['scr_lt1']:>9.1f}% | "
              f"{s['scr_lt3']:>9.1f}% | {s['n_targets']:>6} |")
    print()

    # 额外行：NUDT vs NUAA 比率
    if 'NUDT-SIRST' in results and 'NUAA-SIRST' in results:
        r_nudt = results['NUDT-SIRST']
        r_nuaa = results['NUAA-SIRST']
        if r_nudt and r_nuaa:
            ratio = r_nudt['mean'] / r_nuaa['mean'] * 100
            print(f"  NUDT SCR 均值 = NUAA SCR 均值 的 {ratio:.1f}%")
            print(f"  （NUDT SCR<3 占比比 NUAA 高 {r_nudt['scr_lt3'] - r_nuaa['scr_lt3']:.1f}pp）")


def plot_distributions(results: Dict[str, Dict], out_path: str):
    """可选：绘制 SCR 分布直方图。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['pdf.fonttype'] = 42
        import matplotlib.pyplot as plt

        colors = {'NUAA-SIRST': '#1565C0', 'NUDT-SIRST': '#2E7D32', 'IRSTD-1k': '#E65100'}
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
        fig.patch.set_facecolor('white')

        for ax, (ds, s) in zip(axes, results.items()):
            if not s:
                ax.set_title(ds)
                continue
            arr = s['raw']
            ax.hist(arr, bins=40, color=colors.get(ds, '#555'), alpha=0.75,
                    edgecolor='white', linewidth=0.5)
            ax.axvline(s['mean'],   color='#880E4F', lw=1.5, ls='-',  label=f"Mean={s['mean']:.2f}")
            ax.axvline(s['median'], color='#880E4F', lw=1.2, ls='--', label=f"Median={s['median']:.2f}")
            ax.axvline(3.0, color='#444', lw=1.0, ls=':', alpha=0.8, label='SCR=3 (critical)')
            ax.set_title(f"{ds}\n(SCR<3: {s['scr_lt3']:.1f}%)", fontsize=10, fontweight='bold')
            ax.set_xlabel('SCR', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.legend(fontsize=7.5, framealpha=0.85)
            for sp in ax.spines.values():
                sp.set_linewidth(0.7)

        fig.suptitle('Intrinsic SCR Distribution Across Datasets\n'
                     '(Context Box: expand_ratio=3.0; CA-CFAR definition)',
                     fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  Distribution plot saved → {out_path}")
        plt.close()
    except ImportError:
        print("  [INFO] matplotlib not available, skipping distribution plot.")


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Compute SCR statistics for IRSTD datasets')
    parser.add_argument('--expand_ratio', type=float, default=3.0,
                        help='Context Box expansion ratio (default: 3.0)')
    parser.add_argument('--split', default='all',
                        choices=['all', 'train', 'val', 'test'],
                        help='Dataset split to use (default: all)')
    parser.add_argument('--dataset', default='all',
                        choices=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k', 'all'],
                        help='Dataset to compute (default: all)')
    parser.add_argument('--plot', action='store_true',
                        help='Save SCR distribution histogram to paper/figures/')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress every 100 images')
    args = parser.parse_args()

    datasets = (list(DATASET_CONFIGS.keys()) if args.dataset == 'all'
                else [args.dataset])

    print(f"\n{'='*75}")
    print(f"  HALO SCR Statistics")
    print(f"  expand_ratio={args.expand_ratio}  split={args.split}")
    print(f"{'='*75}")

    results = {}
    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        print(f"\n[{ds}] Computing SCR...")
        stats = compute_dataset_scr(
            ds, cfg,
            expand_ratio=args.expand_ratio,
            split=args.split,
            verbose=args.verbose,
        )
        results[ds] = stats
        if stats:
            print(f"  n_images={stats['n_images']}  n_targets={stats['n_targets']}")
            print(f"  SCR: mean={stats['mean']:.3f}  median={stats['median']:.3f}  "
                  f"std={stats['std']:.3f}")
            print(f"  SCR<1: {stats['scr_lt1']:.1f}%  SCR<3: {stats['scr_lt3']:.1f}%  "
                  f"SCR<5: {stats['scr_lt5']:.1f}%")

    print_markdown_table(results)

    if args.plot:
        out_path = os.path.join(ROOT, 'paper', 'figures', 'fig_scr_distribution.png')
        plot_distributions(results, out_path)


if __name__ == '__main__':
    main()
