#!/usr/bin/env python3
"""
eval_anchor_hitrate.py
======================
评估 p* 定位准确率（Hit Rate）：Ring 方法 vs Window 方法

实验逻辑：
  对每张图，每个 R 值，分别用两种方式估计背景统计量：
    - Ring  方法：μ/σ 从环形区域（排除 YOLO 框内部）计算  ← HALO 的做法
    - Window 方法：μ/σ 从整个扩展框（含目标内部）计算     ← 朴素做法

  然后计算 B-SNR 响应图 W(i)，找到 p* = argmax W(i)，
  判断 p* 是否落在 GT 二值掩码内（Hit）。

输出：
  - 终端打印各数据集、各 R 值的 Hit Rate 对比表格
  - paper/figures/fig_anchor_hitrate.png  ← 折线对比图

用法：
  python paper/eval_anchor_hitrate.py
  python paper/eval_anchor_hitrate.py --datasets NUDT-SIRST NUAA-SIRST --n_samples 200
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from scipy.special import expit   # sigmoid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── 超参数 ───────────────────────────────────────────────────────────────────
TAU          = 1.6          # B-SNR 温度系数（与 HALO 训练保持一致）
EXPAND_RATIOS = [1.1, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
MIN_EXPAND   = 3            # 最小膨胀像素数（小目标保护）

# ─── 数据集配置 ───────────────────────────────────────────────────────────────
DATASET_CFG = {
    'NUDT-SIRST': {
        'img_dir':  'dataset/NUDT-SIRST/images',
        'mask_dir': 'dataset/NUDT-SIRST/masks',
        'split':    'dataset/NUDT-SIRST/50_50/test.txt',
        'img_ext':  '.png',
    },
    'NUAA-SIRST': {
        'img_dir':  'dataset/NUAA-SIRST/images',
        'mask_dir': 'dataset/NUAA-SIRST/masks',
        'split':    'dataset/NUAA-SIRST/50_50/test.txt',
        'img_ext':  '.png',
    },
}

# ─── 基础工具函数 ──────────────────────────────────────────────────────────────

def get_context_box(box, R, H, W):
    """将 YOLO box 按比例 R 膨胀，返回 (cx1, cy1, cx2, cy2)，clip 到图像边界。"""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx = max(MIN_EXPAND, int(bw * (R - 1) / 2))
    my = max(MIN_EXPAND, int(bh * (R - 1) / 2))
    return (max(0, x1 - mx), max(0, y1 - my),
            min(W, x2 + mx), min(H, y2 + my))


def boxes_from_mask(mask):
    """
    从二值掩码（0/255）中提取连通域包围框列表。
    返回 [(x1, y1, x2, y2), ...]
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 127).astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, len(stats)):   # 0 是背景
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if w > 0 and h > 0:
            boxes.append((x, y, x + w, y + h))
    return boxes


def compute_bsnr(img_gray, box, mu_ctx, sigma_ctx):
    """
    在 YOLO box 内逐像素计算 B-SNR 响应：
      W(i) = sigmoid( τ * (I(i) - μ_ctx) / σ_ctx )
    返回 box 内的 W 图（与原图等尺寸，box 外为 0）。
    """
    x1, y1, x2, y2 = box
    H, W = img_gray.shape
    bsnr_map = np.zeros((H, W), np.float32)
    roi = img_gray[y1:y2, x1:x2].astype(np.float32)
    bsnr_map[y1:y2, x1:x2] = expit(TAU * (roi - mu_ctx) / (sigma_ctx + 1e-6))
    return bsnr_map


# ─── 两种统计量估计方式 ────────────────────────────────────────────────────────

def stats_ring(img_gray, box, R):
    """
    Ring 方法：μ, σ 从环形区域（排除 box 内部）估计。
    这是 HALO 的做法，背景统计不受目标污染。
    """
    H, W = img_gray.shape
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = get_context_box(box, R, H, W)

    ctx_mask   = np.zeros((H, W), bool)
    inner_mask = np.zeros((H, W), bool)
    ctx_mask[cy1:cy2, cx1:cx2]  = True
    inner_mask[y1:y2,  x1:x2]   = True
    ring_mask = ctx_mask & ~inner_mask

    px = img_gray[ring_mask].astype(np.float32)
    if len(px) < 4:
        return None, None   # 像素太少，跳过
    return float(np.mean(px)), float(np.std(px)) + 1e-6


def stats_window(img_gray, box, R):
    """
    Window 方法：μ, σ 从整个扩展框（含 box 内部）估计。
    朴素做法，方差会被目标像素污染。
    """
    H, W = img_gray.shape
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = get_context_box(box, R, H, W)

    px = img_gray[cy1:cy2, cx1:cx2].astype(np.float32).ravel()
    if len(px) < 4:
        return None, None
    return float(np.mean(px)), float(np.std(px)) + 1e-6


# ─── 单样本评估 ────────────────────────────────────────────────────────────────

def eval_one_image(img_gray, mask, box, R):
    """
    对一张图、一个目标、一个 R 值，分别用 Ring / Window 计算 p*，
    判断是否命中 GT mask。

    返回 dict:
      { 'ring_hit': bool, 'win_hit': bool,
        'ring_dist': float, 'win_dist': float,
        'p_ring': (px, py), 'p_win': (px, py) }
    """
    x1, y1, x2, y2 = box
    result = {}

    # GT mask 中心（用于距离评估）
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    gt_cx, gt_cy = float(np.mean(xs)), float(np.mean(ys))

    for method_name, stats_fn in [('ring', stats_ring), ('win', stats_window)]:
        mu, sigma = stats_fn(img_gray, box, R)
        if mu is None:
            result[f'{method_name}_hit']  = False
            result[f'{method_name}_dist'] = 9999.0
            result[f'p_{method_name}']    = (x1, y1)
            continue

        bsnr_map = compute_bsnr(img_gray, box, mu, sigma)

        # p* = argmax W(i) within box
        roi_bsnr = bsnr_map[y1:y2, x1:x2]
        flat_idx = int(np.argmax(roi_bsnr))
        bh, bw   = y2 - y1, x2 - x1
        py_rel, px_rel = divmod(flat_idx, bw)
        px_abs = x1 + px_rel
        py_abs = y1 + py_rel

        # Hit：p* 像素在 GT mask 内
        hit = bool(mask[py_abs, px_abs] > 127)

        # 距离：p* 到 GT 最近掩码像素的欧氏距离（像素）
        if hit:
            dist = 0.0
        else:
            # 近似：用到GT中心的距离（快速）
            dist = float(np.hypot(px_abs - gt_cx, py_abs - gt_cy))

        result[f'{method_name}_hit']  = hit
        result[f'{method_name}_dist'] = dist
        result[f'p_{method_name}']    = (px_abs, py_abs)

    return result


# ─── 数据集级别批量评估 ────────────────────────────────────────────────────────

def eval_dataset(ds_name, cfg, n_samples, root):
    """
    返回:
      { R: { 'ring_hits': int, 'win_hits': int,
             'ring_dists': [float,...], 'win_dists': [float,...],
             'total': int } }
    """
    split_path = os.path.join(root, cfg['split'])
    img_dir    = os.path.join(root, cfg['img_dir'])
    mask_dir   = os.path.join(root, cfg['mask_dir'])
    ext        = cfg['img_ext']

    with open(split_path) as f:
        names = [l.strip() for l in f if l.strip()]

    rng = np.random.default_rng(42)
    if n_samples and n_samples < len(names):
        names = rng.choice(names, n_samples, replace=False).tolist()

    records = {R: {'ring_hits': 0, 'win_hits': 0,
                   'ring_dists': [], 'win_dists': [], 'total': 0}
               for R in EXPAND_RATIOS}

    skipped = 0
    for name in names:
        img_path  = os.path.join(img_dir,  name + ext)
        mask_path = os.path.join(mask_dir, name + ext)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            skipped += 1; continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
                   if img_bgr.ndim == 3 else img_bgr

        mask = cv2.imread(mask_path, 0)
        if mask is None or np.max(mask) == 0:
            skipped += 1; continue

        boxes = boxes_from_mask(mask)
        if not boxes:
            skipped += 1; continue

        # 每个连通域（目标）单独评估，取第一个最大目标
        box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

        for R in EXPAND_RATIOS:
            res = eval_one_image(img_gray, mask, box, R)
            if res is None:
                continue
            rec = records[R]
            rec['total']        += 1
            rec['ring_hits']    += int(res['ring_hit'])
            rec['win_hits']     += int(res['win_hit'])
            rec['ring_dists'].append(res['ring_dist'])
            rec['win_dists'].append(res['win_dist'])

    if skipped:
        print(f"  [{ds_name}] 跳过 {skipped}/{len(names)} 张（图像/掩码缺失）")

    return records


# ─── 打印表格 ─────────────────────────────────────────────────────────────────

def print_table(ds_name, records):
    print(f"\n{'='*62}")
    print(f"  数据集: {ds_name}")
    print(f"{'='*62}")
    print(f"  {'R':>5}  {'N':>5}  "
          f"{'Ring Hit%':>10}  {'Win Hit%':>10}  "
          f"{'Ring Dist':>10}  {'Win Dist':>10}")
    print(f"  {'-'*57}")
    for R in EXPAND_RATIOS:
        rec = records[R]
        n   = rec['total']
        if n == 0:
            continue
        rh  = 100 * rec['ring_hits'] / n
        wh  = 100 * rec['win_hits']  / n
        rd  = np.mean(rec['ring_dists']) if rec['ring_dists'] else 0
        wd  = np.mean(rec['win_dists'])  if rec['win_dists']  else 0
        flag = '  ← HALO' if R == 1.5 else ''
        print(f"  {R:>5.2f}  {n:>5}  "
              f"{rh:>9.1f}%  {wh:>9.1f}%  "
              f"{rd:>9.2f}px  {wd:>9.2f}px{flag}")
    print()


# ─── 绘图 ─────────────────────────────────────────────────────────────────────

def plot_results(all_results, out_path):
    """绘制 Ring vs Window 的 Hit Rate 折线图（每个数据集一个子图）。"""
    datasets = list(all_results.keys())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8), sharey=False)
    if n == 1:
        axes = [axes]

    colors = {'Ring (HALO)': '#1565C0', 'Window (Baseline)': '#C62828'}
    markers = {'Ring (HALO)': 'o', 'Window (Baseline)': 's'}

    for ax, ds_name in zip(axes, datasets):
        records = all_results[ds_name]
        ring_rates, win_rates = [], []
        for R in EXPAND_RATIOS:
            rec = records[R]
            n_  = rec['total']
            if n_ == 0:
                ring_rates.append(np.nan)
                win_rates.append(np.nan)
            else:
                ring_rates.append(100 * rec['ring_hits'] / n_)
                win_rates.append(100 * rec['win_hits']   / n_)

        ax.plot(EXPAND_RATIOS, ring_rates,
                color=colors['Ring (HALO)'], marker=markers['Ring (HALO)'],
                lw=2.2, ms=7, label='Ring (HALO)', zorder=4)
        ax.plot(EXPAND_RATIOS, win_rates,
                color=colors['Window (Baseline)'], marker=markers['Window (Baseline)'],
                lw=2.2, ms=7, ls='--', label='Window (Baseline)', zorder=4)

        # 标注 R=1.5 设计点
        ax.axvline(1.5, color='#2E7D32', lw=1.2, ls='--', alpha=0.7,
                   label='R=1.5 (HALO default)')
        ax.axvspan(min(EXPAND_RATIOS), 1.3,
                   alpha=0.08, color='#FF1744', label='_nolegend_')
        ax.text(1.17, 5, 'Danger\nZone',
                fontsize=8, color='#BF360C', ha='center', va='bottom')

        # 在 R=1.5 处标注数值差
        r15_ring = ring_rates[EXPAND_RATIOS.index(1.5)]
        r15_win  = win_rates [EXPAND_RATIOS.index(1.5)]
        if not (np.isnan(r15_ring) or np.isnan(r15_win)):
            ax.annotate(f'Δ={r15_ring - r15_win:.1f}%',
                        xy=(1.5, (r15_ring + r15_win) / 2),
                        xytext=(1.75, (r15_ring + r15_win) / 2 + 3),
                        fontsize=8.5, color='#37474F',
                        arrowprops=dict(arrowstyle='->', color='#37474F', lw=0.9))

        ax.set_title(ds_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Expand Ratio  R', fontsize=10)
        ax.set_ylabel('p* Hit Rate (%)', fontsize=10)
        ax.set_xlim(min(EXPAND_RATIOS) - 0.05, max(EXPAND_RATIOS) + 0.05)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='lower right')

    fig.suptitle(
        r'Physical Anchor $p^*$ Hit Rate: Ring vs Window Background Estimation',
        fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f"\n图像已保存 → {out_path}")
    plt.close()


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='评估 p* 定位 Hit Rate：Ring vs Window')
    parser.add_argument('--root', default=ROOT,
                        help='项目根目录（默认自动检测）')
    parser.add_argument('--datasets', nargs='+',
                        default=['NUDT-SIRST', 'NUAA-SIRST'],
                        choices=list(DATASET_CFG.keys()),
                        help='要评估的数据集（可多选）')
    parser.add_argument('--n_samples', type=int, default=0,
                        help='每个数据集最多评估多少张（0=全部）')
    parser.add_argument('--out', default=os.path.join(
                        ROOT, 'paper', 'figures', 'fig_anchor_hitrate.png'),
                        help='输出图像路径')
    args = parser.parse_args()

    all_results = {}
    for ds_name in args.datasets:
        if ds_name not in DATASET_CFG:
            print(f"警告：数据集 {ds_name} 未配置，跳过")
            continue
        cfg = DATASET_CFG[ds_name]

        # 检查路径是否存在
        img_dir = os.path.join(args.root, cfg['img_dir'])
        if not os.path.isdir(img_dir):
            print(f"警告：{ds_name} 图像目录不存在 ({img_dir})，跳过")
            continue

        print(f"\n[{ds_name}] 开始评估...")
        records = eval_dataset(ds_name, cfg, args.n_samples, args.root)
        all_results[ds_name] = records
        print_table(ds_name, records)

    if not all_results:
        print("没有可用的数据集，退出。")
        return

    plot_results(all_results, args.out)

    # 打印关键结论
    print("\n【核心结论】R=1.5 处 Ring vs Window Hit Rate 差距：")
    for ds_name, records in all_results.items():
        rec = records[1.5]
        n   = rec['total']
        if n == 0:
            continue
        rh = 100 * rec['ring_hits'] / n
        wh = 100 * rec['win_hits']  / n
        print(f"  {ds_name:<15}  Ring={rh:.1f}%  Window={wh:.1f}%  "
              f"差距 Δ={rh - wh:+.1f}%  (N={n})")


if __name__ == '__main__':
    main()
