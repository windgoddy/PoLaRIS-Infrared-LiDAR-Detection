#!/usr/bin/env python3
"""
Figure: Box Perturbation Robustness Curve
==========================================
X 轴: 框扰动像素 δ (0, 2, 4, 6, 8)
Y 轴: Pseudo-Label Quality IoU (%)

三数据集 × 三方法 (Box Fill / B-SNR / HALO) 折线图
来自: ADVISOR_REPORT.md § 框扰动消融实验

用法:
    python paper/fig_robustness_curve.py
    python paper/fig_robustness_curve.py --out paper/figures/fig_robustness.pdf
    python paper/fig_robustness_curve.py --split  # 输出独立 panel
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'DejaVu Sans'
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ────────────────────────────────────────────────────────────────
# 实验数据（来自 ADVISOR_REPORT.md § 框扰动消融实验）
# 评估指标：Pseudo-Label Quality IoU (%)
# 扰动方式：四边独立随机偏移 ±δ，每图 10 次均值（seed=42）
# ────────────────────────────────────────────────────────────────

DELTA = [0, 2, 4, 6, 8]

DATA = {
    'NUAA-SIRST': {
        'Box Fill': [26.53, 28.31, 22.87, 15.23,  9.16],
        'B-SNR':    [51.77, 54.94, 45.33, 31.51, 20.24],
        'HALO':     [68.31, 68.45, 55.52, 39.14, 27.35],
    },
    'NUDT-SIRST': {
        'Box Fill': [23.05, 24.03, 19.95, 14.09,  9.73],
        'B-SNR':    [46.85, 48.96, 40.89, 29.44, 20.58],
        'HALO':     [69.03, 68.53, 54.34, 39.66, 29.27],
    },
    'IRSTD-1k': {
        'Box Fill': [26.28, 27.06, 23.86, 17.53, 12.11],
        'B-SNR':    [54.67, 56.13, 49.47, 37.12, 26.76],
        'HALO':     [61.72, 61.71, 53.16, 41.20, 31.08],
    },
}

STYLES = {
    'Box Fill': dict(color='#E53935', marker='s', ms=7, lw=1.8, ls='--',
                     label='Box Fill (Naive)'),
    'B-SNR':    dict(color='#1565C0', marker='D', ms=7, lw=1.8, ls='-.',
                     label='B-SNR (Ours-C)'),
    'HALO':     dict(color='#6A1B9A', marker='*', ms=11, lw=2.2, ls='-',
                     label='HALO (Ours-D)'),
}

SUBTITLES = {
    'NUAA-SIRST': '(a) NUAA-SIRST\n(Medium target, ~30 px²)',
    'NUDT-SIRST': '(b) NUDT-SIRST\n(Tiny target, <9 px²)',
    'IRSTD-1k':   '(c) IRSTD-1k\n(Diverse BG, ~15 px²)',
}


def _draw_panel(ax, ds_name, show_ylabel=True, show_legend=False):
    ds_data = DATA[ds_name]
    ax.set_axisbelow(True)
    ax.grid(True, ls='--', lw=0.5, alpha=0.45, color='#CCCCCC')

    for method, style in STYLES.items():
        vals = ds_data[method]
        ax.plot(DELTA, vals, **style, zorder=4)
        # 标注 δ=0 起点与 δ=2 终点之间的鲁棒性差异（仅 HALO）
        if method == 'HALO':
            change = vals[1] - vals[0]
            sign = '+' if change >= 0 else ''
            ax.annotate(f'{sign}{change:.2f}%',
                        xy=(2, vals[1]), xytext=(4, vals[1] + 2.5),
                        fontsize=7, color=style['color'],
                        arrowprops=dict(arrowstyle='->', color=style['color'],
                                        lw=0.7),
                        ha='left')

    # δ=2 时 Box Fill / B-SNR 的反常 ~104% 注释
    bf_delta2 = ds_data['Box Fill'][1]
    bf_delta0 = ds_data['Box Fill'][0]
    if bf_delta2 > bf_delta0:
        ax.annotate('Tightness\nbias effect',
                    xy=(2, bf_delta2),
                    xytext=(2.6, bf_delta2 + 5),
                    fontsize=6.5, color='#E53935', ha='left',
                    arrowprops=dict(arrowstyle='->', color='#E53935', lw=0.6))

    ax.set_title(SUBTITLES[ds_name], fontsize=10.5, fontweight='bold', pad=6)
    ax.set_xlabel(r'Perturbation $\delta$ (px)', fontsize=10)
    if show_ylabel:
        ax.set_ylabel('Pseudo-Label Quality IoU (%)', fontsize=10)
    ax.set_xticks(DELTA)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    # Y 轴留白
    all_vals = [v for m in ds_data.values() for v in m]
    lo, hi = min(all_vals), max(all_vals)
    margin = (hi - lo) * 0.18
    ax.set_ylim(lo - margin, hi + margin + 4)

    if show_legend:
        ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#AAAAAA',
                  loc='upper right')
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_edgecolor('#888888')


def draw_combined(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.26, left=0.06, right=0.97, top=0.87, bottom=0.14)

    for i, ds in enumerate(DATA.keys()):
        _draw_panel(axes[i], ds,
                    show_ylabel=(i == 0),
                    show_legend=(i == 2))

    # 共享图例放顶部
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles = [plt.Line2D([0], [0], **{k: v for k, v in s.items() if k != 'label'},
                              label=s['label'])
                   for s in STYLES.values()]
        labels = [s['label'] for s in STYLES.values()]
    fig.legend(handles, labels, loc='upper center', ncol=3,
               fontsize=9, framealpha=0.9, edgecolor='#AAAAAA',
               bbox_to_anchor=(0.5, 1.00))

    fig.suptitle(
        'Box Perturbation Robustness: HALO Decouples from Geometric Boundaries',
        fontsize=12, fontweight='bold', y=1.05,
    )
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_path}")
    plt.close()


def draw_split(out_dir, base, ext):
    os.makedirs(out_dir, exist_ok=True)
    for i, ds in enumerate(DATA.keys()):
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(left=0.14, right=0.95, top=0.88, bottom=0.15)
        _draw_panel(ax, ds, show_ylabel=True, show_legend=True)
        out_path = os.path.join(out_dir, f'{base}_panel_{chr(ord("a")+i)}{ext}')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved → {out_path}")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(
        ROOT, 'paper', 'figures', 'fig_robustness.png'))
    parser.add_argument('--split', action='store_true')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out) or '.'
    base    = os.path.splitext(os.path.basename(args.out))[0]
    ext     = os.path.splitext(args.out)[1] or '.png'

    if args.split:
        draw_split(out_dir, base, ext)
    else:
        draw_combined(args.out)
