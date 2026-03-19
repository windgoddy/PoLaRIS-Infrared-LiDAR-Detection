#!/usr/bin/env python3
"""
Figure: Hyperparameter Sensitivity Analysis
============================================
Left  (a): expand_ratio sweep  (固定 sigma_ratio=1.5)
Right (b): gauss_sigma_ratio sweep (固定 expand_ratio=1.5)
Y 轴: mIoU (%)，三条数据集折线 + 均值粗线
默认值用竖虚线标注

用法:
    python paper/fig_hyperparam_sensitivity.py
    python paper/fig_hyperparam_sensitivity.py --out paper/figures/fig_hyperparam.pdf
    python paper/fig_hyperparam_sensitivity.py --split
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
# 实验数据（来自 ADVISOR_REPORT.md § 超参数敏感性消融）
# 评估指标：Pseudo-Label Quality IoU (%), DNANet 骨干
# ────────────────────────────────────────────────────────────────

# expand_ratio sweep（固定 sigma_ratio=1.5）
EXPAND_X = [1.1, 1.3, 1.5, 2.0, 3.0]
EXPAND_DEFAULT = 1.5
EXPAND_DATA = {
    'NUAA-SIRST': [68.30, 68.30, 68.31, 67.42, 64.06],
    'NUDT-SIRST': [69.39, 69.39, 69.03, 67.62, 65.67],
    'IRSTD-1k':   [61.67, 61.67, 61.72, 61.17, 59.32],
}

# gauss_sigma_ratio sweep（固定 expand_ratio=1.5）
SIGMA_X = [0.8, 1.0, 1.274, 1.5, 2.0, 2.5]
SIGMA_DEFAULT = 1.5
SIGMA_DATA = {
    'NUAA-SIRST': [37.74, 48.12, 61.11, 68.31, 70.53, 66.07],
    'NUDT-SIRST': [44.52, 51.35, 60.29, 69.03, 66.73, 58.58],
    'IRSTD-1k':   [33.61, 42.78, 54.72, 61.72, 66.64, 64.93],
}

DS_STYLES = {
    'NUAA-SIRST': dict(color='#1565C0', marker='o', ms=7,  lw=1.6, ls='-',  label='NUAA-SIRST'),
    'NUDT-SIRST': dict(color='#2E7D32', marker='s', ms=7,  lw=1.6, ls='--', label='NUDT-SIRST'),
    'IRSTD-1k':   dict(color='#E65100', marker='^', ms=7,  lw=1.6, ls='-.', label='IRSTD-1k'),
}
MEAN_STYLE = dict(color='#6A1B9A', marker='D', ms=8, lw=2.2, ls='-', label='Mean', zorder=5)


def _draw_panel(ax, x_vals, data, default_x, xlabel, title,
                show_ylabel=True, show_legend=False):
    ax.set_axisbelow(True)
    ax.grid(True, ls='--', lw=0.5, alpha=0.45, color='#CCCCCC')

    # 三条数据集折线
    for ds, style in DS_STYLES.items():
        ax.plot(x_vals, data[ds], **style, zorder=3)

    # 均值折线
    means = np.mean([data[ds] for ds in data], axis=0)
    ax.plot(x_vals, means, **MEAN_STYLE)

    # 默认值竖线
    ax.axvline(default_x, color='#880E4F', lw=1.2, ls=':', zorder=2, alpha=0.85)
    ax.text(default_x + (x_vals[-1] - x_vals[0]) * 0.015,
            0.97,
            'default', fontsize=7.5, color='#880E4F', va='top',
            transform=ax.get_xaxis_transform())

    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, fontsize=10)
    if show_ylabel:
        ax.set_ylabel('mIoU (%)', fontsize=10)

    ax.set_xticks(x_vals)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    # Y 轴留白
    all_vals = list(means) + [v for ds in data.values() for v in ds]
    lo, hi = min(all_vals), max(all_vals)
    margin = (hi - lo) * 0.18
    ax.set_ylim(lo - margin, hi + margin + 3)

    if show_legend:
        ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#AAAAAA',
                  loc='lower left')
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_edgecolor('#888888')


def draw_combined(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.28, left=0.07, right=0.97, top=0.87, bottom=0.14)

    _draw_panel(axes[0], EXPAND_X, EXPAND_DATA, EXPAND_DEFAULT,
                xlabel=r'Context Expansion Ratio $R$',
                title=r'(a) Sensitivity to expand\_ratio'+'\n(fixed sigma_ratio=1.5)',
                show_ylabel=True, show_legend=True)

    _draw_panel(axes[1], SIGMA_X, SIGMA_DATA, SIGMA_DEFAULT,
                xlabel=r'Gaussian Width Ratio $\kappa$',
                title=r'(b) Sensitivity to gauss\_sigma\_ratio'+'\n(fixed expand_ratio=1.5)',
                show_ylabel=False, show_legend=False)

    # 物理参考线 σ=1.274（理论 PSF 校准值）
    ax_b = axes[1]
    ax_b.axvline(1.274, color='#37474F', lw=0.9, ls='--', alpha=0.6, zorder=2)
    ax_b.text(1.274 + 0.02, 0.04, 'PSF theory\n(1.274)',
              fontsize=6.5, color='#37474F', va='bottom',
              transform=ax_b.get_xaxis_transform())

    # 共享图例放顶部
    handles = [plt.Line2D([0], [0], **{k: v for k, v in s.items() if k != 'label'},
                          label=s['label'])
               for s in list(DS_STYLES.values()) + [MEAN_STYLE]]
    labels = [s['label'] for s in list(DS_STYLES.values()) + [MEAN_STYLE]]
    fig.legend(handles, labels, loc='upper center', ncol=4,
               fontsize=9, framealpha=0.9, edgecolor='#AAAAAA',
               bbox_to_anchor=(0.5, 1.00))

    fig.suptitle(
        'HALO Hyperparameter Sensitivity: Robust Physical Design Constants',
        fontsize=11.5, fontweight='bold', y=1.05,
    )
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_path}")
    plt.close()


def _save_single(draw_fn, out_path):
    draw_fn()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_path}")
    plt.close()


def draw_split(out_dir, base, ext):
    os.makedirs(out_dir, exist_ok=True)

    # Panel a: expand_ratio
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.13, right=0.95, top=0.88, bottom=0.15)
    _draw_panel(ax, EXPAND_X, EXPAND_DATA, EXPAND_DEFAULT,
                xlabel=r'Context Expansion Ratio $R$',
                title=r'(a) Sensitivity to expand\_ratio',
                show_ylabel=True, show_legend=True)
    out_a = os.path.join(out_dir, f'{base}_panel_a{ext}')
    plt.savefig(out_a, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_a}")
    plt.close()

    # Panel b: sigma_ratio
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.13, right=0.95, top=0.88, bottom=0.15)
    _draw_panel(ax, SIGMA_X, SIGMA_DATA, SIGMA_DEFAULT,
                xlabel=r'Gaussian Width Ratio $\kappa$',
                title=r'(b) Sensitivity to gauss\_sigma\_ratio',
                show_ylabel=True, show_legend=True)
    # 物理参考线
    ax.axvline(1.274, color='#37474F', lw=0.9, ls='--', alpha=0.6, zorder=2)
    ax.text(1.274 + 0.02, 0.04, 'PSF theory\n(1.274)',
            fontsize=6.5, color='#37474F', va='bottom',
            transform=ax.get_xaxis_transform())
    out_b = os.path.join(out_dir, f'{base}_panel_b{ext}')
    plt.savefig(out_b, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_b}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(
        ROOT, 'paper', 'figures', 'fig_hyperparam.png'))
    parser.add_argument('--split', action='store_true')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out) or '.'
    base    = os.path.splitext(os.path.basename(args.out))[0]
    ext     = os.path.splitext(args.out)[1] or '.png'

    if args.split:
        draw_split(out_dir, base, ext)
    else:
        draw_combined(args.out)
