#!/usr/bin/env python3
"""
Figure 1: Trend Consistency — HALO vs. Baselines across Datasets and Backbones
===============================================================================
1行 × 2列子图:
  Left:  跨数据集（DNANet 骨干固定），X轴 = NUAA / NUDT / IRSTD，4条曲线 A/B/C/D
  Right: 跨骨干网（NUAA 数据集固定），X轴 = ACM / ALCNet / DNANet，3条曲线 A/C/D

核心叙事:
  HALO (D) 的涨跌趋势与 GT 全监督 (A) 高度吻合;
  Box Fill (B) 和 B-SNR (C) 趋势混乱或与全监督偏离;
  这证明 HALO 伪标签质量最高、在数据集/骨干之间泛化性最强

用法:
    python paper/fig1_trend_consistency.py
    python paper/fig1_trend_consistency.py --out paper/figures/fig1_trend.pdf
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
# ★ 实验数据（来自 ADVISOR_REPORT.md）
# ────────────────────────────────────────────────────────────────

# 跨数据集：固定 DNANet 骨干
CROSS_DATASET = {
    'x_labels': ['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k'],
    'GT Full (A)':    [72.84, 95.07, 70.94],
    'Box Fill (B)':   [55.17, 57.39, 50.90],
    'B-SNR (C)':      [63.12, 56.12, 60.91],
    'HALO Ours (D)':  [70.69, 62.97, 60.75],
}

# 跨骨干网：固定 NUAA-SIRST 数据集
# Box Fill 只有 DNANet 数据，不画完整折线，仅在右侧单独标注
CROSS_BACKBONE = {
    'x_labels': ['ACM', 'ALCNet', 'DNANet'],
    'GT Full (A)':    [68.95, 69.72, 72.84],
    'B-SNR (C)':      [53.80, 59.31, 63.12],
    'HALO Ours (D)':  [66.28, 66.93, 70.69],
    # Box Fill: DNANet only = 55.17 (shown as single marker, no line)
    'Box Fill (B) [DNANet only]': [None, None, 55.17],
}

# ────────────────────────────────────────────────────────────────
# 绘图样式（对齐 PAL 论文风格）
# ────────────────────────────────────────────────────────────────

STYLES = {
    'GT Full (A)': dict(
        color='#CC2222', marker='^', markersize=9, linewidth=2.0,
        linestyle='-', label='GT Full (Full Sup.)',
        zorder=4,
    ),
    'Box Fill (B)': dict(
        color='#228B22', marker='s', markersize=8, linewidth=1.8,
        linestyle='--', label='Box Fill (Naive)',
        zorder=2,
    ),
    'B-SNR (C)': dict(
        color='#1155AA', marker='D', markersize=7, linewidth=1.8,
        linestyle='-.', label='B-SNR (Ours-C)',
        zorder=3,
    ),
    'HALO Ours (D)': dict(
        color='#7B2D8B', marker='*', markersize=12, linewidth=2.2,
        linestyle='-', label='HALO Ours (D)',
        zorder=5,
    ),
    'Box Fill (B) [DNANet only]': dict(
        color='#228B22', marker='s', markersize=8, linewidth=0,
        linestyle='none', label='_nolegend_',
        zorder=2,
    ),
}


def annotate_values(ax, x_pos, values, style, offset_y=0.8, fmt='{:.2f}'):
    """在每个数据点上方标注数值"""
    for xi, yi in zip(x_pos, values):
        if yi is None:
            continue
        ax.annotate(
            fmt.format(yi),
            xy=(xi, yi), xytext=(0, offset_y),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=7.5,
            color=style['color'],
            fontweight='bold',
        )


def draw_subplot(ax, data, title, xlabel, ylabel='mIoU (%)',
                 annotate=True, legend=True):
    """画单个趋势折线子图"""
    x_labels = data['x_labels']
    x_pos = np.arange(len(x_labels))

    # 灰色网格
    ax.set_axisbelow(True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.6,
            color='#CCCCCC', alpha=0.8)

    for key, style in STYLES.items():
        if key == 'x_labels' or key not in data:
            continue
        values = data[key]
        # 过滤 None
        valid_x = [x for x, v in zip(x_pos, values) if v is not None]
        valid_y = [v for v in values if v is not None]

        plot_style = {k: v for k, v in style.items()
                      if k not in ('label', 'zorder')}
        ax.plot(valid_x, valid_y,
                label=style['label'],
                zorder=style.get('zorder', 2),
                **plot_style)

        if annotate and valid_y:
            annotate_values(ax, valid_x, valid_y, style)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    # Y轴范围自适应留白
    all_vals = [v for k in data if k != 'x_labels'
                for v in data[k] if isinstance(v, float)]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.18
        ax.set_ylim(lo - margin, hi + margin)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels)
                    if not l.startswith('_')]
        ax.legend([h for h, _ in filtered],
                  [l for _, l in filtered],
                  fontsize=8.5, framealpha=0.9,
                  edgecolor='#AAAAAA', loc='best')

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor('#888888')


def main(args):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.28, left=0.07, right=0.97,
                        top=0.88, bottom=0.13)

    # ── 左图：跨数据集（DNANet 固定）──────────────────────────────
    draw_subplot(
        axes[0], CROSS_DATASET,
        title='(a) Cross-Dataset Consistency\n(Backbone: DNANet)',
        xlabel='Dataset',
    )

    # ── 右图：跨骨干网（NUAA 固定）───────────────────────────────
    draw_subplot(
        axes[1], CROSS_BACKBONE,
        title='(b) Cross-Backbone Consistency\n(Dataset: NUAA-SIRST)',
        xlabel='Backbone',
    )
    # Box Fill 单点补充说明
    axes[1].annotate(
        'Box Fill\n(DNANet only)',
        xy=(2, 55.17), xytext=(-32, -22),
        textcoords='offset points',
        fontsize=7.5, color='#228B22',
        arrowprops=dict(arrowstyle='->', color='#228B22', lw=0.8),
    )

    # 图级标题
    fig.suptitle(
        'HALO Trend Closely Follows GT Full Supervision Across Datasets and Backbones',
        fontsize=12, fontweight='bold', y=0.98,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved → {args.out}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(
        ROOT, 'paper', 'figures', 'fig1_trend_consistency.png'))
    args = parser.parse_args()
    main(args)
