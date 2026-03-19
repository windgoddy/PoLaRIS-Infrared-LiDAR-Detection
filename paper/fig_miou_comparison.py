#!/usr/bin/env python3
"""
Figure: Main Results — mIoU Comparison (All Methods × 3 Datasets)
==================================================================
分组柱状图：X 轴 = 方法，三组柱（NUAA / NUDT / IRSTD）
水平参考线：LESPS、PAL（单点监督 SOTA）、GT 全监督上界

框监督方法分三层（颜色渐深）：
  GrabCut（传统）→ Box Fill（朴素）→ B-SNR（本文-C）→ HALO（本文-D）
  E组（BoxInst-style）：待结果，当前用占位符跳过

数据来源: ADVISOR_REPORT.md § P1 DNANet 骨干实验

用法:
    python paper/fig_miou_comparison.py
    python paper/fig_miou_comparison.py --out paper/figures/fig_miou.pdf
    python paper/fig_miou_comparison.py --split   # 输出三数据集独立 panel
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
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ────────────────────────────────────────────────────────────────
# 数据（DNANet 骨干，mIoU %）
# 来源：ADVISOR_REPORT.md 主实验表
# ────────────────────────────────────────────────────────────────

DATASETS = ['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k']

# 柱状图方法（框监督，按性能梯度排列）
BAR_METHODS = [
    # (label,              NUAA,  NUDT,  IRSTD, color,    hatch, group)
    ('GrabCut',            57.31, 42.10, 50.43, '#EF9A9A', '',   'box-naive'),
    ('Box Fill',           55.17, 57.39, 50.90, '#EF5350', '',   'box-naive'),
    ('B-SNR\n(Ours-C)',    63.12, 56.12, 60.91, '#9575CD', '',   'ours'),
    ('HALO\n(Ours-D)',     70.69, 62.97, 60.75, '#4527A0', '',   'ours'),
    # E组结果待补充，解注释并填入数字后重新生成
    # ('Proj. Loss\n(E)',  ??,    ??,    ??,    '#7E57C2', '//', 'box-naive'),
]

# 水平参考线：单点监督 SOTA（跨监督形式对比）
HLINES = [
    # (label,                    NUAA,  NUDT,  IRSTD, color,    ls,   lw)
    ('LESPS (Single-point)',     61.13, 58.37, 49.05, '#1565C0', '--', 1.5),
    ('PAL (Single-point)',       66.57, 73.29, 58.40, '#0277BD', ':',  1.8),
    ('GT Full Supervision',      72.84, 95.07, 70.94, '#37474F', '-.', 1.3),
]

DS_IDX = {ds: i for i, ds in enumerate(DATASETS)}


def _bar_vals(method_row):
    _, nuaa, nudt, irstd, *_ = method_row
    return [nuaa, nudt, irstd]


def draw_merged(out_path):
    """1×3 subplot 布局，每个 panel 对应一个数据集。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.28, left=0.06, right=0.97, top=0.87, bottom=0.14)

    for i, ds in enumerate(DATASETS):
        _draw_panel(axes[i], ds,
                    show_ylabel=(i == 0),
                    show_legend=(i == 0))

    # 共享图例（方法颜色，顶部居中）
    method_handles = [
        mpatches.Patch(facecolor=row[4], edgecolor='#555555', linewidth=0.8,
                       label=row[0].replace('\n', ' '))
        for row in BAR_METHODS
    ]
    hline_handles = [
        plt.Line2D([0], [0], color=color, ls=ls, lw=lw, label=label)
        for label, *_, color, ls, lw in HLINES
    ]
    fig.legend(handles=method_handles + hline_handles,
               loc='upper center', ncol=len(method_handles) + len(hline_handles),
               fontsize=8.5, framealpha=0.9, edgecolor='#AAAAAA',
               bbox_to_anchor=(0.5, 1.00))

    fig.suptitle('Main Results: mIoU Comparison  (DNANet backbone)',
                 fontsize=12, fontweight='bold', y=1.05)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_path}")
    plt.close()


def _draw_panel(ax, ds_name, show_ylabel=True, show_legend=False):
    """单数据集柱状图 panel，每方法一根柱。"""
    di = DS_IDX[ds_name]
    vals = [row[1 + di] for row in BAR_METHODS]
    labels = [row[0] for row in BAR_METHODS]
    colors = [row[4] for row in BAR_METHODS]
    groups = [row[6] for row in BAR_METHODS]

    x = np.arange(len(BAR_METHODS))
    width = 0.55

    ax.set_axisbelow(True)
    ax.grid(True, axis='y', ls='--', lw=0.5, alpha=0.4, color='#CCCCCC')

    # 柱
    for i, (v, c, g) in enumerate(zip(vals, colors, groups)):
        ec = '#333333' if g == 'ours' else '#666666'
        ax.bar(i, v, width, color=c, edgecolor=ec, linewidth=0.8, zorder=3)
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom',
                fontsize=7.5, color='#222222', fontweight='bold' if g == 'ours' else 'normal')

    # 参考线（全跨度）
    hline_handles = []
    for label, nuaa, nudt, irstd, color, ls, lw in HLINES:
        v = [nuaa, nudt, irstd][di]
        ax.axhline(v, color=color, ls=ls, lw=lw, zorder=4, alpha=0.85)
        ax.text(len(BAR_METHODS) - 0.45, v + 0.3, label.split('(')[0].strip(),
                fontsize=6.5, color=color, va='bottom', ha='right')
        hline_handles.append(plt.Line2D([0], [0], color=color, ls=ls, lw=lw, label=label))

    # 方法分组背景色
    ax.axvspan(-0.5, 1.5, alpha=0.05, color='#F44336', zorder=0)
    ax.axvspan(1.5, len(BAR_METHODS) - 0.5, alpha=0.05, color='#7B1FA2', zorder=0)

    # 轴
    subtitle_map = {
        'NUAA-SIRST': '(a) NUAA-SIRST  (~30 px²)',
        'NUDT-SIRST': '(b) NUDT-SIRST  (<9 px²)',
        'IRSTD-1k':   '(c) IRSTD-1k  (~15 px²)',
    }
    ax.set_title(subtitle_map[ds_name], fontsize=10.5, fontweight='bold', pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel('mIoU (%)', fontsize=10)
    ax.set_ylim(35, 103)

    if show_legend:
        ax.legend(handles=hline_handles, fontsize=7.5, framealpha=0.9,
                  edgecolor='#AAAAAA', loc='upper left')

    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_edgecolor('#888888')


def draw_split(out_dir, base, ext):
    os.makedirs(out_dir, exist_ok=True)
    for i, ds in enumerate(DATASETS):
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(left=0.13, right=0.96, top=0.88, bottom=0.16)
        _draw_panel(ax, ds, show_ylabel=True, show_legend=(i == 0))
        out_path = os.path.join(out_dir, f'{base}_panel_{chr(ord("a")+i)}{ext}')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved → {out_path}")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(
        ROOT, 'paper', 'figures', 'fig_miou_comparison.png'))
    parser.add_argument('--split', action='store_true')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out) or '.'
    base    = os.path.splitext(os.path.basename(args.out))[0]
    ext     = os.path.splitext(args.out)[1] or '.png'

    if args.split:
        draw_split(out_dir, base, ext)
    else:
        draw_merged(args.out)
