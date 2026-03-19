#!/usr/bin/env python3
"""
Figure: Detection Operating Point Comparison (Pd vs Fa)
=========================================================
单阈值（threshold=0.5）下各方法在三数据集上的 Pd/Fa 散点图
  颜色 → 数据集（NUAA / NUDT / IRSTD）
  形状 → 方法（LESPS / PAL / HALO-DNA / HALO-ACM / HALO-ALCNet）
X 轴（对数）: False Alarm Rate (Fa)
Y 轴: Probability of Detection Pd (%)

数据来源: ADVISOR_REPORT.md § Pd/Fa 对比（threshold=0.5, eval_pdfa_fixed.py）
注: LESPS/PAL 数据来自各自论文，评估协议与本文一致（threshold=0.5）

用法:
    python paper/fig_operating_point.py
    python paper/fig_operating_point.py --out paper/figures/fig_opdpoint.pdf
    python paper/fig_operating_point.py --split   # 输出三数据集独立 panel
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
import matplotlib.lines as mlines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ────────────────────────────────────────────────────────────────
# 数据（来自 ADVISOR_REPORT.md § Pd/Fa 对比）
# ────────────────────────────────────────────────────────────────

DATA = {
    'NUAA-SIRST': [
        dict(method='LESPS',       backbone='DNANet', Pd=93.16, Fa=1.19e-5),
        dict(method='PAL',         backbone='DNANet', Pd=90.49, Fa=2.74e-5),
        dict(method='HALO (Ours)', backbone='DNANet', Pd=99.02, Fa=8.42e-6),
        dict(method='HALO (Ours)', backbone='ACM',    Pd=98.04, Fa=4.77e-7),
        dict(method='HALO (Ours)', backbone='ALCNet', Pd=96.08, Fa=1.43e-6),
    ],
    'NUDT-SIRST': [
        dict(method='LESPS',       backbone='DNANet', Pd=93.76, Fa=2.80e-5),
        dict(method='PAL',         backbone='DNANet', Pd=98.10, Fa=2.21e-5),
        dict(method='HALO (Ours)', backbone='DNANet', Pd=97.14, Fa=1.61e-5),
        dict(method='HALO (Ours)', backbone='ACM',    Pd=93.12, Fa=3.75e-5),
        dict(method='HALO (Ours)', backbone='ALCNet', Pd=94.50, Fa=3.29e-5),
    ],
    'IRSTD-1k': [
        dict(method='LESPS',       backbone='DNANet', Pd=87.54, Fa=1.51e-5),
        dict(method='PAL',         backbone='DNANet', Pd=91.25, Fa=2.45e-5),
        dict(method='HALO (Ours)', backbone='DNANet', Pd=90.14, Fa=2.56e-5),
        dict(method='HALO (Ours)', backbone='ACM',    Pd=83.33, Fa=3.32e-5),
        dict(method='HALO (Ours)', backbone='ALCNet', Pd=87.76, Fa=3.67e-5),
    ],
}

# 数据集 → 颜色（edge / face）
DS_COLOR = {
    'NUAA-SIRST': ('#1565C0', '#90CAF9'),   # 蓝
    'NUDT-SIRST': ('#2E7D32', '#A5D6A7'),   # 绿
    'IRSTD-1k':   ('#E65100', '#FFCC80'),   # 橙
}

# 方法 × 骨干 → 形状 / 尺寸
METHOD_MARKER = {
    ('LESPS',       'DNANet'): dict(marker='^', ms=9,  mew=1.4, zorder=3),
    ('PAL',         'DNANet'): dict(marker='v', ms=9,  mew=1.4, zorder=3),
    ('HALO (Ours)', 'DNANet'): dict(marker='*', ms=15, mew=1.1, zorder=5),
    ('HALO (Ours)', 'ACM'):    dict(marker='D', ms=8,  mew=1.4, zorder=5),
    ('HALO (Ours)', 'ALCNet'): dict(marker='o', ms=8,  mew=1.4, zorder=5),
}

# 需要单独标注的关键点（数据集, 方法, 骨干, label_text, dx_fac, dy）
KEY_LABELS = [
    ('NUAA-SIRST', 'HALO (Ours)', 'ACM',
     'Best $F_a$\n(NUAA-ACM)', 0.25, 0.0),
    ('NUAA-SIRST', 'HALO (Ours)', 'DNANet',
     'Best $P_d$\n(NUAA-DNA)', 1.5, 0.5),
    ('NUDT-SIRST', 'PAL', 'DNANet',
     'PAL (NUDT)', 1.4, 0.4),
]


def _find(ds, method, backbone):
    for pt in DATA[ds]:
        if pt['method'] == method and pt['backbone'] == backbone:
            return pt
    return None


def draw_merged(out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.10, right=0.72, top=0.90, bottom=0.13)

    ax.set_axisbelow(True)
    ax.grid(True, ls='--', lw=0.5, alpha=0.4, color='#CCCCCC')
    ax.set_xscale('log')

    # ── 绘制所有点 ──────────────────────────────────────────
    for ds, pts in DATA.items():
        ec, fc = DS_COLOR[ds]
        for pt in pts:
            key = (pt['method'], pt['backbone'])
            mk = METHOD_MARKER[key]
            ax.plot(pt['Fa'], pt['Pd'],
                    marker=mk['marker'], markersize=mk['ms'],
                    markeredgecolor=ec, markerfacecolor=fc,
                    markeredgewidth=mk['mew'],
                    linestyle='None', zorder=mk['zorder'])

    # ── 关键点标注 ──────────────────────────────────────────
    for ds, method, backbone, label, dx_fac, dy in KEY_LABELS:
        pt = _find(ds, method, backbone)
        if pt is None:
            continue
        ec, _ = DS_COLOR[ds]
        ax.annotate(label,
                    xy=(pt['Fa'], pt['Pd']),
                    xytext=(pt['Fa'] * dx_fac, pt['Pd'] + dy),
                    fontsize=7, color=ec,
                    arrowprops=dict(arrowstyle='->', color=ec,
                                    lw=0.7, shrinkA=3, shrinkB=3),
                    ha='left' if dx_fac > 1 else 'right',
                    va='center')

    # ── 轴设置 ──────────────────────────────────────────────
    all_fa = [pt['Fa'] for pts in DATA.values() for pt in pts]
    all_pd = [pt['Pd'] for pts in DATA.values() for pt in pts]
    ax.set_xlim(min(all_fa) * 0.03, max(all_fa) * 6)
    ax.set_ylim(min(all_pd) - 4, min(max(all_pd) + 2, 101))

    ax.set_xlabel(r'False Alarm Rate $F_a$', fontsize=11)
    ax.set_ylabel(r'Probability of Detection $P_d$ (%)', fontsize=11)
    ax.set_title(
        r'Operating Point Comparison: $P_d$ vs $F_a$ (threshold = 0.5)',
        fontsize=11, fontweight='bold', pad=8)

    # "Better" 方向提示
    ax.text(0.02, 0.97,
            r'$\leftarrow$ Low $F_a$  /  High $P_d$ $\uparrow$' + '\n(top-left = better)',
            fontsize=7.5, color='#455A64', va='top', ha='left',
            transform=ax.transAxes)

    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_edgecolor('#888888')

    # ── 图例（两列：方法 | 数据集）────────────────────────────
    # 方法图例（形状，灰色填充）
    method_entries = [
        mlines.Line2D([], [], marker='^', ms=8,  linestyle='None',
                      color='#555555', mfc='#CCCCCC', mew=1.3,
                      label='LESPS (DNANet)'),
        mlines.Line2D([], [], marker='v', ms=8,  linestyle='None',
                      color='#555555', mfc='#CCCCCC', mew=1.3,
                      label='PAL (DNANet)'),
        mlines.Line2D([], [], marker='*', ms=13, linestyle='None',
                      color='#555555', mfc='#CCCCCC', mew=1.0,
                      label='HALO + DNANet (Ours)'),
        mlines.Line2D([], [], marker='D', ms=7,  linestyle='None',
                      color='#555555', mfc='#CCCCCC', mew=1.3,
                      label='HALO + ACM (Ours)'),
        mlines.Line2D([], [], marker='o', ms=7,  linestyle='None',
                      color='#555555', mfc='#CCCCCC', mew=1.3,
                      label='HALO + ALCNet (Ours)'),
    ]
    # 数据集图例（颜色，圆形标记）
    ds_entries = [
        mlines.Line2D([], [], marker='o', ms=9, linestyle='None',
                      color=ec, mfc=fc, mew=1.5,
                      label=ds)
        for ds, (ec, fc) in DS_COLOR.items()
    ]

    leg1 = ax.legend(handles=method_entries, title='Method',
                     title_fontsize=8.5,
                     loc='upper left', bbox_to_anchor=(1.02, 1.00),
                     fontsize=8, framealpha=0.9, edgecolor='#AAAAAA',
                     borderpad=0.8)
    ax.add_artist(leg1)
    ax.legend(handles=ds_entries, title='Dataset',
              title_fontsize=8.5,
              loc='upper left', bbox_to_anchor=(1.02, 0.40),
              fontsize=8, framealpha=0.9, edgecolor='#AAAAAA',
              borderpad=0.8)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_path}")
    plt.close()


# ── 保留原 split 输出（每数据集独立 panel）────────────────────

SUBTITLES = {
    'NUAA-SIRST': '(a) NUAA-SIRST  (~30 px²)',
    'NUDT-SIRST': '(b) NUDT-SIRST  (<9 px²)',
    'IRSTD-1k':   '(c) IRSTD-1k  (~15 px²)',
}

LABEL_OFFSET = {
    'NUAA-SIRST': {
        ('LESPS',       'DNANet'): (1.6, -0.5),
        ('PAL',         'DNANet'): (1.5,  0.4),
        ('HALO (Ours)', 'DNANet'): (1.3,  0.5),
        ('HALO (Ours)', 'ACM'):    (0.25, 0.0),
        ('HALO (Ours)', 'ALCNet'): (1.4,  0.5),
    },
    'NUDT-SIRST': {
        ('LESPS',       'DNANet'): (1.4, -0.7),
        ('PAL',         'DNANet'): (1.3,  0.5),
        ('HALO (Ours)', 'DNANet'): (1.3,  0.5),
        ('HALO (Ours)', 'ACM'):    (1.2, -0.8),
        ('HALO (Ours)', 'ALCNet'): (1.4,  0.6),
    },
    'IRSTD-1k': {
        ('LESPS',       'DNANet'): (1.4, -0.7),
        ('PAL',         'DNANet'): (1.3,  0.5),
        ('HALO (Ours)', 'DNANet'): (1.5,  0.5),
        ('HALO (Ours)', 'ACM'):    (1.2, -0.8),
        ('HALO (Ours)', 'ALCNet'): (1.4,  0.6),
    },
}
BACKBONE_SHORT = {'DNANet': 'DNA', 'ACM': 'ACM', 'ALCNet': 'ALC'}


def _draw_panel(ax, ds_name, show_ylabel=True, show_legend=False):
    ax.set_axisbelow(True)
    ax.grid(True, ls='--', lw=0.5, alpha=0.45, color='#CCCCCC')
    ax.set_xscale('log')

    ec_ds, fc_ds = DS_COLOR[ds_name]
    offsets = LABEL_OFFSET[ds_name]

    for pt in DATA[ds_name]:
        key = (pt['method'], pt['backbone'])
        mk = METHOD_MARKER[key]
        ax.plot(pt['Fa'], pt['Pd'],
                marker=mk['marker'], markersize=mk['ms'],
                markeredgecolor=ec_ds, markerfacecolor=fc_ds,
                markeredgewidth=mk['mew'],
                linestyle='None', zorder=mk['zorder'])

        dx_factor, dy = offsets.get(key, (1.3, 0.4))
        short = BACKBONE_SHORT[pt['backbone']]
        name = 'LESPS' if pt['method'] == 'LESPS' else \
               'PAL'   if pt['method'] == 'PAL'   else f"HALO\n({short})"
        ha = 'right' if dx_factor < 1 else 'left'
        ax.annotate(name,
                    xy=(pt['Fa'], pt['Pd']),
                    xytext=(pt['Fa'] * dx_factor, pt['Pd'] + dy),
                    fontsize=6.8, color=ec_ds,
                    arrowprops=dict(arrowstyle='-', color=ec_ds,
                                    lw=0.5, shrinkA=2, shrinkB=2),
                    ha=ha, va='center')

    ax.text(0.03, 0.97, 'Better →\n(high Pd, low Fa)',
            fontsize=7, color='#455A64', ha='left', va='top',
            transform=ax.transAxes)

    ax.set_title(SUBTITLES[ds_name], fontsize=10.5, fontweight='bold', pad=6)
    ax.set_xlabel(r'False Alarm Rate $F_a$', fontsize=10)
    if show_ylabel:
        ax.set_ylabel(r'$P_d$ (%)', fontsize=10)

    all_pd = [pt['Pd'] for pt in DATA[ds_name]]
    all_fa = [pt['Fa'] for pt in DATA[ds_name]]
    ax.set_ylim(min(all_pd) - 4, min(max(all_pd) + 2, 101))
    ax.set_xlim(min(all_fa) * 0.05, max(all_fa) * 5)

    if show_legend:
        handles = [
            mlines.Line2D([], [], marker=mk['marker'], ms=mk['ms'] * 0.8,
                          color=ec_ds, mfc=fc_ds, mew=mk['mew'],
                          linestyle='None',
                          label=f"{k[0]} ({k[1]})")
            for k, mk in METHOD_MARKER.items()
        ]
        ax.legend(handles=handles, fontsize=7.5, framealpha=0.9,
                  edgecolor='#AAAAAA', loc='lower left')

    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_edgecolor('#888888')


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
        ROOT, 'paper', 'figures', 'fig_operating_point.png'))
    parser.add_argument('--split', action='store_true')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out) or '.'
    base    = os.path.splitext(os.path.basename(args.out))[0]
    ext     = os.path.splitext(args.out)[1] or '.png'

    if args.split:
        draw_split(out_dir, base, ext)
    else:
        draw_merged(args.out)
