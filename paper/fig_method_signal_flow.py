#!/usr/bin/env python3
"""
Figure: HALO Method Signal Flow Diagram — TIP 严谨信号处理风格
==============================================================
拒绝卡通化，回归信号处理方块图 (Block Diagram)：

  [IR Image + YOLO Box]
        ↓
  [Module 1: Context Statistical Estimation]
    • Box 膨胀图示（R=1.5）
    • 计算 μ_ctx, σ_ctx（Context Ring 上）
        ↓ (μ_ctx, σ_ctx)
  [Module 2: B-SNR Posterior W(i)]
    • W(i) = σ(τ·(I(i)−μ_ctx)/σ_ctx)  [Sigmoid 非线性激活]
    • 仅在框内有效
        ↓ W(i)
  [Module 3: PAG Generation]
    • p* = argmax W(i)            (物理热点)
    • σ_G via FWHM估计            (散布估计)
    • P(i) = exp(−‖r−p*‖²/2σ_G²) (高斯软标签)
        ↓ P(i) [offline pseudo-label]
  [Backbone + Seg Head]
    • 零额外参数，纯物理离线生成

设计要点:
  - 黑白 + 少量强调色（深蓝/橙/紫）
  - 方块间用带箭头的流线连接
  - 关键公式直接嵌入方块内（LaTeX 渲染）
  - 右侧配 3 张缩略图示意: IR图/B-SNR权重/HALO软标签

用法:
    python paper/fig_method_signal_flow.py
    python paper/fig_method_signal_flow.py --out paper/figures/fig_method_flow.pdf
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ────────────────────────────────────────────────────────────────
# 调色板
# ────────────────────────────────────────────────────────────────
C_INPUT   = '#1A237E'   # 深海蓝 — 输入
C_MOD1    = '#0277BD'   # 蓝     — 统计估计
C_MOD2    = '#E65100'   # 橙     — B-SNR
C_MOD3    = '#4A148C'   # 深紫   — PAG
C_OUTPUT  = '#1B5E20'   # 深绿   — 输出/训练
C_ARROW   = '#37474F'   # 深灰   — 流线
C_ANNOT   = '#546E7A'   # 中灰   — 注释文字
C_BG      = 'white'

# ────────────────────────────────────────────────────────────────
# 绘图辅助
# ────────────────────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, color, text, fontsize=9.5,
                text_color='white', alpha=1.0, lw=1.2,
                text_pad=0.0, secondary_text=None, sec_fontsize=8.0):
    """绘制圆角方块，支持主文字 + 小号副文字（公式）"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0.02',
                         facecolor=color, edgecolor='white',
                         linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    if secondary_text:
        ax.text(cx, cy + h * 0.16 + text_pad, text,
                ha='center', va='center', fontsize=fontsize,
                color=text_color, fontweight='bold', zorder=4)
        ax.text(cx, cy - h * 0.18 + text_pad, secondary_text,
                ha='center', va='center', fontsize=sec_fontsize,
                color=text_color, zorder=4, style='italic')
    else:
        ax.text(cx, cy + text_pad, text,
                ha='center', va='center', fontsize=fontsize,
                color=text_color, fontweight='bold', zorder=4)


def arrow(ax, x0, y0, x1, y1, label='', color=C_ARROW, lw=1.5):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.01, my, label, ha='left', va='center',
                fontsize=8, color=C_ANNOT, style='italic', zorder=5)


def side_label(ax, x, y, text, color=C_ANNOT):
    ax.text(x, y, text, ha='left', va='center', fontsize=7.5,
            color=color, style='italic', zorder=5)


def draw_small_image(ax, img, title, cmap='gray', border_color='#555'):
    """在指定子图绘制缩略图"""
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=8, pad=3, color='#333333', fontweight='bold')
    for sp in ax.spines.values():
        sp.set_edgecolor(border_color)
        sp.set_linewidth(1.5)
    ax.set_xticks([]); ax.set_yticks([])


def make_demo_images(size=48):
    """生成合成示意图用于右侧缩略图"""
    from scipy.ndimage import gaussian_filter as gf
    rng = np.random.default_rng(7)

    # IR 图
    bg = rng.normal(0.4, 0.08, (size, size)).astype(np.float32)
    bg = gf(bg, sigma=3)
    # 目标偏上方
    px, py = 30, 16
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    psf = 0.55 * np.exp(-((xx - px)**2 + (yy - py)**2) / (2 * 1.8**2))
    ir = (bg + psf).clip(0, 1)

    # YOLO Box (几何中心在 size//2)
    box = (size//2 - 8, size//2 - 8, size//2 + 8, size//2 + 8)
    x1, y1, x2, y2 = box

    # B-SNR W(i)
    ctx_ring_px = ir[(y1-5 if y1>=5 else 0):y2+5, (x1-5 if x1>=5 else 0):x2+5].ravel()
    mu_ctx = float(np.mean(ctx_ring_px))
    sig_ctx = float(np.std(ctx_ring_px)) + 1e-4
    bsnr = np.zeros((size, size), np.float32)
    snr_roi = (ir[y1:y2, x1:x2] - mu_ctx) / sig_ctx
    bsnr[y1:y2, x1:x2] = 1.0 / (1.0 + np.exp(-1.2 * snr_roi))
    bsnr_n = bsnr / (bsnr.max() + 1e-6)

    # HALO PAG
    roi = bsnr[y1:y2, x1:x2]
    flat_idx = int(np.argmax(roi))
    py_star = y1 + flat_idx // (x2-x1)
    px_star = x1 + flat_idx  % (x2-x1)
    sigma_g = 2.2
    halo = np.exp(-((xx - px_star)**2 + (yy - py_star)**2) / (2 * sigma_g**2))
    halo_in_box = np.zeros((size, size), np.float32)
    halo_in_box[y1:y2, x1:x2] = halo[y1:y2, x1:x2]
    halo_n = halo_in_box / (halo_in_box.max() + 1e-6)

    return ir, bsnr_n, halo_n, box, (px_star, py_star), (size//2, size//2)


# ────────────────────────────────────────────────────────────────
# 主绘图
# ────────────────────────────────────────────────────────────────

def draw(out_path):
    # 画布: 左侧流图 + 右侧 3 张缩略图
    fig = plt.figure(figsize=(16, 8.5))
    fig.patch.set_facecolor(C_BG)

    # GridSpec: 左侧流图占 2/3，右侧 3 张缩略图各占 1 行
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1.0],
                  left=0.03, right=0.97, top=0.91, bottom=0.06, wspace=0.08)

    # ── 左侧：纯 patch 画流图 ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor(C_BG)

    # ─────────────────────── 流图布局参数 ─────────────────────────
    bw  = 0.64   # 方块宽度
    bh  = 0.10   # 方块高度
    bx  = 0.18   # 方块左边 X
    gap = 0.04   # 方块间距

    # Y 坐标（从上到下）
    y_top = 0.87

    def bpos(n):
        """第 n 个方块的左下角 y"""
        return y_top - n * (bh + gap)

    # ── 方块 0: 输入 ────────────────────────────────────────────
    rounded_box(ax, bx, bpos(0), bw, bh, C_INPUT,
                r'Input:  IR Image  +  YOLO Bounding Box  $\mathbf{b}=(x_1,y_1,x_2,y_2)$',
                fontsize=9.5)

    # ── 方块 1: 统计估计 ─────────────────────────────────────────
    arrow(ax, bx + bw/2, bpos(0), bx + bw/2, bpos(1) + bh,
          color=C_ARROW)
    rounded_box(ax, bx, bpos(1), bw, bh, C_MOD1,
                'Module 1 — Context Statistical Estimation',
                secondary_text=(r'Context Ring: $\mathbf{b}$ expanded by $R=1.5$  →  '
                                r'$\mu_{\mathrm{ctx}},\;\sigma_{\mathrm{ctx}}$'),
                fontsize=9.5, sec_fontsize=8.5)

    # 侧注：膨胀示意
    side_label(ax, bx + bw + 0.02, bpos(1) + bh * 0.5,
               'Ring pixels only\n(excludes box interior)',
               color=C_MOD1)

    # ── 方块 2: B-SNR ─────────────────────────────────────────
    arrow(ax, bx + bw/2, bpos(1), bx + bw/2, bpos(2) + bh,
          label=r'$\mu_{\mathrm{ctx}},\;\sigma_{\mathrm{ctx}}$', color=C_ARROW)
    rounded_box(ax, bx, bpos(2), bw, bh, C_MOD2,
                'Module 2 — B-SNR Posterior  (No Learnable Parameters)',
                secondary_text=(r'$W(i) = \sigma\!\left(\tau \cdot '
                                r'\dfrac{I(i)-\mu_{\mathrm{ctx}}}{\sigma_{\mathrm{ctx}}}\right)'
                                r'\quad \forall\, i \in \mathbf{b}$'),
                fontsize=9.5, sec_fontsize=9.0)

    # 侧注：Sigmoid 非线性
    side_label(ax, bx + bw + 0.02, bpos(2) + bh * 0.5,
               'Sigmoid suppresses\nbackground clutter',
               color=C_MOD2)

    # ── 方块 3: PAG ──────────────────────────────────────────────
    arrow(ax, bx + bw/2, bpos(2), bx + bw/2, bpos(3) + bh,
          label=r'$W(i)$', color=C_ARROW)
    rounded_box(ax, bx, bpos(3), bw, bh * 1.25, C_MOD3,
                'Module 3 — PAG Generation  (Physics-Anchored Gaussian)',
                secondary_text=(r'$p^* = \arg\max_{i \in \mathbf{b}} W(i)$'
                                r'     $\sigma_G \leftarrow \mathrm{FWHM}(W)$     '
                                r'$P(i) = \exp\!\left(-\dfrac{\|r_i - p^*\|^2}{2\sigma_G^2}\right)$'),
                fontsize=9.5, sec_fontsize=8.8)

    # 侧注：物理热点
    side_label(ax, bx + bw + 0.02, bpos(3) + bh * 0.7,
               r'$p^*$: Physical Anchor' '\n' r'(≠ geometric center)',
               color=C_MOD3)

    # ── 方块 4: 输出 / 训练 ──────────────────────────────────────
    y_out = bpos(3) - gap - 0.08
    arrow(ax, bx + bw/2, bpos(3), bx + bw/2, y_out + 0.08,
          label=r'$P(i)$  [offline pseudo-label]', color=C_ARROW)
    rounded_box(ax, bx, y_out, bw, 0.08, C_OUTPUT,
                'Backbone + Segmentation Head  (trained with $P(i)$  as soft supervision)',
                fontsize=9.0)

    # ── 零参数标注旗标 ───────────────────────────────────────────
    flag_x = bx - 0.02
    flag_y = (bpos(1) + bpos(3)) / 2 + bh
    ax.annotate('Zero extra\nparameters',
                xy=(bx, bpos(2) + bh/2),
                xytext=(flag_x - 0.07, flag_y),
                fontsize=8, color='#D84315', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#D84315', lw=1.0),
                bbox=dict(boxstyle='round,pad=0.25', fc='#FFF3E0', ec='#D84315', lw=0.8),
                ha='center', va='center')

    # ── 方差污染警示标注 ─────────────────────────────────────────
    warn_txt = (r'$R < 1.3$: Context Ring contaminated' '\n'
                r'$\hat{\sigma}^2_{\mathrm{ctx}} \gg \sigma^2_0$  (Var. Contamination)')
    ax.text(bx - 0.14, bpos(1) + bh * 0.5, warn_txt,
            ha='center', va='center', fontsize=7.5,
            color='#B71C1C', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec='#C62828', lw=0.7))

    # ── 图例：颜色对应模块 ──────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=C_INPUT,  label='Input / YOLO Annotation'),
        mpatches.Patch(color=C_MOD1,   label='Module 1: Context Stat. Estimation'),
        mpatches.Patch(color=C_MOD2,   label='Module 2: B-SNR Posterior W(i)'),
        mpatches.Patch(color=C_MOD3,   label='Module 3: PAG Generation P(i)'),
        mpatches.Patch(color=C_OUTPUT, label='Downstream Backbone Training'),
    ]
    ax.legend(handles=legend_patches, loc='lower left', fontsize=7.5,
              framealpha=0.9, edgecolor='#CCCCCC', ncol=1,
              bbox_to_anchor=(0.0, 0.0))

    # ─────────────────────── 右侧缩略图 ──────────────────────────
    gs_right = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1],
                                       hspace=0.45)

    ir_img, bsnr_img, halo_img, box_demo, p_star, geo_c = make_demo_images(size=48)
    size = 48

    ax_ir   = fig.add_subplot(gs_right[0])
    ax_bsnr = fig.add_subplot(gs_right[1])
    ax_halo = fig.add_subplot(gs_right[2])

    # IR 图 + YOLO box + 几何中心
    draw_small_image(ax_ir, ir_img, '① IR Image + YOLO Box', cmap='gray',
                     border_color=C_INPUT)
    x1, y1, x2, y2 = box_demo
    from matplotlib.patches import Rectangle
    rect = Rectangle((x1-0.5, y1-0.5), x2-x1, y2-y1,
                     edgecolor='yellow', facecolor='none', lw=1.5, ls='--')
    ax_ir.add_patch(rect)
    ax_ir.plot(*geo_c, 'w+', markersize=9, markeredgewidth=1.5)
    ax_ir.text(geo_c[0]+1, geo_c[1]-1, 'Geo. center', fontsize=6,
               color='white', va='bottom')

    # B-SNR 图 + 物理热点
    draw_small_image(ax_bsnr, bsnr_img, '② B-SNR  $W(i)$', cmap='inferno',
                     border_color=C_MOD2)
    ax_bsnr.plot(*p_star, 'r*', markersize=11, markeredgewidth=0.5,
                 markeredgecolor='white')
    ax_bsnr.text(p_star[0]+1, p_star[1]-1, r'$p^*$', fontsize=8,
                 color='#FF8F00', va='bottom', fontweight='bold')

    # HALO PAG
    draw_small_image(ax_halo, halo_img, '③ HALO  $P(i)$', cmap='magma',
                     border_color=C_MOD3)
    ax_halo.plot(*p_star, 'r*', markersize=11, markeredgewidth=0.5,
                 markeredgecolor='white')

    # 缩略图标注横线（指向对应方块）
    # (视觉装饰，用 fig.transFigure 连线太复杂，改用文字说明)
    for ax_t, col, note in [(ax_ir,   C_INPUT,  'Input'),
                             (ax_bsnr, C_MOD2,   'Mod 2'),
                             (ax_halo, C_MOD3,   'Mod 3')]:
        ax_t.set_xlabel(f'→ {note}', fontsize=7, color=col, labelpad=2)

    # ── 全图标题 ─────────────────────────────────────────────────
    fig.suptitle(
        'HALO: Heat-Anchored Label Optimization — Signal Processing Flow',
        fontsize=13, fontweight='bold', color='#1A1A1A', y=0.975,
    )

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=C_BG)
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(
        ROOT, 'paper', 'figures', 'fig_method_flow.png'))
    args = parser.parse_args()
    draw(args.out)
