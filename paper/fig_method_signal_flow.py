#!/usr/bin/env python3
"""
Figure: HALO Method Overview — TIP-Quality Three-Panel Framework Figure
========================================================================
Layout (inspired by Nature MI / EqGPT style):

  (a) Input & Context          (b) Physics-Anchored Processing         (c) Downstream Training
  ─────────────────────        ──────────────────────────────────────   ─────────────────────────
  IR Image + YOLO Box          B-SNR Posterior  │  PAG Generation       Any IRSTD Backbone
  Context Ring Expansion       W(i) formula     │  P(i) formula         Soft Supervision Loss
  μ_ctx, σ_ctx extraction      B-SNR heatmap    │  Soft label image     Output visualization

  Thick arrows: A ──→ B ──→ C   (high-level logic flow)
  "Zero-Parameter / Training-Free" badge prominently on panel b

Usage:
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
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Morandi-inspired color palette (low saturation, professional)
# ─────────────────────────────────────────────────────────────────────────────
C_PA_BG   = '#E8F4FD'   # Panel a bg: pale blue
C_PA_BRD  = '#1565C0'   # Panel a border: deep blue
C_PA_HDR  = '#0D47A1'   # Panel a header text
C_PB_BG   = '#FFF8E1'   # Panel b bg: pale amber
C_PB_BRD  = '#BF360C'   # Panel b border: burnt orange
C_PB_HDR  = '#BF360C'
C_PC_BG   = '#F1F8E9'   # Panel c bg: pale green
C_PC_BRD  = '#2E7D32'   # Panel c border: deep green
C_PC_HDR  = '#1B5E20'
C_BSNR    = '#D84315'   # B-SNR orange
C_PAG     = '#4527A0'   # PAG indigo
C_FBOX_A  = '#1565C0'   # formula box panel a
C_FBOX_B1 = '#BF360C'   # formula box B-SNR
C_FBOX_B2 = '#311B92'   # formula box PAG
C_FBOX_C  = '#1B5E20'   # formula box panel c
C_BADGE   = '#B71C1C'   # zero-param badge
C_ARROW_AB = '#1565C0'
C_ARROW_BC = '#4527A0'
C_ARROW_IN = '#607D8B'  # intra-panel arrows


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic demo images
# ─────────────────────────────────────────────────────────────────────────────

def make_demo_images(size=80):
    rng = np.random.default_rng(42)

    # Smooth IR background
    bg = rng.normal(0.32, 0.065, (size, size)).astype(np.float32)
    bg = gaussian_filter(bg, sigma=5.0)
    xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    bg += 0.04 * np.sin(xx * 10) * np.cos(yy * 7)

    # Bright point target (slightly off-center)
    tx, ty = size * 0.41, size * 0.37
    psf = 0.72 * np.exp(-((xx * size - tx)**2 + (yy * size - ty)**2) / (2 * 1.7**2))
    ir = (bg + psf).clip(0, 1)

    # YOLO bounding box (loose, centered)
    bx1 = int(size * 0.28); by1 = int(size * 0.25)
    bx2 = int(size * 0.56); by2 = int(size * 0.53)

    # Context ring (R≈1.5)
    mg  = max(4, int((bx2 - bx1) * 0.28))
    cx1 = max(0, bx1 - mg); cy1 = max(0, by1 - mg)
    cx2 = min(size, bx2 + mg); cy2 = min(size, by2 + mg)

    # Ring pixels → μ_ctx, σ_ctx
    ctx_m = np.zeros((size, size), bool)
    ctx_m[cy1:cy2, cx1:cx2] = True
    inner = np.zeros((size, size), bool)
    inner[by1:by2, bx1:bx2] = True
    ring_px = ir[ctx_m & ~inner]
    mu_ctx  = float(np.mean(ring_px))
    sg_ctx  = float(np.std(ring_px)) + 1e-4

    # B-SNR weight map
    tau  = 1.6
    bsnr = np.zeros((size, size), np.float32)
    bsnr[by1:by2, bx1:bx2] = 1.0 / (1.0 + np.exp(
        -tau * (ir[by1:by2, bx1:bx2] - mu_ctx) / sg_ctx))
    bsnr /= bsnr.max() + 1e-6

    # p* = argmax W
    roi = bsnr[by1:by2, bx1:bx2]
    flat = int(np.argmax(roi))
    py_s = by1 + flat // (bx2 - bx1)
    px_s = bx1 + flat  % (bx2 - bx1)

    # PAG Gaussian
    sg_g = (bx2 - bx1) * 0.26
    pag  = np.zeros((size, size), np.float32)
    pag[by1:by2, bx1:bx2] = np.exp(
        -((xx[by1:by2, bx1:bx2] * size - px_s)**2 +
          (yy[by1:by2, bx1:bx2] * size - py_s)**2) / (2 * sg_g**2))
    pag /= pag.max() + 1e-6

    geo_c = ((bx1 + bx2) / 2, (by1 + by2) / 2)
    return (ir, bsnr, pag,
            (bx1, by1, bx2, by2),
            (cx1, cy1, cx2, cy2),
            (px_s, py_s), geo_c)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def bg_panel(ax, x, y, w, h, fc, ec, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.010',
                                facecolor=fc, edgecolor=ec,
                                linewidth=lw, alpha=0.55, zorder=1))


def fbox(ax, cx, cy, w, h, color, lines, fsizes=None):
    """Dark formula box centered at (cx, cy); lines = list of strings."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle='round,pad=0.010',
                                facecolor=color, edgecolor='white',
                                linewidth=1.2, alpha=0.93, zorder=4))
    n = len(lines)
    if fsizes is None:
        fsizes = [9.5] * n
    for k, (txt, fs) in enumerate(zip(lines, fsizes)):
        yoff = (n - 1 - 2 * k) / (2 * n) * h * 0.78
        ax.text(cx, cy + yoff, txt, ha='center', va='center',
                fontsize=fs, color='white', fontweight='bold', zorder=5)


def thick_arrow(ax, x0, y0, x1, y1, color, label='', lo=(0, 0.03)):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color,
                                lw=3.5, mutation_scale=24),
                zorder=3)
    if label:
        mx = (x0 + x1) / 2 + lo[0]
        my = (y0 + y1) / 2 + lo[1]
        ax.text(mx, my, label, ha='center', va='bottom',
                fontsize=8.5, color=color, style='italic', zorder=5,
                bbox=dict(boxstyle='round,pad=0.22', fc='white',
                          ec=color, lw=0.9, alpha=0.95))


def thin_arrow(ax, x0, y0, x1, y1, color=C_ARROW_IN, lw=1.8):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=14),
                zorder=3)


def img_ax(fig, bounds, xd, yd, wd, hd):
    """Add image axis; bounds = (left, bottom, width, height) in figure fraction."""
    l, b, w, h = bounds
    return fig.add_axes([l + xd * w, b + yd * h, wd * w, hd * h])


def style_img(ax, title, bcolor, tcolor):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(bcolor); sp.set_linewidth(2.2)
    ax.set_title(title, fontsize=8.5, pad=3, color=tcolor, fontweight='bold')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def draw(out_path):
    FIGW, FIGH = 18.5, 9.8
    fig = plt.figure(figsize=(FIGW, FIGH))
    fig.patch.set_facecolor('#F8F9FA')

    # ── Synthetic images ─────────────────────────────────────────
    ir, bsnr, pag, box, ctx, p_star, geo_c = make_demo_images(size=80)
    bx1, by1, bx2, by2 = box
    cx1, cy1, cx2, cy2 = ctx

    # ── Main canvas (background + arrows + text) ─────────────────
    # Figure-fraction bounds of main axis
    ML, MB, MW, MH = 0.015, 0.04, 0.970, 0.890
    ax = fig.add_axes([ML, MB, MW, MH])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # Helper: data coords → figure coords
    def tf(xd, yd, wd, hd):
        return [ML + xd * MW, MB + yd * MH, wd * MW, hd * MH]

    # ─────────────────────────────────────────────────────────────
    # Panel geometry (data coords)
    # ─────────────────────────────────────────────────────────────
    PAD = 0.008
    GAP = 0.018

    PA = dict(x=PAD,               w=0.228)
    PB = dict(x=PAD+0.228+GAP,     w=0.460)
    PC = dict(x=PAD+0.228+GAP+0.460+GAP, w=1.0 - (PAD+0.228+GAP+0.460+GAP) - PAD)
    PY, PH = PAD, 1.0 - 2 * PAD

    # ── Background panels ─────────────────────────────────────────
    bg_panel(ax, PA['x'], PY, PA['w'], PH, C_PA_BG, C_PA_BRD)
    bg_panel(ax, PB['x'], PY, PB['w'], PH, C_PB_BG, C_PB_BRD)
    bg_panel(ax, PC['x'], PY, PC['w'], PH, C_PC_BG, C_PC_BRD)

    # ── Panel labels ──────────────────────────────────────────────
    for P, lbl, col in [(PA, '(a)', C_PA_HDR), (PB, '(b)', C_PB_HDR), (PC, '(c)', C_PC_HDR)]:
        ax.text(P['x'] + 0.010, PY + PH - 0.012, lbl,
                fontsize=16, fontweight='bold', color=col,
                va='top', zorder=6)

    # ── Panel sub-titles ──────────────────────────────────────────
    ax.text(PA['x'] + PA['w']/2, PY + PH - 0.055,
            'Input & Context\nConstruction',
            ha='center', va='top', fontsize=11, color=C_PA_HDR,
            fontweight='bold', zorder=5)

    ax.text(PB['x'] + PB['w']/2, PY + PH - 0.055,
            'Physics-Anchored Signal Processing',
            ha='center', va='top', fontsize=11, color=C_PB_HDR,
            fontweight='bold', zorder=5)

    ax.text(PC['x'] + PC['w']/2, PY + PH - 0.055,
            'Downstream\nTraining',
            ha='center', va='top', fontsize=11, color=C_PC_HDR,
            fontweight='bold', zorder=5)

    # ─────────────────────────────────────────────────────────────
    # PANEL A  ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────
    pa_cx = PA['x'] + PA['w'] / 2

    # Formula box (bottom): Context Ring → μ_ctx, σ_ctx
    fbox(ax, pa_cx, 0.215, PA['w'] * 0.86, 0.125, C_FBOX_A,
         [r'Context Ring  ($R = 1.5$)',
          r'$\hat{\mu}_{\mathrm{ctx}},\;\hat{\sigma}_{\mathrm{ctx}}$  from ring pixels',
          r'(excludes box interior)'],
         fsizes=[9.0, 9.5, 8.0])

    # Contamination warning badge
    ax.text(pa_cx, 0.095,
            r'$R < 1.3$:  $\hat{\sigma}^2_{\mathrm{ctx}} \gg \sigma^2_0$' '\n'
            r'(Variance Contamination)',
            ha='center', va='center', fontsize=7.5,
            color='#B71C1C', style='italic', zorder=5,
            bbox=dict(boxstyle='round,pad=0.28', fc='#FFEBEE',
                      ec='#C62828', lw=0.9, alpha=0.9))

    # Thin arrow: image → formula box
    thin_arrow(ax, pa_cx, 0.505, pa_cx, 0.285, color=C_PA_BRD)

    # ─────────────────────────────────────────────────────────────
    # PANEL B  ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────

    # Zero-Parameter badge (top-right of panel b)
    bx_r = PB['x'] + PB['w'] - 0.008
    by_t = PY + PH - 0.012
    ax.add_patch(FancyBboxPatch((bx_r - 0.120, by_t - 0.068), 0.120, 0.060,
                                boxstyle='round,pad=0.008',
                                facecolor=C_BADGE, edgecolor='white',
                                linewidth=1.0, alpha=0.93, zorder=6))
    ax.text(bx_r - 0.060, by_t - 0.037,
            'Zero-Parameter\nTraining-Free',
            ha='center', va='center', fontsize=8.5,
            color='white', fontweight='bold', zorder=7)

    # Vertical divider between B-SNR and PAG sub-panels
    b_mid = PB['x'] + PB['w'] / 2
    ax.plot([b_mid, b_mid], [PY + 0.04, PY + PH - 0.14],
            ls='--', color='#BDBDBD', lw=1.2, zorder=2)

    # B-SNR sub-panel
    bsnr_cx = PB['x'] + PB['w'] * 0.255
    ax.text(bsnr_cx, PY + PH - 0.145,
            r'B-SNR Posterior  $W(i)$',
            ha='center', va='center', fontsize=10.5, color=C_BSNR,
            fontweight='bold', zorder=5)

    fbox(ax, bsnr_cx, 0.665, PB['w'] * 0.415, 0.145, C_FBOX_B1,
         [r'$W(i) = \sigma\!\left(\tau \cdot '
          r'\dfrac{I(i)-\hat{\mu}_{\mathrm{ctx}}}{\hat{\sigma}_{\mathrm{ctx}}}\right)$',
          r'$\forall\; i \in \mathbf{b}$   (No Learnable Params)'],
         fsizes=[10.5, 8.5])

    ax.text(bsnr_cx, 0.565,
            'Sigmoid suppresses background clutter',
            ha='center', va='center', fontsize=8.0,
            color=C_BSNR, style='italic', zorder=5)

    # PAG sub-panel
    pag_cx = PB['x'] + PB['w'] * 0.755
    ax.text(pag_cx, PY + PH - 0.145,
            r'PAG Generation  $P(i)$',
            ha='center', va='center', fontsize=10.5, color=C_PAG,
            fontweight='bold', zorder=5)

    fbox(ax, pag_cx, 0.665, PB['w'] * 0.415, 0.145, C_FBOX_B2,
         [r'$p^* = \arg\max_{i \in \mathbf{b}} W(i)$',
          r'$P(i) = \exp\!\left(-\dfrac{\|r_i - p^*\|^2}{2\sigma_G^2}\right)$'],
         fsizes=[10.0, 10.0])

    ax.text(pag_cx, 0.565,
            r'$p^*$: Physics Anchor  $(\neq$ geometric center$)$',
            ha='center', va='center', fontsize=8.0,
            color=C_PAG, style='italic', zorder=5)

    # B-SNR → PAG intra-panel arrow
    thin_arrow(ax, b_mid - 0.005, 0.665, b_mid + 0.005, 0.665,
               color=C_ARROW_IN, lw=2.2)

    # ─────────────────────────────────────────────────────────────
    # PANEL C  ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────
    pc_cx = PC['x'] + PC['w'] / 2

    # Plug-in annotation
    ax.text(pc_cx, PY + PH - 0.145,
            'Plug-in for Any Architecture',
            ha='center', va='center', fontsize=9.0,
            color=C_PC_HDR, style='italic', zorder=5)

    fbox(ax, pc_cx, 0.745, PC['w'] * 0.84, 0.115, C_FBOX_C,
         ['Any IRSTD Backbone',
          '(DNANet / ACM / ALCNet)'],
         fsizes=[10.0, 8.5])

    thin_arrow(ax, pc_cx, 0.685, pc_cx, 0.640, color=C_PC_HDR)

    fbox(ax, pc_cx, 0.590, PC['w'] * 0.84, 0.095, '#37474F',
         [r'$\mathcal{L} = \mathcal{L}_{\mathrm{Dice}}\!\left(f_\theta(x),\; P(i)\right)$',
          'Soft Supervision (offline label)'],
         fsizes=[10.0, 8.0])

    thin_arrow(ax, pc_cx, 0.540, pc_cx, 0.495, color='#37474F')

    # ─────────────────────────────────────────────────────────────
    # Inter-panel thick arrows: A ──→ B ──→ C
    # ─────────────────────────────────────────────────────────────
    arr_y = 0.50
    thick_arrow(ax,
                PA['x'] + PA['w'] + 0.003, arr_y,
                PB['x'] - 0.003,           arr_y,
                color=C_ARROW_AB,
                label=r'$\hat{\mu}_{\mathrm{ctx}},\;\hat{\sigma}_{\mathrm{ctx}}$')

    thick_arrow(ax,
                PB['x'] + PB['w'] + 0.003, arr_y,
                PC['x'] - 0.003,           arr_y,
                color=C_ARROW_BC,
                label=r'$P(i)$  offline pseudo-label')

    # ─────────────────────────────────────────────────────────────
    # Inset image axes
    # ─────────────────────────────────────────────────────────────
    IH = PH * 0.295   # image height in data coords
    IW_A = PA['w'] * 0.84
    IW_B = PB['w'] * 0.36
    IW_C = PC['w'] * 0.84

    img_y_top  = PY + PH * 0.530   # y-bottom for top images (panel A)
    img_y_bot  = PY + PH * 0.190   # y-bottom for bottom images (panels B, C)

    # ── Panel A: IR image with YOLO box + context ring ────────────
    ix_a = PA['x'] + (PA['w'] - IW_A) / 2
    ax_ir = img_ax(fig, (ML, MB, MW, MH), ix_a, img_y_top, IW_A, IH)
    ax_ir.imshow(ir, cmap='inferno', vmin=0, vmax=1, aspect='equal')
    # YOLO box (yellow dashed)
    ax_ir.add_patch(Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                               edgecolor='#FFD740', facecolor='none',
                               lw=2.2, ls='--'))
    # Context ring (cyan dotted)
    ax_ir.add_patch(Rectangle((cx1, cy1), cx2 - cx1, cy2 - cy1,
                               edgecolor='#80DEEA', facecolor='none',
                               lw=1.8, ls=':'))
    # Geometric center
    ax_ir.plot(*geo_c, 'w+', markersize=10, markeredgewidth=2.0)
    ax_ir.text(geo_c[0] + 1, geo_c[1] - 2, 'geo.c', fontsize=6.5,
               color='white', va='bottom')
    style_img(ax_ir,
              r'IR Image + YOLO Box  $\mathbf{b}$' + '\n'
              r'(cyan ring = context, $R=1.5$)',
              C_PA_BRD, C_PA_HDR)

    # ── Panel B left: B-SNR heatmap ───────────────────────────────
    bsnr_img_x = PB['x'] + PB['w'] * 0.045
    ax_bsnr = img_ax(fig, (ML, MB, MW, MH), bsnr_img_x, img_y_bot, IW_B, IH)
    ax_bsnr.imshow(bsnr, cmap='inferno', vmin=0, vmax=1, aspect='equal')
    ax_bsnr.plot(p_star[0], p_star[1], 'r*', markersize=11,
                 markeredgecolor='white', markeredgewidth=0.8)
    ax_bsnr.text(p_star[0] + 1.5, p_star[1] - 1.5, r'$p^*$',
                 fontsize=9, color='#FF8F00', fontweight='bold', va='bottom')
    style_img(ax_bsnr, r'B-SNR Map  $W(i)$', C_BSNR, C_BSNR)

    # ── Panel B right: PAG soft label ─────────────────────────────
    pag_img_x = PB['x'] + PB['w'] * 0.555
    ax_pag = img_ax(fig, (ML, MB, MW, MH), pag_img_x, img_y_bot, IW_B, IH)
    ax_pag.imshow(pag, cmap='magma', vmin=0, vmax=1, aspect='equal')
    ax_pag.plot(p_star[0], p_star[1], 'r*', markersize=11,
                markeredgecolor='white', markeredgewidth=0.8)
    style_img(ax_pag, r'HALO Soft Label  $P(i)$', C_PAG, C_PAG)

    # ── Panel C: output segmentation ─────────────────────────────
    out_x = PC['x'] + (PC['w'] - IW_C) / 2
    ax_out = img_ax(fig, (ML, MB, MW, MH), out_x, img_y_bot, IW_C, IH)
    # Composite: IR (gray) + PAG overlay (hot)
    overlay = np.stack([ir * 0.55 + pag * 0.70,
                        ir * 0.55,
                        ir * 0.55], axis=-1).clip(0, 1)
    ax_out.imshow(overlay, aspect='equal')
    ax_out.plot(p_star[0], p_star[1], 'r*', markersize=11,
                markeredgecolor='white', markeredgewidth=0.8)
    style_img(ax_out, 'Predicted Segmentation\n(IR + soft label overlay)',
              C_PC_BRD, C_PC_HDR)

    # ─────────────────────────────────────────────────────────────
    # Bottom color legend
    # ─────────────────────────────────────────────────────────────
    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(color=C_PA_BRD,  label='(a) Input / YOLO Annotation'),
        mpatches.Patch(color=C_BSNR,    label='(b) B-SNR Posterior  W(i)'),
        mpatches.Patch(color=C_PAG,     label='(b) PAG Generation  P(i)'),
        mpatches.Patch(color=C_PC_HDR,  label='(c) Downstream Backbone Training'),
        mpatches.Patch(color=C_BADGE,   label='Zero-Parameter / Training-Free'),
    ]
    ax.legend(handles=legend_items, loc='lower center',
              ncol=5, fontsize=8.0, framealpha=0.92,
              edgecolor='#CCCCCC', bbox_to_anchor=(0.5, -0.005))

    # ── Figure title ──────────────────────────────────────────────
    fig.text(0.5, 0.975,
             'HALO: Heat-Anchored Label Optimization — Method Overview',
             ha='center', va='top',
             fontsize=14.5, fontweight='bold', color='#1A1A1A')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#F8F9FA')
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(
        ROOT, 'paper', 'figures', 'fig_method_flow.png'))
    args = parser.parse_args()
    draw(args.out)
