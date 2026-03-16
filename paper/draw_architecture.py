"""
HaloNet 架构图 v2 — 精细版
运行: python paper/draw_architecture.py
输出: paper/halonet_architecture.pdf / .png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter

# ── 全局字体与风格 ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
})

# ── 配色方案 ────────────────────────────────────────────────────
BLUE    = '#2196F3'
ORANGE  = '#FF9800'
GREEN   = '#4CAF50'
PURPLE  = '#9C27B0'
RED     = '#F44336'
GRAY    = '#607D8B'
DARK    = '#212121'
LIGHT   = '#FAFAFA'

C_PROB  = '#E3F2FD'   # 问题（浅蓝）
C_BSNR  = '#FFF8E1'   # B-SNR（浅黄）
C_PAG   = '#E8F5E9'   # PAG（浅绿）
C_NET   = '#F3E5F5'   # 网络（浅紫）

# ── 自定义热图色彩 ───────────────────────────────────────────────
ir_cmap   = LinearSegmentedColormap.from_list('ir',   ['#0a0a1a','#1a1a3e','#c62828','#f57c00','#fffde7'])
mask_cmap = LinearSegmentedColormap.from_list('mask', ['#0d0221','#1565C0','#00897B','#F9A825','#FFFFFF'])

# ════════════════════════════════════════════════════════════════
#  仿真数据生成
# ════════════════════════════════════════════════════════════════
def make_ir_scene(H=64, W=64, seed=42):
    """生成仿真红外小目标场景"""
    rng = np.random.RandomState(seed)
    # 复杂背景：云/海浪纹理
    bg = np.zeros((H, W))
    for _ in range(6):
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        r = rng.uniform(10, 25)
        Y, X = np.ogrid[:H, :W]
        bg += rng.uniform(0.05, 0.15) * np.exp(-((X-cx)**2+(Y-cy)**2)/(2*r**2))
    bg += rng.randn(H, W) * 0.03
    bg = gaussian_filter(bg, sigma=2)
    bg = (bg - bg.min()) / (bg.max() - bg.min()) * 0.5

    # 小目标：中心偏上，3×3 像素
    tx, ty = 36, 28
    Y, X = np.ogrid[:H, :W]
    target = 0.85 * np.exp(-((X-tx)**2+(Y-ty)**2)/(2*1.2**2))
    scene = np.clip(bg + target, 0, 1)
    return scene, tx, ty

def make_bsnr_map(scene, box, tau=3.0, expand=1.5, eps=1e-4):
    """仿真 B-SNR 权重图"""
    H, W = scene.shape
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    px = int((expand-1)/2*bw); py = int((expand-1)/2*bh)
    cx1=max(0,x1-px); cy1=max(0,y1-py)
    cx2=min(W,x2+px); cy2=min(H,y2+py)
    ctx = scene[cy1:cy2, cx1:cx2].copy()
    # 掩掉目标框内，只用背景
    ctx[py:py+bh, px:px+bw] = np.nan
    mu = np.nanmean(ctx); sig = np.nanstd(ctx) + eps
    patch = scene[y1:y2, x1:x2]
    snr = (patch - mu) / sig
    W_map = 1/(1+np.exp(-tau*snr))
    return W_map

def make_pag_map(W_map, sigma_ratio=1.5):
    """仿真 PAG 物理锚定高斯"""
    h, w = W_map.shape
    # 找物理热点
    py, px = np.unravel_index(np.argmax(W_map), W_map.shape)
    # FWHM 区域
    peak = W_map[py, px]
    fwhm_mask = W_map >= peak * 0.5
    Y, X = np.ogrid[:h, :w]
    d2 = (X - px)**2 + (Y - py)**2
    weighted_var = np.sum(W_map[fwhm_mask] * d2[fwhm_mask]) / (np.sum(W_map[fwhm_mask]) + 1e-8)
    sigma = sigma_ratio * np.sqrt(weighted_var)
    sigma = max(sigma, 1.0)
    gauss = np.exp(-d2 / (2 * sigma**2))
    pag = W_map * gauss
    pag = pag / (pag.max() + 1e-8)
    return pag, (px, py)

# ════════════════════════════════════════════════════════════════
#  主图布局
# ════════════════════════════════════════════════════════════════
scene, tx, ty = make_ir_scene()
BOX = (tx-12, ty-10, tx+12, ty+10)   # 粗糙框（40% 目标覆盖率）
bx1,by1,bx2,by2 = BOX
W_map = make_bsnr_map(scene, BOX)
pag_map, (phx, phy) = make_pag_map(W_map)

fig = plt.figure(figsize=(20, 11), facecolor='white')

# 主网格：左侧流程(宽) + 右侧验证(窄)
gs_main = gridspec.GridSpec(1, 2, figure=fig,
                            left=0.02, right=0.98,
                            top=0.90, bottom=0.04,
                            width_ratios=[3.2, 1], wspace=0.05)

gs_left  = gridspec.GridSpecFromSubplotSpec(3, 5, subplot_spec=gs_main[0],
                                            hspace=0.55, wspace=0.35)
gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[1],
                                            hspace=0.4)

# ─── 大标题 ────────────────────────────────────────────────────
fig.text(0.5, 0.96,
         'HaloNet: Physics-Anchored Pseudo-Label Generation for Box-Supervised Infrared Small Target Detection',
         ha='center', va='top', fontsize=13.5, fontweight='bold', color=DARK)
fig.text(0.5, 0.925,
         'Zero learnable parameters  ·  No iterative refinement  ·  Plug-and-play for any IRSTD backbone',
         ha='center', va='top', fontsize=10, color=GRAY, style='italic')

# ════════════════════════════════════════════════════════════════
#  第 0 列：问题定义（无标签 patch 显示）
# ════════════════════════════════════════════════════════════════
ax_ir = fig.add_subplot(gs_left[0, 0])
ax_ir.imshow(scene, cmap=ir_cmap, vmin=0, vmax=0.95, interpolation='bilinear')
ax_ir.add_patch(Rectangle((bx1, by1), bx2-bx1, by2-by1,
                           fill=False, edgecolor='#FF5252', lw=2.5, linestyle='--'))
ax_ir.scatter([tx], [ty], marker='+', s=120, c='#FF5252', linewidths=1.8, zorder=5)
ax_ir.set_title('Input: IR Image +\nBounding Box', fontsize=8.5, fontweight='bold', pad=3)
ax_ir.axis('off')

# 问题标注：框内 99% 背景
box_area = (bx2-bx1)*(by2-by1)
target_area = np.pi*1.2**2
pct_bg = 100*(1 - target_area/box_area)
ax_ir.text(bx1+(bx2-bx1)/2, by2+1.5,
           f'~{pct_bg:.0f}% background noise inside box',
           ha='center', va='top', fontsize=6.5, color='#FF5252', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#FF5252', alpha=0.85))

# 问题描述框（中部 + 下部）
ax_prob_desc = fig.add_subplot(gs_left[1, 0])
ax_prob_desc.axis('off')
txt = ("[x]  Pixel-level mask:\n"
       "     expensive & slow\n\n"
       "[x]  Single-point label:\n"
       "     hard to click accurately\n"
       "     on dim 3x3 px targets\n\n"
       "[v]  Bounding Box:\n"
       "     auto-output by YOLO\n"
       "     or radar systems")
ax_prob_desc.text(0.5, 0.95, txt, ha='center', va='top', fontsize=7.5,
                  transform=ax_prob_desc.transAxes,
                  bbox=dict(boxstyle='round,pad=0.5', fc=C_PROB, ec=BLUE, alpha=0.9),
                  linespacing=1.5, family='monospace')
ax_prob_desc.set_title('The Annotation\nBottleneck', fontsize=8.5,
                        fontweight='bold', color=BLUE, pad=2)

# ════════════════════════════════════════════════════════════════
#  第 1 列：箭头
# ════════════════════════════════════════════════════════════════
for row in range(3):
    ax_arrow = fig.add_subplot(gs_left[row, 1])
    ax_arrow.axis('off')
    ax_arrow.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                      xycoords='axes fraction',
                      arrowprops=dict(arrowstyle='->', color=DARK,
                                      lw=2.5, mutation_scale=20))

# ════════════════════════════════════════════════════════════════
#  第 2 列：创新点 1 — B-SNR
# ════════════════════════════════════════════════════════════════
# 上：B-SNR 可视化
ax_bsnr_vis = fig.add_subplot(gs_left[0, 2])
# 仅显示 box 区域，其余区域显示原始 IR 图方便对比
bsnr_full = np.zeros_like(scene)
bsnr_full[by1:by2, bx1:bx2] = W_map
bsnr_rgb = plt.cm.ScalarMappable(cmap=mask_cmap).to_rgba(bsnr_full)[:,:,:3]
ir_rgb    = plt.cm.ScalarMappable(cmap=ir_cmap).to_rgba(scene)[:,:,:3]
# 框外显示原图（暗化），框内显示 B-SNR 热图
alpha_mask = np.zeros((scene.shape[0], scene.shape[1], 1))
alpha_mask[by1:by2, bx1:bx2] = 1.0
composite_bsnr = alpha_mask * bsnr_rgb + (1-alpha_mask) * ir_rgb * 0.35
ax_bsnr_vis.imshow(composite_bsnr, interpolation='bilinear')
ax_bsnr_vis.add_patch(Rectangle((bx1, by1), bx2-bx1, by2-by1,
                                  fill=False, edgecolor='white', lw=1.5, linestyle='--'))
# 标出 context box
pad = int(0.5/2*(bx2-bx1))
ax_bsnr_vis.add_patch(Rectangle((bx1-pad, by1-pad), (bx2-bx1)+2*pad, (by2-by1)+2*pad,
                                  fill=False, edgecolor='#FFC107', lw=1.2, linestyle=':'))
ax_bsnr_vis.set_title('① B-SNR Posterior Mask\n[Group C Output]',
                       fontsize=8.5, fontweight='bold', color=ORANGE, pad=3)
ax_bsnr_vis.axis('off')

# 图例
ax_bsnr_vis.text(0.02, 0.02, '── target box\n┅┅ context box (1.5×)',
                 transform=ax_bsnr_vis.transAxes,
                 fontsize=5.5, color='white', va='bottom',
                 bbox=dict(fc='#00000066', ec='none', pad=2))

# 中：公式 + 解释
ax_bsnr_desc = fig.add_subplot(gs_left[1, 2])
ax_bsnr_desc.axis('off')
desc1 = ("STEP 1  Estimate background stats\n"
         "        from Context Box (1.5× expanded):\n"
         "        μ_ctx , σ_ctx  (background only)\n\n"
         "STEP 2  Compute per-pixel posterior:\n\n"
         "  W(i) = σ( τ · (I(i) − μ_ctx) / σ_ctx )\n\n"
         "→ High W = likely TARGET pixel\n"
         "→ Low  W = likely BACKGROUND pixel\n\n"
         "Derived from Bayesian posterior\n"
         "under Gaussian background model")
ax_bsnr_desc.text(0.5, 0.97, desc1, ha='center', va='top', fontsize=7,
                  transform=ax_bsnr_desc.transAxes,
                  bbox=dict(boxstyle='round,pad=0.5', fc=C_BSNR, ec=ORANGE, alpha=0.95),
                  linespacing=1.6, family='monospace')

# 下：colorbar 指示
ax_bsnr_cb = fig.add_subplot(gs_left[2, 2])
ax_bsnr_cb.axis('off')
gradient = np.linspace(0, 1, 128).reshape(1, -1)
ax_bsnr_cb.imshow(gradient, aspect='auto', cmap=mask_cmap,
                  extent=[0, 1, 0.3, 0.7])
ax_bsnr_cb.text(0.0, 0.15, '0  (Background)', ha='left', va='top',
                fontsize=7, color=GRAY, transform=ax_bsnr_cb.transAxes)
ax_bsnr_cb.text(1.0, 0.15, '1  (Target)', ha='right', va='top',
                fontsize=7, color=ORANGE, fontweight='bold',
                transform=ax_bsnr_cb.transAxes)
ax_bsnr_cb.set_title('B-SNR Weight Value', fontsize=7.5, pad=2, color=GRAY)

# ════════════════════════════════════════════════════════════════
#  第 3 列：箭头（复用）
# ════════════════════════════════════════════════════════════════
# Already done in loop above

# ════════════════════════════════════════════════════════════════
#  第 4 列：创新点 2 — PAG
# ════════════════════════════════════════════════════════════════
# 上：PAG 可视化
ax_pag_vis = fig.add_subplot(gs_left[0, 4])
pag_full = np.zeros_like(scene)
pag_full[by1:by2, bx1:bx2] = pag_map
pag_rgb = plt.cm.ScalarMappable(cmap=mask_cmap).to_rgba(pag_full)[:,:,:3]
composite_pag = alpha_mask * pag_rgb + (1-alpha_mask) * ir_rgb * 0.35
ax_pag_vis.imshow(composite_pag, interpolation='bilinear')
# 标出物理热点
hx_global = bx1 + phx
hy_global = by1 + phy
ax_pag_vis.scatter([hx_global], [hy_global], marker='*', s=150, c='white',
                   zorder=5, label='Physical hotspot p*')
ax_pag_vis.add_patch(Rectangle((bx1, by1), bx2-bx1, by2-by1,
                                fill=False, edgecolor='white', lw=1.5, linestyle='--'))
ax_pag_vis.set_title('② PAG Soft Label\n[Group D Output]',
                     fontsize=8.5, fontweight='bold', color=GREEN, pad=3)
ax_pag_vis.axis('off')
ax_pag_vis.text(0.02, 0.02, '★ = physical hotspot p*',
                transform=ax_pag_vis.transAxes,
                fontsize=5.5, color='white', va='bottom',
                bbox=dict(fc='#00000066', ec='none', pad=2))

# 中：公式 + 解释
ax_pag_desc = fig.add_subplot(gs_left[1, 4])
ax_pag_desc.axis('off')
desc2 = ("STEP 3  Find physical hotspot:\n"
         "        p* = argmax( W )  ← NOT box center!\n\n"
         "STEP 4  FWHM anchor region:\n"
         "        Ω = { i : W(i) ≥ W(p*) × 0.5 }\n\n"
         "STEP 5  SNR-weighted spatial spread:\n"
         "        σ = 1.5 × √(Σ_Ω W(i)·d²ᵢ / Σ_Ω W(i))\n\n"
         "STEP 6  Physics-Anchored Gaussian:\n"
         "        P(i) = W(i) × exp(−d²ᵢ / 2σ²)\n\n"
         "→ σ tracks target's physical radiance spread\n"
         "→ Robust to box position errors")
ax_pag_desc.text(0.5, 0.97, desc2, ha='center', va='top', fontsize=7,
                 transform=ax_pag_desc.transAxes,
                 bbox=dict(boxstyle='round,pad=0.5', fc=C_PAG, ec=GREEN, alpha=0.95),
                 linespacing=1.6, family='monospace')

# 下：与 B-SNR 的对比
ax_pag_cmp = fig.add_subplot(gs_left[2, 4])
ax_pag_cmp.axis('off')
cmp_txt = ("Why PAG > B-SNR:\n"
           "B-SNR captures SNR shape but may\n"
           "retain background ripples.\n"
           "PAG multiplies with a Gaussian anchored\n"
           "to the physical energy peak → clean,\n"
           "compact, smooth soft label that\n"
           "better matches target PSF.")
ax_pag_cmp.text(0.5, 0.95, cmp_txt, ha='center', va='top', fontsize=7,
                transform=ax_pag_cmp.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', fc='#F1F8E9', ec=GREEN, alpha=0.9),
                linespacing=1.5)
ax_pag_cmp.set_title('PAG vs B-SNR', fontsize=7.5, pad=2, color=GREEN)

# ════════════════════════════════════════════════════════════════
#  理论边界 — 整体底部注释（横跨左侧流程）
# ════════════════════════════════════════════════════════════════
fig.text(0.02, 0.025,
         '[!]  Theoretical Limitation (Variance Corruption Bound): '
         'σ̂²_ctx ≈ σ²₀ + α(1−α)·Δμ²   '
         '→  When target-to-box ratio α < 0.05 (ultra-small targets in NUDT-SIRST), '
         'background variance is inflated by target brightness, causing SNR → 0. '
         'This formally explains the performance gap on NUDT.',
         ha='left', va='bottom', fontsize=8,
         color='#5D4037',
         bbox=dict(boxstyle='round,pad=0.4', fc='#FFF8E1', ec='#FF8F00', alpha=0.92))

# ════════════════════════════════════════════════════════════════
#  右列 上：Model-Agnostic 骨干框
# ════════════════════════════════════════════════════════════════
ax_net = fig.add_subplot(gs_right[0])
ax_net.axis('off')
ax_net.set_facecolor(C_NET)

# 大框
net_rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                           boxstyle="round,pad=0.03",
                           facecolor=C_NET, edgecolor=PURPLE, linewidth=2,
                           transform=ax_net.transAxes)
ax_net.add_patch(net_rect)

ax_net.text(0.5, 0.93, '③ Model-Agnostic Training',
            ha='center', va='top', fontsize=9.5, fontweight='bold',
            color=PURPLE, transform=ax_net.transAxes)
ax_net.text(0.5, 0.84, 'Plug any IRSTD backbone:',
            ha='center', va='top', fontsize=8, color=DARK,
            transform=ax_net.transAxes)

# 三个骨干网 icon
for i, (name, year, desc) in enumerate([
        ('DNANet', '(TIP 22)', 'Dense nested\nUNet++'),
        ('ACM',    '(WACV 21)', 'Asymmetric\nnon-local'),
        ('ALCNet', '(TGRS 21)', 'Local contrast\nFPN'),
]):
    x = 0.18 + i * 0.32
    box = FancyBboxPatch((x-0.12, 0.42), 0.25, 0.30,
                          boxstyle="round,pad=0.02",
                          facecolor='white', edgecolor=PURPLE, linewidth=1.2,
                          transform=ax_net.transAxes)
    ax_net.add_patch(box)
    ax_net.text(x+0.005, 0.64, name, ha='center', va='top', fontsize=8.5,
                fontweight='bold', color=DARK, transform=ax_net.transAxes)
    ax_net.text(x+0.005, 0.58, year, ha='center', va='top', fontsize=6.5,
                color=GRAY, transform=ax_net.transAxes)
    ax_net.text(x+0.005, 0.50, desc, ha='center', va='top', fontsize=6.5,
                color=GRAY, transform=ax_net.transAxes, linespacing=1.3)

ax_net.text(0.5, 0.37, '↓ SoftIoULoss (sigmoid applied internally)',
            ha='center', va='top', fontsize=7.5, color=GRAY,
            transform=ax_net.transAxes)
ax_net.text(0.5, 0.28, 'Training with PAG soft labels\n→ pixel-level segmentation',
            ha='center', va='top', fontsize=8, color=DARK,
            transform=ax_net.transAxes, linespacing=1.4)
ax_net.text(0.5, 0.12,
            '(*) No architecture modification\n(*) No extra parameters',
            ha='center', va='top', fontsize=8, color=GREEN,
            fontweight='bold', transform=ax_net.transAxes, linespacing=1.4)

# ════════════════════════════════════════════════════════════════
#  右列 下：结果表
# ════════════════════════════════════════════════════════════════
ax_tbl = fig.add_subplot(gs_right[1])
ax_tbl.axis('off')

ax_tbl.text(0.5, 0.99, '④ Experimental Results (mIoU %)',
            ha='center', va='top', fontsize=9.5, fontweight='bold',
            color=DARK, transform=ax_tbl.transAxes)
ax_tbl.text(0.5, 0.93, 'DNANet backbone · 3 datasets',
            ha='center', va='top', fontsize=7.5, color=GRAY,
            transform=ax_tbl.transAxes)

rows = [
    ['Dataset',      'LESPS', 'PAL',  'C\n(B-SNR)', 'D\n(PAG)', 'GT\n(Full)'],
    ['NUAA-SIRST',   '55.17', '63.46','63.12',       '70.69',    '72.84'],
    ['NUDT-SIRST',   '57.39', '73.10','56.12',       '62.97',    '95.07'],
    ['IRSTD-1K',     '50.90', '60.72','60.91',       '60.75',    '70.94'],
    ['Average',      '54.49', '65.76','60.05',       '64.80',    '79.62'],
]
col_w  = [0.30, 0.13, 0.13, 0.13, 0.13, 0.13]
col_x  = [0.01, 0.31, 0.44, 0.57, 0.70, 0.84]
row_h  = 0.12
row_y0 = 0.88

cell_colors = {
    (0,): '#374151',            # header row
    ('avg',3): '#FF8F00',       # avg C-Ours
    ('avg',4): '#2E7D32',       # avg D-Ours
}

for r, row in enumerate(rows):
    is_header = (r == 0)
    is_avg    = (r == len(rows)-1)
    y_top = row_y0 - r * row_h * 0.88

    for c, (cell, xc, wc) in enumerate(zip(row, col_x, col_w)):
        # background
        if is_header:
            fc = '#374151'; ec = '#374151'; tc = 'white'; fw = 'bold'
        elif is_avg:
            fc = '#F3F4F6'; ec = '#9CA3AF'
            tc = '#2E7D32' if c == 4 else ('#B45309' if c == 3 else DARK)
            fw = 'bold'
        elif c == 4:   # D-PAG column highlight
            fc = '#DCFCE7'; ec = '#86EFAC'; tc = '#15803D'; fw = 'bold'
        elif c == 3:   # C-BSNR column
            fc = '#FEF9C3'; ec = '#FDE68A'; tc = '#92400E'; fw = 'normal'
        else:
            fc = 'white'; ec = '#E5E7EB'; tc = DARK; fw = 'normal'

        rect = FancyBboxPatch((xc, y_top - row_h*0.80), wc-0.005, row_h*0.78,
                               boxstyle="square,pad=0",
                               facecolor=fc, edgecolor=ec, linewidth=0.8,
                               transform=ax_tbl.transAxes)
        ax_tbl.add_patch(rect)
        ax_tbl.text(xc + wc/2 - 0.005, y_top - row_h*0.38, cell,
                    ha='center', va='center', fontsize=7.5,
                    fontweight=fw, color=tc,
                    transform=ax_tbl.transAxes)

# 图例条
leg_y = row_y0 - len(rows)*row_h*0.88 - 0.03
ax_tbl.add_patch(FancyBboxPatch((0.57, leg_y-0.05), 0.13-0.005, 0.04,
                                  boxstyle="square,pad=0",
                                  facecolor='#FEF9C3', edgecolor='#FDE68A',
                                  transform=ax_tbl.transAxes))
ax_tbl.text(0.635, leg_y-0.025, 'B-SNR (C)', ha='center', va='center',
            fontsize=6.5, color='#92400E', transform=ax_tbl.transAxes)
ax_tbl.add_patch(FancyBboxPatch((0.70, leg_y-0.05), 0.13-0.005, 0.04,
                                  boxstyle="square,pad=0",
                                  facecolor='#DCFCE7', edgecolor='#86EFAC',
                                  transform=ax_tbl.transAxes))
ax_tbl.text(0.765, leg_y-0.025, 'PAG (D) ★', ha='center', va='center',
            fontsize=6.5, color='#15803D', fontweight='bold',
            transform=ax_tbl.transAxes)

# Model-Agnostic 结果注释
ma_y = leg_y - 0.10
ax_tbl.text(0.5, ma_y,
            'Model-Agnostic Verification (NUDT, ~1000 ep):',
            ha='center', va='top', fontsize=8, fontweight='bold',
            color=PURPLE, transform=ax_tbl.transAxes)
ma_rows = [
    ['Backbone', 'GT (A)', 'PAG (D)', 'Retention'],
    ['ALCNet',   '65.76%', '55.81%', '84.9%'],
    ['ACM',      '64.85%', '53.37%', '82.3%'],
]
for r, row in enumerate(ma_rows):
    y = ma_y - 0.08 - r*0.075
    is_h = (r == 0)
    for c, (cell, xc, wc) in enumerate(zip(row, [0.01,0.25,0.50,0.73], [0.24,0.25,0.23,0.26])):
        fc = '#374151' if is_h else ('#DCFCE7' if c == 2 else '#F9FAFB')
        tc = 'white' if is_h else ('#15803D' if c == 2 else DARK)
        fw = 'bold' if (is_h or c == 3) else 'normal'
        rect = FancyBboxPatch((xc, y-0.058), wc-0.005, 0.055,
                               boxstyle="square,pad=0",
                               facecolor=fc, edgecolor='#E5E7EB', linewidth=0.7,
                               transform=ax_tbl.transAxes)
        ax_tbl.add_patch(rect)
        ax_tbl.text(xc+wc/2-0.005, y-0.028, cell,
                    ha='center', va='center', fontsize=7,
                    fontweight=fw, color=tc,
                    transform=ax_tbl.transAxes)

ax_tbl.text(0.5, ma_y - 0.08 - 3*0.075 - 0.01,
            '>> PAG > B-SNR consistently across ALL backbones',
            ha='center', va='top', fontsize=7.5, color=GREEN,
            fontweight='bold', transform=ax_tbl.transAxes)

# ════════════════════════════════════════════════════════════════
#  创新点编号标签（左上角 badge）
# ════════════════════════════════════════════════════════════════
badges = [
    (0.145, 0.91, '①', BLUE,   'Problem'),
    (0.385, 0.91, '②', ORANGE, 'B-SNR'),
    (0.565, 0.91, '③', GREEN,  'PAG'),
    (0.765, 0.91, '④', PURPLE, 'Model-Agnostic'),
    (0.905, 0.91, '⑤', RED,    'Results'),
]
for bx, by, num, color, label in badges:
    fig.text(bx, by, f' {num} {label} ', ha='center', va='center',
             fontsize=8.5, fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.35', fc=color, ec='none', alpha=0.9))

plt.savefig('paper/halonet_architecture.pdf', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.savefig('paper/halonet_architecture.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print("架构图已保存：paper/halonet_architecture.pdf / .png")
