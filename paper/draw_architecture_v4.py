"""
HaloNet 架构图 v4 — 基于 Mermaid 逻辑的专业流程图
按照：输入 → 伪标签生成器(B-SNR + PAG) → 模型无关训练 的三段式结构
运行: python paper/draw_architecture_v4.py
输出: paper/halonet_architecture_v4.pdf / .png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
})

# ── 配色方案 ──
C_INPUT_BG   = '#E3F2FD'
C_INPUT_BD   = '#1E88E5'
C_BSNR_BG    = '#FFF8E1'
C_BSNR_BD    = '#FFB300'
C_PAG_BG     = '#E8F5E9'
C_PAG_BD     = '#43A047'
C_TRAIN_BG   = '#F3E5F5'
C_TRAIN_BD   = '#8E24AA'
C_THEORY_BG  = '#FFEBEE'
C_THEORY_BD  = '#E53935'
C_DARK       = '#1a1a2e'
C_GRAY       = '#546E7A'
C_WHITE      = '#FFFFFF'
C_GENERATOR_BG = '#FAFAFA'
C_GENERATOR_BD = '#90A4AE'

# 红外自定义 colormap
ir_cmap   = LinearSegmentedColormap.from_list('ir',   ['#0a0a1a','#1a1a3e','#c62828','#f57c00','#fffde7'])
mask_cmap = LinearSegmentedColormap.from_list('mask', ['#0d0221','#1565C0','#00897B','#F9A825','#FFFFFF'])
pag_cmap  = LinearSegmentedColormap.from_list('pag',  ['#0d0221','#1B5E20','#4CAF50','#CDDC39','#FFFFFF'])


# ════════════════════════════════════════════════════════════
# 仿真红外场景数据
# ════════════════════════════════════════════════════════════
def make_scene(H=80, W=80, seed=42):
    rng = np.random.RandomState(seed)
    bg = np.zeros((H, W))
    for _ in range(8):
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        r = rng.uniform(8, 28)
        Y, X = np.ogrid[:H, :W]
        bg += rng.uniform(0.04, 0.12) * np.exp(-((X-cx)**2+(Y-cy)**2)/(2*r**2))
    bg += rng.randn(H, W) * 0.025
    bg = gaussian_filter(bg, sigma=2.5)
    bg = (bg - bg.min()) / (bg.max() - bg.min()) * 0.45
    tx, ty = 44, 38
    Y, X = np.ogrid[:H, :W]
    target = 0.9 * np.exp(-((X-tx)**2+(Y-ty)**2)/(2*1.3**2))
    clutter = 0.25 * np.exp(-((X-36)**2+(Y-42)**2)/(2*2.0**2))
    return np.clip(bg + target + clutter, 0, 1), tx, ty

scene, tx, ty = make_scene()
H_s, W_s = scene.shape
BOX = (tx-14, ty-12, tx+14, ty+12)
bx1, by1, bx2, by2 = BOX

# B-SNR 计算
exp_ratio = 1.5
pad_x = int((exp_ratio-1)/2*(bx2-bx1)); pad_y = int((exp_ratio-1)/2*(by2-by1))
cx1, cy1 = max(0, bx1-pad_x), max(0, by1-pad_y)
cx2, cy2 = min(W_s, bx2+pad_x), min(H_s, by2+pad_y)
ctx = scene[cy1:cy2, cx1:cx2].copy()
mu_ctx = np.nanmean(ctx); sig_ctx = np.nanstd(ctx) + 1e-4
patch = scene[by1:by2, bx1:bx2]
snr_raw = (patch - mu_ctx) / sig_ctx
W_map = 1/(1+np.exp(-3.0*snr_raw))
wmin, wmax = W_map.min(), W_map.max()
if wmax - wmin > 1e-8:
    W_map = (W_map - wmin) / (wmax - wmin)

# PAG 计算
bh, bw = by2-by1, bx2-bx1
peak_flat = snr_raw.argmax()
py_local, px_local = divmod(peak_flat, bw)
yy, xx = np.mgrid[0:bh, 0:bw]
dist_sq = ((yy-py_local)**2 + (xx-px_local)**2).astype(np.float32)
peak_val = snr_raw[py_local, px_local]
fwhm_mask = snr_raw >= peak_val * 0.5
snr_pos = np.where(fwhm_mask, np.maximum(snr_raw, 0), 0).astype(np.float32)
s = snr_pos.sum()
sig_px = max(1.5 * np.sqrt((snr_pos * dist_sq).sum() / (s+1e-8)), 1.0)
gauss = np.exp(-dist_sq / (2*sig_px**2))
pag_map = W_map * gauss
pag_map = pag_map / (pag_map.max() + 1e-8)

def make_rgb(data, cmap, vmin=0, vmax=1):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    return sm.to_rgba(data)[:,:,:3]

ir_rgb = make_rgb(scene, ir_cmap, 0, 0.9)


# ════════════════════════════════════════════════════════════
# 主图 — 竖向流程图布局
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 24), facecolor='white')

# 整个图的坐标系
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18)
ax.set_ylim(0, 24)
ax.axis('off')

# ── 辅助绘图函数 ──
def rounded_box(x, y, w, h, fc, ec, lw=2, alpha=1.0, ls='-', zorder=1):
    """绘制圆角矩形"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                          facecolor=fc, edgecolor=ec, linewidth=lw,
                          alpha=alpha, linestyle=ls, zorder=zorder)
    ax.add_patch(box)
    return box

def arrow_down(x, y1, y2, color=C_DARK, lw=2.5, label=None, label_side='right'):
    """垂直向下箭头"""
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=20))
    if label:
        offset = 0.15 if label_side == 'right' else -0.15
        ha = 'left' if label_side == 'right' else 'right'
        ax.text(x + offset, (y1+y2)/2, label, ha=ha, va='center',
                fontsize=8, color=C_GRAY, style='italic')

def arrow_right(x1, x2, y, color=C_DARK, lw=2.5, label=None):
    """水平向右箭头"""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=20))
    if label:
        ax.text((x1+x2)/2, y+0.2, label, ha='center', va='bottom',
                fontsize=8, color=C_GRAY, style='italic')

def arrow_curved(x1, y1, x2, y2, color=C_DARK, lw=1.5, ls='--', rad=0.3):
    """曲线箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls, mutation_scale=15,
                                connectionstyle=f'arc3,rad={rad}'))

def node_box(x, y, w, h, text, fc, ec, fontsize=10, text_color=C_DARK, bold=True):
    """带文字的节点框"""
    rounded_box(x, y, w, h, fc, ec)
    weight = 'bold' if bold else 'normal'
    ax.text(x+w/2, y+h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=text_color,
            linespacing=1.5)

def section_label(x, y, text, color, fontsize=13):
    """分区标题标签"""
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.5', fc=color, ec='none', alpha=0.92))

def innovation_label(x, y, num, text, color=C_THEORY_BD):
    """创新点标注"""
    ax.text(x, y, f'Innovation {num}: {text}',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round,pad=0.5', fc=C_THEORY_BG, ec=color,
                      alpha=0.95, linestyle='--', linewidth=2.0))


# ════════════════════════════════════════════════════════════
# SECTION 1: 输入空间 (y: 22.0 ~ 23.5)
# ════════════════════════════════════════════════════════════
section_label(9.0, 23.5, '  INPUT SPACE  ', C_INPUT_BD)

# 输入区域大背景框
rounded_box(1.5, 21.2, 15.0, 2.1, C_INPUT_BG+'33', C_INPUT_BD, lw=1.5, alpha=0.4)

# IR 图像节点
node_box(2.0, 21.4, 5.0, 1.6, 'IR Image  $I$\n(Single-channel Infrared)',
         C_INPUT_BG, C_INPUT_BD, fontsize=11)

# YOLO Box 节点
node_box(10.5, 21.4, 5.5, 1.6, 'Bounding Box  $B$\n(YOLO / Radar / Manual)',
         C_INPUT_BG, C_INPUT_BD, fontsize=11)

# IR image 小图 (更大)
ax_ir = fig.add_axes([0.13, 0.855, 0.1, 0.045])
ax_ir.imshow(scene, cmap=ir_cmap, vmin=0, vmax=0.9, interpolation='bilinear')
ax_ir.add_patch(Rectangle((bx1, by1), bx2-bx1, by2-by1,
                           fill=False, edgecolor='#FF5252', lw=1.5, linestyle='--'))
ax_ir.scatter([tx], [ty], marker='+', s=60, c='#FF5252', linewidths=1, zorder=5)
ax_ir.axis('off')

# Box 小图 (更大)
ax_box = fig.add_axes([0.63, 0.855, 0.1, 0.045])
ax_box.imshow(scene[max(0,by1-5):min(H_s,by2+5), max(0,bx1-5):min(W_s,bx2+5)],
              cmap=ir_cmap, vmin=0, vmax=0.9)
ax_box.add_patch(Rectangle((5, 5), bx2-bx1, by2-by1,
                            fill=False, edgecolor='#FF5252', lw=2))
ax_box.set_title('>99% pixels = background', fontsize=7, color='red', pad=2)
ax_box.axis('off')


# ═══════ 连接箭头：输入 → 伪标签生成器 ═══════
arrow_down(4.5, 21.2, 20.4, color=C_INPUT_BD, label='Image I')
arrow_down(13.25, 21.2, 20.4, color=C_INPUT_BD, label='Box B')


# ════════════════════════════════════════════════════════════
# SECTION 2: 伪标签生成器大框 (y: 10.5 ~ 20.4)
# ════════════════════════════════════════════════════════════
section_label(9.0, 20.7, '  PHYSICS-ANCHORED PSEUDO-LABEL GENERATOR (Zero Parameters, No Iteration)  ',
              C_GENERATOR_BD)

# 生成器大背景
rounded_box(0.5, 10.4, 17.0, 9.8, '#F5F5F5', C_GENERATOR_BD, lw=2.5, alpha=0.5)

# ──────────────────────────────────────────────
# Innovation 1: B-SNR (y: 15.5 ~ 19.8)
# ──────────────────────────────────────────────
innovation_label(9.0, 19.6, '1',
                 'B-SNR: Background Statistics / Target Action Domain Separation')

# Innovation 1 背景框 (虚线)
rounded_box(1.0, 14.8, 16.0, 4.6, C_BSNR_BG+'22', C_BSNR_BD, lw=2, alpha=0.4, ls='--')

# Step 1: Context Box 膨胀
node_box(2.0, 17.8, 5.5, 1.6,
         'Context Box Expansion\n'
         'Ratio = 1.5x  (each side +25%)',
         C_BSNR_BG, C_BSNR_BD, fontsize=10)

# Context Box 可视化 (更大)
ax_ctx = fig.add_axes([0.42, 0.748, 0.08, 0.038])
ctx_vis = ir_rgb.copy() * 0.5
ctx_vis[cy1:cy2, cx1:cx2] *= 2.0
ctx_vis = np.clip(ctx_vis, 0, 1)
ctx_vis[by1:by2, bx1:bx2] = ctx_vis[by1:by2, bx1:bx2] * 0.3 + np.array([0.4, 0.15, 0.1]) * 0.7
ax_ctx.imshow(ctx_vis, interpolation='bilinear')
ax_ctx.add_patch(Rectangle((cx1, cy1), cx2-cx1, cy2-cy1,
                            fill=False, edgecolor='#FFC107', lw=1.5, linestyle='--'))
ax_ctx.add_patch(Rectangle((bx1, by1), bx2-bx1, by2-by1,
                            fill=False, edgecolor='#FF5252', lw=1, linestyle=':'))
ax_ctx.axis('off')

# Step 2: 背景统计量
node_box(10.0, 17.8, 5.5, 1.6,
         'Background Statistics\n'
         r'$\mu_{ctx}$ = mean,   $\sigma_{ctx}$ = std',
         C_BSNR_BG, C_BSNR_BD, fontsize=10)

# 分离标注
ax.text(12.75, 17.5, 'Statistics from context ring\n(excludes target box interior)',
        ha='center', va='top', fontsize=8, color='#E65100', style='italic')

# Step 1 → Step 2 箭头
arrow_right(7.7, 9.8, 18.6, color=C_BSNR_BD, label='exclude target pollution')

# Step 3: B-SNR 后验
node_box(5.0, 15.2, 8.0, 1.6,
         r'B-SNR Posterior Mask:   $W(i) = \sigma\!\left(\tau \cdot \frac{I(i) - \mu_{ctx}}{\sigma_{ctx}}\right)$'
         '\n'
         r'Temperature $\tau = 3.0$   |   Bayesian posterior under Gaussian shift',
         C_BSNR_BG, C_BSNR_BD, fontsize=9.5, bold=False)

# B-SNR 可视化 (更大)
ax_bsnr = fig.add_axes([0.73, 0.635, 0.08, 0.038])
ax_bsnr.imshow(W_map, cmap=mask_cmap, vmin=0, vmax=1, interpolation='bilinear')
ax_bsnr.set_title('W(i) B-SNR', fontsize=7, pad=2, color=C_BSNR_BD, fontweight='bold')
ax_bsnr.axis('off')

# 箭头 Step2 → Step3, Image I → Step3
arrow_down(7.5, 17.6, 17.0, color=C_BSNR_BD)
arrow_down(12.75, 17.6, 17.0, color=C_BSNR_BD)
# Image I 直接进入 W 计算 (曲线从左侧)
arrow_curved(4.5, 20.4, 5.5, 17.0, color=C_INPUT_BD, rad=-0.3)


# ──────────────────────────────────────────────
# Innovation 2: PAG (y: 10.8 ~ 14.5)
# ──────────────────────────────────────────────
innovation_label(9.0, 14.4, '2',
                 'PAG: Physics-Anchored Gaussian Refinement (NOT Geometric Center)')

# Innovation 2 背景框 (虚线)
rounded_box(1.0, 10.7, 16.0, 3.5, C_PAG_BG+'22', C_PAG_BD, lw=2, alpha=0.4, ls='--')

# B-SNR → PAG 连接
arrow_down(9.0, 15.0, 14.0, color=C_BSNR_BD, label='W(i)')

# PAG 4个步骤 — 水平排列
pag_steps = [
    ('p* = argmax(W)\nPhysical Hotspot', 1.5, 3.0),
    ('FWHM Region\n{i: W(i)>=0.5*W(p*)}', 5.5, 3.5),
    (r'Adaptive $\sigma$' + '\nSNR-weighted std', 9.7, 3.0),
    ('P(i) = W * G(d,sigma)\nPAG Soft Label', 13.3, 3.2),
]

for text, x, w in pag_steps:
    node_box(x, 11.0, w, 1.5, text, C_PAG_BG, C_PAG_BD, fontsize=9)

# PAG 步骤间箭头
arrow_right(4.7, 5.3, 11.75, color=C_PAG_BD)
arrow_right(9.2, 9.5, 11.75, color=C_PAG_BD)
arrow_right(12.9, 13.1, 11.75, color=C_PAG_BD)

# PAG 可视化 (更大)
ax_pag = fig.add_axes([0.82, 0.48, 0.08, 0.038])
ax_pag.imshow(pag_map, cmap=pag_cmap, vmin=0, vmax=1, interpolation='bilinear')
hx_g = px_local; hy_g = py_local
ax_pag.scatter([hx_g], [hy_g], marker='*', s=100, c='white', edgecolors='black', linewidth=0.5, zorder=5)
ax_pag.contour(fwhm_mask.astype(float), levels=[0.5], colors=['#FFEB3B'], linewidths=1.0)
ax_pag.set_title('P(i) PAG', fontsize=7, pad=2, color=C_PAG_BD, fontweight='bold')
ax_pag.axis('off')


# ═══════ 连接箭头：伪标签生成器 → 模型训练 ═══════
arrow_down(9.0, 10.4, 9.5, color=C_PAG_BD, lw=3)
ax.text(9.5, 9.9, 'PAG High-Quality\nSoft Labels', ha='left', va='center',
        fontsize=9, fontweight='bold', color=C_PAG_BD)


# ════════════════════════════════════════════════════════════
# SECTION 3: 模型无关训练 (y: 6.5 ~ 9.5)
# ════════════════════════════════════════════════════════════
section_label(9.0, 9.8, '  MODEL-AGNOSTIC SUPERVISED TRAINING  ', C_TRAIN_BD)

# 训练区域大背景
rounded_box(1.5, 6.5, 15.0, 3.0, C_TRAIN_BG+'33', C_TRAIN_BD, lw=1.5, alpha=0.4)

# 三个骨干网
for i, (name, desc, year) in enumerate([
    ('DNANet', 'Dense Nested UNet++\n+ Res-CBAM', 'TIP 2022'),
    ('ACM', 'Asymmetric Contextual\nModulation', 'WACV 2021'),
    ('ALCNet', 'Local Contrast\nAttention FPN', 'TGRS 2021'),
]):
    x_pos = 2.0 + i * 5.0
    rounded_box(x_pos, 7.3, 4.5, 1.8, C_TRAIN_BG, C_TRAIN_BD, lw=1.5)
    ax.text(x_pos + 2.25, 8.6, name, ha='center', va='center',
            fontsize=12, fontweight='bold', color=C_TRAIN_BD)
    ax.text(x_pos + 2.25, 7.8, f'{desc}\n({year})', ha='center', va='center',
            fontsize=8, color=C_GRAY, linespacing=1.3)

# Loss 标注
ax.text(9.0, 6.8, 'SoftIoU Loss  |  No network modification  |  No extra parameters',
        ha='center', va='center', fontsize=9.5, color=C_TRAIN_BD, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc=C_TRAIN_BG, ec=C_TRAIN_BD, alpha=0.7))

# Image I → 网络 (曲线虚线箭头，表示前向传播)
arrow_curved(3.0, 21.2, 4.25, 9.3, color=C_INPUT_BD, rad=-0.6, lw=1.5)
ax.text(0.6, 15.0, 'Image I\nForward Pass', ha='center', va='center',
        fontsize=8.5, color=C_INPUT_BD, rotation=90, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc=C_INPUT_BG, ec=C_INPUT_BD, alpha=0.6))


# ═══════ 输出箭头 ═══════
arrow_down(9.0, 6.3, 5.5, color=C_TRAIN_BD, lw=3)
ax.text(9.0, 5.2, 'Pixel-level Segmentation Prediction',
        ha='center', va='center', fontsize=12, fontweight='bold', color=C_DARK,
        bbox=dict(boxstyle='round,pad=0.5', fc='#E0E0E0', ec=C_DARK, lw=2))


# ════════════════════════════════════════════════════════════
# 理论支撑标注 (右侧浮动)
# ════════════════════════════════════════════════════════════
# Theory 1: 方差污染边界 → 连接 Innovation 1
theory1_y = 17.0
ax.text(17.5, theory1_y,
        'Theoretical\nGuarantee',
        ha='center', va='center', fontsize=9, fontweight='bold',
        color=C_THEORY_BD,
        bbox=dict(boxstyle='round,pad=0.4', fc=C_THEORY_BG, ec=C_THEORY_BD,
                  lw=1.5, linestyle='--'))
ax.text(17.5, theory1_y - 1.2,
        r'$\hat{\sigma}^2_{ctx} \approx \sigma_0^2$'
        '\n'
        r'$+ \alpha(1\!-\!\alpha)\Delta\mu^2$'
        '\n\n'
        'Variance corruption\n'
        r'bound when $\alpha < 0.05$',
        ha='center', va='center', fontsize=8.5, color='#5D4037',
        bbox=dict(boxstyle='round,pad=0.4', fc='#FFF3E0', ec='#FF8F00',
                  alpha=0.9), linespacing=1.4)
arrow_curved(17.2, theory1_y - 0.5, 15.7, 16.5, color=C_THEORY_BD, rad=0.3, ls='--')

# Theory 2: 物理热点 → 连接 Innovation 2
theory2_y = 12.5
ax.text(17.5, theory2_y,
        'Physical\nIntuition',
        ha='center', va='center', fontsize=9, fontweight='bold',
        color=C_THEORY_BD,
        bbox=dict(boxstyle='round,pad=0.4', fc=C_THEORY_BG, ec=C_THEORY_BD,
                  lw=1.5, linestyle='--'))
ax.text(17.5, theory2_y - 1.2,
        'Energy peak p* replaces\n'
        'geometric box center\n\n'
        'FWHM-based sigma\n'
        'tracks target PSF',
        ha='center', va='center', fontsize=8.5, color='#5D4037',
        bbox=dict(boxstyle='round,pad=0.4', fc='#F1F8E9', ec='#66BB6A',
                  alpha=0.9), linespacing=1.4)
arrow_curved(17.2, theory2_y - 0.5, 15.7, 12.0, color=C_THEORY_BD, rad=0.3, ls='--')


# ════════════════════════════════════════════════════════════
# 底部：核心优势总结 (y: 2.5 ~ 4.5)
# ════════════════════════════════════════════════════════════
summary_items = [
    ('Zero Parameters', 'Pseudo-labels from\nphysics formulas only', C_BSNR_BD),
    ('No Iteration', 'Single forward-pass\nlabel generation', C_PAG_BD),
    ('Model-Agnostic', 'Works with any\nIRSTD backbone', C_TRAIN_BD),
    ('Theoretically Grounded', 'Variance corruption\nbound analysis', C_THEORY_BD),
]

for i, (title, desc, color) in enumerate(summary_items):
    x_pos = 1.5 + i * 4.2
    rounded_box(x_pos, 2.5, 3.7, 2.2, '#FAFAFA', color, lw=2)
    ax.text(x_pos + 1.85, 4.1, title, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=color)
    ax.text(x_pos + 1.85, 3.1, desc, ha='center', va='center',
            fontsize=9, color=C_GRAY, linespacing=1.4)

# 底部分隔线
ax.plot([2, 16], [5.0, 5.0], color='#E0E0E0', lw=1, ls='--')
ax.text(9.0, 4.9, 'KEY ADVANTAGES', ha='center', va='bottom',
        fontsize=10, fontweight='bold', color=C_GRAY)

# ── 大标题 ──
ax.text(9.0, 23.9,
        'HaloNet: Physics-Anchored Pseudo-Label Generation for Box-Supervised IRSTD',
        ha='center', va='center', fontsize=16, fontweight='bold', color=C_DARK)

# ── 图例 ──
legend_items = [
    (C_INPUT_BG, C_INPUT_BD, 'Input'),
    (C_BSNR_BG, C_BSNR_BD, 'B-SNR (Ours)'),
    (C_PAG_BG, C_PAG_BD, 'PAG (Ours)'),
    (C_TRAIN_BG, C_TRAIN_BD, 'Training'),
    (C_THEORY_BG, C_THEORY_BD, 'Theory'),
]
for i, (fc, ec, label) in enumerate(legend_items):
    x_pos = 3.0 + i * 2.8
    rounded_box(x_pos, 1.5, 0.6, 0.5, fc, ec, lw=1.5)
    ax.text(x_pos + 0.8, 1.75, label, ha='left', va='center',
            fontsize=8.5, color=C_DARK)


# ════════════════════════════════════════════════════════════
# 保存
# ════════════════════════════════════════════════════════════
plt.savefig('paper/halonet_architecture_v4.pdf', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('paper/halonet_architecture_v4.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Done: paper/halonet_architecture_v4.pdf / .png")
