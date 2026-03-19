#!/usr/bin/env python3
"""
Figure: Variance Contamination Theory — TIP-Style 1×3 Empirical Validation
============================================================================
排版: 1 行 × 3 列

  Panel A (左) ── 极小目标重度污染区 (NUDT, expand_ratio=1.1)
    Context Ring PDF (红) 严重右移/展宽 vs True BG (蓝虚线)
    图内标注: α·(1-α)·Δμ² = XXX  (LaTeX)

  Panel B (中) ── 中等目标安全平台区 (NUAA, expand_ratio=1.5)
    Context Ring PDF (绿) 与 True BG (蓝虚线) 高度贴合
    图内标注: 方法设计合理性佐证

  Panel C (右) ── 污染曲线 (两数据集叠加)
    X 轴: expand_ratio,  Y 轴: σ²_ctx / σ²₀
    r < 1.3 红底阴影 "Danger Zone"
    r ≥ 1.5 绿底阴影 "Safe Platform"
    两组折线: NUDT (红) vs NUAA (绿)
    右 Y 轴: α 值

用法:
    # 单样本实证（服务器上运行）
    python paper/fig_variance_contamination.py \\
        --nudt 000259 --nuaa Misc_87 \\
        --out paper/figures/fig_variance_theory.pdf

    # 批量统计（输出 CSV 供论文引用）
    python paper/fig_variance_contamination.py \\
        --batch --dataset NUDT-SIRST --n 50 --out_dir paper/figures/variance_batch

    # 本地演示（使用合成数据，无需数据集）
    python paper/fig_variance_contamination.py --demo
"""

import os
import argparse
import numpy as np
import cv2
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'DejaVu Sans'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_CONFIGS = {
    'NUAA-SIRST': {
        'img_dir':   'dataset/NUAA-SIRST/images',
        'mask_dir':  'dataset/NUAA-SIRST/masks',
        'label_dir': 'dataset/NUAA-SIRST/labels_box',
        'split':     'dataset/NUAA-SIRST/50_50/test.txt',
        'img_ext':   '.png',
    },
    'NUDT-SIRST': {
        'img_dir':   'dataset/NUDT-SIRST/images',
        'mask_dir':  'dataset/NUDT-SIRST/masks',
        'label_dir': 'dataset/NUDT-SIRST/labels_box',
        'split':     'dataset/NUDT-SIRST/50_50/test.txt',
        'img_ext':   '.png',
    },
    'IRSTD-1k': {
        'img_dir':   'dataset/IRSTD-1k/IRSTD1k_Img',
        'mask_dir':  'dataset/IRSTD-1k/masks',
        'label_dir': 'dataset/IRSTD-1k/labels_box',
        'split':     'dataset/IRSTD-1k/50_50/test.txt',
        'img_ext':   '.png',
    },
}

EXPAND_RATIOS = [1.1, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]

# ────────────────────────────────────────────────────────────────
# 数据提取工具
# ────────────────────────────────────────────────────────────────

def load_yolo_boxes(label_path, img_h, img_w):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = (int(parts[0]), float(parts[1]),
                               float(parts[2]), float(parts[3]), float(parts[4]))
            x1 = max(0, int((cx - w / 2) * img_w))
            y1 = max(0, int((cy - h / 2) * img_h))
            x2 = min(img_w, int((cx + w / 2) * img_w))
            y2 = min(img_h, int((cy + h / 2) * img_h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def get_context_box(box, expand_ratio, H, W, min_expand=3):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx = max(min_expand, int(bw * (expand_ratio - 1) / 2))
    my = max(min_expand, int(bh * (expand_ratio - 1) / 2))
    return (max(0, x1 - mx), max(0, y1 - my),
            min(W, x2 + mx), min(H, y2 + my))


def extract_regions(img_gray, mask, box, expand_ratio, true_bg_margin=40):
    """
    返回 (true_bg_pixels, ctx_ring_pixels, box_pixels, stats_dict)
    stats_dict 包含解耦后的各项统计量（σ²₀、污染项、α、Δμ 均独立输出）
    """
    H, W = img_gray.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    # ── True Background ─────────────────────────────────────────
    bg_margin = true_bg_margin
    true_bg_mask = np.ones((H, W), dtype=bool)
    true_bg_mask[max(0, y1 - bg_margin):min(H, y2 + bg_margin),
                 max(0, x1 - bg_margin):min(W, x2 + bg_margin)] = False
    if mask is not None:
        true_bg_mask[mask > 127] = False
    true_bg_pixels = img_gray[true_bg_mask].astype(np.float32)

    # ── Context Ring ─────────────────────────────────────────────
    cx1, cy1, cx2, cy2 = get_context_box(box, expand_ratio, H, W)
    ctx_mask   = np.zeros((H, W), dtype=bool)
    inner_mask = np.zeros((H, W), dtype=bool)
    ctx_mask[cy1:cy2, cx1:cx2] = True
    inner_mask[y1:y2, x1:x2]   = True
    ring_mask = ctx_mask & ~inner_mask
    if mask is not None:
        ring_mask[mask > 127] = False
    ctx_ring_pixels = img_gray[ring_mask].astype(np.float32)

    # ── Box Interior ─────────────────────────────────────────────
    box_pixels = img_gray[y1:y2, x1:x2].astype(np.float32).ravel()

    # ── Target pixels for Δμ ────────────────────────────────────
    if mask is not None and np.any(mask[y1:y2, x1:x2] > 127):
        target_pixels = img_gray[y1:y2, x1:x2][mask[y1:y2, x1:x2] > 127].astype(np.float32)
    else:
        target_pixels = np.array([], dtype=np.float32)

    # ── Statistics ───────────────────────────────────────────────
    s = {}
    s['mu_true']  = float(np.mean(true_bg_pixels))  if len(true_bg_pixels)  > 0 else np.nan
    s['std_true'] = float(np.std(true_bg_pixels))   if len(true_bg_pixels)  > 0 else np.nan
    s['var_true'] = s['std_true'] ** 2              if not np.isnan(s['std_true']) else np.nan
    s['mu_ctx']   = float(np.mean(ctx_ring_pixels)) if len(ctx_ring_pixels) > 0 else np.nan
    s['std_ctx']  = float(np.std(ctx_ring_pixels))  if len(ctx_ring_pixels) > 0 else np.nan
    s['var_ctx']  = s['std_ctx'] ** 2               if not np.isnan(s['std_ctx'])  else np.nan

    # α — box / ctx_total
    ctx_area  = max(1, int(np.sum(ring_mask)))
    box_area  = bw * bh
    ctx_total = ctx_area + box_area
    alpha = box_area / ctx_total

    # Δμ — 解耦：目标亮度 vs 真实背景均值
    if len(target_pixels) > 0:
        mu_target = float(np.mean(target_pixels))
    else:
        mu_target = float(np.mean(box_pixels)) if len(box_pixels) > 0 else s['mu_true']
    delta_mu = mu_target - s['mu_true'] if not np.isnan(s['mu_true']) else 0.0

    # 解耦输出 ─ 三项独立
    contamination_term = alpha * (1.0 - alpha) * delta_mu ** 2   # α(1-α)Δμ²
    s['alpha']              = alpha
    s['delta_mu']           = delta_mu
    s['mu_target']          = mu_target
    s['contamination_term'] = contamination_term                  # 纯污染贡献
    s['var_ctx_theory']     = s['var_true'] + contamination_term  # 理论预测
    s['target_area']        = int(np.sum(mask > 127)) if mask is not None else 0
    s['box_area']           = box_area
    s['expand_ratio']       = expand_ratio
    s['n_ring_pixels']      = len(ctx_ring_pixels)
    return true_bg_pixels, ctx_ring_pixels, box_pixels, s


# ────────────────────────────────────────────────────────────────
# 合成数据演示（无需本地数据集）
# ────────────────────────────────────────────────────────────────

def make_synthetic_sample(target_brightness, bg_mu, bg_sigma, img_size=256,
                           box_size=6, seed=42):
    """
    生成合成红外图像和对应 GT mask / YOLO box。
    target_brightness: 目标灰度均值（可小于 bg_mu 模拟暗目标）
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(bg_mu, bg_sigma, (img_size, img_size)).clip(0, 255).astype(np.float32)
    cx, cy = img_size // 2, img_size // 2
    half = box_size // 2
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    # 目标区域添加点扩散函数（2D Gaussian）
    yy, xx = np.mgrid[y1:y2, x1:x2]
    psf = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (half * 0.6) ** 2))
    img[y1:y2, x1:x2] += (target_brightness - bg_mu) * psf
    img = img.clip(0, 255).astype(np.uint8)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return img, mask, (x1, y1, x2, y2)


# ────────────────────────────────────────────────────────────────
# 核心绘图：TIP 风格 1×3
# ────────────────────────────────────────────────────────────────

def _kde_line(pixels, x_range):
    if len(pixels) < 5:
        return None
    if np.std(pixels) < 1e-3:   # 零方差时 KDE 退化为 delta，跳过
        return None
    try:
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            kde = gaussian_kde(pixels, bw_method='silverman')
            return kde(x_range)
    except Exception:
        return None


def draw_panel_a(ax, img_gray, mask, box, ds_label):
    """
    Panel A: 极小目标 / 紧 expand_ratio=1.1 → 严重污染
    展示 Context Ring PDF 大幅右移/展宽 vs True BG
    图内注释 α(1-α)Δμ²
    """
    ratio = 1.1
    true_bg, ctx_ring, _, stats = extract_regions(img_gray, mask, box, expand_ratio=ratio)

    mu0   = stats['mu_true']
    std0  = stats['std_true']
    lo = max(0,   mu0 - 5.5 * std0)
    hi = min(255, mu0 + 5.5 * max(std0, stats['std_ctx']))
    x_range = np.linspace(lo, hi, 600)

    y_bg  = _kde_line(true_bg,  x_range)
    y_ctx = _kde_line(ctx_ring, x_range)

    if y_bg is not None:
        ax.plot(x_range, y_bg, color='#1565C0', lw=2.5, ls='--',
                label=fr'True BG  ($\sigma_0={std0:.1f}$)', zorder=5)
        ax.axvline(mu0, color='#1565C0', lw=1.0, ls=':', alpha=0.55)

    if y_ctx is not None:
        ax.plot(x_range, y_ctx, color='#C62828', lw=2.5,
                label=fr'Ctx Ring (r={ratio})  $\sigma={stats["std_ctx"]:.1f}$', zorder=4)
        ax.fill_between(x_range, y_ctx, alpha=0.18, color='#C62828')
        ax.axvline(stats['mu_ctx'], color='#C62828', lw=1.0, ls=':', alpha=0.55)

    # 标注污染项（LaTeX-style text）
    term_val = stats['contamination_term']
    alpha_v  = stats['alpha']
    dmu_v    = stats['delta_mu']
    var0     = stats['var_true']
    var_ctx  = stats['var_ctx']
    ann = (rf"$\alpha={alpha_v:.3f}$"  "\n"
           rf"$\Delta\mu={dmu_v:.1f}$"  "\n"
           rf"$\alpha(1-\alpha)\Delta\mu^2={term_val:.1f}$"  "\n"
           rf"$\hat{{\sigma}}^2_{{\mathrm{{ctx}}}}/\sigma^2_0={var_ctx/var0:.2f}\times$")
    ax.text(0.97, 0.97, ann, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', alpha=0.88,
                      edgecolor='#C62828', lw=0.8),
            color='#B71C1C')

    ax.set_title(f'(a) Severe Contamination\n{ds_label}  $R={ratio}$',
                 fontsize=10.5, fontweight='bold', pad=6)
    ax.set_xlabel('Pixel Intensity', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.9, loc='upper left')
    ax.grid(True, ls='--', lw=0.5, alpha=0.45)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_edgecolor('#888')


def draw_panel_b(ax, img_gray, mask, box, ds_label):
    """
    Panel B: 中等目标 / 默认 expand_ratio=1.5 → 安全平台
    Context Ring PDF 贴合 True BG
    """
    ratio = 1.5
    true_bg, ctx_ring, _, stats = extract_regions(img_gray, mask, box, expand_ratio=ratio)

    mu0   = stats['mu_true']
    std0  = stats['std_true']
    lo = max(0,   mu0 - 5.0 * std0)
    hi = min(255, mu0 + 5.0 * std0)
    x_range = np.linspace(lo, hi, 600)

    y_bg  = _kde_line(true_bg,  x_range)
    y_ctx = _kde_line(ctx_ring, x_range)

    if y_bg is not None:
        ax.plot(x_range, y_bg, color='#1565C0', lw=2.5, ls='--',
                label=fr'True BG  ($\sigma_0={std0:.1f}$)', zorder=5)

    if y_ctx is not None:
        ax.plot(x_range, y_ctx, color='#2E7D32', lw=2.5,
                label=fr'Ctx Ring (r={ratio})  $\sigma={stats["std_ctx"]:.1f}$', zorder=4)
        ax.fill_between(x_range, y_ctx, alpha=0.18, color='#2E7D32')

    # KL 距离估计（均值/方差差）
    term_val = stats['contamination_term']
    var0     = stats['var_true']
    var_ctx  = stats['var_ctx']
    ann = (rf"$\alpha={stats['alpha']:.3f}$"  "\n"
           rf"$\alpha(1-\alpha)\Delta\mu^2={term_val:.1f}$"  "\n"
           rf"$\hat{{\sigma}}^2_{{\mathrm{{ctx}}}}/\sigma^2_0={var_ctx/var0:.2f}\times$"  "\n"
           r"$\approx$ No significant bias")
    ax.text(0.97, 0.97, ann, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', alpha=0.88,
                      edgecolor='#2E7D32', lw=0.8),
            color='#1B5E20')

    ax.set_title(f'(b) Safe Platform (Design Choice)\n{ds_label}  $R={ratio}$',
                 fontsize=10.5, fontweight='bold', pad=6)
    ax.set_xlabel('Pixel Intensity', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.9, loc='upper left')
    ax.grid(True, ls='--', lw=0.5, alpha=0.45)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_edgecolor('#888')


def draw_panel_c(ax, data_nudt, data_nuaa):
    """
    Panel C: 污染曲线 — σ²_ctx/σ²₀ vs expand_ratio
    红色折线=NUDT(极小目标), 绿色折线=NUAA(中等目标)
    背景: r<1.3 危险区(浅红), r≥1.5 安全平台(浅绿)
    右 Y 轴: α 值
    """
    # 危险区 / 安全平台 背景
    ax.axvspan(min(EXPAND_RATIOS), 1.3,  alpha=0.12, color='#FF1744', label='_nolegend_')
    ax.axvspan(1.5, max(EXPAND_RATIOS), alpha=0.10, color='#00C853', label='_nolegend_')

    # 临界线（标签用轴坐标系，不影响数据 Y 轴范围）
    ax.axvline(1.3, color='#FF1744', lw=1.0, ls='--', alpha=0.7)
    ax.axvline(1.5, color='#00C853', lw=1.0, ls='--', alpha=0.7)
    # 使用 ax.get_xaxis_transform() 让 x 是数据坐标、y 是轴比例坐标
    trans = ax.get_xaxis_transform()
    ax.text(1.3 + 0.02, 0.97, r'$R_{\min}=1.3$', fontsize=7.5, color='#BF360C',
            va='top', transform=trans)
    ax.text(1.5 + 0.02, 0.97, r'$R^*=1.5$', fontsize=7.5, color='#1B5E20',
            va='top', transform=trans)

    def plot_curve(ax, ratio_list, contamination_data, color, label, marker):
        ratios = [r for r, _ in contamination_data]
        vals   = [v for _, v in contamination_data]
        ax.plot(ratios, vals, marker=marker, color=color, lw=2.2,
                markersize=7, label=label, zorder=4)
        for r, v in zip(ratios, vals):
            ax.annotate(f'{v:.2f}', xy=(r, v), xytext=(0, 5),
                        textcoords='offset points', ha='center',
                        fontsize=7, color=color)

    # 基准线 σ²₀/σ²₀ = 1
    ax.axhline(1.0, color='#37474F', lw=1.5, ls='-', alpha=0.7,
               label=r'Baseline $\sigma^2_0 / \sigma^2_0 = 1$', zorder=3)

    if data_nudt:
        plot_curve(ax, EXPAND_RATIOS, data_nudt, '#C62828',
                   'NUDT-SIRST (tiny target)', 'o')
    if data_nuaa:
        plot_curve(ax, EXPAND_RATIOS, data_nuaa, '#2E7D32',
                   'NUAA-SIRST (medium target)', '^')

    # "Danger Zone" / "Safe Platform" — 轴坐标系，居中放置
    ax.text(0.12, 0.50, 'Danger\nZone', fontsize=8, color='#BF360C',
            ha='center', va='center', fontweight='bold', alpha=0.8,
            transform=ax.transAxes)
    ax.text(0.65, 0.50, 'Safe Platform', fontsize=8, color='#1B5E20',
            ha='center', va='center', fontweight='bold', alpha=0.8,
            transform=ax.transAxes)

    ax.set_xlabel(r'expand\_ratio $R$', fontsize=10)
    ax.set_ylabel(r'$\hat{\sigma}^2_{\mathrm{ctx}} \;/\; \sigma^2_0$', fontsize=11)
    ax.set_title('(c) Contamination Factor vs. Expand Ratio', fontsize=10.5,
                 fontweight='bold', pad=6)
    ax.legend(fontsize=8.5, framealpha=0.9, loc='upper right')
    ax.grid(True, ls='--', lw=0.5, alpha=0.45)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_edgecolor('#888')


# ────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────

def load_sample(ds_name, sample_name, root):
    cfg = DATASET_CONFIGS[ds_name]
    img_bgr  = cv2.imread(os.path.join(root, cfg['img_dir'], sample_name + cfg['img_ext']))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {sample_name}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape
    mask = cv2.imread(os.path.join(root, cfg['mask_dir'], sample_name + cfg['img_ext']), 0)
    boxes = load_yolo_boxes(
        os.path.join(root, cfg['label_dir'], sample_name + '.txt'), H, W)
    return img_gray, mask, boxes


def compute_contamination_curve(img_gray, mask, box):
    """返回 [(expand_ratio, var_ctx/var0)] 列表"""
    result = []
    for ratio in EXPAND_RATIOS:
        _, ctx_ring, _, stats = extract_regions(img_gray, mask, box, expand_ratio=ratio)
        if len(ctx_ring) < 5 or np.isnan(stats['var_true']) or stats['var_true'] < 1:
            continue
        result.append((ratio, stats['var_ctx'] / stats['var_true']))
    return result


def save_csv(img_gray, mask, box, out_dir, ds_name, sample_name):
    """解耦输出：σ²₀、污染项、α、Δμ 各自独立列"""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'stats_{ds_name}_{sample_name}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'expand_ratio',
            'mu_true', 'std_true', 'var_true',       # σ²₀ 基准
            'mu_ctx',  'std_ctx',  'var_ctx',          # 实测
            'var_ctx_theory',                          # 理论预测
            'contamination_term',                      # 纯污染 α(1-α)Δμ²  (解耦!)
            'alpha', 'delta_mu', 'mu_target',          # 各项独立
            'contamination_ratio',                     # var_ctx/var_true
            'target_area_px2', 'box_area_px2', 'n_ring_px',
        ])
        for ratio in EXPAND_RATIOS:
            _, ctx_ring, _, s = extract_regions(img_gray, mask, box, expand_ratio=ratio)
            if len(ctx_ring) < 5 or np.isnan(s['var_true']):
                continue
            writer.writerow([
                ratio,
                f"{s['mu_true']:.3f}",  f"{s['std_true']:.3f}",  f"{s['var_true']:.3f}",
                f"{s['mu_ctx']:.3f}",   f"{s['std_ctx']:.3f}",   f"{s['var_ctx']:.3f}",
                f"{s['var_ctx_theory']:.3f}",
                f"{s['contamination_term']:.3f}",
                f"{s['alpha']:.5f}",    f"{s['delta_mu']:.3f}",   f"{s['mu_target']:.3f}",
                f"{s['var_ctx']/s['var_true']:.4f}",
                s['target_area'], s['box_area'], s['n_ring_pixels'],
            ])
    print(f"CSV  → {csv_path}")
    return csv_path


def batch_analysis(ds_name, root, n_samples, out_dir):
    """批量统计：平均 σ²_ctx/σ²₀ 与理论预测的 Pearson 相关"""
    cfg = DATASET_CONFIGS[ds_name]
    split_file = os.path.join(root, cfg['split'])
    with open(split_file) as f:
        names = [l.strip() for l in f if l.strip()]

    records = {r: {'measured': [], 'theory': [], 'alpha': [], 'delta_mu': [], 'var0': []}
               for r in EXPAND_RATIOS}
    count = 0
    for name in names:
        if count >= n_samples:
            break
        lp = os.path.join(root, cfg['label_dir'], name + '.txt')
        if not os.path.exists(lp) or os.path.getsize(lp) == 0:
            continue
        img_bgr = cv2.imread(os.path.join(root, cfg['img_dir'], name + cfg['img_ext']))
        if img_bgr is None:
            continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W = img_gray.shape
        mask = cv2.imread(os.path.join(root, cfg['mask_dir'], name + cfg['img_ext']), 0)
        boxes = load_yolo_boxes(lp, H, W)
        if not boxes:
            continue
        for box in boxes:
            for ratio in EXPAND_RATIOS:
                _, ctx_ring, _, s = extract_regions(img_gray, mask, box, expand_ratio=ratio)
                if len(ctx_ring) < 5 or np.isnan(s['var_true']) or s['var_true'] < 1:
                    continue
                records[ratio]['measured'].append(s['var_ctx']        / s['var_true'])
                records[ratio]['theory'].append(s['var_ctx_theory']   / s['var_true'])
                records[ratio]['alpha'].append(s['alpha'])
                records[ratio]['delta_mu'].append(abs(s['delta_mu']))
                records[ratio]['var0'].append(s['var_true'])
        count += 1

    print(f"\n=== Batch Analysis: {ds_name}  (n={count}) ===")
    hdr = (f"{'R':>5} | {'σ²_ctx/σ²₀':>18} | {'Theory':>10} | "
           f"{'α mean':>8} | {'|Δμ| mean':>10} | {'σ²₀ mean':>10} | Pearson-r")
    print(hdr); print('-' * len(hdr))
    for ratio in EXPAND_RATIOS:
        m = np.array(records[ratio]['measured'])
        t = np.array(records[ratio]['theory'])
        if len(m) == 0:
            continue
        r_corr = np.corrcoef(m, t)[0, 1] if len(m) > 2 else float('nan')
        print(f"  {ratio:<5.1f} | {np.mean(m):>8.3f} ± {np.std(m):>5.3f}    "
              f"| {np.mean(t):>10.3f} | {np.mean(records[ratio]['alpha']):>8.4f} "
              f"| {np.mean(records[ratio]['delta_mu']):>10.2f} "
              f"| {np.mean(records[ratio]['var0']):>10.2f} "
              f"| {r_corr:.4f}")

    # 批量曲线图
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ratios_plot = [r for r in EXPAND_RATIOS if records[r]['measured']]
    means = [np.mean(records[r]['measured']) for r in ratios_plot]
    stds  = [np.std(records[r]['measured'])  for r in ratios_plot]
    ax.errorbar(ratios_plot, means, yerr=stds, fmt='o-', color='#C62828',
                capsize=5, lw=2, markersize=7, label='Measured mean ± std')
    ax.axhline(1.0, color='#37474F', lw=1.5, ls='--', label='σ²₀ baseline')
    ax.axvspan(min(EXPAND_RATIOS), 1.3, alpha=0.12, color='#FF1744')
    ax.axvspan(1.5, max(EXPAND_RATIOS), alpha=0.10, color='#00C853')
    ax.set_xlabel(r'expand\_ratio $R$', fontsize=11)
    ax.set_ylabel(r'$\hat{\sigma}^2_{\mathrm{ctx}} / \sigma^2_0$', fontsize=11)
    ax.set_title(f'Contamination Factor — {ds_name} (n={count})', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.4)
    out_path = os.path.join(out_dir, f'contamination_batch_{ds_name.replace("-","_")}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Batch plot → {out_path}")


# ────────────────────────────────────────────────────────────────
# 合成演示模式（无数据集）
# ────────────────────────────────────────────────────────────────

def demo_mode(out_path):
    """
    用合成数据生成 TIP 1×3 图，验证脚本逻辑正确性
    NUDT 场景: 极小目标 (6×6px), 极暗目标, 背景纹理丰富
    NUAA 场景: 中等目标 (20×20px), 较亮目标
    """
    # ── NUDT-like ────────────────────────────────────────────────
    # 挑战性: 目标比背景更暗 (暗目标), 背景高噪声
    img_nudt, mask_nudt, box_nudt = make_synthetic_sample(
        target_brightness=90, bg_mu=120, bg_sigma=22, img_size=256,
        box_size=6, seed=7)

    # ── NUAA-like ────────────────────────────────────────────────
    img_nuaa, mask_nuaa, box_nuaa = make_synthetic_sample(
        target_brightness=220, bg_mu=110, bg_sigma=18, img_size=256,
        box_size=20, seed=42)

    # 污染曲线数据
    curve_nudt = compute_contamination_curve(img_nudt, mask_nudt, box_nudt)
    curve_nuaa = compute_contamination_curve(img_nuaa, mask_nuaa, box_nuaa)

    # ── 绘图 ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.32, left=0.06, right=0.97, top=0.88, bottom=0.13)

    draw_panel_a(axes[0], img_nudt, mask_nudt, box_nudt, 'NUDT-SIRST (tiny, dark)')
    draw_panel_b(axes[1], img_nuaa, mask_nuaa, box_nuaa, 'NUAA-SIRST (medium, bright)')

    draw_panel_c(axes[2], curve_nudt, curve_nuaa)

    fig.suptitle(
        'Variance Contamination Theory: '
        r'$\hat{\sigma}^2_{\mathrm{ctx}} \approx \sigma^2_0 + \alpha(1-\alpha)\Delta\mu^2$',
        fontsize=12, fontweight='bold', y=0.995,
    )
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Demo saved → {out_path}")
    plt.close()


def main_single(args):
    """单样本 / 真实数据模式"""
    # ── 加载 NUDT 样本（Panel A：污染严重）────────────────────────
    if args.nudt:
        img_nudt, mask_nudt, boxes_nudt = load_sample('NUDT-SIRST', args.nudt, args.root)
        box_nudt = boxes_nudt[0]
        label_nudt = f'NUDT  {args.nudt}'
        curve_nudt = compute_contamination_curve(img_nudt, mask_nudt, box_nudt)
        save_csv(img_nudt, mask_nudt, box_nudt, args.out_dir, 'NUDT', args.nudt)
    else:
        img_nudt, mask_nudt, box_nudt, label_nudt, curve_nudt = None, None, None, 'NUDT', []

    # ── 加载 NUAA 样本（Panel B：安全平台）──────────────────────
    if args.nuaa:
        img_nuaa, mask_nuaa, boxes_nuaa = load_sample('NUAA-SIRST', args.nuaa, args.root)
        box_nuaa = boxes_nuaa[0]
        label_nuaa = f'NUAA  {args.nuaa}'
        curve_nuaa = compute_contamination_curve(img_nuaa, mask_nuaa, box_nuaa)
        save_csv(img_nuaa, mask_nuaa, box_nuaa, args.out_dir, 'NUAA', args.nuaa)
    else:
        img_nuaa, mask_nuaa, box_nuaa, label_nuaa, curve_nuaa = None, None, None, 'NUAA', []

    # ── 绘图 ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.32, left=0.06, right=0.97, top=0.88, bottom=0.13)

    if img_nudt is not None:
        draw_panel_a(axes[0], img_nudt, mask_nudt, box_nudt, label_nudt)
    if img_nuaa is not None:
        draw_panel_b(axes[1], img_nuaa, mask_nuaa, box_nuaa, label_nuaa)
    draw_panel_c(axes[2], curve_nudt, curve_nuaa)

    fig.suptitle(
        'Variance Contamination Theory: '
        r'$\hat{\sigma}^2_{\mathrm{ctx}} \approx \sigma^2_0 + \alpha(1-\alpha)\Delta\mu^2$',
        fontsize=12, fontweight='bold', y=0.995,
    )

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',    default=ROOT)
    parser.add_argument('--out', default=os.path.join(
        ROOT, 'paper', 'figures', 'fig_variance_theory.png'))
    parser.add_argument('--out_dir', default=os.path.join(
        ROOT, 'paper', 'figures', 'variance_contamination'))
    parser.add_argument('--nuaa',  default='')
    parser.add_argument('--nudt',  default='')
    parser.add_argument('--irstd', default='')
    parser.add_argument('--demo',  action='store_true',
                        help='Generate demo figure with synthetic data (no dataset needed)')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--dataset', default='NUDT-SIRST', choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--n', type=int, default=50)
    args = parser.parse_args()

    if args.demo:
        demo_mode(args.out)
    elif args.batch:
        batch_analysis(args.dataset, args.root, args.n, args.out_dir)
    else:
        main_single(args)
