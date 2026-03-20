#!/usr/bin/env python3
"""
Figure: Variance Contamination Theory — TIP-Style 1×3 Empirical Validation
============================================================================
排版: 1 行 × 3 列

  Panel A (左) ── 极小目标重度污染区 (NUDT, expand_ratio=1.1)
    Ctx Window PDF (红, 含目标像素) 严重右移/展宽 vs True BG (蓝虚线)
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

  ── 【推荐】服务器：用真实数据生成 panel_a/b/c 三张独立图 ──────────────────
    python paper/fig_variance_contamination.py \
        --nudt 000259 --nudt 000259 \
        --out paper/figures/fig_variance_theory.png \
        --split
    # 同时在 paper/figures/variance_contamination/ 下生成
    #   stats_NUDT_000259.csv  和  stats_NUAA_Misc_87.csv

  ── 服务器：用真实数据生成合并版（单张 1×3）────────────────────────────────
    python paper/fig_variance_contamination.py \
        --nudt 000259 --nuaa Misc_87 \
        --out paper/figures/fig_variance_theory.png

  ── 服务器：先生成 CSV，本地再绘图（适合图像不在本地的情况）─────────────────
    # Step 1 在服务器：生成 CSV（会自动保存到 --out_dir）
    python paper/fig_variance_contamination.py \
        --nudt 000259 --nuaa Misc_87 \
        --out paper/figures/fig_variance_theory.png
    # Step 2 把 CSV 拷到本地后，本地绘图（Panel A/B 需要图像，Panel C 仅需 CSV）
    python paper/fig_variance_contamination.py \
        --csv_nudt paper/figures/variance_contamination/stats_NUDT_000259.csv \
        --csv_nuaa paper/figures/variance_contamination/stats_NUAA_Misc_87.csv \
        --out paper/figures/fig_variance_theory.png \
        --split

  ── 服务器：批量统计（输出平均污染曲线，供论文引用）────────────────────────
    python paper/fig_variance_contamination.py \
        --batch --dataset NUDT-SIRST --n 50 \
        --out_dir paper/figures/variance_batch

  ── 本地演示（合成数据，无需数据集，验证脚本逻辑）──────────────────────────
    python paper/fig_variance_contamination.py --demo --split
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


def extract_ctx_window(img_gray, box, expand_ratio, exclude_mask=None):
    """
    提取整个 ctx window（box + ring）的像素，含目标像素。
    这是 HALO 方法在没有 GT mask 时的实际估计场景：
    目标亮斑混入 ctx_total → 方差被污染。

    exclude_mask: 若提供，则排除该区域（用于获取纯背景对比基准）。
    返回 (pixels, alpha, n_total)
    """
    H, W = img_gray.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    cx1, cy1, cx2, cy2 = get_context_box(box, expand_ratio, H, W)
    window_mask = np.zeros((H, W), dtype=bool)
    window_mask[cy1:cy2, cx1:cx2] = True

    if exclude_mask is not None:
        window_mask[exclude_mask > 127] = False

    pixels = img_gray[window_mask].astype(np.float32)

    # α = box_area / ctx_total_area（目标在 ctx window 中的占比）
    box_area  = bw * bh
    n_window  = max(1, int(np.sum(window_mask)))
    if exclude_mask is not None:
        n_window += int(np.sum(exclude_mask[cy1:cy2, cx1:cx2] > 127))
    alpha = box_area / max(1, n_window)

    return pixels, alpha, n_window


def extract_true_bg(img_gray, mask, box, bg_margin=40):
    """
    提取图像中远离目标的纯背景像素（排除目标 GT mask）。
    用于计算 σ²₀（真实背景方差基准）。
    """
    H, W = img_gray.shape
    x1, y1, x2, y2 = box
    bg_mask = np.ones((H, W), dtype=bool)
    bg_mask[max(0, y1 - bg_margin):min(H, y2 + bg_margin),
            max(0, x1 - bg_margin):min(W, x2 + bg_margin)] = False
    if mask is not None:
        bg_mask[mask > 127] = False
    return img_gray[bg_mask].astype(np.float32)


def extract_regions(img_gray, mask, box, expand_ratio, true_bg_margin=40):
    """
    返回 (true_bg_pixels, ctx_ring_pixels, box_pixels, stats_dict)
    stats_dict 包含解耦后的各项统计量（σ²₀、污染项、α、Δμ 均独立输出）
    """
    H, W = img_gray.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    true_bg_pixels = extract_true_bg(img_gray, mask, box, true_bg_margin)

    # ── Context Ring（排除目标像素）────────────────────────────
    cx1, cy1, cx2, cy2 = get_context_box(box, expand_ratio, H, W)
    ctx_mask   = np.zeros((H, W), dtype=bool)
    inner_mask = np.zeros((H, W), dtype=bool)
    ctx_mask[cy1:cy2, cx1:cx2] = True
    inner_mask[y1:y2, x1:x2]   = True
    ring_mask = ctx_mask & ~inner_mask
    if mask is not None:
        ring_mask[mask > 127] = False
    ctx_ring_pixels = img_gray[ring_mask].astype(np.float32)

    # ── Ctx Window（含目标，不排除）─── 理论验证用 ────────────
    ctx_win_pixels, alpha, _ = extract_ctx_window(img_gray, box, expand_ratio)

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

    # 含目标的 ctx window 统计量（方差污染实测）
    s['var_ctx_win'] = float(np.var(ctx_win_pixels)) if len(ctx_win_pixels) > 0 else np.nan

    box_area  = bw * bh
    ctx_total = max(1, int(np.sum(ring_mask))) + box_area

    # Δμ — 解耦：目标亮度 vs 真实背景均值
    if len(target_pixels) > 0:
        mu_target = float(np.mean(target_pixels))
    else:
        mu_target = float(np.mean(box_pixels)) if len(box_pixels) > 0 else s['mu_true']
    delta_mu = mu_target - s['mu_true'] if not np.isnan(s['mu_true']) else 0.0

    contamination_term = alpha * (1.0 - alpha) * delta_mu ** 2
    s['alpha']              = alpha
    s['delta_mu']           = delta_mu
    s['mu_target']          = mu_target
    s['contamination_term'] = contamination_term
    s['var_ctx_theory']     = s['var_true'] + contamination_term
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


def _extract_ring_and_window(img_gray, box, expand_ratio):
    """
    返回 (ring_pixels, window_pixels, alpha, mu_ring, delta_mu)
    ─ ring_pixels:   ctx box 内排除 box interior 的像素（局部纯背景估计，不依赖 GT mask）
    ─ window_pixels: ctx box 内全部像素，含 box interior（含目标，被污染）
    ─ alpha:         box_area / ctx_window_area（混入比例）
    ─ mu_ring:       ring 均值
    ─ delta_mu:      box interior 均值 - ring 均值（正=亮目标，负=暗目标）
    """
    H, W = img_gray.shape
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = get_context_box(box, expand_ratio, H, W)

    ctx_mask   = np.zeros((H, W), dtype=bool)
    inner_mask = np.zeros((H, W), dtype=bool)
    ctx_mask[cy1:cy2, cx1:cx2]   = True
    inner_mask[y1:y2,  x1:x2]    = True
    ring_mask   = ctx_mask & ~inner_mask

    ring_pixels   = img_gray[ring_mask].astype(np.float32)
    window_pixels = img_gray[ctx_mask].astype(np.float32)

    box_area   = int(inner_mask.sum())
    n_window   = max(1, int(ctx_mask.sum()))
    alpha      = box_area / n_window

    mu_ring    = float(np.mean(ring_pixels))   if len(ring_pixels)   > 0 else 128.0
    box_pixels = img_gray[inner_mask].astype(np.float32)
    mu_box     = float(np.mean(box_pixels))    if len(box_pixels)    > 0 else mu_ring
    delta_mu   = mu_box - mu_ring

    return ring_pixels, window_pixels, alpha, mu_ring, delta_mu


def draw_panel_a(ax, img_gray, mask, box, ds_label):  # noqa: mask kept for API compat
    """
    Panel A: 极小目标，紧 expand_ratio=1.1 → box内像素混入 ctx window，方差被污染。

    比较对象：同一局部区域 (R=1.1) 的两种采样方式：
    ─ 蓝色虚线：Ctx Ring (R=1.1，排除 box interior) — 局部纯背景，代表 σ²₀
    ─ 红色实线：Ctx Window (R=1.1，含 box interior) — 被目标污染的估计
      → σ²_win = σ²_ring + α(1-α)Δμ²，始终 ≥ σ²_ring（无论亮/暗目标）

    此设计消除了全图异质性问题：两者来自同一局部 patch，差异纯粹来自 box 内目标混入。
    """
    ratio_small = 1.1
    ring_px, win_px, alpha, mu_ring, delta_mu = _extract_ring_and_window(
        img_gray, box, ratio_small)

    std_ring = float(np.std(ring_px)) if len(ring_px) > 5 else 1.0
    std_win  = float(np.std(win_px))  if len(win_px)  > 5 else 1.0
    ct = alpha * (1.0 - alpha) * delta_mu ** 2

    lo = max(0,   min(mu_ring, float(np.mean(win_px))) - 5.0 * max(std_ring, std_win, 1.0))
    hi = min(255, max(mu_ring, float(np.mean(win_px))) + 5.0 * max(std_ring, std_win, 1.0))
    x_range = np.linspace(lo, hi, 600)

    y_ring = _kde_line(ring_px, x_range)
    y_win  = _kde_line(win_px,  x_range)

    if y_ring is not None:
        ax.plot(x_range, y_ring, color='#1565C0', lw=2.5, ls='--', zorder=5)
        ax.fill_between(x_range, y_ring, alpha=0.10, color='#1565C0')

    if y_win is not None:
        ax.plot(x_range, y_win, color='#C62828', lw=2.5, zorder=4)
        ax.fill_between(x_range, y_win, alpha=0.18, color='#C62828')

    # 右上角文字标注（替代图例，避免遮挡峰值）
    # ax.text(0.97, 0.97, fr'Ctx Ring  $\sigma_0={std_ring:.1f}$',
    #         fontsize=8, color='#1565C0', ha='right', va='top', transform=ax.transAxes)
    # ax.text(0.97, 0.88, fr'Ctx Window  $\sigma={std_win:.1f}$',
    #         fontsize=8, color='#C62828', ha='right', va='top', transform=ax.transAxes)
    # ax.text(0.97, 0.76,
    #         r'$\alpha(1-\alpha)\Delta\mu^2$' + f' = {ct:.1f}',
    #         fontsize=8, color='#C62828', ha='right', va='top', transform=ax.transAxes)
    # ax.text(0.97, 0.67, fr'$\alpha = {alpha:.3f}$',
    #         fontsize=8, color='#555555', ha='right', va='top', transform=ax.transAxes)

    ax.set_title(f'(a) Severe Contamination\n{ds_label}  $R={ratio_small}$',
                 fontsize=10.5, fontweight='bold', pad=6)
    ax.set_xlabel('Pixel Intensity', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.grid(True, ls='--', lw=0.5, alpha=0.45)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_edgecolor('#888')


def draw_panel_b(ax, img_gray, mask, box, ds_label, ratio_design=1.5):  # noqa: mask kept for API compat
    """
    Panel B: 充分扩展 ctx (默认 R=1.5，演示用 R=3.0) → α 小，window 与 ring 高度重叠（安全平台）。

    ─ 蓝色虚线：Ctx Ring (ratio_design，排除 box) — 局部纯背景基准
    ─ 绿色实线：Ctx Window (ratio_design，含 box) — 污染已被充分稀释

    对比 Panel A（同一目标，更小 R）→ Ring 与 Window 的差距缩小，说明污染缓解
    """
    ring_px, win_px, alpha, mu_ring, delta_mu = _extract_ring_and_window(
        img_gray, box, ratio_design)

    std_ring = float(np.std(ring_px)) if len(ring_px) > 5 else 1.0
    std_win  = float(np.std(win_px))  if len(win_px)  > 5 else 1.0
    ct = alpha * (1.0 - alpha) * delta_mu ** 2

    lo = max(0,   min(mu_ring, float(np.mean(win_px))) - 5.0 * max(std_ring, std_win, 1.0))
    hi = min(255, max(mu_ring, float(np.mean(win_px))) + 5.0 * max(std_ring, std_win, 1.0))
    x_range = np.linspace(lo, hi, 600)

    y_ring = _kde_line(ring_px, x_range)
    y_win  = _kde_line(win_px,  x_range)

    if y_ring is not None:
        ax.plot(x_range, y_ring, color='#1565C0', lw=2.5, ls='--', zorder=5)
        ax.fill_between(x_range, y_ring, alpha=0.10, color='#1565C0')

    if y_win is not None:
        ax.plot(x_range, y_win, color='#2E7D32', lw=2.5, zorder=4)
        ax.fill_between(x_range, y_win, alpha=0.18, color='#2E7D32')

    # 右上角文字标注（替代图例）
    # ax.text(0.97, 0.97, fr'Ctx Ring  $\sigma_0={std_ring:.1f}$',
    #         fontsize=8, color='#1565C0', ha='right', va='top', transform=ax.transAxes)
    # ax.text(0.97, 0.88, fr'Ctx Window  $\sigma={std_win:.1f}$',
    #         fontsize=8, color='#2E7D32', ha='right', va='top', transform=ax.transAxes)
    # ax.text(0.97, 0.76,
    #         r'$\alpha(1-\alpha)\Delta\mu^2$' + f' = {ct:.1f}',
    #         fontsize=8, color='#2E7D32', ha='right', va='top', transform=ax.transAxes)
    # ax.text(0.97, 0.67, fr'$\alpha = {alpha:.3f}$',
    #         fontsize=8, color='#555555', ha='right', va='top', transform=ax.transAxes)

    ax.set_title(f'(b) Mitigated: Larger Context\n{ds_label}  $R={ratio_design}$',
                 fontsize=10.5, fontweight='bold', pad=6)
    ax.set_xlabel('Pixel Intensity', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.grid(True, ls='--', lw=0.5, alpha=0.45)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_edgecolor('#888')


def draw_panel_c(ax, data_nudt, data_nuaa):
    """
    Panel C: 污染曲线 — σ²_win/σ²_ring vs expand_ratio

    data 格式支持两种：
      单样本：[(ratio, val), ...]
      批量：  [(ratio, mean, std, n), ...]  — 画均值线 + ±1std 阴影置信区间
    """
    ax.axvspan(min(EXPAND_RATIOS), 1.3,  alpha=0.12, color='#FF1744', label='_nolegend_')
    ax.axvspan(1.5, max(EXPAND_RATIOS), alpha=0.10, color='#00C853', label='_nolegend_')

    ax.axvline(1.3, color='#FF1744', lw=1.0, ls='--', alpha=0.7)
    ax.axvline(1.5, color='#00C853', lw=1.0, ls='--', alpha=0.7)

    def plot_curve(ax_, data, color, label, marker):
        is_batch = len(data[0]) == 5   # (ratio, median, q25, q75, n)
        ratios  = [d[0] for d in data]
        medians = [d[1] for d in data]
        ax_.plot(ratios, medians, marker=marker, color=color, lw=2.2,
                 markersize=7, label=label, zorder=4)
        if is_batch:
            q25s = [d[2] for d in data]
            q75s = [d[3] for d in data]
            ax_.fill_between(ratios, q25s, q75s,
                             alpha=0.20, color=color, zorder=3,
                             label='_nolegend_')
        else:
            for r, v in zip(ratios, medians):
                ax_.annotate(f'{v:.2f}', xy=(r, v), xytext=(0, 5),
                             textcoords='offset points', ha='center',
                             fontsize=7, color=color)

    # 基准线 = 1
    ax.axhline(1.0, color='#37474F', lw=1.5, ls='-', alpha=0.7,
               label=r'Baseline  $\sigma^2_{\mathrm{ring}}$', zorder=3)

    if data_nudt:
        is_b = len(data_nudt[0]) == 5
        n_str = f', N={data_nudt[0][4]}' if is_b else ''
        lbl   = f'NUDT-SIRST (tiny target{n_str})' + (' median±IQR' if is_b else '')
        plot_curve(ax, data_nudt, '#C62828', lbl, 'o')
    if data_nuaa:
        is_b = len(data_nuaa[0]) == 5
        n_str = f', N={data_nuaa[0][4]}' if is_b else ''
        lbl   = f'NUAA-SIRST (medium target{n_str})' + (' median±IQR' if is_b else '')
        plot_curve(ax, data_nuaa, '#2E7D32', lbl, '^')

    ax.text(0.12, 0.75, 'Danger\nZone', fontsize=8, color='#BF360C',
            ha='center', va='center', fontweight='bold', alpha=0.8,
            transform=ax.transAxes)
    ax.text(0.72, 0.25, 'Design\nRegion', fontsize=8, color='#1B5E20',
            ha='center', va='center', fontweight='bold', alpha=0.8,
            transform=ax.transAxes)

    ax.set_xlabel(r'expand\_ratio $R$', fontsize=10)
    ax.set_ylabel(r'$\sigma^2_{\mathrm{win}} \;/\; \sigma^2_{\mathrm{ring}}$', fontsize=11)
    ax.set_title('(c) Contamination Factor vs. Expand Ratio', fontsize=10.5,
                 fontweight='bold', pad=6)
    ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
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


def compute_contamination_curve(img_gray, mask, box):  # noqa: mask kept for API compat
    """
    返回 [(expand_ratio, var_ctx_win / var_ctx_ring)] 列表

    ─ 分母：Ctx Ring（同 R，排除 box interior）的方差 = 局部纯背景方差 σ²_ring
    ─ 分子：Ctx Window（同 R，含 box interior）的方差 = 被污染的估计 σ²_win

    由总方差公式：σ²_win = σ²_ring + α(1-α)Δμ² ≥ σ²_ring
    → 比值始终 ≥ 1（亮/暗目标均成立），R 越小 α 越大 → 比值越高（危险区）
                                           R 越大 α 越小 → 比值 → 1 （安全区）
    此设计无需 GT mask，也不受全图背景异质性影响。
    """
    result = []
    for ratio in EXPAND_RATIOS:
        ring_px, win_px, _, _, _ = _extract_ring_and_window(img_gray, box, ratio)
        if len(ring_px) < 5 or len(win_px) < 5:
            continue
        var_ring = float(np.var(ring_px))
        if var_ring < 0.1:      # 极度均匀背景，比值无意义
            continue
        result.append((ratio, float(np.var(win_px)) / var_ring))
    return result


def compute_batch_contamination_curve(ds_name, root, n_samples=100):
    """
    批量扫描数据集，返回 [(ratio, mean, std, n)] 格式的平均污染曲线。
    只统计 var_ring > 1 的有效样本，自动剔除异常值（>mean+3std）。
    """
    cfg = DATASET_CONFIGS.get(ds_name)
    if cfg is None:
        return []
    split_file = os.path.join(root, cfg['split'])
    if not os.path.exists(split_file):
        return []
    with open(split_file) as f:
        names = [l.strip() for l in f if l.strip()]

    records = {r: [] for r in EXPAND_RATIOS}
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
        boxes = load_yolo_boxes(lp, H, W)
        if not boxes:
            continue
        for box in boxes:
            for ratio in EXPAND_RATIOS:
                ring_px, win_px, _, _, _ = _extract_ring_and_window(img_gray, box, ratio)
                if len(ring_px) < 5 or len(win_px) < 5:
                    continue
                var_ring = float(np.var(ring_px))
                if var_ring < 1.0:
                    continue
                records[ratio].append(float(np.var(win_px)) / var_ring)
        count += 1

    print(f"[Batch] {ds_name}: {count} images scanned")
    result = []
    for ratio in EXPAND_RATIOS:
        vals = np.array(records[ratio])
        if len(vals) < 3:
            continue
        # 用中位数 + IQR（25%-75%），对极端值鲁棒，且下界自然 ≥ 1
        median = float(np.median(vals))
        q25    = float(np.percentile(vals, 25))
        q75    = float(np.percentile(vals, 75))
        result.append((ratio, median, q25, q75, len(vals)))
        print(f"  R={ratio}: median={median:.3f}  IQR=[{q25:.3f}, {q75:.3f}]  (n={len(vals)})")
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
            'mu_ctx',  'std_ctx',  'var_ctx',          # ctx ring (排除目标)
            'var_ctx_win',                             # ctx window (含目标) — 理论验证用
            'var_ctx_theory',                          # 理论预测
            'contamination_term',                      # 纯污染 α(1-α)Δμ²
            'alpha', 'delta_mu', 'mu_target',
            'contamination_ratio',                     # var_ctx_win/var_true ← 正确比值
            'target_area_px2', 'box_area_px2', 'n_ring_px',
        ])
        for ratio in EXPAND_RATIOS:
            _, ctx_ring, _, s = extract_regions(img_gray, mask, box, expand_ratio=ratio)
            if len(ctx_ring) < 5 or np.isnan(s['var_true']):
                continue
            var_win = s.get('var_ctx_win', np.nan)
            contamination_ratio = (var_win / s['var_true']) if not np.isnan(var_win) else np.nan
            writer.writerow([
                ratio,
                f"{s['mu_true']:.3f}",  f"{s['std_true']:.3f}",  f"{s['var_true']:.3f}",
                f"{s['mu_ctx']:.3f}",   f"{s['std_ctx']:.3f}",   f"{s['var_ctx']:.3f}",
                f"{var_win:.3f}" if not np.isnan(var_win) else '',
                f"{s['var_ctx_theory']:.3f}",
                f"{s['contamination_term']:.3f}",
                f"{s['alpha']:.5f}",    f"{s['delta_mu']:.3f}",   f"{s['mu_target']:.3f}",
                f"{contamination_ratio:.4f}" if not np.isnan(contamination_ratio) else '',
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
                ring_px, win_px, alpha, _, delta_mu = _extract_ring_and_window(
                    img_gray, box, ratio)
                if len(ring_px) < 5 or len(win_px) < 5:
                    continue
                var_ring = float(np.var(ring_px))
                if var_ring < 0.1:
                    continue
                var_win = float(np.var(win_px))
                # measured: σ²_win/σ²_ring（实测，始终≥1）
                # theory:  1 + α(1-α)Δμ²/σ²_ring（理论预测）
                theory_ratio = 1.0 + alpha * (1.0 - alpha) * delta_mu ** 2 / var_ring
                records[ratio]['measured'].append(var_win   / var_ring)
                records[ratio]['theory'].append(theory_ratio)
                records[ratio]['alpha'].append(alpha)
                records[ratio]['delta_mu'].append(abs(delta_mu))
                records[ratio]['var0'].append(var_ring)
        count += 1

    print(f"\n=== Batch Analysis: {ds_name}  (n={count}) ===")
    hdr = (f"{'R':>5} | {'σ²_win/σ²_ring':>18} | {'Theory':>10} | "
           f"{'α mean':>8} | {'|Δμ| mean':>10} | {'σ²_ring mean':>12} | Pearson-r")
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
    ax.set_ylabel(r'$\sigma^2_{\mathrm{win}} / \sigma^2_{\mathrm{ring}}$', fontsize=11)
    ax.set_title(f'Contamination Factor — {ds_name} (n={count})', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.4)
    out_path = os.path.join(out_dir, f'contamination_batch_{ds_name.replace("-","_")}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Batch plot → {out_path}")


# ────────────────────────────────────────────────────────────────
# 合成演示模式（无数据集）
# ────────────────────────────────────────────────────────────────

def demo_mode(out_path, split=False):
    """
    用合成数据生成 TIP 1×3 图，验证脚本逻辑正确性

    NUDT 场景: 极小目标 (8×8px), 亮目标 Δμ=35 (SNR≈1.9)，背景均匀 σ₀=18
      Panel A: R=1.1 → ctx_win 含目标，σ 显著 > σ₀（方差污染）
      Panel B: R=3.0 → ctx_win 充分稀释，σ ≈ σ₀（安全平台）
    NUAA 场景: 中等目标 (20×20px)，用于 Panel C 第二条曲线
      → 大目标 α 更大，污染曲线始终高于小目标，但趋势相同
    """
    # 模拟真实红外场景：σ₀=22，Δμ_nominal=80（PSF 有效 Δμ≈52，SNR≈2.4）
    # Panel A (R=1.1, α≈0.33): σ²≈σ²₀+595, σ≈30 >> σ₀=22（明显展宽 36%）
    # Panel B (R=3.0, α≈0.06): σ²≈σ²₀+158, σ≈25 ≈ σ₀=22（缓解至 9%，视觉接近重叠）
    img_nudt, mask_nudt, box_nudt = make_synthetic_sample(
        target_brightness=200, bg_mu=120, bg_sigma=22, img_size=256,
        box_size=8, seed=7)
    img_nuaa, mask_nuaa, box_nuaa = make_synthetic_sample(
        target_brightness=200, bg_mu=120, bg_sigma=22, img_size=256,
        box_size=20, seed=42)
    curve_nudt = compute_contamination_curve(img_nudt, mask_nudt, box_nudt)
    curve_nuaa = compute_contamination_curve(img_nuaa, mask_nuaa, box_nuaa)

    out_dir = os.path.dirname(out_path) or '.'
    base    = os.path.splitext(os.path.basename(out_path))[0]
    ext     = os.path.splitext(out_path)[1] or '.png'
    os.makedirs(out_dir, exist_ok=True)

    # Panel A: 小目标 R=1.1（严重污染），Panel B: 同一小目标 R=3.0（缓解）
    panel_b_label = f'NUDT-SIRST (same target, R=3.0)'
    if split:
        save_single_panel(draw_panel_a,
                          (img_nudt, mask_nudt, box_nudt, 'NUDT-SIRST (tiny target, R=1.1)'),
                          os.path.join(out_dir, f'{base}_panel_a{ext}'))
        save_single_panel(draw_panel_b,
                          (img_nudt, mask_nudt, box_nudt, panel_b_label, 3.0),
                          os.path.join(out_dir, f'{base}_panel_b{ext}'))
        save_single_panel(draw_panel_c,
                          (curve_nudt, curve_nuaa),
                          os.path.join(out_dir, f'{base}_panel_c{ext}'))
        return

    # ── 合并图 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.32, left=0.06, right=0.97, top=0.88, bottom=0.13)
    draw_panel_a(axes[0], img_nudt, mask_nudt, box_nudt, 'NUDT-SIRST (tiny target, R=1.1)')
    draw_panel_b(axes[1], img_nudt, mask_nudt, box_nudt, panel_b_label, ratio_design=3.0)
    draw_panel_c(axes[2], curve_nudt, curve_nuaa)
    fig.suptitle(
        'Variance Contamination Theory: '
        r'$\hat{\sigma}^2_{\mathrm{ctx}} \approx \sigma^2_0 + \alpha(1-\alpha)\Delta\mu^2$',
        fontsize=12, fontweight='bold', y=0.995,
    )
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Demo saved → {out_path}")
    plt.close()


def load_curve_from_csv(csv_path):
    """从 stats_*.csv 加载污染曲线 [(expand_ratio, contamination_ratio)]"""
    curve = []
    if not csv_path or not os.path.exists(csv_path):
        return curve
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                curve.append((float(row['expand_ratio']), float(row['contamination_ratio'])))
            except (KeyError, ValueError):
                pass
    return curve


def save_single_panel(draw_fn, draw_args, out_path, figsize=(5.5, 5.0)):
    """将单个 draw_panel_* 函数输出为独立 PNG"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.13, right=0.95, top=0.90, bottom=0.14)
    draw_fn(ax, *draw_args)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_path}")
    plt.close()


def main_single(args):
    """单样本 / 真实数据模式"""
    # ── 加载 NUDT 样本（Panel A：污染严重）────────────────────────
    if args.nudt:
        img_nudt, mask_nudt, boxes_nudt = load_sample('NUDT-SIRST', args.nudt, args.root)
        box_nudt = boxes_nudt[0]
        label_nudt = f'NUDT  {args.nudt}'
        save_csv(img_nudt, mask_nudt, box_nudt, args.out_dir, 'NUDT', args.nudt)
    else:
        img_nudt, mask_nudt, box_nudt, label_nudt = None, None, None, 'NUDT'

    # Panel C 曲线：优先批量平均，fallback 单样本
    if args.batch_n > 0:
        curve_nudt = compute_batch_contamination_curve('NUDT-SIRST', args.root, args.batch_n)
    elif args.nudt:
        curve_nudt = compute_contamination_curve(img_nudt, mask_nudt, box_nudt)
    else:
        curve_nudt = load_curve_from_csv(args.csv_nudt)

    # ── 加载 NUAA 样本（Panel B：安全平台）──────────────────────
    if args.nuaa:
        img_nuaa, mask_nuaa, boxes_nuaa = load_sample('NUAA-SIRST', args.nuaa, args.root)
        box_nuaa = boxes_nuaa[0]
        label_nuaa = f'NUAA  {args.nuaa}'
        save_csv(img_nuaa, mask_nuaa, box_nuaa, args.out_dir, 'NUAA', args.nuaa)
    else:
        img_nuaa, mask_nuaa, box_nuaa, label_nuaa = None, None, None, 'NUAA'

    if args.batch_n > 0:
        curve_nuaa = compute_batch_contamination_curve('NUAA-SIRST', args.root, args.batch_n)
    elif args.nuaa:
        curve_nuaa = compute_contamination_curve(img_nuaa, mask_nuaa, box_nuaa)
    else:
        curve_nuaa = load_curve_from_csv(args.csv_nuaa)

    out_dir = os.path.dirname(args.out) or '.'
    base    = os.path.splitext(os.path.basename(args.out))[0]
    ext     = os.path.splitext(args.out)[1] or '.png'

    # Panel A/B 优先使用 NUDT 样本（小目标、背景均匀，污染效果更典型）
    # 若 NUDT 未提供则 fallback 到 NUAA
    panel_ab_img  = img_nudt  if img_nudt  is not None else img_nuaa
    panel_ab_mask = mask_nudt if img_nudt  is not None else mask_nuaa
    panel_ab_box  = box_nudt  if img_nudt  is not None else box_nuaa
    panel_ab_lbl  = label_nudt if img_nudt is not None else label_nuaa

    if args.split:
        # ── 独立输出三个 Panel ──────────────────────────────────
        if panel_ab_img is not None:
            # Panel A: R=1.1（严重污染）
            save_single_panel(draw_panel_a,
                              (panel_ab_img, panel_ab_mask, panel_ab_box, panel_ab_lbl),
                              os.path.join(out_dir, f'{base}_panel_a{ext}'))
            # Panel B: R=3.0（充分缓解，安全平台）
            save_single_panel(draw_panel_b,
                              (panel_ab_img, panel_ab_mask, panel_ab_box,
                               panel_ab_lbl, 3.0),
                              os.path.join(out_dir, f'{base}_panel_b{ext}'))
        save_single_panel(draw_panel_c,
                          (curve_nudt, curve_nuaa),
                          os.path.join(out_dir, f'{base}_panel_c{ext}'))
    else:
        # ── 合并输出（默认）──────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(wspace=0.32, left=0.06, right=0.97, top=0.88, bottom=0.13)

        if panel_ab_img is not None:
            draw_panel_a(axes[0], panel_ab_img, panel_ab_mask, panel_ab_box, panel_ab_lbl)
            draw_panel_b(axes[1], panel_ab_img, panel_ab_mask, panel_ab_box,
                         panel_ab_lbl, ratio_design=3.0)
        draw_panel_c(axes[2], curve_nudt, curve_nuaa)

        fig.suptitle(
            'Variance Contamination Theory: '
            r'$\hat{\sigma}^2_{\mathrm{ctx}} \approx \sigma^2_0 + \alpha(1-\alpha)\Delta\mu^2$',
            fontsize=12, fontweight='bold', y=0.995,
        )
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved → {args.out}")
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
    # 从已有 CSV 直接加载 Panel C 污染曲线（无需数据集）
    parser.add_argument('--csv_nudt', default='', metavar='PATH',
                        help='stats_NUDT_*.csv path for Panel C curve')
    parser.add_argument('--csv_nuaa', default='', metavar='PATH',
                        help='stats_NUAA_*.csv path for Panel C curve')
    parser.add_argument('--split', action='store_true',
                        help='Output each panel as a separate PNG (for manual assembly)')
    parser.add_argument('--demo',  action='store_true',
                        help='Generate demo figure with synthetic data (no dataset needed)')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--batch_n', type=int, default=100,
                        help='Panel C 批量平均样本数（0=用单样本曲线，默认100）')
    parser.add_argument('--dataset', default='NUDT-SIRST', choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--n', type=int, default=50)
    args = parser.parse_args()

    if args.demo:
        demo_mode(args.out, split=args.split)
    elif args.batch:
        batch_analysis(args.dataset, args.root, args.n, args.out_dir)
    else:
        main_single(args)
