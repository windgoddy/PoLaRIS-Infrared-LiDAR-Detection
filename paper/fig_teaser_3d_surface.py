#!/usr/bin/env python3
"""
Figure: 3D Energy Surface Teaser — TIP 首页镇楼图
===================================================
展示三种伪标签策略在极小目标上的三维能量曲面：

  列 1: Box Fill       → 平坦方块台地 (表征坍塌 Representation Collapse)
  列 2: B-SNR W(i)     → 起伏丘陵，含噪声旁瓣，标出物理热点
  列 3: HALO PAG P(i)  → 物理热点上的完美高斯钟形曲面

每列下方附对应的 2D 灰度视图（热图）供对比。

用法:
    # 合成演示（无需数据集）
    python paper/fig_teaser_3d_surface.py --demo
    python paper/fig_teaser_3d_surface.py --demo --out paper/figures/fig_teaser_3d.pdf

    # 真实样本
    python paper/fig_teaser_3d_surface.py --nudt 000259
    python paper/fig_teaser_3d_surface.py --nuaa Misc_87
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'DejaVu Sans'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_CONFIGS = {
    'NUAA-SIRST': dict(
        img_dir='dataset/NUAA-SIRST/images',
        mask_dir='dataset/NUAA-SIRST/masks',
        label_dir='dataset/NUAA-SIRST/labels_box',
        img_ext='.png'),
    'NUDT-SIRST': dict(
        img_dir='dataset/NUDT-SIRST/images',
        mask_dir='dataset/NUDT-SIRST/masks',
        label_dir='dataset/NUDT-SIRST/labels_box',
        img_ext='.png'),
}

# ────────────────────────────────────────────────────────────────
# 伪标签生成工具（与 HALO 主流程保持数学一致）
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
            _, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = max(0, int((cx - w/2) * img_w))
            y1 = max(0, int((cy - h/2) * img_h))
            x2 = min(img_w, int((cx + w/2) * img_w))
            y2 = min(img_h, int((cy + h/2) * img_h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def make_box_fill(H, W, box):
    """Box Fill: 框内均匀填充 1.0"""
    label = np.zeros((H, W), dtype=np.float32)
    x1, y1, x2, y2 = box
    label[y1:y2, x1:x2] = 1.0
    return label


def make_bsnr(img_gray_f, box, expand_ratio=1.5, tau=1.0):
    """
    B-SNR 权重图: W(i) = sigmoid(τ · (I(i) - μ_ctx) / σ_ctx)
    仅在框内有效，框外置 0
    """
    H, W = img_gray_f.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx = max(2, int(bw * (expand_ratio - 1) / 2))
    my = max(2, int(bh * (expand_ratio - 1) / 2))
    cx1 = max(0, x1 - mx); cy1 = max(0, y1 - my)
    cx2 = min(W, x2 + mx); cy2 = min(H, y2 + my)

    # Context Ring
    ctx_mask  = np.zeros((H, W), dtype=bool)
    inner     = np.zeros((H, W), dtype=bool)
    ctx_mask[cy1:cy2, cx1:cx2] = True
    inner[y1:y2, x1:x2] = True
    ring = ctx_mask & ~inner
    ctx_pixels = img_gray_f[ring]
    if len(ctx_pixels) < 5:
        mu_ctx, sigma_ctx = float(np.mean(img_gray_f)), float(np.std(img_gray_f)) + 1e-4
    else:
        mu_ctx    = float(np.mean(ctx_pixels))
        sigma_ctx = float(np.std(ctx_pixels)) + 1e-4

    W_map = np.zeros((H, W), dtype=np.float32)
    snr_vals = (img_gray_f[y1:y2, x1:x2] - mu_ctx) / sigma_ctx
    W_map[y1:y2, x1:x2] = 1.0 / (1.0 + np.exp(-tau * snr_vals))
    return W_map


def make_halo_pag(H, W, box, bsnr_map):
    """
    HALO PAG: 以 B-SNR 权重图的 argmax 为中心，FWHM 估计 σ 的高斯软标签
    σ 由目标区域高亮度像素分布宽度（FWHM = 2.355σ）估算
    """
    x1, y1, x2, y2 = box
    roi = bsnr_map[y1:y2, x1:x2]
    # argmax in roi → 全图坐标
    flat_idx = int(np.argmax(roi))
    py_star  = y1 + flat_idx // (x2 - x1)
    px_star  = x1 + flat_idx  % (x2 - x1)

    # σ 估算：目标区域高响应像素的分布半径（FWHM/2.355）
    bw, bh = x2 - x1, y2 - y1
    threshold = 0.5 * roi.max()
    high_px = np.sum(roi > threshold)
    sigma = max(0.8, np.sqrt(high_px / np.pi) / 2.355)

    # 生成高斯
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    halo = np.exp(-((xx - px_star)**2 + (yy - py_star)**2) / (2 * sigma**2))
    halo[halo < 1e-4] = 0.0
    # 框外置 0
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    halo[~mask] = 0.0
    return halo, px_star, py_star, sigma


# ────────────────────────────────────────────────────────────────
# 3D 曲面绘制工具
# ────────────────────────────────────────────────────────────────

def draw_3d_surface(ax, Z, title, cmap, elev=28, azim=-55,
                    highlight_pos=None, zlabel='Response',
                    alpha_surf=0.88, color_wireframe=None):
    """
    绘制 3D 曲面。Z: 2D numpy array [0, 1] 范围
    highlight_pos: (px, py) 在曲面上标记物理热点红色星号
    """
    H, W = Z.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    surf = ax.plot_surface(xx, yy, Z, cmap=cmap, alpha=alpha_surf,
                           linewidth=0, antialiased=True, rcount=60, ccount=60)

    if color_wireframe:
        ax.plot_wireframe(xx, yy, Z, color=color_wireframe, lw=0.2, alpha=0.25,
                          rcount=15, ccount=15)

    if highlight_pos is not None:
        px, py = highlight_pos
        if 0 <= px < W and 0 <= py < H:
            z_val = float(Z[py, px])
            ax.scatter([px], [py], [z_val + 0.06], color='red',
                       s=120, zorder=10, marker='*', depthshade=False)
            ax.plot([px, px], [py, py], [0, z_val], color='red',
                    lw=0.8, ls='--', alpha=0.6)

    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=4)
    ax.set_xlabel('x', fontsize=8, labelpad=2)
    ax.set_ylabel('y', fontsize=8, labelpad=2)
    ax.set_zlabel(zlabel, fontsize=8, labelpad=2)
    ax.set_zlim(0, 1.05)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=6, pad=1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#CCCCCC')
    ax.yaxis.pane.set_edgecolor('#CCCCCC')
    ax.zaxis.pane.set_edgecolor('#CCCCCC')
    ax.grid(True, lw=0.3, alpha=0.4)
    return surf


def draw_2d_heatmap(ax, Z, title, cmap, highlight_pos=None, box=None):
    """底部 2D 热图（imshow）"""
    ax.imshow(Z, cmap=cmap, vmin=0, vmax=1, interpolation='nearest', aspect='equal')
    if box is not None:
        x1, y1, x2, y2 = box
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1-0.5, y1-0.5), x2-x1, y2-y1,
                         edgecolor='yellow', facecolor='none', lw=1.2, ls='--')
        ax.add_patch(rect)
    if highlight_pos is not None:
        px, py = highlight_pos
        ax.plot(px, py, 'r*', markersize=10, markeredgewidth=0.5, markeredgecolor='white')
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis('off')


# ────────────────────────────────────────────────────────────────
# 合成数据生成
# ────────────────────────────────────────────────────────────────

def make_synthetic_ir(size=64, target_pos=None, target_sigma=1.8,
                       target_peak=0.85, bg_mu=0.42, bg_noise=0.08,
                       seed=0):
    """
    生成合成红外 patch（归一化到 [0,1]）
    target_pos: (px, py) 目标中心（像素）
    返回 float32 [0,1] 数组
    """
    rng = np.random.default_rng(seed)
    # 背景：低频纹理 + 高斯噪声
    bg_low = rng.normal(bg_mu, bg_noise * 0.4, (size, size)).astype(np.float32)
    bg_low = gaussian_filter(bg_low, sigma=4.0)
    noise  = rng.normal(0, bg_noise, (size, size)).astype(np.float32)
    img = (bg_low + noise).clip(0, 1)

    if target_pos is None:
        target_pos = (size // 2 + rng.integers(-4, 5), size // 2 + rng.integers(-4, 5))
    px, py = target_pos
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    psf = target_peak * np.exp(-((xx - px)**2 + (yy - py)**2) / (2 * target_sigma**2))
    img = (img + psf).clip(0, 1)
    return img


# ────────────────────────────────────────────────────────────────
# 主绘图入口
# ────────────────────────────────────────────────────────────────

def _prepare_surfaces(img_patch, box_in_patch):
    """计算三种标签曲面及相关元数据，供 make_figure / make_figure_split 共用"""
    H, W = img_patch.shape
    box  = box_in_patch
    box_fill = make_box_fill(H, W, box)
    bsnr     = make_bsnr(img_patch, box, expand_ratio=1.5, tau=1.2)
    halo, px_star, py_star, sigma = make_halo_pag(H, W, box, bsnr)
    bsnr_vis = bsnr / bsnr.max() if bsnr.max() > 0 else bsnr
    x1, y1, x2, y2 = box
    geo_cx = (x1 + x2) // 2
    geo_cy = (y1 + y2) // 2
    return (box_fill, bsnr_vis, halo,
            px_star, py_star, sigma,
            geo_cx, geo_cy, box)


COL_SPECS = [
    ('Box Fill\n(Representation Collapse)',    '#E53935', 'plasma',  False),
    ('B-SNR  $W(i)$\n(Physical Anchor)',       '#FB8C00', 'inferno', True),
    ('HALO PAG  $P(i)$\n(Gaussian Refinement)','#7B1FA2', 'magma',   True),
]
# (title, color, cmap, show_anchor)


def make_figure(img_patch, box_in_patch, out_path, sample_label, split=False):
    """
    img_patch: float32 [0,1], shape (H, W)
    split=True  → 输出 6 个独立 PNG（3×3D + 3×2D），文件名以 out_path 为前缀
    split=False → 输出合并图（默认）
    """
    (box_fill, bsnr_vis, halo,
     px_star, py_star, sigma,
     geo_cx, geo_cy, box) = _prepare_surfaces(img_patch, box_in_patch)

    x1, y1, x2, y2 = box
    surfaces  = [box_fill, bsnr_vis, halo]
    highlight = [None, (px_star, py_star), (px_star, py_star)]

    out_dir  = os.path.dirname(out_path) or '.'
    base     = os.path.splitext(os.path.basename(out_path))[0]
    ext      = os.path.splitext(out_path)[1] or '.png'
    os.makedirs(out_dir, exist_ok=True)

    col_names = ['boxfill', 'bsnr', 'halo']

    if split:
        # ── 独立输出 6 个文件 ────────────────────────────────────
        for col, (title, col_color, cmap, show_anchor) in enumerate(COL_SPECS):
            Z   = surfaces[col]
            hlp = highlight[col]

            # 3D 曲面
            fig3d = plt.figure(figsize=(5, 5))
            fig3d.patch.set_facecolor('#0A0A0A')
            ax3d = fig3d.add_subplot(111, projection='3d')
            ax3d.set_facecolor('#0A0A0A')
            wire_color = '#444444' if col < 2 else '#6A1B9A'
            draw_3d_surface(ax3d, Z, title='', cmap=cmap, elev=28, azim=-50,
                            highlight_pos=hlp, zlabel='Label Confidence',
                            alpha_surf=0.92, color_wireframe=wire_color)
            ax3d.set_title(title, fontsize=11, fontweight='bold', pad=6, color=col_color)
            p3d = os.path.join(out_dir, f'{base}_{col_names[col]}_3d{ext}')
            plt.savefig(p3d, dpi=300, bbox_inches='tight',
                        facecolor='#0A0A0A', edgecolor='none')
            print(f"Saved → {p3d}")
            plt.close()

            # 2D 热图
            fig2d, ax2d = plt.subplots(figsize=(4, 4))
            fig2d.patch.set_facecolor('#0A0A0A')
            ax2d.set_facecolor('#0A0A0A')
            draw_2d_heatmap(ax2d, Z, title='', cmap=cmap,
                            highlight_pos=hlp, box=box if col == 0 else None)
            p2d = os.path.join(out_dir, f'{base}_{col_names[col]}_2d{ext}')
            plt.savefig(p2d, dpi=300, bbox_inches='tight',
                        facecolor='#0A0A0A', edgecolor='none')
            print(f"Saved → {p2d}")
            plt.close()
        return

    # ── 合并图（默认）────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor('#0A0A0A')
    gs = fig.add_gridspec(2, 3, hspace=0.06, wspace=0.04,
                          left=0.03, right=0.97, top=0.91, bottom=0.04,
                          height_ratios=[1.65, 1.0])

    for col, (title, col_color, cmap, _) in enumerate(COL_SPECS):
        Z   = surfaces[col]
        hlp = highlight[col]

        ax3d = fig.add_subplot(gs[0, col], projection='3d')
        ax3d.set_facecolor('#0A0A0A')
        wire_color = '#444444' if col < 2 else '#6A1B9A'
        draw_3d_surface(ax3d, Z, title='', cmap=cmap, elev=28, azim=-50,
                        highlight_pos=hlp, zlabel='Label Confidence',
                        alpha_surf=0.92, color_wireframe=wire_color)
        ax3d.set_title(title, fontsize=11, fontweight='bold', pad=6, color=col_color)

        ax2d = fig.add_subplot(gs[1, col])
        ax2d.set_facecolor('#0A0A0A')
        draw_2d_heatmap(ax2d, Z, title='', cmap=cmap,
                        highlight_pos=hlp, box=box if col == 0 else None)

    offset = np.sqrt((px_star - geo_cx)**2 + (py_star - geo_cy)**2)
    diag   = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    offset_ratio = offset / max(diag, 1)
    info_txt = (f"Box: {x2-x1}×{y2-y1} px  |  "
                f"Physical anchor offset: {offset:.1f} px "
                f"({offset_ratio*100:.0f}% of diag)  |  "
                f"HALO σ = {sigma:.2f} px")
    fig.text(0.5, 0.005, info_txt, ha='center', va='bottom',
             fontsize=8.5, color='#AAAAAA', style='italic')
    fig.suptitle(
        f'HALO: From Coarse Box to Physics-Anchored Gaussian Soft Label — {sample_label}',
        fontsize=12.5, fontweight='bold', color='white', y=0.975,
    )
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=11, label='Physical Anchor (argmax of B-SNR)', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markersize=7, label='Geometric Center (box centroid)', linestyle='None',
               alpha=0.5),
    ]
    fig.legend(handles=legend_elems, loc='upper right', fontsize=9,
               framealpha=0.25, edgecolor='#555', labelcolor='white',
               bbox_to_anchor=(0.97, 0.96))
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='#0A0A0A', edgecolor='none')
    print(f"Saved → {out_path}")
    plt.close()


# ────────────────────────────────────────────────────────────────
# 合成演示
# ────────────────────────────────────────────────────────────────

def demo_mode(out_path, split=False):
    """合成数据演示：目标偏离框几何中心"""
    size = 64
    target_px, target_py = 38, 22
    box = (size//2 - 10, size//2 - 10, size//2 + 10, size//2 + 10)
    img = make_synthetic_ir(size=size, target_pos=(target_px, target_py),
                            target_sigma=2.0, target_peak=0.75,
                            bg_mu=0.38, bg_noise=0.09, seed=13)
    make_figure(img, box, out_path,
                sample_label='Synthetic Demo (target offset from box center)',
                split=split)


# ────────────────────────────────────────────────────────────────
# 真实数据模式
# ────────────────────────────────────────────────────────────────

def load_and_crop(ds_name, sample_name, root, crop_size=64):
    cfg = DATASET_CONFIGS[ds_name]
    img_bgr = cv2.imread(os.path.join(root, cfg['img_dir'], sample_name + cfg['img_ext']))
    if img_bgr is None:
        raise FileNotFoundError(sample_name)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    H, W = img_gray.shape
    boxes = load_yolo_boxes(
        os.path.join(root, cfg['label_dir'], sample_name + '.txt'), H, W)
    if not boxes:
        raise ValueError("No box labels found")

    # 选主框（用 B-SNR argmax 确定主目标）
    primary_box = boxes[0]
    best_snr = -1
    for box in boxes:
        roi = img_gray[box[1]:box[3], box[0]:box[2]]
        if roi.size == 0:
            continue
        snr = float(roi.max() - roi.mean())
        if snr > best_snr:
            best_snr = snr
            primary_box = box

    x1, y1, x2, y2 = primary_box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = crop_size // 2
    ox = max(0, min(cx - half, W - crop_size))
    oy = max(0, min(cy - half, H - crop_size))

    img_patch = img_gray[oy:oy+crop_size, ox:ox+crop_size]
    box_in_patch = (
        max(0, x1 - ox), max(0, y1 - oy),
        min(crop_size, x2 - ox), min(crop_size, y2 - oy),
    )
    return img_patch, box_in_patch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=ROOT)
    parser.add_argument('--out', default=os.path.join(
        ROOT, 'paper', 'figures', 'fig_teaser_3d.png'))
    parser.add_argument('--demo', action='store_true',
                        help='Use synthetic data (no dataset needed)')
    parser.add_argument('--split', action='store_true',
                        help='Output each panel as a separate PNG (for manual assembly)')
    parser.add_argument('--nudt', default='')
    parser.add_argument('--nuaa', default='')
    parser.add_argument('--crop_size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    if args.demo or (not args.nudt and not args.nuaa):
        demo_mode(args.out, split=args.split)
    elif args.nudt:
        img_p, box_p = load_and_crop('NUDT-SIRST', args.nudt, args.root, args.crop_size)
        make_figure(img_p, box_p, args.out, f'NUDT  {args.nudt}', split=args.split)
    elif args.nuaa:
        img_p, box_p = load_and_crop('NUAA-SIRST', args.nuaa, args.root, args.crop_size)
        make_figure(img_p, box_p, args.out, f'NUAA  {args.nuaa}', split=args.split)
