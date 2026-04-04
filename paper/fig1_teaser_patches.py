#!/usr/bin/env python3
"""
Figure 1 Teaser: Generate Component Patches for Illustrator/PPT Assembly
=========================================================================
输出每个叙事组件的独立高清 PNG，供后续在 Illustrator/PPT 中排版拼接。

生成的文件（全部输出到 --out_dir）:
  01_ir.png               — 原始 IR 灰度图切片（无任何标注）
  02_ir_box.png           — IR + YOLO 粗糙红框
  03_ir_context.png       — IR + YOLO 框 + Context Box（橙色虚线）
  04_geo_gaussian.png     — 【错误路径】IR + 几何中心高斯软标签（几何先验失败）
  05_bsnr.png             — 【Step A】IR + B-SNR 物理热力图
  06_argmax_marker.png    — 【Step B】IR + 物理热点 p* 准星锁定
  07_halo.png             — 【Step C】IR + HALO PAG 软标签（物理先验成功）
  08_fail_pred.png        — 【输出对比】失败预测（用 BoxFill 代理：覆盖整个框区域）
  09_success_pred.png     — 【输出对比】成功预测（用 HALO 软标签二值化代理）
  10_robustness.png       — 【鲁棒性】两个偏移框 + 不变的 HALO 软标签

此外会输出一张 composite_preview.png 供快速预览所有组件。

用法:
    python paper/fig1_teaser_patches.py --nuaa Misc_399
    python paper/fig1_teaser_patches.py --nudt 000314
    python paper/fig1_teaser_patches.py --irstd XDU369

    # 自动寻找几何中心偏移最大（argmax 偏离框中心最远）的样本：
    python paper/fig1_teaser_patches.py --mine --dataset NUAA-SIRST --mine_n 5
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
import matplotlib.patches as mpatches

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

# 上下文切片大小（01-04, 08-10）：显示完整场景 + YOLO 框
CROP_SIZE = 128
# 放大切片大小（05-07, 09）：紧凑裁剪到目标附近，再由 matplotlib 撑满显示
# 值 = max(box_w, box_h) * ZOOM_PAD_FACTOR，最小 32px
ZOOM_PAD_FACTOR = 3     # 目标框尺寸的 N 倍作为 zoom crop
ZOOM_SIZE_MIN   = 32    # zoom crop 最小像素数

DPI      = 300
FIG_INCH = 4.0          # 固定输出尺寸（所有 patch 统一 4 英寸）


# ────────────────────────────────────────────────────────────────

def load_bsnr_utils(root):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'bsnr_mask_utils',
        os.path.join(root, 'model_Mamba', 'dataset', 'bsnr_mask_utils.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_bsnr_weight


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
            x1 = max(0, round((cx - w / 2) * img_w))
            y1 = max(0, round((cy - h / 2) * img_h))
            x2 = min(img_w, round((cx + w / 2) * img_w))
            y2 = min(img_h, round((cy + h / 2) * img_h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def get_union_center(boxes, H, W):
    if not boxes:
        return W // 2, H // 2
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1 + x2) // 2, (y1 + y2) // 2


def crop_patch(arr, cx, cy, size):
    """No-black-band crop: clamp offset so patch stays in bounds."""
    h, w = arr.shape[:2]
    ox = max(0, min(cx - size // 2, w - size))
    oy = max(0, min(cy - size // 2, h - size))
    patch = arr[oy:oy + size, ox:ox + size]
    if patch.shape[0] < size or patch.shape[1] < size:
        out = np.zeros((size, size) + arr.shape[2:], dtype=arr.dtype)
        out[:patch.shape[0], :patch.shape[1]] = patch
        patch = out
    return patch, ox, oy


def make_geo_gaussian(boxes, H, W):
    """框几何中心为锚点的高斯软标签（传统失败路径）。"""
    geo_map = np.zeros((H, W), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        cx_geo = (x1 + x2) / 2.0
        cy_geo = (y1 + y2) / 2.0
        sigma_x = (x2 - x1) / 3.0   # 用框宽/高估计 sigma
        sigma_y = (y2 - y1) / 3.0
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        g = np.exp(-((xs - cx_geo)**2 / (2 * sigma_x**2 + 1e-6) +
                     (ys - cy_geo)**2 / (2 * sigma_y**2 + 1e-6)))
        # 只在框内有效
        mask_box = np.zeros((H, W), dtype=bool)
        mask_box[y1:y2, x1:x2] = True
        geo_map = np.maximum(geo_map, g * mask_box)
    if geo_map.max() > 0:
        geo_map /= geo_map.max()
    return geo_map


def load_data(ds_name, sample_name, root, compute_bsnr_weight):
    cfg = DATASET_CONFIGS[ds_name]
    img_path   = os.path.join(root, cfg['img_dir'],   sample_name + cfg['img_ext'])
    mask_path  = os.path.join(root, cfg['mask_dir'],  sample_name + cfg['img_ext'])
    label_path = os.path.join(root, cfg['label_dir'], sample_name + '.txt')

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape

    mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_full is None:
        mask_full = np.zeros((H, W), dtype=np.uint8)
    mask_full = (mask_full > 127).astype(np.uint8) * 255

    boxes = load_yolo_boxes(label_path, H, W)

    # Step 1: 先对每个 box 单独计算 B-SNR，找到全局最强热点所在的 box
    img_float = img_gray.astype(np.float32) / 255.0
    bsnr_full = np.zeros((H, W), dtype=np.float32)
    halo_full = np.zeros((H, W), dtype=np.float32)
    for box in boxes:
        bx1, by1, bx2, by2 = box
        w_map = compute_bsnr_weight(
            img_float, box, expand_ratio=1.5, temperature=3.0,
            spatial_gaussian=False, gauss_sigma_ratio=1.5
        )
        bsnr_full[by1:by2, bx1:bx2] = np.maximum(bsnr_full[by1:by2, bx1:bx2], w_map)
        w_halo = compute_bsnr_weight(
            img_float, box, expand_ratio=1.5, temperature=3.0,
            spatial_gaussian=True, gauss_sigma_ratio=1.5
        )
        halo_full[by1:by2, bx1:bx2] = np.maximum(halo_full[by1:by2, bx1:bx2], w_halo)

    # Step 2: 找全局 argmax（物理热点 p*）
    argmax_idx = np.unravel_index(np.argmax(bsnr_full), bsnr_full.shape)
    py_star, px_star = argmax_idx   # (row, col) → (y, x)

    # Step 3: 找包含 p* 的那个 box（主 box），以它的中心为裁剪中心
    primary_box = boxes[0]  # 默认第一个
    for box in boxes:
        bx1, by1, bx2, by2 = box
        if bx1 <= px_star <= bx2 and by1 <= py_star <= by2:
            primary_box = box
            break
    bx1, by1, bx2, by2 = primary_box
    cx = (bx1 + bx2) // 2
    cy = (by1 + by2) // 2

    _, sigma_px = compute_bsnr_weight(
        img_float, primary_box, expand_ratio=1.5, temperature=3.0,
        spatial_gaussian=True, gauss_sigma_ratio=1.5, return_sigma=True
    )

    geo_full = make_geo_gaussian([primary_box], H, W)

    # Step 4: 以主 box 中心裁剪（上下文大图，用于 01-04, 08, 10）
    img_p,   ox, oy = crop_patch(img_gray,   cx, cy, CROP_SIZE)
    mask_p,  _,  _  = crop_patch(mask_full,  cx, cy, CROP_SIZE)
    bsnr_p,  _,  _  = crop_patch(bsnr_full,  cx, cy, CROP_SIZE)
    halo_p,  _,  _  = crop_patch(halo_full,  cx, cy, CROP_SIZE)
    geo_p,   _,  _  = crop_patch(geo_full,   cx, cy, CROP_SIZE)

    # 只展示主 box（在 patch 坐标系内）
    boxes_p = [(primary_box[0] - ox, primary_box[1] - oy,
                primary_box[2] - ox, primary_box[3] - oy)]
    px_star_p = px_star - ox
    py_star_p = py_star - oy

    # Step 5: 以物理热点 p* 为中心的紧凑 zoom 切片（用于 05-07, 09）
    bw = primary_box[2] - primary_box[0]
    bh = primary_box[3] - primary_box[1]
    zoom_size = max(ZOOM_SIZE_MIN, int(max(bw, bh) * ZOOM_PAD_FACTOR))
    img_z,   zox, zoy = crop_patch(img_gray,  px_star, py_star, zoom_size)
    bsnr_z,  _,   _   = crop_patch(bsnr_full, px_star, py_star, zoom_size)
    halo_z,  _,   _   = crop_patch(halo_full, px_star, py_star, zoom_size)
    geo_z,   _,   _   = crop_patch(geo_full,  px_star, py_star, zoom_size)
    mask_z,  _,   _   = crop_patch(mask_full.astype(np.float32), px_star, py_star, zoom_size)
    boxes_z = [(primary_box[0] - zox, primary_box[1] - zoy,
                primary_box[2] - zox, primary_box[3] - zoy)]
    px_star_z = px_star - zox
    py_star_z = py_star - zoy

    return {
        # 上下文切片（大图）
        'img_p': img_p, 'mask_p': mask_p,
        'bsnr_p': bsnr_p, 'halo_p': halo_p, 'geo_p': geo_p,
        'boxes_p': boxes_p,
        'px_star_p': px_star_p, 'py_star_p': py_star_p,
        # 放大切片（紧凑）
        'img_z': img_z, 'bsnr_z': bsnr_z, 'halo_z': halo_z, 'geo_z': geo_z,
        'mask_z': mask_z,
        'boxes_z': boxes_z,
        'px_star_z': px_star_z, 'py_star_z': py_star_z,
        'zoom_size': zoom_size,
        'sigma_px': sigma_px,
        # meta
        'H': H, 'W': W, 'ox': ox, 'oy': oy,
        'primary_box': primary_box,
        # zoom 裁剪原点（供外部 preds 加载时复用）
        'zox': zox, 'zoy': zoy,
        'zoom_size': zoom_size,
        'px_star': px_star, 'py_star': py_star,
    }


# ────────────────────────────────────────────────────────────────
# 单张切片保存函数（无坐标轴、无边距，干净的像素图）
# ────────────────────────────────────────────────────────────────

def save_patch(path, render_fn, dpi=DPI):
    """render_fn(fig, ax) -> draws onto ax; saved as tight PNG."""
    fig = plt.figure(figsize=(FIG_INCH, FIG_INCH), dpi=dpi)
    ax  = fig.add_axes([0, 0, 1, 1])   # 占满整个 figure
    ax.set_axis_off()
    render_fn(fig, ax)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"  → {os.path.basename(path)}")


def draw_box_in_patch(ax, boxes_p, color='#FF3333', lw=2.5, ls='-'):
    for (x1, y1, x2, y2) in boxes_p:
        w, h = x2 - x1, y2 - y1
        # -0.5 aligns the rectangle edge with the pixel boundary
        # (imshow centers pixel i at coordinate i, so edge is at i-0.5)
        rect = mpatches.Rectangle((x1 - 0.5, y1 - 0.5), w, h,
                                   linewidth=lw, edgecolor=color,
                                   facecolor='none', linestyle=ls)
        ax.add_patch(rect)


def draw_context_box(ax, boxes_p, expand_ratio=1.5, color='#FF8800', lw=1.8):
    """在 patch 坐标系中画 Context Box（膨胀框）。"""
    for (x1, y1, x2, y2) in boxes_p:
        bw, bh = x2 - x1, y2 - y1
        margin_x = bw * (expand_ratio - 1) / 2
        margin_y = bh * (expand_ratio - 1) / 2
        cx1 = x1 - margin_x
        cy1 = y1 - margin_y
        cw  = bw + 2 * margin_x
        ch  = bh + 2 * margin_y
        # -0.5 aligns the rectangle edge with the pixel boundary
        rect = mpatches.Rectangle((cx1 - 0.5, cy1 - 0.5), cw, ch,
                                   linewidth=1.8, edgecolor=color,
                                   facecolor='none', linestyle='--')
        ax.add_patch(rect)


def draw_crosshair(ax, px, py, size=6, color='#FFFF00', lw=1.8, radius=None):
    """在 (px, py) 画准星（十字+圆圈）标注物理热点 p*。
    radius: 圆圈半径；若为 None 则退回到 size * 0.8（经验值）。
            建议传入 PAG 的 sigma_px，使圆圈对应高斯软标签的 e^(-1/2) 等高线。
    """
    ax.plot([px - size, px + size], [py, py], color=color, lw=lw, zorder=10)
    ax.plot([px, px], [py - size, py + size], color=color, lw=lw, zorder=10)
    r = radius if radius is not None else size * 0.8
    circle = mpatches.Circle((px, py), radius=r,
                               linewidth=lw, edgecolor=color,
                               facecolor='none', zorder=10)
    ax.add_patch(circle)


def overlay_heatmap(ax, img_p, heatmap, cmap='inferno', alpha=0.72):
    ax.imshow(img_p, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    ax.imshow(heatmap, cmap=cmap, alpha=alpha,
              interpolation='nearest', vmin=0, vmax=1)


# ────────────────────────────────────────────────────────────────

def generate_patches(data, out_dir):
    # 上下文（大图）变量 — 仅 01 使用
    img_p    = data['img_p']
    px_s     = data['px_star_p']
    py_s     = data['py_star_p']
    CS       = CROP_SIZE

    # 放大（zoom）变量 — 02-11 全部使用
    img_z    = data['img_z']
    bsnr_z   = data['bsnr_z']
    halo_z   = data['halo_z']
    geo_z    = data['geo_z']
    mask_z   = data['mask_z']
    boxes_z  = data['boxes_z']
    px_sz    = data['px_star_z']
    py_sz    = data['py_star_z']
    ZS       = data['zoom_size']
    sigma_px = data['sigma_px']

    os.makedirs(out_dir, exist_ok=True)

    # 准星十字臂长 —— 经验尺寸；圆圈半径用 PAG 的物理 σ
    mk_zoom = max(2, ZS // 8)
    lw_zoom = 1.2

    # ── 01 完整原图 + 黄色虚线框标出放大区域（唯一的上下文图）─────
    def r01(fig, ax):
        ax.imshow(img_p, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        # 在 context 坐标系中标出 zoom 区域对应的位置
        # zoom 裁剪是以 p* 为中心，大小为 ZS，转换到 context 坐标系
        zx1 = px_s - ZS // 2
        zy1 = py_s - ZS // 2
        rect = mpatches.Rectangle((zx1, zy1), ZS, ZS,
                                   linewidth=1.5, edgecolor='#FFEE55',
                                   facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.set_xlim(0, CS); ax.set_ylim(CS, 0)
    save_patch(os.path.join(out_dir, '01_ir.png'), r01)

    # ── 02 IR 放大切片（无标注） ──────────────────────────────────
    def r02(fig, ax):
        ax.imshow(img_z, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
    save_patch(os.path.join(out_dir, '02_ir.png'), r02)

    # ── 03 放大 + YOLO 框 ────────────────────────────────────────
    def r03(fig, ax):
        ax.imshow(img_z, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        draw_box_in_patch(ax, boxes_z, color='#FF3333', lw=lw_zoom + 0.3)
        ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
    save_patch(os.path.join(out_dir, '03_ir_box.png'), r03)

    # ── 04 放大 + YOLO 框 + Context Box ─────────────────────────
    def r04(fig, ax):
        ax.imshow(img_z, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        draw_context_box(ax, boxes_z)
        draw_box_in_patch(ax, boxes_z, color='#FF3333', lw=lw_zoom + 0.3)
        ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
    save_patch(os.path.join(out_dir, '04_ir_context.png'), r04)

    # ── 05 几何先验高斯【错误路径】（放大）
    # 叠在原图上；geo_z 归一化到 [0,1]，inferno colormap 低值为黑
    def r05_geo(fig, ax):
        overlay_heatmap(ax, img_z, geo_z, cmap='inferno', alpha=0.80)
        draw_box_in_patch(ax, boxes_z, color='#FF3333', lw=lw_zoom)
        # 几何中心：白色叉号（错误锚点）
        for (x1, y1, x2, y2) in boxes_z:
            gcx, gcy = (x1 + x2) / 2, (y1 + y2) / 2
            s = mk_zoom
            ax.plot([gcx-s, gcx+s], [gcy-s, gcy+s], color='white', lw=lw_zoom, zorder=10)
            ax.plot([gcx-s, gcx+s], [gcy+s, gcy-s], color='white', lw=lw_zoom, zorder=10)
        ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
    save_patch(os.path.join(out_dir, '05_geo_gaussian.png'), r05_geo)

    # ── 06 B-SNR 物理热力图【ZOOM】 ───────────────────────────────
    def r06(fig, ax):
        overlay_heatmap(ax, img_z, bsnr_z, cmap='inferno', alpha=0.72)
        draw_box_in_patch(ax, boxes_z, color='#FF3333', lw=lw_zoom, ls='--')
        ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
    save_patch(os.path.join(out_dir, '06_bsnr.png'), r06)

    # ── 07 物理热点 p* 准星锁定【ZOOM】 ──────────────────────────
    def r07(fig, ax):
        overlay_heatmap(ax, img_z, bsnr_z, cmap='inferno', alpha=0.65)
        draw_box_in_patch(ax, boxes_z, color='#FF3333', lw=lw_zoom, ls='--')
        draw_crosshair(ax, px_sz, py_sz, size=mk_zoom, color='#FFFF00', lw=lw_zoom, radius=sigma_px)
        ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
    save_patch(os.path.join(out_dir, '07_argmax_marker.png'), r07)

    # ── 08 HALO PAG 软标签【ZOOM】 ────────────────────────────────
    def r08(fig, ax):
        overlay_heatmap(ax, img_z, halo_z, cmap='inferno', alpha=0.75)
        draw_box_in_patch(ax, boxes_z, color='#FF3333', lw=lw_zoom, ls='--')
        draw_crosshair(ax, px_sz, py_sz, size=mk_zoom, color='#FFFF00', lw=lw_zoom, radius=sigma_px)
        ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
    save_patch(os.path.join(out_dir, '08_halo.png'), r08)

    # ── 08b HALO PAG 软标签（无准星）【ZOOM】 ────────────────────────
    def r08b(fig, ax):
        overlay_heatmap(ax, img_z, halo_z, cmap='inferno', alpha=0.75)
        draw_box_in_patch(ax, boxes_z, color='#FF3333', lw=lw_zoom, ls='--')
        ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
    save_patch(os.path.join(out_dir, '08b_halo_nomarker.png'), r08b)

    # ── 09 失败预测（BoxFill 代理）【ZOOM】 ───────────────────────
    def r09(fig, ax):
        ax.imshow(img_z, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        fail_overlay = np.zeros((*img_z.shape, 4), dtype=np.float32)
        for (x1, y1, x2, y2) in boxes_z:
            px1 = max(0, x1); py1 = max(0, y1)
            px2 = min(ZS, x2); py2 = min(ZS, y2)
            if px2 > px1 and py2 > py1:
                fail_overlay[py1:py2, px1:px2, :] = [1.0, 0.2, 0.2, 0.70]
        ax.imshow(fail_overlay, interpolation='nearest')
        ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
    save_patch(os.path.join(out_dir, '09_fail_pred.png'), r09)

    # ── 11 鲁棒性：±2px 偏移框 + 不变的 HALO【ZOOM】 ─────────────
    def r11(fig, ax):
        overlay_heatmap(ax, img_z, halo_z, cmap='inferno', alpha=0.75)
        draw_crosshair(ax, px_sz, py_sz, size=mk_zoom, color='#FFFF00', lw=lw_zoom, radius=sigma_px)
        draw_box_in_patch(ax, boxes_z, color='#FF3333', lw=lw_zoom)
        for delta in [-2, +2]:
            shifted = [(x1+delta, y1+delta, x2+delta, y2+delta)
                       for (x1, y1, x2, y2) in boxes_z]
            draw_box_in_patch(ax, shifted, color='#FF9999', lw=0.8, ls='--')
        ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
    save_patch(os.path.join(out_dir, '11_robustness.png'), r11)

    # ── 12a-12d 真实预测对比【ZOOM】（需要 preds_maps 已加载）─────
    # preds_maps: dict of {method: (H,W) float32 [0,1]} in zoom coords
    # 颜色方案：GT=白，HALO=青，B-SNR=绿，BoxFill=橙
    # ── 12a-12e 真实预测对比【ZOOM】（需要 preds_maps 已加载）─────
    # 颜色方案：GT annotation=白，GT-pred=白，HALO=青，B-SNR=绿，BoxFill=橙

    # 12a: 原始 GT 像素标注（直接来自 mask_z，非预测结果）
    def r12a(fig, ax):
        ax.imshow(img_z, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        gt_overlay = np.zeros((*img_z.shape, 4), dtype=np.float32)
        gt_overlay[mask_z > 127, :] = [1.0, 1.0, 1.0, 0.90]
        ax.imshow(gt_overlay, interpolation='nearest')
        ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
    save_patch(os.path.join(out_dir, '12a_gt_annotation.png'), r12a)

    _pred_cfg = [
        ('12b_pred_gt.png',      'gt',      [1.0, 1.0, 1.0, 0.85]),
        ('12c_pred_halo.png',    'halo',    [0.0, 1.0, 1.0, 0.85]),
        ('12d_pred_bsnr.png',    'bsnr',    [0.2, 1.0, 0.2, 0.85]),
        ('12e_pred_boxfill.png', 'boxfill', [1.0, 0.6, 0.0, 0.85]),
    ]
    preds_maps = data.get('preds_maps', {})
    for fname, method, rgba in _pred_cfg:
        if method not in preds_maps:
            continue
        pred_z = preds_maps[method]
        def _make_r(pz=pred_z, c=rgba):
            def _r(fig, ax):
                ax.imshow(img_z, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
                overlay = np.zeros((*img_z.shape, 4), dtype=np.float32)
                overlay[pz > 0.5, :] = c
                ax.imshow(overlay, interpolation='nearest')
                ax.set_xlim(0, ZS); ax.set_ylim(ZS, 0)
            return _r
        save_patch(os.path.join(out_dir, fname), _make_r())

    # ── 13/14  B-SNR & HALO 3D/2D 能量曲面【ZOOM】──────────────────
    # 利用真实 compute_bsnr_weight 算出的 bsnr_z / halo_z，
    # 复用 fig_teaser_3d_surface.py 的 draw_3d_surface / draw_2d_heatmap。
    import importlib.util as _ilu
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401 触发 3d 注册
    _spec3d = _ilu.spec_from_file_location(
        '_3d_utils',
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'fig_teaser_3d_surface.py')
    )
    _mod3d = _ilu.module_from_spec(_spec3d)
    _spec3d.loader.exec_module(_mod3d)
    _draw3d = _mod3d.draw_3d_surface
    _draw2d = _mod3d.draw_2d_heatmap
    _T      = _mod3d.THEME['dark']
    _FIG3D  = 4.5   # 英寸，3D patch 固定尺寸

    bz = boxes_z[0]  # (x1, y1, x2, y2) 在 zoom 坐标系
    _bx1, _by1, _bx2, _by2 = bz

    # ── 裁剪到 YOLO 框内，p* 换算为框内局部坐标 ──────────────────
    bsnr_box = bsnr_z[_by1:_by2, _bx1:_bx2].copy()
    halo_box  = halo_z[_by1:_by2, _bx1:_bx2].copy()
    px_box = px_sz - _bx1   # p* 在框内 x
    py_box = py_sz - _by1   # p* 在框内 y

    _surf_cfg = [
        # (文件前缀,  Z 数据(框内),  cmap,      标题颜色,   p*星号颜色)
        # B-SNR inferno 峰值为亮黄，红色不可见 → 用亮青色
        ('13_bsnr',  bsnr_box, 'inferno', '#FB8C00', '#00FFFF'),
        # HALO magma 峰值为深色，红色可见 → 沿用红色
        ('14_halo',  halo_box, 'magma',   '#7B1FA2', '#FF3333'),
    ]
    _wire = [_T['wire_colors'][1], _T['wire_colors'][2]]  # bsnr→idx1, halo→idx2

    for _i, (pref, Z, cmap, col_color, star_color) in enumerate(_surf_cfg):
        # ── 3D 曲面（仅框内像素） ─────────────────────────────────
        fig3d = plt.figure(figsize=(_FIG3D, _FIG3D), dpi=DPI)
        fig3d.patch.set_facecolor(_T['bg'])
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.set_facecolor(_T['ax_bg'])
        # 先画曲面（不带 highlight，避免 highlight_color='red' 在 inferno 上不可见）
        _draw3d(ax3d, Z, title='', cmap=cmap, elev=28, azim=-50,
                highlight_pos=None,
                zlabel='Label Confidence', alpha_surf=0.92,
                color_wireframe=_wire[_i], theme='dark')
        # 再单独画 p* 星号（自定义颜色，保证在各 cmap 上均清晰可见）
        if 0 <= px_box < Z.shape[1] and 0 <= py_box < Z.shape[0]:
            _z_star = float(Z[py_box, px_box])
            ax3d.scatter([px_box], [py_box], [_z_star + 0.06],
                         color=star_color, s=160, zorder=10,
                         marker='*', depthshade=False)
            ax3d.plot([px_box, px_box], [py_box, py_box], [0, _z_star],
                      color=star_color, lw=0.9, ls='--', alpha=0.7)
        ax3d.set_title(pref.split('_', 1)[1].upper(),
                       fontsize=11, fontweight='bold', pad=6, color=col_color)
        _p3d = os.path.join(out_dir, f'{pref}_3d.png')
        plt.savefig(_p3d, dpi=DPI, bbox_inches='tight',
                    facecolor=_T['bg'], edgecolor='none')
        plt.close(fig3d)
        print(f"  → {os.path.basename(_p3d)}")

        # ── 2D 热图（仅框内像素，无需画框轮廓） ───────────────────
        fig2d, ax2d = plt.subplots(figsize=(_FIG3D, _FIG3D), dpi=DPI)
        fig2d.patch.set_facecolor(_T['bg'])
        ax2d.set_facecolor(_T['ax_bg'])
        # highlight_pos=None，避免 draw_2d_heatmap 内部硬编码的 'r*' 覆盖颜色
        _draw2d(ax2d, Z, title='', cmap=cmap,
                highlight_pos=None,
                box=None, show_box_outline=False, theme='dark')
        # 单独画 p* 星号（与 3D 图保持相同自定义颜色）
        ax2d.plot(px_box, py_box, marker='*', color=star_color,
                  markersize=12, markeredgewidth=0.6,
                  markeredgecolor='black', zorder=10)
        _p2d = os.path.join(out_dir, f'{pref}_2d.png')
        plt.savefig(_p2d, dpi=DPI, bbox_inches='tight',
                    facecolor=_T['bg'], edgecolor='none')
        plt.close(fig2d)
        print(f"  → {os.path.basename(_p2d)}")


# ────────────────────────────────────────────────────────────────
# 预览合图（3行 × 4列，仅供核查）
# ────────────────────────────────────────────────────────────────

def make_preview(data, out_dir):
    files = [
        '01_ir.png',
        '02_ir.png', '03_ir_box.png', '04_ir_context.png',
        '05_geo_gaussian.png', '06_bsnr.png', '07_argmax_marker.png',
        '08_halo.png', '09_fail_pred.png',
        '11_robustness.png',
        '13_bsnr_3d.png', '13_bsnr_2d.png',
        '14_halo_3d.png', '14_halo_2d.png',
    ]
    captions = [
        '01 Full Image\n(zoom region)',
        '02 Zoom IR', '03 Zoom+Box', '04 Zoom+Context',
        '05 Geo-Gaussian\n(Wrong)', '06 B-SNR', '07 Argmax p*',
        '08 HALO PAG\n(Ours)', '09 Fail Pred\n(BoxFill)',
        '11 Robustness\n(±2px)',
        '13 B-SNR 3D', '13 B-SNR 2D',
        '14 HALO 3D', '14 HALO 2D',
    ]
    n = len(files)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    fig.patch.set_facecolor('#1a1a1a')
    axes = axes.flatten()
    for i, (fn, cap) in enumerate(zip(files, captions)):
        path = os.path.join(out_dir, fn)
        img = plt.imread(path)
        axes[i].imshow(img, interpolation='nearest')
        axes[i].set_title(cap, color='white', fontsize=8, pad=3)
        axes[i].axis('off')
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(pad=0.4)
    out_path = os.path.join(out_dir, 'composite_preview.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"\nPreview → {out_path}")


# ────────────────────────────────────────────────────────────────
# Mine mode: 找几何中心偏离 argmax 最大的样本
# ────────────────────────────────────────────────────────────────

def mine_offset_samples(ds_name, root, compute_bsnr_weight, top_n=5):
    """找框内 argmax 相对几何中心偏移最大的样本（适合 Teaser 展示）。"""
    cfg = DATASET_CONFIGS[ds_name]
    split_file = os.path.join(root, cfg['split'])
    with open(split_file) as f:
        names = [l.strip() for l in f if l.strip()]

    results = []
    for name in names:
        label_path = os.path.join(root, cfg['label_dir'], name + '.txt')
        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            continue
        img_path = os.path.join(root, cfg['img_dir'], name + cfg['img_ext'])
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W = img_gray.shape
        boxes = load_yolo_boxes(label_path, H, W)
        if not boxes:
            continue

        img_float = img_gray.astype(np.float32) / 255.0
        max_offset = 0.0
        for box in boxes:
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            if bw < 4 or bh < 4:
                continue
            try:
                w_map = compute_bsnr_weight(
                    img_float, box, expand_ratio=1.5, temperature=3.0,
                    spatial_gaussian=False, gauss_sigma_ratio=1.5
                )
            except Exception:
                continue
            bsnr_region = np.zeros((H, W), dtype=np.float32)
            bsnr_region[y1:y2, x1:x2] = w_map
            aidx = np.unravel_index(np.argmax(bsnr_region), bsnr_region.shape)
            py_a, px_a = aidx
            geo_cx, geo_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            offset = np.sqrt((px_a - geo_cx)**2 + (py_a - geo_cy)**2)
            # 归一化为框对角线的比例
            diag = np.sqrt(bw**2 + bh**2)
            offset_ratio = offset / (diag + 1e-6)
            max_offset = max(max_offset, offset_ratio)

        results.append((max_offset, name))

    results.sort(reverse=True)
    print(f"\nTop-{top_n} samples with largest geometric-vs-argmax offset ({ds_name}):")
    for ratio, name in results[:top_n]:
        print(f"  {name}  offset_ratio={ratio:.3f}")
    return results


# ────────────────────────────────────────────────────────────────

def main(args):
    compute_bsnr_weight = load_bsnr_utils(args.root)

    # Mine mode
    if args.mine:
        mine_offset_samples(args.dataset, args.root, compute_bsnr_weight,
                            top_n=args.mine_n)
        return

    # 确定 dataset 和 sample_name
    ds_name = args.dataset
    if args.nuaa:
        ds_name = 'NUAA-SIRST'
        sample_name = args.nuaa
    elif args.nudt:
        ds_name = 'NUDT-SIRST'
        sample_name = args.nudt
    elif args.irstd:
        ds_name = 'IRSTD-1k'
        sample_name = args.irstd
    else:
        # 自动选 split 第一张
        cfg = DATASET_CONFIGS[ds_name]
        split_file = os.path.join(args.root, cfg['split'])
        with open(split_file) as f:
            for line in f:
                name = line.strip()
                lp = os.path.join(args.root, cfg['label_dir'], name + '.txt')
                if os.path.exists(lp) and os.path.getsize(lp) > 0:
                    sample_name = name
                    break

    print(f"Dataset: {ds_name}  Sample: {sample_name}")

    data = load_data(ds_name, sample_name, args.root, compute_bsnr_weight)

    # ── 加载真实预测掩码，裁出 zoom 区域 ──────────────────────────
    DS_SUFFIX = {'NUAA-SIRST': 'nuaa', 'NUDT-SIRST': 'nudt', 'IRSTD-1k': 'irstd'}
    preds_root = args.preds_root
    ds_suffix  = DS_SUFFIX.get(ds_name, '')
    preds_maps = {}
    for method in ['gt', 'halo', 'bsnr', 'boxfill']:
        pred_path = os.path.join(preds_root, f'{method}_{ds_suffix}',
                                 sample_name + '.png')
        if not os.path.exists(pred_path):
            print(f"  [WARN] preds not found: {pred_path}")
            continue
        pred_full = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred_full is None:
            continue
        # 按与 img_z 相同的 zoom 裁剪参数裁切
        pred_z, _, _ = crop_patch(pred_full.astype(np.float32),
                                  data['px_star'], data['py_star'],
                                  data['zoom_size'])
        preds_maps[method] = pred_z / 255.0
    data['preds_maps'] = preds_maps

    out_dir = args.out_dir
    print(f"\nGenerating patches → {out_dir}/")
    generate_patches(data, out_dir)
    make_preview(data, out_dir)

    # 打印主 box 的热点偏移信息
    x1, y1, x2, y2 = data['boxes_p'][0]
    gcx, gcy = (x1 + x2) / 2, (y1 + y2) / 2
    px_s, py_s = data['px_star_p'], data['py_star_p']
    offset = np.sqrt((px_s - gcx)**2 + (py_s - gcy)**2)
    diag = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    print(f"\nPrimary box geo-center (patch coords): ({gcx:.1f}, {gcy:.1f})")
    print(f"Physical argmax p* (patch coords):     ({px_s}, {py_s})")
    print(f"Offset: {offset:.1f} px  (box size: {x2-x1}×{y2-y1}, "
          f"offset_ratio={offset/(diag+1e-6):.3f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=ROOT)
    parser.add_argument('--out_dir', default=os.path.join(
        ROOT, 'paper', 'figures', 'teaser_patches'))
    parser.add_argument('--preds_root', default=os.path.join(
        ROOT, 'paper', 'figures', 'preds'),
        help='Root dir containing gt_*/halo_*/bsnr_*/boxfill_* prediction folders')
    parser.add_argument('--nuaa',  default='')
    parser.add_argument('--nudt',  default='')
    parser.add_argument('--irstd', default='')
    parser.add_argument('--dataset', default='NUAA-SIRST',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--mine', action='store_true',
                        help='Find samples with max argmax-vs-geocenter offset')
    parser.add_argument('--mine_n', type=int, default=5)
    args = parser.parse_args()
    main(args)
