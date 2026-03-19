#!/usr/bin/env python3
"""
Figure 0: Dataset Motivation — From Cheap Box to Precise Pseudo-label
======================================================================
Teaser figure: 3行（数据集） × 4列（叙事角色）

  (a) Full Image           — 完整原图，黄色虚线框标注放大区域
  (b) Box Supervision      — 放大局部 + YOLO 框（粗糙的弱监督）
  (c) Pixel-level GT       — 放大局部 + 像素级掩码（昂贵的强监督）
  (d) HALO Pseudo-label    — 放大局部 + 我们离线生成的高斯软伪标签

叙事核心："用便宜的 (b) 生成媲美 (c) 的 (d)"
  - (a) 给读者空间定位感，黄色框标出后三列的放大区域
  - (b) 展示框的粗糙性（Box/GT 面积比标注）
  - (c) 展示目标的微小性（像素数标注）
  - (d) 展示高斯软标签的精准集中

用法:
    python paper/fig0_dataset_samples.py
    python paper/fig0_dataset_samples.py --nuaa Misc_33 --nudt 000032 --irstd XDU343
    python paper/fig0_dataset_samples.py --out paper/figures/fig0.pdf

制图铁律:
    - 所有 imshow 使用 interpolation='nearest'
    - 伪彩色统一使用 'inferno'
    - pdf.fonttype=42 确保 PDF 字体可嵌入
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
import matplotlib.patches as patches

# ────────────────────────────────────────────────────────────────
# ★ 路径配置
# ────────────────────────────────────────────────────────────────
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

# ★ 手动指定每个数据集要展示的样本文件名（不含扩展名）
# 如果留空 ''，脚本会自动选 test split 第一张有目标的图
SAMPLE_NAMES = {
    'NUAA-SIRST': '',
    'NUDT-SIRST': '',
    'IRSTD-1k':   '',
}

# 局部裁剪框大小（以目标中心为中心的正方形 patch，像素）
CROP_SIZE = 64

# 图像输出分辨率
DPI = 300

# 列边框颜色 —— 视觉上区分四个叙事角色
# (a) 全图定位: 黄  (b) 框监督: 红  (c) GT: 青  (d) 伪标签: 橙
COL_SPINE_COLORS   = ['#CCAA33', '#DD4444', '#33BBBB', '#EE8822']
COL_TITLE_COLORS   = ['#FFDD66', '#FF8888', '#66DDDD', '#FFAA44']
# 全图中标注裁剪区域的虚线框颜色（与标题颜色联动）
CROP_BOX_COLOR     = '#FFEE55'

# ────────────────────────────────────────────────────────────────


def load_bsnr_utils(root):
    """直接加载 bsnr_mask_utils 避免触发 torch 导入"""
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
    with open(label_path, 'r') as f:
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


def auto_select_sample(cfg, root):
    """自动从 test split 中选第一张有 YOLO label 且目标非空的图"""
    split_file = os.path.join(root, cfg['split'])
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, 'r') as f:
        names = [l.strip() for l in f if l.strip()]
    for name in names:
        label_path = os.path.join(root, cfg['label_dir'], name + '.txt')
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            return name
    raise RuntimeError("No valid sample found in split.")


def get_target_center(boxes, img_h, img_w):
    """返回所有 boxes 的联合中心"""
    if not boxes:
        return img_w // 2, img_h // 2
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1 + x2) // 2, (y1 + y2) // 2


def crop_patch(arr, cx, cy, size, mode='gray'):
    """以 (cx, cy) 为中心裁剪 size×size 的 patch，nearest 填边。
    返回 (patch, ox, oy) 其中 ox/oy 是 patch 左上角在原图中的坐标。
    """
    h, w = arr.shape[:2]
    half = size // 2
    ox = max(0, min(cx - half, w - size))   # clamp 使 patch 不越界
    oy = max(0, min(cy - half, h - size))
    patch = arr[oy:oy + size, ox:ox + size]
    # 边缘极端情况：图像本身小于 size
    if patch.shape[0] < size or patch.shape[1] < size:
        if mode == 'gray':
            out = np.zeros((size, size), dtype=patch.dtype)
        else:
            out = np.zeros((size, size, patch.shape[2]), dtype=patch.dtype)
        out[:patch.shape[0], :patch.shape[1]] = patch
        patch = out
    return patch, ox, oy


def prepare_row(dataset_name, cfg, root, sample_name, compute_bsnr_weight):
    """
    返回:
      img_full        — 完整原始灰度图（用于第 0 列全局展示）
      img_patch       — 放大局部块（以目标为中心，CROP_SIZE×CROP_SIZE）
      mask_patch      — GT 掩码局部块
      halo_patch      — HALO 离线高斯软伪标签局部块
      boxes_in_patch  — patch 坐标系内的 YOLO 框列表 [(x,y,w,h),...]
      box_area        — 全部 YOLO 框总面积（像素²）
      target_area     — GT 掩码前景面积（像素²）
      crop_ox, crop_oy — 裁剪块左上角在原图中的坐标（用于绘制定位框）
    """
    img_path   = os.path.join(root, cfg['img_dir'],   sample_name + cfg['img_ext'])
    mask_path  = os.path.join(root, cfg['mask_dir'],  sample_name + cfg['img_ext'])
    label_path = os.path.join(root, cfg['label_dir'], sample_name + '.txt')

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros((H, W), dtype=np.uint8)
    mask = (mask > 127).astype(np.uint8) * 255

    boxes = load_yolo_boxes(label_path, H, W)
    cx, cy = get_target_center(boxes, H, W)

    # 裁剪 patch（不会产生黑边）
    img_patch,  ox, oy = crop_patch(img_gray, cx, cy, CROP_SIZE)
    mask_patch, _,  _  = crop_patch(mask,     cx, cy, CROP_SIZE)

    # boxes 在 patch 坐标系内的位置
    boxes_in_patch = []
    for (x1, y1, x2, y2) in boxes:
        boxes_in_patch.append((x1 - ox, y1 - oy, (x2 - x1), (y2 - y1)))

    # 面积统计（用于叙事标注）
    box_area    = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes)
    target_area = int(np.sum(mask > 127))

    # 生成 HALO 离线高斯软伪标签（全图，然后裁剪）
    img_float = img_gray.astype(np.float32) / 255.0
    halo_full = np.zeros((H, W), dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = box
        w_map = compute_bsnr_weight(
            img_float, box,
            expand_ratio=1.5, temperature=3.0,
            spatial_gaussian=True, gauss_sigma_ratio=1.5
        )
        halo_full[y1:y2, x1:x2] = np.maximum(halo_full[y1:y2, x1:x2], w_map)
    halo_patch, _, _ = crop_patch(halo_full, cx, cy, CROP_SIZE)

    return (img_gray, img_patch, mask_patch, halo_patch,
            boxes_in_patch, box_area, target_area, ox, oy)


def _annotate(ax, text, color, loc='bottom'):
    """在 subplot 角落添加统计标注（带黑底半透明衬底）"""
    y  = CROP_SIZE - 2 if loc == 'bottom' else 2
    va = 'bottom'      if loc == 'bottom' else 'top'
    ax.text(2, y, text, color=color, fontsize=6.5, va=va, ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.65,
                      edgecolor='none'))


def _set_spines(ax, color, lw=1.5):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


def main(args):
    compute_bsnr_weight = load_bsnr_utils(args.root)

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    fig.patch.set_facecolor('black')

    # 列标题：强调叙事角色
    col_titles = [
        '(a) Full Image\n(with zoom region)',
        '(b) Box Supervision\n(Cheap & Coarse)',
        '(c) Pixel-level GT\n(Expensive)',
        '(d) HALO Pseudo-label\n(Ours)',
    ]

    row_labels = [
        'NUAA-SIRST\n(medium target)',
        'NUDT-SIRST\n(dim tiny target)',
        'IRSTD-1k\n(complex BG)',
    ]

    datasets = ['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k']

    for row_idx, (ds_name, row_label) in enumerate(zip(datasets, row_labels)):
        cfg  = DATASET_CONFIGS[ds_name]
        name = SAMPLE_NAMES[ds_name]
        if not name:
            name = auto_select_sample(cfg, args.root)
            print(f"[{ds_name}] auto-selected: {name}")
        else:
            print(f"[{ds_name}] using: {name}")

        (img_full, img_patch, mask_patch, halo_patch,
         boxes_in_patch, box_area, target_area, crop_ox, crop_oy) = prepare_row(
            ds_name, cfg, args.root, name, compute_bsnr_weight
        )

        # ── (a) 完整原图 + 黄色虚线框标注放大区域 ──────────────────
        ax = axes[row_idx][0]
        ax.imshow(img_full, cmap='gray', interpolation='nearest',
                  vmin=0, vmax=255)
        # 黄色虚线框：标出后三列的放大区域
        rect = patches.Rectangle(
            (crop_ox, crop_oy), CROP_SIZE, CROP_SIZE,
            linewidth=1.5, edgecolor=CROP_BOX_COLOR,
            facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        # 四角连接线（小L形），增强定位感
        tick = max(3, CROP_SIZE // 8)
        for dx, dy in [(0, 0), (CROP_SIZE, 0), (0, CROP_SIZE), (CROP_SIZE, CROP_SIZE)]:
            sx = 1 if dx == 0 else -1
            sy = 1 if dy == 0 else -1
            ax.plot([crop_ox + dx, crop_ox + dx + sx * tick],
                    [crop_oy + dy, crop_oy + dy],
                    color=CROP_BOX_COLOR, lw=1.0)
            ax.plot([crop_ox + dx, crop_ox + dx],
                    [crop_oy + dy, crop_oy + dy + sy * tick],
                    color=CROP_BOX_COLOR, lw=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(col_titles[0], color=COL_TITLE_COLORS[0], fontsize=10, pad=5)
        ax.set_ylabel(row_label, color='white', fontsize=8, labelpad=4)
        _set_spines(ax, COL_SPINE_COLORS[0])

        # ── (b) 放大局部 + YOLO 红框（粗糙的弱监督）────────────────
        ax = axes[row_idx][1]
        ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        for (bx, by, bw, bh) in boxes_in_patch:
            rect = patches.Rectangle((bx, by), bw, bh,
                                     linewidth=2.0, edgecolor='#FF3333', facecolor='none')
            ax.add_patch(rect)
        # 标注 Box/GT 面积比 → 强化"框很粗糙"
        if target_area > 0 and box_area > 0:
            ratio = box_area / target_area
            _annotate(ax, f'Box/GT \u2248 {ratio:.0f}\u00d7', '#FF6666')
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(col_titles[1], color=COL_TITLE_COLORS[1], fontsize=10, pad=5)
        _set_spines(ax, COL_SPINE_COLORS[1])

        # ── (c) 放大局部 + GT 像素掩码（昂贵的强监督）──────────────
        ax = axes[row_idx][2]
        ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        gt_overlay = np.zeros((*img_patch.shape, 4), dtype=np.float32)
        gt_overlay[mask_patch > 127, :] = [0.0, 1.0, 1.0, 0.85]  # cyan
        ax.imshow(gt_overlay, interpolation='nearest')
        # 标注目标像素数 → 强化"目标极小"
        if target_area > 0:
            _annotate(ax, f'Target: {target_area}\u202fpx\u00b2', '#66FFFF')
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(col_titles[2], color=COL_TITLE_COLORS[2], fontsize=10, pad=5)
        _set_spines(ax, COL_SPINE_COLORS[2])

        # ── (d) 放大局部 + HALO 离线高斯软伪标签 ────────────────────
        ax = axes[row_idx][3]
        ax.imshow(img_patch, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        ax.imshow(halo_patch, cmap='inferno', alpha=0.75,
                  interpolation='nearest', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(col_titles[3], color=COL_TITLE_COLORS[3], fontsize=10, pad=5)
        _set_spines(ax, COL_SPINE_COLORS[3])

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.10, hspace=0.08, wspace=0.06)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=DPI, bbox_inches='tight', facecolor='black')
    print(f"Saved \u2192 {args.out}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=ROOT, help='Project root directory')
    parser.add_argument('--out',  default=os.path.join(ROOT, 'paper', 'figures', 'fig0_samples.png'))
    parser.add_argument('--nuaa',  default='', help='NUAA-SIRST sample name (e.g. Misc_33)')
    parser.add_argument('--nudt',  default='', help='NUDT-SIRST sample name (e.g. 000032)')
    parser.add_argument('--irstd', default='', help='IRSTD-1k sample name (e.g. XDU343)')
    args = parser.parse_args()
    if args.nuaa:  SAMPLE_NAMES['NUAA-SIRST'] = args.nuaa
    if args.nudt:  SAMPLE_NAMES['NUDT-SIRST'] = args.nudt
    if args.irstd: SAMPLE_NAMES['IRSTD-1k']   = args.irstd
    main(args)
