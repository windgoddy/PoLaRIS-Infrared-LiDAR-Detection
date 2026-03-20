#!/usr/bin/env python3
"""
eval_anchor_vs_geocenter.py
===========================
核心消融：物理锚点 p* (HALO) vs 几何中心 (传统假设)

【实验逻辑】
  HALO 的真正贡献是：用 B-SNR 找到框内最亮像素 p*，
  而非盲目假设目标在 YOLO 框几何中心。

  对于小目标（IR 图像中目标往往偏离框中心）：
    p*       = argmax I(i) within box  ← HALO 数据驱动定位
    geo-center = ((x1+x2)/2, (y1+y2)/2) ← 传统几何假设

  以 p* / geo-center 为中心生成高斯软标签 P(i)，
  对比其与 GT mask 的质量差异。

【评估指标】
  1. 锚点命中率 (Hit Rate)：锚点像素是否在 GT mask 内
  2. 锚点偏移距离 (Dist)：锚点到 GT mask 最近像素的距离 (px)
  3. 软标签 IoU：P(i) 以 0.5 为阈值二值化后与 GT mask 的 IoU
  4. 软标签 Dice：同上，用 Dice 系数衡量

【为什么不需要多个 R 值】
  p* = argmax I(i) 与 μ/σ 无关，因此与 R 无关。
  几何中心也与 R 无关。本实验与 R 的选取解耦。

用法：
  python paper/eval_anchor_vs_geocenter.py
  python paper/eval_anchor_vs_geocenter.py --n_samples 300 --sigma_g_beta 0.30
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── σ_G 参数：P(i) 高斯宽度 = β * sqrt(box_area) ──────────────────────────
SIGMA_G_BETA = 0.26   # 与 HALO 训练一致

DATASET_CFG = {
    'NUDT-SIRST': {
        'img_dir':  'dataset/NUDT-SIRST/images',
        'mask_dir': 'dataset/NUDT-SIRST/masks',
        'split':    'dataset/NUDT-SIRST/50_50/test.txt',
        'img_ext':  '.png',
        'label':    'NUDT-SIRST\n(tiny target)',
    },
    'NUAA-SIRST': {
        'img_dir':  'dataset/NUAA-SIRST/images',
        'mask_dir': 'dataset/NUAA-SIRST/masks',
        'split':    'dataset/NUAA-SIRST/50_50/test.txt',
        'img_ext':  '.png',
        'label':    'NUAA-SIRST\n(medium target)',
    },
    'IRSTD-1k': {
        'img_dir':  'dataset/IRSTD-1k/IRSTD1k_Img',
        'mask_dir': 'dataset/IRSTD-1k/masks',
        'split':    'dataset/IRSTD-1k/test.txt',
        'img_ext':  '.png',
        'label':    'IRSTD-1k\n(mixed target)',
    },
}


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def boxes_from_mask(mask):
    """从 GT 二值掩码提取包围框列表。"""
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 127).astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, len(stats)):
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if w > 0 and h > 0:
            boxes.append((x, y, x + w, y + h))
    return boxes


def make_gaussian_label(H, W, anchor_x, anchor_y, sigma_g):
    """
    在 H×W 图上以 (anchor_x, anchor_y) 为中心生成高斯软标签 P(i)。
    返回 float32 图，最大值归一化到 1.0。
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    P = np.exp(-((xx - anchor_x)**2 + (yy - anchor_y)**2) / (2 * sigma_g**2))
    return P  # 最大值已为 1.0（中心处）


def mask_to_binary(P, threshold=0.5):
    return (P >= threshold).astype(np.uint8)


def iou_dice(pred_bin, gt_bin):
    """计算二值图的 IoU 和 Dice。"""
    inter = int((pred_bin & gt_bin).sum())
    union = int((pred_bin | gt_bin).sum())
    sum_  = int(pred_bin.sum()) + int(gt_bin.sum())
    iou  = inter / (union + 1e-6)
    dice = 2 * inter / (sum_  + 1e-6)
    return iou, dice


def dist_to_mask(px, py, mask):
    """计算点 (px, py) 到 GT mask 最近像素的欧氏距离。"""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return 9999.0
    dists = np.hypot(xs - px, ys - py)
    return float(dists.min())


def gt_centroid(mask):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None, None
    return float(np.mean(xs)), float(np.mean(ys))


# ─── 单图评估 ─────────────────────────────────────────────────────────────────

def eval_one(img_gray, mask, box, sigma_g_beta):
    """
    返回 dict 包含 p* 和 geo-center 的各项指标。
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None

    H, W = img_gray.shape
    sigma_g = sigma_g_beta * np.sqrt(bw * bh)

    # ── 1. 确定两种锚点 ───────────────────────────────────────────────────
    # p*：框内强度最大的像素（argmax I(i)）
    roi = img_gray[y1:y2, x1:x2].astype(np.float32)
    flat = int(np.argmax(roi))
    py_rel, px_rel = divmod(flat, bw)
    px_star = x1 + px_rel
    py_star = y1 + py_rel

    # 几何中心（框中心，传统假设）
    px_geo = (x1 + x2) / 2.0
    py_geo = (y1 + y2) / 2.0

    result = {}
    gt_bin = (mask > 127).astype(np.uint8)

    for name, ax, ay in [('pstar', px_star, py_star),
                          ('geo',   px_geo,  py_geo)]:
        # 命中率：锚点像素（取整）是否在 GT mask 内
        ax_int, ay_int = int(round(ax)), int(round(ay))
        ax_int = np.clip(ax_int, 0, W-1)
        ay_int = np.clip(ay_int, 0, H-1)
        hit = bool(gt_bin[ay_int, ax_int] > 0)

        # 偏移距离：锚点到 GT mask 最近像素
        dist = 0.0 if hit else dist_to_mask(ax, ay, mask)

        # 软标签 P(i) 生成（整图，高斯中心在锚点）
        P = make_gaussian_label(H, W, ax, ay, sigma_g)

        # 只评估框内区域（软标签只在框内有意义）
        P_box = np.zeros((H, W), np.float32)
        P_box[y1:y2, x1:x2] = P[y1:y2, x1:x2]

        # IoU / Dice（P 二值化后 vs GT）
        P_bin = mask_to_binary(P_box, threshold=0.5)
        iou, dice = iou_dice(P_bin, gt_bin)

        result[f'{name}_hit']  = hit
        result[f'{name}_dist'] = dist
        result[f'{name}_iou']  = iou
        result[f'{name}_dice'] = dice

    return result


# ─── 数据集批量评估 ───────────────────────────────────────────────────────────

def eval_dataset(ds_name, cfg, n_samples, root, sigma_g_beta):
    split_path = os.path.join(root, cfg['split'])
    img_dir    = os.path.join(root, cfg['img_dir'])
    mask_dir   = os.path.join(root, cfg['mask_dir'])
    ext        = cfg['img_ext']

    with open(split_path) as f:
        names = [l.strip() for l in f if l.strip()]

    rng = np.random.default_rng(42)
    if n_samples and n_samples < len(names):
        names = rng.choice(names, n_samples, replace=False).tolist()

    keys = ['pstar_hit', 'pstar_dist', 'pstar_iou', 'pstar_dice',
            'geo_hit',   'geo_dist',   'geo_iou',   'geo_dice']
    records = {k: [] for k in keys}

    skipped = 0
    for name in names:
        img_bgr = cv2.imread(os.path.join(img_dir,  name + ext))
        if img_bgr is None:
            skipped += 1; continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
                   if img_bgr.ndim == 3 else img_bgr.copy()

        mask = cv2.imread(os.path.join(mask_dir, name + ext), 0)
        if mask is None or np.max(mask) == 0:
            skipped += 1; continue

        # 确保 mask 与 img 尺寸一致
        if mask.shape != img_gray.shape:
            mask = cv2.resize(mask, (img_gray.shape[1], img_gray.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        boxes = boxes_from_mask(mask)
        if not boxes:
            skipped += 1; continue
        box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

        res = eval_one(img_gray, mask, box, sigma_g_beta)
        if res is None:
            skipped += 1; continue

        for k in keys:
            records[k].append(res[k])

    if skipped:
        print(f"  [{ds_name}] 跳过 {skipped}/{len(names)} 张")

    return records


# ─── 打印表格 ─────────────────────────────────────────────────────────────────

def print_table(ds_name, records):
    n = len(records['pstar_hit'])
    if n == 0:
        print(f"  [{ds_name}] 无有效样本")
        return

    ph  = 100 * np.mean(records['pstar_hit'])
    gh  = 100 * np.mean(records['geo_hit'])
    pd  = np.mean(records['pstar_dist'])
    gd  = np.mean(records['geo_dist'])
    pi  = np.mean(records['pstar_iou'])
    gi  = np.mean(records['geo_iou'])
    pdi = np.mean(records['pstar_dice'])
    gdi = np.mean(records['geo_dice'])

    print(f"\n{'='*68}")
    print(f"  数据集: {ds_name}  (N={n})")
    print(f"{'='*68}")
    print(f"  {'指标':<22}  {'p* (HALO)':>12}  {'Geo-Center':>12}  {'Δ (p*-Geo)':>12}")
    print(f"  {'-'*64}")
    print(f"  {'Hit Rate (%)':<22}  {ph:>12.1f}  {gh:>12.1f}  {ph-gh:>+12.1f}")
    print(f"  {'Dist to GT (px)':<22}  {pd:>12.3f}  {gd:>12.3f}  {pd-gd:>+12.3f}")
    print(f"  {'Soft Label IoU':<22}  {pi:>12.4f}  {gi:>12.4f}  {pi-gi:>+12.4f}")
    print(f"  {'Soft Label Dice':<22}  {pdi:>12.4f}  {gdi:>12.4f}  {pdi-gdi:>+12.4f}")
    print()


# ─── 绘图 ─────────────────────────────────────────────────────────────────────

def plot_results(all_results, out_path):
    datasets = list(all_results.keys())
    n_ds = len(datasets)

    metrics = [
        ('hit',  'Hit Rate (%)',         100, True),
        ('dist', 'Dist to GT (px)',       1,  False),
        ('iou',  'Soft Label IoU',        1,  True),
        ('dice', 'Soft Label Dice',       1,  True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    C_PSTAR = '#1565C0'
    C_GEO   = '#C62828'
    x = np.arange(n_ds)
    bw = 0.32

    for idx, (key, ylabel, scale, higher_is_better) in enumerate(metrics):
        ax = axes[idx]
        pstar_vals = []
        geo_vals   = []
        for ds_name in datasets:
            rec = all_results[ds_name]
            n   = len(rec[f'pstar_{key}'])
            if n == 0:
                pstar_vals.append(0); geo_vals.append(0)
            else:
                pstar_vals.append(np.mean(rec[f'pstar_{key}']) * scale)
                geo_vals.append  (np.mean(rec[f'geo_{key}'])   * scale)

        bars_p = ax.bar(x - bw/2, pstar_vals, bw,
                        color=C_PSTAR, alpha=0.85, label=r'$p^*$ (HALO)',
                        zorder=3, edgecolor='white', linewidth=0.8)
        bars_g = ax.bar(x + bw/2, geo_vals,   bw,
                        color=C_GEO,   alpha=0.85, label='Geo-Center',
                        zorder=3, edgecolor='white', linewidth=0.8)

        # 数值标签
        for bar in bars_p:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001 * scale,
                    f'{h:.2f}', ha='center', va='bottom',
                    fontsize=7.5, color=C_PSTAR, fontweight='bold')
        for bar in bars_g:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001 * scale,
                    f'{h:.2f}', ha='center', va='bottom',
                    fontsize=7.5, color=C_GEO, fontweight='bold')

        # Δ 标注（p* - Geo）
        for i, (pv, gv) in enumerate(zip(pstar_vals, geo_vals)):
            delta = pv - gv
            color = '#1B5E20' if (delta > 0) == higher_is_better else '#B71C1C'
            sym   = '+' if delta >= 0 else ''
            y_pos = max(pv, gv) + 0.025 * scale
            ax.text(x[i], y_pos, f'Δ={sym}{delta:.2f}',
                    ha='center', va='bottom', fontsize=8,
                    color=color, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([all_results[d] if isinstance(all_results[d], str)
                            else DATASET_CFG[d]['label']
                            for d in datasets],
                           fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        arrow = '↑ better' if higher_is_better else '↓ better'
        ax.text(0.98, 0.02, arrow, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='#607D8B')

    fig.suptitle(
        r'Physical Anchor $p^*$ (HALO) vs Geometric Center' '\n'
        r'Soft Label Quality: $P(i) = \exp(-\|r_i - \mathrm{anchor}\|^2 / 2\sigma_G^2)$',
        fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f"\n图像已保存 → {out_path}")
    plt.close()


# ─── 可视化案例（失败对比图）────────────────────────────────────────────────

def visualize_cases(all_results_raw, out_path, n_cases=6):
    """
    挑选 geo-center 失败（不在 mask 内）但 p* 成功的典型样本，
    并排展示：IR 图 + p* 标注 + geo-center 标注 + GT mask
    """
    # all_results_raw: {ds_name: [(img, mask, box, res), ...]}
    fail_cases = []
    for ds_name, cases in all_results_raw.items():
        for img_gray, mask, box, res in cases:
            if res is None:
                continue
            if res['pstar_hit'] and not res['geo_hit']:
                fail_cases.append((ds_name, img_gray, mask, box, res))

    if not fail_cases:
        print("  未找到 p* 命中但 Geo 失败的典型样本（数据集目标可能都在框中心）")
        return

    n = min(n_cases, len(fail_cases))
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, (ds_name, img_gray, mask, box, res) in zip(axes, fail_cases[:n]):
        x1, y1, x2, y2 = box
        # 裁剪到框附近（稍微扩大一点）
        pad = max(10, (x2-x1))
        cx1 = max(0, x1-pad); cy1 = max(0, y1-pad)
        cx2 = min(img_gray.shape[1], x2+pad)
        cy2 = min(img_gray.shape[0], y2+pad)
        crop_img  = img_gray[cy1:cy2, cx1:cx2]
        crop_mask = mask[cy1:cy2, cx1:cx2]

        ax.imshow(crop_img, cmap='gray', vmin=0, vmax=255)
        ax.contour(crop_mask, levels=[127], colors=['#00E676'], linewidths=1.5)

        # YOLO 框
        rect_x = x1-cx1; rect_y = y1-cy1
        ax.add_patch(plt.Rectangle((rect_x, rect_y), x2-x1, y2-y1,
                                    edgecolor='#FFD740', facecolor='none',
                                    lw=1.8, ls='--'))

        # p* 标注（蓝色五角星）
        ax.plot(res['pstar_ax']-cx1, res['pstar_ay']-cy1,
                '*', color='#1565C0', markersize=12,
                markeredgecolor='white', markeredgewidth=0.8,
                label=r'$p^*$ (HALO)', zorder=5)

        # Geo-center 标注（红色叉）
        ax.plot(res['geo_ax']-cx1, res['geo_ay']-cy1,
                'x', color='#C62828', markersize=10,
                markeredgewidth=2.5, label='Geo-Center', zorder=5)

        ax.set_title(f'{ds_name}\np* hit={res["pstar_hit"]}  '
                     f'geo hit={res["geo_hit"]}',
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc='lower right')

    fig.suptitle(r'Typical Cases: $p^*$ Correct, Geo-Center Wrong',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    case_path = out_path.replace('.png', '_cases.png')
    plt.savefig(case_path, dpi=180, bbox_inches='tight')
    print(f"案例图已保存 → {case_path}")
    plt.close()


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=r'消融：p* (HALO) vs 几何中心 的软标签质量')
    parser.add_argument('--root', default=ROOT)
    parser.add_argument('--datasets', nargs='+',
                        default=['NUDT-SIRST', 'NUAA-SIRST', 'IRSTD-1k'],
                        choices=list(DATASET_CFG.keys()))
    parser.add_argument('--n_samples', type=int, default=0,
                        help='每数据集最多评估张数（0=全部）')
    parser.add_argument('--sigma_g_beta', type=float, default=SIGMA_G_BETA,
                        help='高斯软标签宽度系数 σ_G = β·sqrt(box_area)')
    parser.add_argument('--out', default=os.path.join(
                        ROOT, 'paper', 'figures', 'fig_anchor_vs_geocenter.png'))
    args = parser.parse_args()

    all_results     = {}   # 用于绘图（聚合统计）
    all_results_raw = {}   # 用于案例可视化（保存每张图的详情）

    for ds_name in args.datasets:
        cfg = DATASET_CFG.get(ds_name)
        if cfg is None:
            continue
        img_dir = os.path.join(args.root, cfg['img_dir'])
        if not os.path.isdir(img_dir):
            print(f"[{ds_name}] 图像目录不存在，跳过: {img_dir}"); continue

        print(f"\n[{ds_name}] 开始评估 (σ_G_beta={args.sigma_g_beta})...")

        # ── 标准统计 ──────────────────────────────────────────────────────
        records = eval_dataset(ds_name, cfg, args.n_samples,
                               args.root, args.sigma_g_beta)
        all_results[ds_name] = records
        print_table(ds_name, records)

        # ── 原始数据（用于案例图）─────────────────────────────────────────
        split_path = os.path.join(args.root, cfg['split'])
        img_dir_   = os.path.join(args.root, cfg['img_dir'])
        mask_dir_  = os.path.join(args.root, cfg['mask_dir'])
        ext        = cfg['img_ext']
        with open(split_path) as f:
            names = [l.strip() for l in f if l.strip()]
        rng = np.random.default_rng(42)
        if args.n_samples and args.n_samples < len(names):
            names = rng.choice(names, args.n_samples, replace=False).tolist()

        raw_cases = []
        for name in names[:200]:   # 案例图只需要检查前200张
            img_bgr = cv2.imread(os.path.join(img_dir_,  name + ext))
            if img_bgr is None: continue
            ig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
                 if img_bgr.ndim == 3 else img_bgr.copy()
            mask = cv2.imread(os.path.join(mask_dir_, name + ext), 0)
            if mask is None or np.max(mask) == 0: continue
            if mask.shape != ig.shape:
                mask = cv2.resize(mask, (ig.shape[1], ig.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            boxes = boxes_from_mask(mask)
            if not boxes: continue
            box = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            res = eval_one(ig, mask, box, args.sigma_g_beta)
            if res is None: continue

            # 把锚点坐标也存进去（给可视化用）
            x1, y1, x2, y2 = box
            bw, bh = x2-x1, y2-y1
            roi = ig[y1:y2, x1:x2].astype(np.float32)
            flat = int(np.argmax(roi))
            py_rel, px_rel = divmod(flat, bw)
            res['pstar_ax'] = x1 + px_rel
            res['pstar_ay'] = y1 + py_rel
            res['geo_ax']   = (x1 + x2) / 2.0
            res['geo_ay']   = (y1 + y2) / 2.0
            raw_cases.append((ig, mask, box, res))

        all_results_raw[ds_name] = raw_cases

    if not all_results:
        print("没有可用数据集，退出。"); return

    # ── 绘制主统计图 ──────────────────────────────────────────────────────
    plot_results(all_results, args.out)

    # ── 绘制典型失败案例图 ────────────────────────────────────────────────
    visualize_cases(all_results_raw, args.out)

    # ── 核心结论汇总 ──────────────────────────────────────────────────────
    print("\n【核心结论】p* vs Geo-Center 在 Soft Label IoU 上的差距：")
    print(f"  {'数据集':<15}  {'p* IoU':>8}  {'Geo IoU':>8}  "
          f"{'Δ IoU':>8}  {'p* Hit%':>8}  {'Geo Hit%':>8}  N")
    for ds_name, rec in all_results.items():
        n = len(rec['pstar_iou'])
        if n == 0: continue
        pi = np.mean(rec['pstar_iou'])
        gi = np.mean(rec['geo_iou'])
        ph = 100 * np.mean(rec['pstar_hit'])
        gh = 100 * np.mean(rec['geo_hit'])
        print(f"  {ds_name:<15}  {pi:>8.4f}  {gi:>8.4f}  "
              f"{pi-gi:>+8.4f}  {ph:>7.1f}%  {gh:>7.1f}%  {n}")


if __name__ == '__main__':
    main()
