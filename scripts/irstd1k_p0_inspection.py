"""
P0 风险排查：IRSTD-1k 强杂波场景 PAG 定位准确性巡检
================================================================

功能：
  1. 从 IRSTD-1k 像素掩码推导 YOLO Box（模拟弱监督标注）
  2. 计算 B-SNR 和 PAG (B-SNR × 物理锚定高斯) 软标签
  3. 验证 SNR argmax（物理热点）是否准确落在 GT 目标区域内
  4. 输出可视化对比图 + 定位准确率统计

选样策略：优先选背景方差最高的图像（强杂波 = 最难场景）

可视化图例：
  - 绿框 = 该 box 的 SNR 热点准确落在 GT 区域  ✓
  - 红框 = 热点偏移，可能被云层/海面杂波误吸引  ✗
  - 蓝点 = SNR argmax 的绝对坐标

输出结论：
  - 准确率 ≥ 85%  → P1 可继续推进
  - 准确率 70-85% → 考虑加入距离惩罚项
  - 准确率 < 70%  → 方法需要修复

用法：
    python scripts/irstd1k_p0_inspection.py \\
        --dataset_dir /home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/IRSTD-1k \\
        --n_samples 50 \\
        --output_dir pilot_vis/irstd1k_p0
"""

import os
import sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.nudt_mask_to_yolo_box import mask_to_yolo_boxes
from model_Mamba.dataset.bsnr_mask_utils import compute_bsnr_weight


# ======================================================
# 辅助：选取强杂波困难样本
# ======================================================

def select_hard_samples(dataset_dir, img_ids, n_samples=50):
    """
    按背景区域标准差从高到低排序，选取最困难的 n_samples 个样本。
    背景方差越高 = 海面/云层杂波越强 = 对 PAG 最具挑战性。
    """
    img_dir = os.path.join(dataset_dir, 'IRSTD1k_Img')
    lbl_dir = os.path.join(dataset_dir, 'IRSTD1k_Label')

    scores = []
    for img_id in img_ids:
        img_path = os.path.join(img_dir, img_id + '.png')
        lbl_path = os.path.join(lbl_dir, img_id + '.png')

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

        if img is None or lbl is None:
            continue

        # 仅统计背景区域（排除目标像素）的方差
        bg_mask = (lbl < 128)
        if bg_mask.sum() < 100:
            continue

        bg_std = float(img[bg_mask].std())
        scores.append((bg_std, img_id))

    scores.sort(reverse=True)
    selected = [s[1] for s in scores[:n_samples]]
    print(f"  背景方差范围: [{scores[min(n_samples,len(scores))-1][0]:.1f}, {scores[0][0]:.1f}]")
    return selected


# ======================================================
# 核心：逐 box 检查 SNR argmax 准确性
# ======================================================

def check_peak_in_gt(snr_map, gt_crop):
    """
    检查 SNR 的最大值像素是否落在 GT 掩码区域内。

    Args:
        snr_map: (bh, bw) float32，框内原始 SNR 值（未经 sigmoid）
        gt_crop: (bh, bw) uint8，GT 掩码在 box 区域内的裁剪

    Returns:
        is_accurate: bool
        peak_y, peak_x: int，SNR 峰值在 box 内的局部坐标
    """
    bh, bw = snr_map.shape
    peak_flat = int(snr_map.argmax())
    peak_y = peak_flat // bw
    peak_x = peak_flat % bw

    is_accurate = False
    if gt_crop is not None and gt_crop.shape == snr_map.shape:
        is_accurate = bool(gt_crop[peak_y, peak_x] > 0)

    return is_accurate, peak_y, peak_x


# ======================================================
# 主巡检函数
# ======================================================

def run_p0_inspection(
    dataset_dir,
    n_samples=50,
    output_dir=None,
    pad=2,
    expand_ratio=1.5,
    temperature=3.0,
    gauss_sigma_ratio=1.5,
):
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, 'p0_inspection_vis')
    os.makedirs(output_dir, exist_ok=True)

    img_dir = os.path.join(dataset_dir, 'IRSTD1k_Img')
    lbl_dir = os.path.join(dataset_dir, 'IRSTD1k_Label')

    # 读取全部样本 ID
    all_txt = os.path.join(dataset_dir, 'trainvaltest.txt')
    with open(all_txt) as f:
        all_ids = [l.strip() for l in f if l.strip()]
    print(f"总样本数: {len(all_ids)}")

    # 选取强杂波困难样本
    print(f"按背景方差选取前 {n_samples} 个强杂波样本...")
    hard_ids = select_hard_samples(dataset_dir, all_ids, n_samples)
    print(f"已选 {len(hard_ids)} 个样本\n")

    # 统计计数器
    total_boxes    = 0
    peak_accurate  = 0
    peak_mislocate = []  # [(img_id, box_idx, gt_pixels, peak描述)]

    # 对比度分层分析：每个有效 box 的统计记录
    # snr_peak: 目标对比度指标（原始 SNR 峰值）
    # bsnr_gt_frac: B-SNR 权重落在 GT 区域内的比例
    # pag_gt_frac:  PAG 权重落在 GT 区域内的比例
    contrast_records = []

    for img_id in tqdm(hard_ids, desc='P0巡检'):
        img_path = os.path.join(img_dir, img_id + '.png')
        lbl_path = os.path.join(lbl_dir, img_id + '.png')

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        lbl     = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or lbl is None:
            continue

        # RGB → 灰度（用于 SNR 计算）
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W     = img_gray.shape
        gray_f   = img_gray.astype(np.float32) / 255.0

        # 从像素 GT 推导 YOLO Box（模拟弱监督标注，pad=2 像素膨胀）
        boxes = mask_to_yolo_boxes(lbl, pad=pad)
        if len(boxes) == 0:
            continue

        # 用于可视化的图层
        box_vis      = img_bgr.copy()
        bsnr_mask    = np.zeros((H, W), dtype=np.float32)  # C 组 B-SNR
        pag_mask     = np.zeros((H, W), dtype=np.float32)  # D 组 PAG

        box_flags = []  # 每个 box 的准确性标记

        for b_idx, box in enumerate(boxes):
            _, cx_n, cy_n, w_n, h_n = box[:5]
            cx = cx_n * W;  cy = cy_n * H
            bw = w_n  * W;  bh = h_n  * H
            x1 = int(max(0, cx - bw / 2))
            y1 = int(max(0, cy - bh / 2))
            x2 = int(min(W, cx + bw / 2))
            y2 = int(min(H, cy + bh / 2))
            if x2 <= x1 or y2 <= y1:
                continue

            total_boxes += 1

            # ─── 计算原始 SNR（用于热点定位验证）───
            pad_x = max(int((expand_ratio - 1.0) / 2.0 * (x2 - x1)), 3)
            pad_y = max(int((expand_ratio - 1.0) / 2.0 * (y2 - y1)), 3)
            ctx_x1 = max(0, x1 - pad_x);  ctx_y1 = max(0, y1 - pad_y)
            ctx_x2 = min(W, x2 + pad_x);  ctx_y2 = min(H, y2 + pad_y)

            ctx  = gray_f[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
            mu   = ctx.mean()
            sigma = ctx.std()
            snr  = (gray_f[y1:y2, x1:x2] - mu) / (sigma + 1e-4)

            # ─── 验证热点是否在 GT 区域 ───
            gt_crop   = (lbl[y1:y2, x1:x2] > 128).astype(np.uint8)
            is_acc, peak_y, peak_x = check_peak_in_gt(snr, gt_crop)

            box_flags.append(is_acc)
            if is_acc:
                peak_accurate += 1
            else:
                gt_px = int(gt_crop.sum())
                peak_mislocate.append(
                    f"{img_id}[box{b_idx}]: "
                    f"peak_abs=({x1+peak_x},{y1+peak_y}), "
                    f"gt_pixels_in_box={gt_px}, "
                    f"bg_std={sigma*255:.1f}"
                )

            # ─── 生成 C 组 B-SNR 软标签 ───
            w_c = compute_bsnr_weight(
                gray_f, (x1, y1, x2, y2),
                expand_ratio=expand_ratio,
                temperature=temperature,
                spatial_gaussian=False,
            )
            bsnr_mask[y1:y2, x1:x2] = np.maximum(bsnr_mask[y1:y2, x1:x2], w_c)

            # ─── 生成 D 组 PAG 软标签 ───
            w_d = compute_bsnr_weight(
                gray_f, (x1, y1, x2, y2),
                expand_ratio=expand_ratio,
                temperature=temperature,
                spatial_gaussian=True,
                gauss_sigma_ratio=gauss_sigma_ratio,
            )
            pag_mask[y1:y2, x1:x2] = np.maximum(pag_mask[y1:y2, x1:x2], w_d)

            # ─── 对比度分层：记录权重集中度指标 ───
            # 只对 GT 区域非空的 box 计算（排除 gt_pixels=0 的标注问题样本）
            gt_bool = (gt_crop > 0)
            if gt_bool.sum() > 0:
                total_w_c = float(w_c.sum()) + 1e-8
                total_w_d = float(w_d.sum()) + 1e-8
                bsnr_gt_frac = float(w_c[gt_bool].sum()) / total_w_c
                pag_gt_frac  = float(w_d[gt_bool].sum()) / total_w_d
                contrast_records.append({
                    'snr_peak':     float(snr.max()),  # 目标对比度（越大=目标越亮）
                    'bsnr_gt_frac': bsnr_gt_frac,
                    'pag_gt_frac':  pag_gt_frac,
                    'pag_gain':     pag_gt_frac - bsnr_gt_frac,  # PAG 相对 B-SNR 的提升
                    'is_accurate':  is_acc,
                    'img_id':       img_id,
                })

            # ─── 标注可视化 ───
            color = (0, 255, 0) if is_acc else (0, 0, 255)   # 绿=准确, 红=偏移
            cv2.rectangle(box_vis, (x1, y1), (x2, y2), color, 1)
            cv2.circle(box_vis, (x1 + peak_x, y1 + peak_y), 3, (255, 0, 0), -1)  # 蓝点=热点

        # ─── 生成并保存对比图 ───
        def to_heatmap(m):
            return cv2.applyColorMap((m * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)

        gt_vis   = to_heatmap(lbl.astype(np.float32) / 255.0)
        bsnr_vis = to_heatmap(bsnr_mask)
        pag_vis  = to_heatmap(pag_mask)

        panel_w = W
        acc_n   = sum(box_flags)
        total_n = len(box_flags)
        status  = "ALL_OK" if acc_n == total_n else f"MISS_{total_n-acc_n}"

        row = np.hstack([box_vis, gt_vis, bsnr_vis, pag_vis])
        # 标题条
        bar = np.zeros((22, row.shape[1], 3), dtype=np.uint8)
        titles = [f"Img+Box [{status}]", "GT Mask", "C: B-SNR", "D: PAG Soft"]
        for j, t in enumerate(titles):
            cv2.putText(bar, t, (j * panel_w + 4, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        out = np.vstack([bar, row])
        cv2.imwrite(os.path.join(output_dir, f"{img_id}_p0.png"), out)

    # ======================================================
    # 统计摘要
    # ======================================================
    print(f"\n{'='*62}")
    print(f"P0 巡检摘要 — IRSTD-1k 强杂波（最困难 {len(hard_ids)} 张）")
    print(f"{'='*62}")
    print(f"检查样本数  : {len(hard_ids)}")
    print(f"总 Box 数   : {total_boxes}")
    print(f"热点准确数  : {peak_accurate} / {total_boxes}")

    if total_boxes > 0:
        acc_rate = peak_accurate / total_boxes * 100
        print(f"定位准确率  : {acc_rate:.1f}%")
        print(f"偏移案例数  : {len(peak_mislocate)}")

        if peak_mislocate:
            print(f"\n偏移案例（前 10 条，供分析）：")
            for case in peak_mislocate[:10]:
                print(f"  {case}")

        print(f"\n图例：绿框=热点准确, 红框=热点偏移, 蓝点=SNR argmax")
        print(f"可视化保存至: {output_dir}")

        print(f"\n{'─'*62}")
        if acc_rate >= 85:
            print(f"[结论] ✓  PAG 机制鲁棒 ({acc_rate:.1f}% 准确)")
            print(f"         → 强杂波场景下物理锚定有效，可继续 P1 阶段")
        elif acc_rate >= 70:
            print(f"[结论] ⚠  PAG 机制存在部分偏移 ({acc_rate:.1f}%)")
            print(f"         → 建议引入距离惩罚项再启动 P1")
        else:
            print(f"[结论] ✗  PAG 机制偏移严重 ({acc_rate:.1f}%)")
            print(f"         → 需要修复方法后才能推进 P1")
        print(f"{'─'*62}")
    else:
        print("未找到有效 Box，请检查数据集路径和标签格式")

    # ======================================================
    # 对比度分层分析：验证"低对比度时 PAG 更有价值"假设
    # ======================================================
    if len(contrast_records) >= 4:
        print(f"\n{'='*62}")
        print(f"对比度分层分析（仅含 GT 区域非空的 box，共 {len(contrast_records)} 个）")
        print(f"{'='*62}")
        print(f"指标：GT权重集中度 = 落在GT区域内的权重 / 总权重（越高越好）")
        print(f"{'─'*62}")

        # 按 snr_peak 排序后分为 4 组（低→高对比度）
        records_sorted = sorted(contrast_records, key=lambda r: r['snr_peak'])
        n = len(records_sorted)
        q_size = n // 4
        quartile_labels = ['Q1(低对比度)', 'Q2', 'Q3', 'Q4(高对比度)']

        all_snr_peaks   = [r['snr_peak']     for r in records_sorted]
        all_bsnr_fracs  = [r['bsnr_gt_frac'] for r in records_sorted]
        all_pag_fracs   = [r['pag_gt_frac']  for r in records_sorted]
        all_pag_gains   = [r['pag_gain']      for r in records_sorted]

        print(f"{'分组':<14} {'样本数':>5}  {'SNR峰值':>8}  {'B-SNR集中度':>11}  {'PAG集中度':>9}  {'PAG增益':>8}")
        print(f"{'─'*62}")

        for qi in range(4):
            s = qi * q_size
            e = (qi + 1) * q_size if qi < 3 else n
            grp = records_sorted[s:e]
            snr_mean   = sum(r['snr_peak']     for r in grp) / len(grp)
            bsnr_mean  = sum(r['bsnr_gt_frac'] for r in grp) / len(grp)
            pag_mean   = sum(r['pag_gt_frac']  for r in grp) / len(grp)
            gain_mean  = sum(r['pag_gain']      for r in grp) / len(grp)
            print(f"  {quartile_labels[qi]:<12} {len(grp):>5}  {snr_mean:>8.2f}  "
                  f"{bsnr_mean:>11.3f}  {pag_mean:>9.3f}  {gain_mean:>+8.3f}")

        print(f"{'─'*62}")
        # 全局均值
        bsnr_all = sum(all_bsnr_fracs) / n
        pag_all  = sum(all_pag_fracs)  / n
        gain_all = sum(all_pag_gains)  / n
        print(f"  {'全局均值':<12} {n:>5}  {'':>8}  {bsnr_all:>11.3f}  {pag_all:>9.3f}  {gain_all:>+8.3f}")
        print(f"\n解读：")
        print(f"  PAG增益 > 0 → PAG 在该组比 B-SNR 更集中，有效")
        print(f"  PAG增益 < 0 → PAG 在该组不如 B-SNR，Gaussian 偏离目标")
        if all_pag_gains[0] > all_pag_gains[-1]:
            print(f"\n[结论] 假设成立 ✓  低对比度时 PAG 增益更大")
        elif abs(all_pag_gains[0] - all_pag_gains[-1]) < 0.02:
            print(f"\n[结论] 差异不显著 —  PAG 增益与对比度相关性弱")
        else:
            print(f"\n[结论] 假设不成立 ✗  高对比度时 PAG 反而增益更大（需要进一步分析）")

    return peak_accurate, total_boxes, peak_mislocate


def main():
    parser = argparse.ArgumentParser(description='P0 IRSTD-1k PAG 强杂波定位巡检')
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/b311/data2/25-zhangxizhe/code/'
                                'PoLaRIS-Infrared-LiDAR-Detection/dataset/IRSTD-1k',
                        help='IRSTD-1k 数据集根目录')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='巡检样本数（按背景方差排序，默认50）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='可视化输出目录（默认: dataset_dir/p0_inspection_vis）')
    parser.add_argument('--pad', type=int, default=2,
                        help='Box 膨胀像素（模拟标注松散度，默认2）')
    parser.add_argument('--expand_ratio', type=float, default=1.5,
                        help='Context Box 膨胀倍率（默认1.5）')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='SNR sigmoid 温度系数（默认3.0）')
    parser.add_argument('--gauss_sigma_ratio', type=float, default=1.5,
                        help='PAG 高斯 σ = ratio × max(box_h,box_w)/2（默认1.5）')
    args = parser.parse_args()

    run_p0_inspection(
        dataset_dir=args.dataset_dir,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        pad=args.pad,
        expand_ratio=args.expand_ratio,
        temperature=args.temperature,
        gauss_sigma_ratio=args.gauss_sigma_ratio,
    )


if __name__ == '__main__':
    main()
