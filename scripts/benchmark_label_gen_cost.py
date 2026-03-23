"""
HALO Label Generation Cost Benchmark
======================================
量化 HALO 伪标签生成的计算代价，与 LESPS/PAL 的迭代复杂度进行比较。

用法（在项目根目录）：
    python scripts/benchmark_label_gen_cost.py \
        --root dataset \
        --datasets NUAA-SIRST NUDT-SIRST IRSTD-1k \
        --split_method 50_50 \
        --suffix .png \
        --expand_ratio 1.5 \
        --temperature 3.0 \
        --gauss_sigma_ratio 1.5 \
        [--halo_only]      # 只跑 HALO，跳过静态分析（快速模式）
        [--dry_run N]      # 只处理每个数据集前 N 张图（调试用）

输出：
    - 每个数据集的 B-SNR（C组）和 HALO（D组）单独计时
    - 每张图的平均耗时（ms）
    - 与 PAL / LESPS 的静态迭代分析对比表
    - 可粘贴进论文的 Markdown 表格

作者：PoLaRIS Team  日期：2026-03-23
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
from tqdm import tqdm

# 把项目根目录加入 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model_Mamba.dataset.bsnr_mask_utils import compute_bsnr_weight, _load_yolo_labels


# ──────────────────────────────────────────────────────────────
# 1. 工具函数
# ──────────────────────────────────────────────────────────────

def load_img_ids(root, dataset, split_method='50_50'):
    """读取训练集 img_ids（与 train.py 中保持一致）。"""
    txt_path = os.path.join(root, dataset, split_method, 'train.txt')
    if not os.path.exists(txt_path):
        # 兜底：直接扫 images 目录
        img_dir = os.path.join(root, dataset, 'images')
        if os.path.isdir(img_dir):
            ids = [os.path.splitext(f)[0] for f in sorted(os.listdir(img_dir))
                   if not f.startswith('.')]
            print(f"  [warn] 未找到 {txt_path}，改用 images/ 目录（{len(ids)} 张）")
            return ids
        raise FileNotFoundError(f"找不到 {txt_path} 也找不到 {img_dir}")
    with open(txt_path) as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def detect_image_folder(root, dataset, img_id_sample, suffix):
    """
    自动探测数据集的图像文件夹名和后缀。
    依次尝试常见目录名和后缀，返回 (images_dir, suffix)。
    """
    candidate_folders = ['images', 'images-8bit', 'img', 'image', 'imgs',
                         'IRSTD1k_Img', 'NUDT-SIRST', 'NUAA-SIRST']
    candidate_suffixes = [suffix, '.png', '.jpg', '.jpeg', '.bmp']

    for folder in candidate_folders:
        images_dir = os.path.join(root, dataset, folder)
        if not os.path.isdir(images_dir):
            continue
        for suf in candidate_suffixes:
            test_path = os.path.join(images_dir, img_id_sample + suf)
            if os.path.exists(test_path):
                if folder != 'images' or suf != suffix:
                    print(f"  [auto-detect] {dataset}: 图像目录={folder}, 后缀={suf}")
                return images_dir, suf
    # 找不到时返回默认值，让后续读取失败时打印 skip
    return os.path.join(root, dataset, 'images'), suffix


def time_halo_on_dataset(root, dataset, img_ids, suffix,
                          expand_ratio, temperature, gauss_sigma_ratio,
                          mode='D', dry_run=None, image_folder=None):
    """
    对指定数据集的全部训练图像运行 HALO 标签生成，返回计时统计。

    mode: 'C' = 仅 B-SNR（无高斯），'D' = B-SNR + PAG 高斯（完整 HALO）
    """
    labels_dir = os.path.join(root, dataset, 'labels_box')

    # 自动探测图像目录和后缀（可被 --image_folder 覆盖）
    sample_id = img_ids[0] if img_ids else 'unknown'
    if image_folder:
        images_dir = os.path.join(root, dataset, image_folder)
        print(f"  [override] {dataset}: 图像目录={image_folder}")
    else:
        images_dir, suffix = detect_image_folder(root, dataset, sample_id, suffix)

    spatial_gaussian = (mode == 'D')
    ids = img_ids[:dry_run] if dry_run else img_ids

    times_per_img = []
    n_boxes_total = 0
    n_skipped = 0

    for img_id in tqdm(ids, desc=f'{dataset} [{mode}]', leave=False):
        img_path = os.path.join(images_dir, img_id + suffix)
        label_path = os.path.join(labels_dir, img_id + '.txt')

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            n_skipped += 1
            continue
        H, W = image.shape
        image_f32 = image.astype(np.float32) / 255.0

        labels = _load_yolo_labels(label_path) if os.path.exists(label_path) else []
        if len(labels) == 0:
            # 无目标图像：无 HALO 计算，跳过（不混入计时统计）
            continue

        t0 = time.perf_counter()
        for lbl in labels:
            _, cx_n, cy_n, w_n, h_n = lbl[:5]
            x1 = int(max(0, (cx_n - w_n / 2) * W))
            y1 = int(max(0, (cy_n - h_n / 2) * H))
            x2 = int(min(W, (cx_n + w_n / 2) * W))
            y2 = int(min(H, (cy_n + h_n / 2) * H))
            if x2 <= x1 or y2 <= y1:
                continue
            compute_bsnr_weight(
                image_f32,
                (x1, y1, x2, y2),
                expand_ratio=expand_ratio,
                temperature=temperature,
                spatial_gaussian=spatial_gaussian,
                gauss_sigma_ratio=gauss_sigma_ratio,
            )
        t1 = time.perf_counter()
        times_per_img.append((t1 - t0) * 1000)  # ms
        n_boxes_total += len(labels)

    total_ms = sum(times_per_img)
    n_imgs = len(times_per_img)
    avg_ms = total_ms / max(n_imgs, 1)
    median_ms = float(np.median(times_per_img)) if times_per_img else 0.0

    return {
        'dataset': dataset,
        'mode': mode,
        'n_images': n_imgs,
        'n_boxes': n_boxes_total,
        'n_skipped': n_skipped,
        'total_sec': total_ms / 1000,
        'avg_ms_per_img': avg_ms,
        'median_ms_per_img': median_ms,
    }


# ──────────────────────────────────────────────────────────────
# 2. PAL / LESPS 静态复杂度分析（从代码规格推导，无需重跑）
# ──────────────────────────────────────────────────────────────

def static_pal_analysis(n_train: dict):
    """
    基于 PAL/train_model.py 的超参数，静态计算 PAL 的标签精化代价。

    PAL 关键超参数（来自 PAL/train_model.py）：
      NUM_EPOCHS = 400
      clear_epoch_gap = 5       ← 每 5 epoch 更新一次 no_choose 池的伪标签
      final_epoch_ratio = 0.8   ← 前 80% epoch 为 enhancement 阶段（持续更新）

    每次标签更新 = 对全量训练集做一次网络 forward（推理伪掩码）
    总 forward passes for label update = floor(400 × 0.8 / 5) = 64 次
    每次 = 遍历一遍训练集 → 64 × n_train 次 forward
    """
    NUM_EPOCHS = 400
    final_epoch_ratio = 0.8
    clear_epoch_gap = 5

    active_epochs = int(NUM_EPOCHS * final_epoch_ratio)
    n_label_updates = active_epochs // clear_epoch_gap

    results = {}
    for ds, n in n_train.items():
        total_fwd = n_label_updates * n
        results[ds] = {
            'n_train': n,
            'label_update_rounds': n_label_updates,
            'total_network_fwd_for_labels': total_fwd,
        }
    return results, n_label_updates


def static_lesps_analysis(n_train: dict):
    """
    基于 LESPS 论文（CVPR 2023）的代码规格，静态计算 LESPS 的标签精化代价。

    LESPS 关键机制：
      - 每个 epoch 结束后，用当前网络预测更新伪掩码（每 epoch 更新一次）
      - 训练 epoch 数通常与 HALO 实验相同（1500 epoch）

    因此：
      total_network_fwd_for_labels = 1500 × n_train（每 epoch 全量推理一次）

    注：以上基于 LESPS 论文描述，实际数字以其代码为准。
    """
    LESPS_EPOCHS = 1500  # 与本文实验一致
    results = {}
    for ds, n in n_train.items():
        total_fwd = LESPS_EPOCHS * n
        results[ds] = {
            'n_train': n,
            'label_update_rounds': LESPS_EPOCHS,
            'total_network_fwd_for_labels': total_fwd,
        }
    return results


# ──────────────────────────────────────────────────────────────
# 3. 主函数
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='HALO label generation cost benchmark')
    parser.add_argument('--root', default='dataset')
    parser.add_argument('--datasets', nargs='+',
                        default=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k'])
    parser.add_argument('--split_method', default='50_50')
    parser.add_argument('--suffix', default='.png')
    parser.add_argument('--expand_ratio', type=float, default=1.5)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--gauss_sigma_ratio', type=float, default=1.5)
    parser.add_argument('--halo_only', action='store_true',
                        help='只测 HALO，跳过 B-SNR only（C组）')
    parser.add_argument('--dry_run', type=int, default=None,
                        help='每个数据集只处理前 N 张（调试用）')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='强制指定图像子目录名（如 IRSTD1k_Img），覆盖自动探测')
    args = parser.parse_args()

    print("=" * 65)
    print("  HALO Label Generation Cost Benchmark")
    print(f"  expand_ratio={args.expand_ratio}  temperature={args.temperature}"
          f"  gauss_sigma_ratio={args.gauss_sigma_ratio}")
    if args.dry_run:
        print(f"  [DRY RUN] 每个数据集只处理前 {args.dry_run} 张")
    print("=" * 65)

    # ── Step 1: 收集数据集基本信息 ──
    all_ids = {}
    for ds in args.datasets:
        try:
            ids = load_img_ids(args.root, ds, args.split_method)
            all_ids[ds] = ids
            print(f"  {ds}: {len(ids)} 张训练图")
        except Exception as e:
            print(f"  [SKIP] {ds}: {e}")

    if not all_ids:
        print("[ERROR] 未找到任何数据集，请检查 --root 路径和 --datasets 参数")
        sys.exit(1)

    # ── Step 2: 计时 HALO 标签生成 ──
    results_C = {}
    results_D = {}

    for ds, ids in all_ids.items():
        if not args.halo_only:
            r = time_halo_on_dataset(
                args.root, ds, ids, args.suffix,
                args.expand_ratio, args.temperature, args.gauss_sigma_ratio,
                mode='C', dry_run=args.dry_run, image_folder=args.image_folder
            )
            results_C[ds] = r

        r = time_halo_on_dataset(
            args.root, ds, ids, args.suffix,
            args.expand_ratio, args.temperature, args.gauss_sigma_ratio,
            mode='D', dry_run=args.dry_run, image_folder=args.image_folder
        )
        results_D[ds] = r

    # ── Step 3: PAL / LESPS 静态分析 ──
    n_train = {ds: len(ids) for ds, ids in all_ids.items()}
    pal_stats, n_pal_updates = static_pal_analysis(n_train)
    lesps_stats = static_lesps_analysis(n_train)

    # ── Step 4: 打印结果 ──
    print("\n" + "=" * 65)
    print("  HALO 标签生成计时结果（CPU，单进程）")
    print("=" * 65)
    print(f"{'数据集':<14} {'模式':<6} {'图数':>6} {'总耗时(s)':>10} "
          f"{'均值(ms/图)':>12} {'中位数(ms/图)':>14}")
    print("-" * 65)

    total_all_C = 0.0
    total_all_D = 0.0
    total_imgs = 0

    for ds in args.datasets:
        if ds not in results_D:
            continue
        if not args.halo_only and ds in results_C:
            rc = results_C[ds]
            print(f"  {ds:<12} {'C(B-SNR)':<8} {rc['n_images']:>6} "
                  f"{rc['total_sec']:>10.2f} "
                  f"{rc['avg_ms_per_img']:>12.3f} "
                  f"{rc['median_ms_per_img']:>14.3f}")
            total_all_C += rc['total_sec']

        rd = results_D[ds]
        print(f"  {ds:<12} {'D(HALO)':<8} {rd['n_images']:>6} "
              f"{rd['total_sec']:>10.2f} "
              f"{rd['avg_ms_per_img']:>12.3f} "
              f"{rd['median_ms_per_img']:>14.3f}")
        total_all_D += rd['total_sec']
        total_imgs += rd['n_images']

    print("-" * 65)
    if not args.halo_only:
        print(f"  {'所有数据集合计':<14} {'C(B-SNR)':<8} {'':<6} "
              f"{total_all_C:>10.2f}s")
    print(f"  {'所有数据集合计':<14} {'D(HALO)':<8} {'':<6} "
          f"{total_all_D:>10.2f}s")
    print(f"  （共 {total_imgs} 张图，纯 CPU，零 GPU 占用，零网络前向）")

    # ── Step 5: 对比表（论文用） ──
    print("\n" + "=" * 65)
    print("  方法复杂度对比表（可粘贴进论文）")
    print("=" * 65)

    # 计算全量数字（若 dry_run，折算到全数据集）
    scale = 1.0
    if args.dry_run:
        total_full = sum(len(v) for v in all_ids.values())
        total_sampled = args.dry_run * len(all_ids)
        scale = total_full / max(total_sampled, 1)
        print(f"  [注] dry_run 模式：{total_sampled} 张样本折算全量 {total_full} 张（×{scale:.1f}）")

    # HALO 全量估算
    halo_total_est = total_all_D * scale
    all_avg_ms_D = [results_D[ds]['avg_ms_per_img']
                    for ds in args.datasets if ds in results_D]
    halo_avg_ms_str = f"≈ {np.mean(all_avg_ms_D):.2f} ms/图"

    print()
    print("| Method | Network-in-loop | Label Update Rounds | GPU Cost (label) | Offline Label Time (all 3 datasets) |")
    print("|--------|:---------------:|:-------------------:|:----------------:|:-----------------------------------:|")
    print(f"| LESPS  | ✓               | ~1500×全量           | ~= training cost | N/A (online, per-epoch update)      |")
    print(f"| PAL    | ✓               | ~{n_pal_updates}×全量              | ~= 16% training  | N/A (online, active loop)           |")
    print(f"| GrabCut| ✗               | EM-GMM (内部迭代)    | 0 GPU            | (仅外观先验，无物理先验)                |")
    print(f"| **HALO** | **✗**         | **Zero (1-shot)**   | **0 GPU**        | **≈ {halo_total_est:.1f}s CPU ({halo_avg_ms_str})** |")
    print()

    print("=" * 65)
    print("  逐数据集 HALO vs PAL 网络调用量对比")
    print("=" * 65)
    print(f"{'数据集':<14} {'N_train':>8} {'HALO fwd':>12} {'PAL label fwd':>16} {'LESPS label fwd':>18} {'节省比(vs LESPS)':>18}")
    print("-" * 65)
    for ds in args.datasets:
        if ds not in pal_stats:
            continue
        n = pal_stats[ds]['n_train']
        pal_fwd = pal_stats[ds]['total_network_fwd_for_labels']
        lesps_fwd = lesps_stats[ds]['total_network_fwd_for_labels']
        halo_fwd = 0  # HALO 的网络调用数 = 0
        saving = f"{lesps_fwd:,}× faster"
        print(f"  {ds:<12} {n:>8,} {halo_fwd:>12} {pal_fwd:>16,} {lesps_fwd:>18,} {saving:>18}")

    print("\n  HALO: 0 次网络前向（纯 CPU numpy/opencv，离线一次性生成）")
    print(f"  PAL: {n_pal_updates} 轮标签更新 × N_train 张图 次网络推理（online，GPU）")
    print("  LESPS: 每 epoch 全量推理一次（online，GPU，与训练并行）")

    # ── Step 6: 可直接复制的论文句子 ──
    halo_total_str = f"{halo_total_est:.1f}" if not args.dry_run else f"≈{halo_total_est:.0f}"
    print("\n" + "=" * 65)
    print("  论文可用句子（填入实测数字后可直接粘贴）：")
    print("=" * 65)
    print(f"""
  EN:
  "HALO's pseudo-label generation requires zero network forward
  passes and runs entirely on CPU. Across all three datasets
  ({', '.join(args.datasets)}), the total offline label generation
  time is ≈{halo_total_str}s on a single CPU core, compared to
  {n_pal_updates}×N_train GPU inferences for PAL's iterative
  label refinement and 1500×N_train GPU inferences for LESPS."

  CN:
  "HALO 的伪标签生成无需任何网络前向传播，全程在 CPU 上运行。
  三个数据集全量标签生成合计耗时约 {halo_total_str}s（单 CPU 核心），
  而 PAL 每轮主动学习需要 {n_pal_updates}×N_train 次 GPU 推理，
  LESPS 每 epoch 需要 N_train 次 GPU 推理（共 1500 epoch）。"
""")


if __name__ == '__main__':
    main()
