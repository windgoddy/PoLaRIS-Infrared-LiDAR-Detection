"""
NUDT-SIRST 模拟弱监督 Pilot 实验 — 伪掩码生成
=====================================================

实验分组：
  A 组（上界）  ：直接用原始 masks/ 像素 GT 监督（不需要本脚本）
  B 组（矩形框）：用 labels_box/ YOLO box → 矩形填充掩码 → masks_box_fill/
                  这是弱监督的朴素基线（什么都不做，直接把框内全标为前景）
  C 组（B-SNR） ：用 labels_box/ YOLO box + 图像纹理 → B-SNR 净化掩码 → masks_bsnr/
                  本文方法：物理 SNR 门控去除框内背景像素

论文核心论证：
  C >> B → B-SNR 从相同的 box 标注中提取出了显著更优的监督信号
  C ≈ A  → 用 box + B-SNR 几乎可以恢复 GT 像素级监督性能

用法：
    # 同时生成 B 组和 C 组（全量）
    python scripts/nudt_generate_pseudo_masks.py \
        --dataset_dir dataset/NUDT-SIRST \
        --generate box_fill bsnr \
        --visualize

    # 只生成某一组
    python scripts/nudt_generate_pseudo_masks.py \
        --dataset_dir dataset/NUDT-SIRST \
        --generate bsnr

依赖：
    model_Mamba/dataset/bsnr_mask_utils.py  (generate_bsnr_mask, _load_yolo_labels)
"""

import os
import sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# 把项目根目录加入 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model_Mamba.dataset.bsnr_mask_utils import generate_bsnr_mask, _load_yolo_labels


# ======================================================
# B 组：矩形框填充掩码（朴素弱监督基线）
# ======================================================

def generate_box_fill_masks(
    dataset_dir,
    img_ids,
    label_folder='labels_box',
    image_folder='images',
    output_folder='masks_box_fill',
    suffix='.png',
):
    """
    从 YOLO box 生成矩形填充掩码（B 组，朴素弱监督基线）。

    把每个 bounding box 内部全部填充为 1（二值），是弱监督场景下
    "什么都不做"的零假设基线。
    """
    labels_dir = os.path.join(dataset_dir, label_folder)
    images_dir = os.path.join(dataset_dir, image_folder)
    output_dir = os.path.join(dataset_dir, output_folder)
    os.makedirs(output_dir, exist_ok=True)

    has_target = 0
    no_label = 0

    print(f"\n[B组] 生成矩形框填充掩码: {output_dir}")
    for img_id in tqdm(img_ids, desc='Box Fill'):
        # 读取图像尺寸
        img_path = os.path.join(images_dir, img_id + suffix)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape

        # 读取 YOLO box 标签
        label_path = os.path.join(labels_dir, img_id + '.txt')
        labels = _load_yolo_labels(label_path)

        # 矩形填充：box 内全 1，框外全 0
        mask = np.zeros((H, W), dtype=np.float32)
        for label in labels:
            _, cx_n, cy_n, w_n, h_n = label[:5]
            x1 = int(max(0, (cx_n - w_n / 2) * W))
            y1 = int(max(0, (cy_n - h_n / 2) * H))
            x2 = int(min(W, (cx_n + w_n / 2) * W))
            y2 = int(min(H, (cy_n + h_n / 2) * H))
            mask[y1:y2, x1:x2] = 1.0

        # 保存为 uint8 PNG（0 或 255）
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, img_id + suffix), mask_uint8)

        if len(labels) > 0:
            has_target += 1
        else:
            no_label += 1

    print(f"  完成: 有目标={has_target}, 无目标={no_label}")
    return output_dir


# ======================================================
# C 组：B-SNR 净化掩码（本文方法）
# ======================================================

def generate_bsnr_masks(
    dataset_dir,
    img_ids,
    label_folder='labels_box',
    image_folder='images',
    output_folder='masks_bsnr',
    suffix='.png',
    expand_ratio=1.5,
    temperature=3.0,
    fg_threshold=0.5,
    soft_label=True,
):
    """
    从 YOLO box + 原始图像生成 B-SNR 净化掩码（C 组，本文方法）。

    核心：利用框外背景统计量 (μ, σ) 计算框内每像素的物理 SNR，
    sigmoid 门控后阈值化，保留高亮辐射像素，去除框内背景噪声。
    """
    labels_dir = os.path.join(dataset_dir, label_folder)
    images_dir = os.path.join(dataset_dir, image_folder)
    output_dir = os.path.join(dataset_dir, output_folder)
    os.makedirs(output_dir, exist_ok=True)

    has_target = 0
    no_label = 0
    empty_bsnr = 0

    print(f"\n[C组] 生成 B-SNR 净化掩码: {output_dir}")
    print(f"  expand_ratio={expand_ratio}, temperature={temperature}, fg_threshold={fg_threshold}")

    for img_id in tqdm(img_ids, desc='B-SNR'):
        # 读取图像
        img_path = os.path.join(images_dir, img_id + suffix)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        H, W = image.shape

        # 读取 YOLO box 标签
        label_path = os.path.join(labels_dir, img_id + '.txt')
        labels = _load_yolo_labels(label_path)

        # 生成 B-SNR 掩码
        mask = generate_bsnr_mask(
            labels, (H, W), image,
            expand_ratio=expand_ratio,
            temperature=temperature,
            fg_threshold=fg_threshold,
            soft_label=soft_label,
            max_value=1.0,
        )

        # 保存为 uint8 PNG
        mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, img_id + suffix), mask_uint8)

        if len(labels) > 0:
            has_target += 1
            if mask.max() == 0:
                empty_bsnr += 1
        else:
            no_label += 1

    print(f"  完成: 有目标={has_target}, 无目标={no_label}")
    if has_target > 0:
        print(f"  B-SNR 生成空掩码: {empty_bsnr}/{has_target} "
              f"({100*empty_bsnr/has_target:.1f}%) — 这些目标可能对比度太低")
    return output_dir


# ======================================================
# 可视化对比（四列：原图+Box | A:GT | B:矩形框 | C:B-SNR）
# ======================================================

def visualize_comparison(
    dataset_dir,
    img_ids,
    label_folder='labels_box',
    image_folder='images',
    output_dir=None,
    n_samples=30,
    suffix='.png',
):
    """
    生成四列对比图：原图(+box) | A:GT掩码 | B:矩形框填充 | C:B-SNR净化
    """
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, 'pilot_vis')
    os.makedirs(output_dir, exist_ok=True)

    labels_dir = os.path.join(dataset_dir, label_folder)
    images_dir = os.path.join(dataset_dir, image_folder)
    gt_dir       = os.path.join(dataset_dir, 'masks')
    box_fill_dir = os.path.join(dataset_dir, 'masks_box_fill')
    bsnr_dir     = os.path.join(dataset_dir, 'masks_bsnr')

    panel_configs = [
        (None,        'Image+Box'),
        (gt_dir,      'A: GT Mask'),
        (box_fill_dir,'B: Box Fill'),
        (bsnr_dir,    'C: B-SNR'),
    ]

    count = 0
    for img_id in img_ids:
        label_path = os.path.join(labels_dir, img_id + '.txt')
        labels = _load_yolo_labels(label_path)
        if len(labels) == 0:
            continue

        img_path = os.path.join(images_dir, img_id + suffix)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        H, W = image.shape
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 画 box（绿色）
        for label in labels:
            _, cx_n, cy_n, w_n, h_n = label[:5]
            x1 = int(max(0, (cx_n - w_n / 2) * W))
            y1 = int(max(0, (cy_n - h_n / 2) * H))
            x2 = int(min(W, (cx_n + w_n / 2) * W))
            y2 = int(min(H, (cy_n + h_n / 2) * H))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        def load_heatmap(mask_dir):
            path = os.path.join(mask_dir, img_id + suffix)
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                m = np.zeros((H, W), dtype=np.uint8)
            return cv2.applyColorMap(m, cv2.COLORMAP_JET)

        panels = []
        titles = []
        for mask_dir, title in panel_configs:
            if mask_dir is None:
                panels.append(vis_img)
            elif os.path.exists(mask_dir):
                panels.append(load_heatmap(mask_dir))
            else:
                continue
            titles.append(title)

        row = np.hstack(panels)

        for j, txt in enumerate(titles):
            cv2.putText(row, txt, (j * W + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(output_dir, img_id + '_compare.png'), row)
        count += 1
        if count >= n_samples:
            break

    print(f"\n  可视化: {count} 张对比图 → {output_dir}")


# ======================================================
# 主函数
# ======================================================

def main():
    parser = argparse.ArgumentParser(description='生成 Pilot 实验伪掩码（A/B/C 三组）')
    parser.add_argument('--dataset_dir', type=str, default='dataset/NUDT-SIRST')
    parser.add_argument('--label_folder', type=str, default='labels_box',
                        help='YOLO box 标注目录（由 nudt_mask_to_yolo_box.py 生成）')
    parser.add_argument('--image_folder', type=str, default='images')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split', type=str, default=None,
                        help='Split 文件路径（相对于 dataset_dir 或绝对路径）；'
                             '不传则处理 label_folder 下所有图像')
    parser.add_argument('--generate', nargs='+', default=['box_fill', 'bsnr'],
                        choices=['box_fill', 'bsnr'],
                        help='要生成的掩码类型: box_fill=B组, bsnr=C组')
    parser.add_argument('--visualize', action='store_true',
                        help='生成四列对比可视化（Image+Box | A:GT | B:BoxFill | C:B-SNR）')
    parser.add_argument('--vis_samples', type=int, default=30,
                        help='可视化样本数量')
    # B-SNR 超参数（C 组）
    parser.add_argument('--expand_ratio', type=float, default=1.5,
                        help='Context Box 膨胀倍率')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='sigmoid 温度系数，越大分离越尖锐')
    parser.add_argument('--fg_threshold', type=float, default=0.5,
                        help='B-SNR 前景阈值，低于此值置零')

    args = parser.parse_args()
    dataset_dir = args.dataset_dir

    # 读取图像 ID 列表
    if args.split:
        split_path = args.split if os.path.isabs(args.split) else \
                     (args.split if os.path.exists(args.split) else
                      os.path.join(dataset_dir, args.split))
        with open(split_path) as f:
            img_ids = [l.strip() for l in f if l.strip()]
        print(f"Split: {split_path} → {len(img_ids)} images")
    else:
        labels_dir = os.path.join(dataset_dir, args.label_folder)
        img_ids = [f[:-4] for f in sorted(os.listdir(labels_dir)) if f.endswith('.txt')]
        print(f"All labels: {len(img_ids)} images")

    # 验证 label_folder 存在
    labels_dir = os.path.join(dataset_dir, args.label_folder)
    if not os.path.exists(labels_dir):
        print(f"[ERROR] label_folder 不存在: {labels_dir}")
        print("请先运行: python scripts/nudt_mask_to_yolo_box.py")
        sys.exit(1)

    print(f"Dataset: {dataset_dir}")
    print(f"Labels:  {labels_dir}")
    print(f"Images:  {os.path.join(dataset_dir, args.image_folder)}")
    print(f"Tasks:   {args.generate}")

    # B 组：矩形框填充
    if 'box_fill' in args.generate:
        generate_box_fill_masks(
            dataset_dir=dataset_dir,
            img_ids=img_ids,
            label_folder=args.label_folder,
            image_folder=args.image_folder,
            output_folder='masks_box_fill',
            suffix=args.suffix,
        )

    # C 组：B-SNR 净化
    if 'bsnr' in args.generate:
        generate_bsnr_masks(
            dataset_dir=dataset_dir,
            img_ids=img_ids,
            label_folder=args.label_folder,
            image_folder=args.image_folder,
            output_folder='masks_bsnr',
            suffix=args.suffix,
            expand_ratio=args.expand_ratio,
            temperature=args.temperature,
            fg_threshold=args.fg_threshold,
        )

    # 可视化对比
    if args.visualize:
        print("\n生成可视化对比图...")
        visualize_comparison(
            dataset_dir=dataset_dir,
            img_ids=img_ids,
            label_folder=args.label_folder,
            image_folder=args.image_folder,
            n_samples=args.vis_samples,
            suffix=args.suffix,
        )

    print("\n全部完成！")
    print(f"  A 组训练用: --mask_folder masks         （GT 上界）")
    print(f"  B 组训练用: --mask_folder masks_box_fill （矩形框朴素基线）")
    print(f"  C 组训练用: --mask_folder masks_bsnr     （B-SNR 净化，本文方法）")


if __name__ == '__main__':
    main()
