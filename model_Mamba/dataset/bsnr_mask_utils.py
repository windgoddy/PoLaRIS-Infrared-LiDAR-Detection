"""
B-SNR (Box-Constrained SNR) Pseudo-Mask Generation
====================================================

基于物理 SNR 先验的伪标签净化工具。

核心思想:
  给定一个 Bounding Box，利用框外的局部背景统计量 (μ, σ) 来计算框内
  每个像素的物理 SNR，从而将模糊的矩形/高斯伪标签精炼为紧贴目标
  热辐射轮廓的精细掩码。

关键设计 — "统计域与作用域分离":
  - 统计域 (Context Box): 将原始 Box 向外膨胀 expand_ratio 倍，用于
    计算背景的 μ 和 σ。膨胀后包含足够的纯背景像素，保证统计量稳定。
  - 作用域 (Target Box): SNR 权重只在原始 Box 内部生效，框外强制为 0。

公式:
  W_physics(i) = sigmoid(τ · (I(i) - μ_ctx) / (σ_ctx + ε))
  Y_target = W_physics  (在 Box 内), 0 (在 Box 外)

Author: PoLaRIS Team
Date: 2026-03-10
"""

import numpy as np
import cv2
import os
from typing import List, Tuple, Optional


def compute_bsnr_weight(
    image_gray: np.ndarray,
    box: Tuple[int, int, int, int],
    expand_ratio: float = 1.5,
    temperature: float = 3.0,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    计算单个 Box 内的 B-SNR 物理权重图。

    Args:
        image_gray: (H, W) 灰度图，float32，值域 [0, 1]
        box: (x1, y1, x2, y2) 原始目标 Box 的像素坐标
        expand_ratio: Context Box 相对于原始 Box 的膨胀倍率
                      例如 1.5 表示每边各向外扩展 25% 的 box 尺寸
        temperature: sigmoid 温度系数 τ，控制前景/背景分离的锐度
                     τ 越大，分离越尖锐；τ 越小，过渡越平滑
        epsilon: 防止除零的小常数

    Returns:
        weight: (box_h, box_w) float32，值域 [0, 1]，物理 SNR 权重
    """
    H, W = image_gray.shape
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return np.zeros((max(1, box_h), max(1, box_w)), dtype=np.float32)

    # === Step 1: 计算膨胀后的 Context Box ===
    # 每边向外扩展 (expand_ratio - 1) / 2 * box_dim
    pad_x = int((expand_ratio - 1.0) / 2.0 * box_w)
    pad_y = int((expand_ratio - 1.0) / 2.0 * box_h)
    # 最少膨胀 3 像素，保证极小目标也有足够背景
    pad_x = max(pad_x, 3)
    pad_y = max(pad_y, 3)

    ctx_x1 = max(0, x1 - pad_x)
    ctx_y1 = max(0, y1 - pad_y)
    ctx_x2 = min(W, x2 + pad_x)
    ctx_y2 = min(H, y2 + pad_y)

    # === Step 2: 在 Context Box 区域计算背景统计量 ===
    # 使用整个 context 区域（包含目标 + 周围背景）计算 μ 和 σ
    # 由于目标像素占比很小（红外小目标），μ 和 σ 主要反映背景
    ctx_region = image_gray[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
    mu = ctx_region.mean()
    sigma = ctx_region.std()

    # === Step 3: 在原始 Box 内计算 SNR 权重 ===
    target_region = image_gray[y1:y2, x1:x2]  # (box_h, box_w)

    # 核心公式: W = sigmoid(τ · (I - μ) / (σ + ε))
    # 当像素亮度 >> μ（目标）：SNR >> 0 → sigmoid → ~1
    # 当像素亮度 ≈ μ（背景）：SNR ≈ 0 → sigmoid → ~0.5
    # 当像素亮度 << μ（暗背景）：SNR << 0 → sigmoid → ~0
    snr = (target_region - mu) / (sigma + epsilon)
    weight = 1.0 / (1.0 + np.exp(-temperature * snr))  # sigmoid

    # 归一化到 [0, 1]（确保最大值为 1）
    w_min = weight.min()
    w_max = weight.max()
    if w_max - w_min > 1e-8:
        weight = (weight - w_min) / (w_max - w_min)
    else:
        # 框内像素几乎均匀（极端情况），返回全 0.5
        weight = np.full_like(weight, 0.5)

    return weight.astype(np.float32)


def generate_bsnr_mask(
    labels: List[List[float]],
    img_size: Tuple[int, int],
    image: Optional[np.ndarray] = None,
    expand_ratio: float = 1.5,
    temperature: float = 3.0,
    fg_threshold: float = 0.5,
    soft_label: bool = True,
    max_value: float = 1.0,
) -> np.ndarray:
    """
    从 YOLO 标签 + 原始图像生成 B-SNR 物理净化伪掩码。

    Args:
        labels: YOLO 格式标签列表 [[class_id, cx, cy, w, h], ...]
                坐标均为归一化值 [0, 1]
        img_size: (H, W) 图像尺寸
        image: (H, W) 或 (H, W, 3) 原始图像（uint8 或 uint16）
               如果为 None，退化为简单的矩形填充
        expand_ratio: Context Box 膨胀倍率（默认 1.5）
        temperature: sigmoid 温度系数（默认 3.0）
        fg_threshold: B-SNR 权重低于此阈值的像素视为背景，置零
                      设为 0 则保留所有像素的连续权重
        soft_label: True 返回连续值掩码 [0, max_value]
                    False 返回二值掩码 {0, max_value}
        max_value: 前景最大值（默认 1.0，可设为 0.8 配合 soft label 训练）

    Returns:
        mask: (H, W) float32 伪掩码
    """
    H, W = img_size
    mask = np.zeros((H, W), dtype=np.float32)

    if image is None or len(labels) == 0:
        # 无图像或无标签，返回全零
        return mask

    # 将图像转为灰度 float32 [0, 1]
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype == np.uint16:
        gray = gray.astype(np.float32) / 65535.0
    elif gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255.0
    else:
        # 已经是 float
        gray = gray.astype(np.float32)

    for label in labels:
        if len(label) < 5:
            continue

        class_id, cx_norm, cy_norm, w_norm, h_norm = label[:5]

        # 转为像素坐标
        cx = cx_norm * W
        cy = cy_norm * H
        bw = w_norm * W
        bh = h_norm * H

        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(W, cx + bw / 2))
        y2 = int(min(H, cy + bh / 2))

        if x2 <= x1 or y2 <= y1:
            continue

        # 计算 B-SNR 物理权重
        weight = compute_bsnr_weight(
            gray, (x1, y1, x2, y2),
            expand_ratio=expand_ratio,
            temperature=temperature,
        )

        # 应用阈值
        if fg_threshold > 0:
            weight[weight < fg_threshold] = 0.0

        # 二值化或保持连续值
        if soft_label:
            refined = weight * max_value
        else:
            refined = (weight >= fg_threshold).astype(np.float32) * max_value

        # 写入全局掩码（多目标取 max）
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], refined)

    return mask


def generate_bsnr_mask_batch_offline(
    dataset_dir: str,
    img_ids: List[str],
    output_dir: str,
    image_folder: str = 'images-8bit',
    suffix: str = '.png',
    expand_ratio: float = 1.5,
    temperature: float = 3.0,
    fg_threshold: float = 0.5,
    soft_label: bool = True,
    max_value: float = 1.0,
    visualize: bool = False,
    vis_output_dir: Optional[str] = None,
):
    """
    离线批量生成 B-SNR 伪掩码并保存为 PNG。

    用于 Pilot 实验：预生成掩码后直接替换 masks/ 目录进行训练。

    Args:
        dataset_dir: 数据集根目录（如 dataset/Pohang-Canal-3k）
        img_ids: 图像 ID 列表（不含后缀）
        output_dir: 输出目录（如 dataset/Pohang-Canal-3k/masks_bsnr）
        image_folder: 图像文件夹名
        suffix: 图像后缀
        expand_ratio: Context Box 膨胀倍率
        temperature: sigmoid 温度系数
        fg_threshold: 前景阈值
        soft_label: 是否输出连续值
        max_value: 前景最大值
        visualize: 是否输出可视化对比图
        vis_output_dir: 可视化输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    if visualize and vis_output_dir:
        os.makedirs(vis_output_dir, exist_ok=True)

    labels_dir = os.path.join(dataset_dir, 'labels')
    images_dir = os.path.join(dataset_dir, image_folder)

    # 统计信息
    total = len(img_ids)
    success = 0
    empty = 0

    for i, img_id in enumerate(img_ids):
        # 加载图像
        img_path = os.path.join(images_dir, img_id + suffix)
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        H, W = image.shape[:2]

        # 加载 YOLO 标签
        label_path = os.path.join(labels_dir, img_id + '.txt')
        labels = _load_yolo_labels(label_path)

        # 生成 B-SNR 掩码
        mask = generate_bsnr_mask(
            labels, (H, W), image,
            expand_ratio=expand_ratio,
            temperature=temperature,
            fg_threshold=fg_threshold,
            soft_label=soft_label,
            max_value=max_value,
        )

        # 保存为 PNG (uint8, 0-255)
        mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(output_dir, img_id + '.png')
        cv2.imwrite(out_path, mask_uint8)

        if mask.max() > 0:
            success += 1
        else:
            empty += 1

        # 可视化对比（只对有目标的图像生成）
        if visualize and vis_output_dir and len(labels) > 0 and success <= 50:
            _save_visualization(
                image, mask, labels, (H, W),
                os.path.join(vis_output_dir, img_id + '_vis.png'),
                expand_ratio=expand_ratio,
            )

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{total}] success={success}, empty={empty}")

    print(f"Done: {success} masks generated, {empty} empty, saved to {output_dir}")


def _load_yolo_labels(label_path: str) -> List[List[float]]:
    """加载 YOLO 格式标签文件。"""
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    labels.append([class_id, cx, cy, w, h])
    except FileNotFoundError:
        pass
    return labels


def _save_visualization(
    image: np.ndarray,
    bsnr_mask: np.ndarray,
    labels: List[List[float]],
    img_size: Tuple[int, int],
    save_path: str,
    expand_ratio: float = 1.5,
):
    """
    保存对比可视化：原图(带box) | 矩形填充掩码 | 高斯掩码 | B-SNR掩码
    """
    H, W = img_size

    # 确保图像是 3 通道 uint8
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image.astype(np.uint8) if image.dtype != np.uint8 else image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
    if vis_img.dtype != np.uint8:
        vis_img = (vis_img * 255).clip(0, 255).astype(np.uint8)

    # 画 Box
    for label in labels:
        if len(label) < 5:
            continue
        _, cx_n, cy_n, w_n, h_n = label[:5]
        x1 = int(max(0, cx_n * W - w_n * W / 2))
        y1 = int(max(0, cy_n * H - h_n * H / 2))
        x2 = int(min(W, cx_n * W + w_n * W / 2))
        y2 = int(min(H, cy_n * H + h_n * H / 2))
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # 生成矩形填充掩码（对比用）
    box_mask = np.zeros((H, W), dtype=np.float32)
    for label in labels:
        if len(label) < 5:
            continue
        _, cx_n, cy_n, w_n, h_n = label[:5]
        x1 = int(max(0, cx_n * W - w_n * W / 2))
        y1 = int(max(0, cy_n * H - h_n * H / 2))
        x2 = int(min(W, cx_n * W + w_n * W / 2))
        y2 = int(min(H, cy_n * H + h_n * H / 2))
        box_mask[y1:y2, x1:x2] = 1.0

    # 生成高斯掩码（对比用）
    from model_Mamba.dataset.gaussian_utils import generate_gaussian_target
    gauss_mask = generate_gaussian_target(labels, (H, W), downscale=1, normalize=True)

    # 转为彩色热力图
    def to_heatmap(m):
        m_u8 = (m * 255).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(m_u8, cv2.COLORMAP_JET)

    box_heat = to_heatmap(box_mask)
    gauss_heat = to_heatmap(gauss_mask)
    bsnr_heat = to_heatmap(bsnr_mask)

    # 拼接: 原图 | 矩形 | 高斯 | B-SNR
    row = np.hstack([vis_img, box_heat, gauss_heat, bsnr_heat])
    cv2.imwrite(save_path, row)


# ==========================================
# 独立运行：批量生成 + 可视化
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='B-SNR Pseudo-Mask Generator')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset root (e.g., dataset/Pohang-Canal-3k)')
    parser.add_argument('--split_file', type=str, default=None,
                        help='Train/test split file (e.g., 50_50_cat2/train.txt). '
                             'If None, process all images in labels/')
    parser.add_argument('--image_folder', type=str, default='images-8bit')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--output_folder', type=str, default='masks_bsnr',
                        help='Output folder name within dataset_dir')
    parser.add_argument('--expand_ratio', type=float, default=1.5)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--fg_threshold', type=float, default=0.5)
    parser.add_argument('--soft_label', action='store_true', default=True)
    parser.add_argument('--max_value', type=float, default=1.0)
    parser.add_argument('--visualize', action='store_true',
                        help='Save comparison visualizations')
    parser.add_argument('--vis_samples', type=int, default=20,
                        help='Number of visualization samples')

    args = parser.parse_args()

    # 获取图像 ID 列表
    if args.split_file:
        # 支持绝对路径和相对于 dataset_dir 的路径
        if os.path.isabs(args.split_file) or os.path.exists(args.split_file):
            split_path = args.split_file
        else:
            split_path = os.path.join(args.dataset_dir, args.split_file)
        with open(split_path, 'r') as f:
            img_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(img_ids)} image IDs from {split_path}")
    else:
        # 从 labels/ 目录获取
        labels_dir = os.path.join(args.dataset_dir, 'labels')
        img_ids = [f[:-4] for f in sorted(os.listdir(labels_dir)) if f.endswith('.txt')]
        print(f"Found {len(img_ids)} label files in {labels_dir}")

    output_dir = os.path.join(args.dataset_dir, args.output_folder)
    vis_dir = os.path.join(args.dataset_dir, args.output_folder + '_vis') if args.visualize else None

    print(f"\nB-SNR Mask Generation")
    print(f"  Dataset:       {args.dataset_dir}")
    print(f"  Images:        {args.image_folder}")
    print(f"  Output:        {output_dir}")
    print(f"  expand_ratio:  {args.expand_ratio}")
    print(f"  temperature:   {args.temperature}")
    print(f"  fg_threshold:  {args.fg_threshold}")
    print(f"  soft_label:    {args.soft_label}")
    print(f"  max_value:     {args.max_value}")
    print()

    generate_bsnr_mask_batch_offline(
        dataset_dir=args.dataset_dir,
        img_ids=img_ids,
        output_dir=output_dir,
        image_folder=args.image_folder,
        suffix=args.suffix,
        expand_ratio=args.expand_ratio,
        temperature=args.temperature,
        fg_threshold=args.fg_threshold,
        soft_label=args.soft_label,
        max_value=args.max_value,
        visualize=args.visualize,
        vis_output_dir=vis_dir,
    )
