#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理工具集 - 整合所有数据预处理和分析功能

功能：
1. filter-lidar: 过滤有效的激光雷达图像（投影点数>2）
2. analyze-bbox: 分析边缘截断标注框
3. visualize: 可视化YOLO标注框
4. filter-labels: 过滤小目标标签（基于有效区域）

用法：
    python scripts/data_tools.py filter-lidar --min_points 2
    python scripts/data_tools.py analyze-bbox
    python scripts/data_tools.py visualize --num_samples 100
    python scripts/data_tools.py filter-labels --preview
"""

import os
import cv2
import numpy as np
import argparse
import shutil
import random
from tqdm import tqdm
from datetime import datetime

# ==================== 全局配置 ====================
DATASET_ROOT = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all'
LABEL_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/00/all/00_inf_labels'
LIDAR_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/00/lidar'
CALIB_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/00/calib'

IMAGES_DIR = os.path.join(DATASET_ROOT, 'images')
MASKS_DIR = os.path.join(DATASET_ROOT, 'masks')
SPLIT_DIR = os.path.join(DATASET_ROOT, 'split_data')

IMG_W, IMG_H = 640, 512

# LiDAR过滤配置
FILTER_CONFIG = {
    'min_depth': 3.0,
    'max_depth': 150.0,
    'min_height': -160.0,
    'max_height': 50.0,
    'use_intensity_filter': True,
    'min_intensity': 5.0,
    'filter_zero_intensity': True,
}
# ==================================================


# ==================== 功能1: 过滤有效激光雷达图像 ====================
def filter_lidar_points(points, config=FILTER_CONFIG):
    """过滤LiDAR点云"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    intensity = points[:, 3]

    distances = np.sqrt(x**2 + y**2 + z**2)
    mask = np.ones(points.shape[0], dtype=bool)

    mask &= (distances >= config['min_depth']) & (distances <= config['max_depth'])
    mask &= (y >= config['min_height']) & (y <= config['max_height'])

    if config.get('use_intensity_filter', False):
        mask &= intensity > config['min_intensity']

    if config.get('filter_zero_intensity', False):
        mask &= intensity > 0.1

    return points[mask]


def project_lidar_to_image(lidar_points, K_cam, T_cam_to_lidar, img_shape):
    """投影LiDAR点到图像并统计有效点数"""
    if lidar_points.shape[0] == 0:
        return 0

    img_height, img_width = img_shape

    # 转换为齐次坐标
    homo = np.hstack([lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))])
    P_cam = (T_cam_to_lidar @ homo.T).T

    # 过滤相机后方的点
    z_coords = P_cam[:, 2]
    valid_mask = z_coords >= 0.1

    if not np.any(valid_mask):
        return 0

    P_cam_valid = P_cam[valid_mask]
    xyz_cam = P_cam_valid[:, :3]
    Z = xyz_cam[:, 2]

    # 投影到图像平面
    P_img = (K_cam @ xyz_cam.T).T
    u = P_img[:, 0] / Z
    v = P_img[:, 1] / Z

    # 统计在图像范围内的点
    valid = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    return np.sum(valid)


def filter_valid_lidar_images(min_points=2):
    """过滤有效的激光雷达图像"""
    print("=" * 80)
    print("🔍 过滤有效激光雷达图像")
    print("=" * 80)
    print(f"激光雷达目录: {LIDAR_DIR}")
    print(f"标定目录: {CALIB_DIR}")
    print(f"最小投影点数: {min_points}")
    print("=" * 80)

    # 读取标定文件
    K_cam_path = os.path.join(CALIB_DIR, 'calib_cam_to_cam.txt')
    T_path = os.path.join(CALIB_DIR, 'calib_lidar_to_cam.txt')

    K_cam = np.loadtxt(K_cam_path)
    T_cam_to_lidar = np.loadtxt(T_path)

    # 获取所有点云文件
    lidar_files = sorted([f for f in os.listdir(LIDAR_DIR) if f.endswith('.bin')])
    frame_ids = [f.replace('.bin', '') for f in lidar_files]

    print(f"\n找到 {len(frame_ids)} 个点云文件")

    valid_frames = []
    stats = {'total': 0, 'valid': 0, 'invalid': 0}

    for frame_id in tqdm(frame_ids, desc="处理进度"):
        stats['total'] += 1

        lidar_path = os.path.join(LIDAR_DIR, f"{frame_id}.bin")
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

        # 过滤点云
        points_filtered = filter_lidar_points(points, FILTER_CONFIG)

        # 投影并统计
        num_projected = project_lidar_to_image(
            points_filtered, K_cam, T_cam_to_lidar, (IMG_H, IMG_W)
        )

        if num_projected > min_points:
            valid_frames.append(frame_id)
            stats['valid'] += 1
        else:
            stats['invalid'] += 1

    # 保存结果
    output_path = os.path.join(DATASET_ROOT, 'valid_lidar_images.txt')
    with open(output_path, 'w') as f:
        for frame_id in valid_frames:
            f.write(f"{frame_id}\n")

    # 打印统计
    print("\n" + "=" * 80)
    print("📊 统计结果")
    print("=" * 80)
    print(f"总图像数: {stats['total']}")
    print(f"✅ 有效图像: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"❌ 无效图像: {stats['invalid']} ({stats['invalid']/stats['total']*100:.1f}%)")
    print(f"\n输出文件: {output_path}")
    print("=" * 80)


# ==================== 功能2: 分析边缘截断标注框 ====================
def analyze_clipped_bbox():
    """分析边缘截断目标"""
    print("=" * 80)
    print("🔍 边缘截断标注框分析")
    print("=" * 80)
    print(f"标签目录: {LABEL_DIR}")
    print(f"图像目录: {IMAGES_DIR}")
    print("\n过滤策略：")
    print("  内部目标：仅过滤极小噪点（面积 < 5）")
    print("  边缘目标（严格）：")
    print("    - 有效面积 < 25 像素")
    print("    - 有效短边 < 5 像素")
    print("    - 有效长宽比 > 4.0")
    print("=" * 80)

    output_dir = os.path.join(DATASET_ROOT, 'debug_clipped_analysis')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    trash_dir = os.path.join(output_dir, 'boundary_trash')
    kept_dir = os.path.join(output_dir, 'boundary_kept')
    inside_dir = os.path.join(output_dir, 'inside_safe')
    os.makedirs(trash_dir, exist_ok=True)
    os.makedirs(kept_dir, exist_ok=True)
    os.makedirs(inside_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]

    stats = {
        'total': 0,
        'inside_safe': 0,
        'inside_noise': 0,
        'boundary_kept': 0,
        'boundary_trash': 0,
    }

    samples_saved = {'trash': 0, 'kept': 0, 'inside': 0}
    MAX_SAMPLES = 50

    for txt_file in tqdm(txt_files, desc="分析进度"):
        img_name = txt_file.replace('.txt', '.png')
        img_path = os.path.join(IMAGES_DIR, img_name)
        img = None

        txt_path = os.path.join(LABEL_DIR, txt_file)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            stats['total'] += 1
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            x_c = float(parts[1]) * IMG_W
            y_c = float(parts[2]) * IMG_H
            w = float(parts[3]) * IMG_W
            h = float(parts[4]) * IMG_H

            x1 = x_c - w/2
            y1 = y_c - h/2
            x2 = x_c + w/2
            y2 = y_c + h/2

            is_boundary = (x1 < 1) or (y1 < 1) or (x2 > IMG_W - 1) or (y2 > IMG_H - 1)

            if not is_boundary:
                if w * h < 5:
                    stats['inside_noise'] += 1
                else:
                    stats['inside_safe'] += 1

                    if samples_saved['inside'] < MAX_SAMPLES:
                        if img is None and os.path.exists(img_path):
                            img = cv2.imread(img_path)
                        if img is not None:
                            save_sample(img, int(x1), int(y1), int(x2), int(y2),
                                      inside_dir, f"INSIDE_{samples_saved['inside']:03d}",
                                      f"Area={int(w*h)}")
                            samples_saved['inside'] += 1
                continue

            # 边缘截断目标
            x1_clip = max(0, x1)
            y1_clip = max(0, y1)
            x2_clip = min(IMG_W, x2)
            y2_clip = min(IMG_H, y2)

            eff_w = x2_clip - x1_clip
            eff_h = y2_clip - y1_clip
            eff_area = eff_w * eff_h

            if eff_area <= 0:
                stats['boundary_trash'] += 1
                continue

            eff_min_side = min(eff_w, eff_h)
            eff_max_side = max(eff_w, eff_h)
            eff_ratio = eff_max_side / (eff_min_side + 1e-6)

            is_trash = False
            reasons = []

            if eff_area < 25:
                is_trash = True
                reasons.append(f"EffArea={int(eff_area)}")
            if eff_min_side < 5:
                is_trash = True
                reasons.append(f"EffSide={eff_min_side:.1f}")
            if eff_ratio > 4.0:
                is_trash = True
                reasons.append(f"EffRatio={eff_ratio:.1f}")

            if is_trash:
                stats['boundary_trash'] += 1
                if samples_saved['trash'] < MAX_SAMPLES:
                    if img is None and os.path.exists(img_path):
                        img = cv2.imread(img_path)
                    if img is not None:
                        reason_str = '_'.join(reasons)
                        save_sample(img, int(x1_clip), int(y1_clip), int(x2_clip), int(y2_clip),
                                  trash_dir, f"TRASH_{samples_saved['trash']:03d}",
                                  reason_str, is_clipped=True)
                        samples_saved['trash'] += 1
            else:
                stats['boundary_kept'] += 1
                if samples_saved['kept'] < MAX_SAMPLES:
                    if img is None and os.path.exists(img_path):
                        img = cv2.imread(img_path)
                    if img is not None:
                        info = f"Eff={int(eff_area)}_Ratio={eff_ratio:.1f}"
                        save_sample(img, int(x1_clip), int(y1_clip), int(x2_clip), int(y2_clip),
                                  kept_dir, f"KEPT_{samples_saved['kept']:03d}",
                                  info, is_clipped=True)
                        samples_saved['kept'] += 1

    # 统计报告
    print("\n" + "=" * 80)
    print("📊 分析报告")
    print("=" * 80)
    print(f"总目标数: {stats['total']}")
    print(f"\n内部目标:")
    print(f"  ✅ 安全（保留）: {stats['inside_safe']} ({stats['inside_safe']/stats['total']*100:.1f}%)")
    print(f"  ❌ 噪点（删除）: {stats['inside_noise']} ({stats['inside_noise']/stats['total']*100:.1f}%)")
    print(f"\n边缘目标:")
    print(f"  ✅ 有效（保留）: {stats['boundary_kept']} ({stats['boundary_kept']/stats['total']*100:.1f}%)")
    print(f"  ❌ 垃圾（删除）: {stats['boundary_trash']} ({stats['boundary_trash']/stats['total']*100:.1f}%)")
    print(f"\n样本已保存到: {output_dir}")
    print("=" * 80)


def save_sample(img, x1, y1, x2, y2, save_dir, prefix, info, is_clipped=False):
    """保存样本图像"""
    pad = 30
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    crop_x1 = max(0, cx - pad)
    crop_y1 = max(0, cy - pad)
    crop_x2 = min(IMG_W, cx + pad)
    crop_y2 = min(IMG_H, cy + pad)

    crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    if crop.size == 0:
        return None

    draw_x1 = x1 - crop_x1
    draw_y1 = y1 - crop_y1
    draw_x2 = x2 - crop_x1
    draw_y2 = y2 - crop_y1

    color = (0, 255, 0) if is_clipped else (0, 255, 255)
    cv2.rectangle(crop, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)
    cv2.putText(crop, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
               0.4, (255, 255, 255), 1, cv2.LINE_AA)

    fname = f"{prefix}_{info}.png"
    cv2.imwrite(os.path.join(save_dir, fname), crop)

    return crop


# ==================== 功能3: 可视化标注 ====================
def visualize_annotations(num_samples=100):
    """生成可视化图像"""
    print("=" * 80)
    print("🎨 YOLO标注可视化")
    print("=" * 80)
    print(f"图像目录: {IMAGES_DIR}")
    print(f"标签目录: {LABEL_DIR}")
    print(f"样本数量: {num_samples}")
    print("=" * 80)

    output_dir = os.path.join(DATASET_ROOT, 'vis_annotations')
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')]

    if len(image_files) == 0:
        print("❌ 未找到图像文件")
        return

    print(f"\n找到 {len(image_files)} 张图像")

    if len(image_files) > num_samples:
        sampled_files = random.sample(image_files, num_samples)
    else:
        sampled_files = image_files

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    success_count = 0
    total_boxes = 0

    for img_file in tqdm(sampled_files, desc="生成进度"):
        img_path = os.path.join(IMAGES_DIR, img_file)
        label_path = os.path.join(LABEL_DIR, img_file.replace('.png', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            continue

        vis_img = img.copy()
        num_boxes = 0

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_c = float(parts[1]) * IMG_W
                y_c = float(parts[2]) * IMG_H
                w = float(parts[3]) * IMG_W
                h = float(parts[4]) * IMG_H

                x1 = int(x_c - w/2)
                y1 = int(y_c - h/2)
                x2 = int(x_c + w/2)
                y2 = int(y_c + h/2)

                x1 = max(0, min(x1, IMG_W - 1))
                y1 = max(0, min(y1, IMG_H - 1))
                x2 = max(0, min(x2, IMG_W - 1))
                y2 = max(0, min(y2, IMG_H - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                color = colors[class_id % len(colors)]
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                num_boxes += 1

        cv2.imwrite(os.path.join(output_dir, img_file), vis_img)
        success_count += 1
        total_boxes += num_boxes

    print("\n" + "=" * 80)
    print("📊 可视化完成")
    print("=" * 80)
    print(f"✅ 成功生成: {success_count} 张图像")
    print(f"总标注框数: {total_boxes}")
    print(f"平均每张: {total_boxes/success_count:.1f} 个标注框")
    print(f"\n输出目录: {output_dir}")
    print("=" * 80)


# ==================== 功能4: 过滤标签 ====================
def filter_labels(area_threshold=25, ratio_threshold=4.0, min_side=5, preview=False, backup=True):
    """过滤小目标标签（基于有效区域）"""
    print("=" * 80)
    print("🔍 过滤YOLO标签（基于有效区域）")
    print("=" * 80)
    print(f"标签目录: {LABEL_DIR}")
    print(f"\n过滤规则:")
    print(f"  内部目标：仅过滤极小噪点（面积 < 5 像素）")
    print(f"  边缘截断目标（严格）：")
    print(f"    - 有效面积 < {area_threshold} 像素")
    print(f"    - 有效最小边 < {min_side} 像素")
    print(f"    - 有效长宽比 > {ratio_threshold}")
    print(f"\n模式: {'预览模式（不修改文件）' if preview else '执行模式（会修改文件）'}")
    print("=" * 80)

    if backup and not preview:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{LABEL_DIR}_backup_{timestamp}"
        print(f"\n>>> 备份原始标签到: {backup_dir}")
        shutil.copytree(LABEL_DIR, backup_dir)
        print(f"✅ 备份完成")

    txt_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]

    stats = {
        'total_files': len(txt_files),
        'total_objects': 0,
        'filtered_objects': 0,
        'inside_safe': 0,
        'inside_noise': 0,
        'boundary_kept': 0,
        'boundary_trash': 0,
        'files_modified': 0,
        'files_deleted': 0,
        'files_unchanged': 0,
    }

    INSIDE_TH_AREA = 5
    files_to_delete = []

    for txt_file in tqdm(txt_files, desc="处理进度"):
        txt_path = os.path.join(LABEL_DIR, txt_file)

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            continue

        kept_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            stats['total_objects'] += 1

            x_c_norm = float(parts[1])
            y_c_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])

            x_c = x_c_norm * IMG_W
            y_c = y_c_norm * IMG_H
            w_pix = w_norm * IMG_W
            h_pix = h_norm * IMG_H

            x1 = x_c - w_pix/2
            y1 = y_c - h_pix/2
            x2 = x_c + w_pix/2
            y2 = y_c + h_pix/2

            is_boundary = (x1 < 1) or (y1 < 1) or (x2 > IMG_W - 1) or (y2 > IMG_H - 1)

            if not is_boundary:
                area = w_pix * h_pix
                if area < INSIDE_TH_AREA:
                    stats['filtered_objects'] += 1
                    stats['inside_noise'] += 1
                else:
                    kept_lines.append(line)
                    stats['inside_safe'] += 1
            else:
                x1_clip = max(0, x1)
                y1_clip = max(0, y1)
                x2_clip = min(IMG_W, x2)
                y2_clip = min(IMG_H, y2)

                eff_w = x2_clip - x1_clip
                eff_h = y2_clip - y1_clip
                eff_area = eff_w * eff_h

                if eff_area <= 0:
                    stats['filtered_objects'] += 1
                    stats['boundary_trash'] += 1
                    continue

                eff_min_side = min(eff_w, eff_h)
                eff_max_side = max(eff_w, eff_h)
                eff_ratio = eff_max_side / (eff_min_side + 1e-6)

                is_trash = False
                if eff_area < area_threshold:
                    is_trash = True
                elif eff_min_side < min_side:
                    is_trash = True
                elif eff_ratio > ratio_threshold:
                    is_trash = True

                if is_trash:
                    stats['filtered_objects'] += 1
                    stats['boundary_trash'] += 1
                else:
                    kept_lines.append(line)
                    stats['boundary_kept'] += 1

        if len(kept_lines) == 0:
            files_to_delete.append(txt_path)
            stats['files_deleted'] += 1
        elif len(kept_lines) < len(lines):
            if not preview:
                with open(txt_path, 'w') as f:
                    f.writelines(kept_lines)
            stats['files_modified'] += 1
        else:
            stats['files_unchanged'] += 1

    if not preview:
        for txt_path in files_to_delete:
            os.remove(txt_path)

    # 打印统计
    print("\n" + "=" * 80)
    print("📊 过滤统计")
    print("=" * 80)
    print(f"总文件数: {stats['total_files']}")
    print(f"总目标数: {stats['total_objects']}")
    print(f"\n内部目标:")
    print(f"  ✅ 安全（保留）: {stats['inside_safe']} ({stats['inside_safe']/stats['total_objects']*100:.1f}%)")
    print(f"  ❌ 噪点（删除）: {stats['inside_noise']} ({stats['inside_noise']/stats['total_objects']*100:.1f}%)")
    print(f"\n边缘目标:")
    print(f"  ✅ 有效（保留）: {stats['boundary_kept']} ({stats['boundary_kept']/stats['total_objects']*100:.1f}%)")
    print(f"  ❌ 垃圾（删除）: {stats['boundary_trash']} ({stats['boundary_trash']/stats['total_objects']*100:.1f}%)")
    print(f"\n文件处理:")
    print(f"  - 未修改: {stats['files_unchanged']}")
    print(f"  - 已修改: {stats['files_modified']}")
    print(f"  - 已删除: {stats['files_deleted']}")
    print("=" * 80)


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(
        description='数据处理工具集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 过滤有效激光雷达图像
  python scripts/data_tools.py filter-lidar --min_points 2

  # 分析边缘截断标注框
  python scripts/data_tools.py analyze-bbox

  # 可视化标注（100张样本）
  python scripts/data_tools.py visualize --num_samples 100

  # 预览标签过滤结果
  python scripts/data_tools.py filter-labels --preview

  # 执行标签过滤（带备份）
  python scripts/data_tools.py filter-labels --backup
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='选择功能')

    # filter-lidar命令
    parser_lidar = subparsers.add_parser('filter-lidar', help='过滤有效激光雷达图像')
    parser_lidar.add_argument('--min_points', type=int, default=2,
                             help='最小投影点数（默认2）')

    # analyze-bbox命令
    parser_bbox = subparsers.add_parser('analyze-bbox', help='分析边缘截断标注框')

    # visualize命令
    parser_vis = subparsers.add_parser('visualize', help='可视化YOLO标注')
    parser_vis.add_argument('--num_samples', type=int, default=100,
                           help='生成样本数量（默认100）')

    # filter-labels命令
    parser_filter = subparsers.add_parser('filter-labels', help='过滤小目标标签')
    parser_filter.add_argument('--area_threshold', type=int, default=25,
                              help='边缘目标有效面积阈值（默认25）')
    parser_filter.add_argument('--ratio_threshold', type=float, default=4.0,
                              help='边缘目标有效长宽比阈值（默认4.0）')
    parser_filter.add_argument('--min_side', type=int, default=5,
                              help='边缘目标有效最小边阈值（默认5）')
    parser_filter.add_argument('--preview', action='store_true',
                              help='预览模式（不修改文件）')
    parser_filter.add_argument('--backup', action='store_true', default=True,
                              help='执行前备份（默认开启）')
    parser_filter.add_argument('--no-backup', dest='backup', action='store_false',
                              help='不备份原始文件')

    args = parser.parse_args()

    if args.command == 'filter-lidar':
        filter_valid_lidar_images(min_points=args.min_points)
    elif args.command == 'analyze-bbox':
        analyze_clipped_bbox()
    elif args.command == 'visualize':
        visualize_annotations(num_samples=args.num_samples)
    elif args.command == 'filter-labels':
        filter_labels(
            area_threshold=args.area_threshold,
            ratio_threshold=args.ratio_threshold,
            min_side=args.min_side,
            preview=args.preview,
            backup=args.backup
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
