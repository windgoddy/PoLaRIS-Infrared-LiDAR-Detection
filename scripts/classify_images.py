#!/usr/bin/env python3
"""
自动分类打标签脚本
根据点云和标注框特征对红外图像进行分类
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# LiDAR 点云过滤配置
# ==========================================
LIDAR_FILTER_CONFIG = {
    'min_depth': 3.0,
    'max_depth': 150.0,
    'min_height': -160.0,
    'max_height': 50.0,
    'use_intensity_filter': True,
    'min_intensity': 5.0,
    'filter_zero_intensity': True,
}

# ==========================================
# 工具函数（复用自 generate_oracle_masks.py）
# ==========================================

def filter_lidar_points(points, config=LIDAR_FILTER_CONFIG):
    """过滤 LiDAR 点云以移除噪声"""
    if len(points) == 0:
        return points

    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    # 1. 深度过滤（欧式距离）
    depth = np.sqrt(x**2 + y**2 + z**2)
    depth_mask = (depth >= config['min_depth']) & (depth <= config['max_depth'])

    # 2. 高度过滤（Z 轴）
    height_mask = (z >= config['min_height']) & (z <= config['max_height'])

    # 3. 强度过滤
    if config['use_intensity_filter']:
        intensity_mask = intensity >= config['min_intensity']
        if config['filter_zero_intensity']:
            intensity_mask = intensity_mask & (intensity > 0)
    else:
        intensity_mask = np.ones(len(points), dtype=bool)

    # 组合所有过滤条件
    final_mask = depth_mask & height_mask & intensity_mask

    return points[final_mask]


def get_transform_matrix(extrinsics):
    """从外参字典构建 4x4 变换矩阵"""
    q = extrinsics['quaternion']
    t = extrinsics['translation']

    # Quaternion to Rotation Matrix
    x, y, z, w = q
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def project_lidar_to_image(lidar_points, K_cam, T_cam_to_lidar, img_shape):
    """将 LiDAR 点投影到图像平面"""
    if len(lidar_points) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

    # 1. 转齐次坐标
    xyz_points = lidar_points[:, :3]
    ones = np.ones((xyz_points.shape[0], 1))
    pts_homo = np.hstack([xyz_points, ones])

    # 2. 变换到相机坐标系
    pts_cam = (T_cam_to_lidar @ pts_homo.T).T

    # 3. 深度过滤 (Z > 0.1)
    valid_z = pts_cam[:, 2] >= 0.1
    pts_cam = pts_cam[valid_z]
    if len(pts_cam) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

    # 4. 投影
    pts_img_homo = (K_cam @ pts_cam[:, :3].T).T
    u = pts_img_homo[:, 0] / pts_img_homo[:, 2]
    v = pts_img_homo[:, 1] / pts_img_homo[:, 2]

    # 5. 图像边界过滤
    H, W = img_shape[:2]
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u = u[valid_uv]
    v = v[valid_uv]
    depths = pts_cam[valid_z][valid_uv, 2]

    pixels = np.column_stack([u.astype(int), v.astype(int)])
    return pixels, depths


def parse_label_file(label_path):
    """
    解析YOLO格式的标注文件
    返回：bbox列表 [(x, y, w, h), ...]（像素坐标）
    """
    if not os.path.exists(label_path):
        return []

    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO格式: class_id x_center y_center width height (归一化)
                # 这里需要图像尺寸来转换，暂时存储归一化值
                boxes.append([float(x) for x in parts[1:5]])

    return boxes


def yolo_to_pixel_boxes(yolo_boxes, img_w, img_h):
    """
    将YOLO格式（归一化）转换为像素坐标的bbox

    Args:
        yolo_boxes: [(x_center_norm, y_center_norm, w_norm, h_norm), ...]
        img_w, img_h: 图像尺寸

    Returns:
        boxes: [(x, y, w, h), ...] 像素坐标，左上角+宽高
    """
    boxes = []
    for (x_c_norm, y_c_norm, w_norm, h_norm) in yolo_boxes:
        x_center = x_c_norm * img_w
        y_center = y_c_norm * img_h
        width = w_norm * img_w
        height = h_norm * img_h

        # 转为左上角坐标
        x1 = x_center - width / 2
        y1 = y_center - height / 2

        boxes.append((x1, y1, width, height))

    return boxes


def compute_box_valid_area(box, img_w, img_h):
    """
    计算标注框与图像边界的交集面积（有效区域）

    Args:
        box: (x, y, w, h) 像素坐标
        img_w, img_h: 图像尺寸

    Returns:
        valid_area: 有效区域面积
        valid_box: (x1, y1, x2, y2) 有效区域的边界
    """
    x, y, w, h = box

    # 原始框的边界
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    # 与图像边界求交集
    valid_x1 = max(0, x1)
    valid_y1 = max(0, y1)
    valid_x2 = min(img_w, x2)
    valid_y2 = min(img_h, y2)

    # 计算有效面积
    valid_w = max(0, valid_x2 - valid_x1)
    valid_h = max(0, valid_y2 - valid_y1)
    valid_area = valid_w * valid_h

    return valid_area, (valid_x1, valid_y1, valid_x2, valid_y2)


def count_points_in_box(pixels, box, img_w, img_h):
    """
    计算落在标注框内的点云数量（考虑图像边界）

    Args:
        pixels: (N, 2) 投影点坐标 [(u, v), ...]
        box: (x, y, w, h) 像素坐标
        img_w, img_h: 图像尺寸

    Returns:
        count: 框内点数
        fill_ratio: 点云填充率（框内点数 / 有效区域面积）
                    注意：fill_ratio 可能 > 1.0（多个点投影到同一像素时）
                    这里的定义是：平均每个像素有多少个点
                    如果需要"像素覆盖率"，应该用 unique 像素数 / 总像素数
    """
    if len(pixels) == 0:
        return 0, 0.0

    # 计算有效区域
    valid_area, (x1, y1, x2, y2) = compute_box_valid_area(box, img_w, img_h)

    if valid_area == 0:
        return 0, 0.0

    # 筛选框内的点
    in_box = (pixels[:, 0] >= x1) & (pixels[:, 0] < x2) & \
             (pixels[:, 1] >= y1) & (pixels[:, 1] < y2)

    count = np.sum(in_box)
    fill_ratio = count / valid_area if valid_area > 0 else 0.0

    return count, fill_ratio


# ==========================================
# 分类逻辑
# ==========================================

def classify_image(pixels, boxes, img_w, img_h, verbose=False):
    """
    根据规则对图像进行分类

    Args:
        pixels: (N, 2) 投影后的有效点云坐标
        boxes: [(x, y, w, h), ...] 像素坐标的bbox列表
        img_w, img_h: 图像尺寸
        verbose: 是否打印详细信息

    Returns:
        label: 0/1/2/3
        stats: 统计信息字典
    """
    total_points = len(pixels)
    num_boxes = len(boxes)

    # 计算每个框的统计信息
    box_stats = []
    total_points_in_boxes = 0

    for idx, box in enumerate(boxes):
        valid_area, _ = compute_box_valid_area(box, img_w, img_h)
        points_in_box, fill_ratio = count_points_in_box(pixels, box, img_w, img_h)

        box_stats.append({
            'area': valid_area,
            'points': points_in_box,
            'fill_ratio': fill_ratio
        })

        total_points_in_boxes += points_in_box

    if verbose:
        print(f"  全图有效点云数: {total_points}")
        print(f"  标注框总数: {num_boxes}")
        print(f"  所有框内点云总数: {total_points_in_boxes}")
        for idx, stats in enumerate(box_stats):
            print(f"  框{idx+1}: 面积={stats['area']:.0f}, 点数={stats['points']}, 填充率={stats['fill_ratio']*100:.2f}%")

    # 优先级 3: 困难岸边样本
    if total_points >= 10000 and num_boxes <= 5:
        # 检查是否存在至少一个框的填充率 > 60%
        has_high_fill = any(s['fill_ratio'] > 0.6 for s in box_stats)
        if has_high_fill:
            if verbose:
                print("  → 判定为标签3（困难岸边样本）")
            return 3, {
                'total_points': total_points,
                'num_boxes': num_boxes,
                'box_stats': box_stats
            }

    # 优先级 2: 困难极小样本
    if num_boxes > 0:
        # 检查是否存在至少一个框内部无点云
        has_empty_box = any(s['points'] == 0 for s in box_stats)

        # 检查70%以上的框面积 < 32*32
        small_boxes = [s for s in box_stats if s['area'] < 32 * 32]
        small_ratio = len(small_boxes) / num_boxes if num_boxes > 0 else 0

        if has_empty_box and small_ratio > 0.7:
            if verbose:
                print(f"  → 判定为标签2（困难极小样本）：无点云框存在，小框占比={small_ratio*100:.1f}%")
            return 2, {
                'total_points': total_points,
                'num_boxes': num_boxes,
                'box_stats': box_stats
            }

    # 优先级 1: 简单样本
    if num_boxes <= 5 and total_points > 0:
        points_ratio = total_points_in_boxes / total_points
        if points_ratio > 0.7:
            if verbose:
                print(f"  → 判定为标签1（简单样本）：框内点云占比={points_ratio*100:.1f}%")
            return 1, {
                'total_points': total_points,
                'num_boxes': num_boxes,
                'box_stats': box_stats
            }

    # 优先级 0: 其他
    if verbose:
        print("  → 判定为标签0（其他）")
    return 0, {
        'total_points': total_points,
        'num_boxes': num_boxes,
        'box_stats': box_stats
    }


# ==========================================
# 主处理流程
# ==========================================

def process_single_image(image_name, base_dir, verbose=False):
    """
    处理单张图像

    Args:
        image_name: 例如 "00_43.png"
        base_dir: 数据集根目录
        verbose: 是否打印详细信息

    Returns:
        label: 分类标签
        stats: 统计信息
        error: 错误信息（如果有）
    """
    try:
        # 1. 解析文件名
        name_parts = image_name.replace('.png', '').split('_')
        if len(name_parts) != 2:
            return None, None, f"文件名格式错误: {image_name}"

        dataset_id = name_parts[0]
        frame_id = name_parts[1]

        if verbose:
            print(f"\n处理图像: {image_name} (数据集{dataset_id}, 帧{frame_id})")

        # 2. 构建路径
        dataset_path = os.path.join(base_dir, dataset_id)

        ir_path = os.path.join(dataset_path, 'infrared', f'{frame_id}.png')
        lidar_path = os.path.join(dataset_path, 'lidar_front', f'{frame_id}.bin')

        # 标注文件路径：需要从 base_dir 上溯到 PoLaRIS 目录
        # base_dir: /home/.../Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset
        # label: /home/.../Pohang Canal Dataset And PoLaRIS/PoLaRIS/{ID}/all/tir/{frame}.txt
        polaris_root = os.path.dirname(base_dir)  # 得到 .../Pohang Canal Dataset And PoLaRIS
        label_path = os.path.join(polaris_root, 'PoLaRIS', 'PoLaRIS', dataset_id, 'all', 'tir', f'{frame_id}.txt')

        calib_dir = os.path.join(dataset_path, 'calibration')

        if verbose:
            print(f"  路径信息:")
            print(f"    红外图像: {ir_path}")
            print(f"    LiDAR: {lidar_path}")
            print(f"    标注: {label_path}")
            print(f"    标定: {calib_dir}")

        # 3. 检查文件存在性
        if not os.path.exists(ir_path):
            return None, None, f"红外图像不存在: {ir_path}"
        if not os.path.exists(lidar_path):
            return None, None, f"LiDAR文件不存在: {lidar_path}"
        if not os.path.exists(label_path):
            return None, None, f"标注文件不存在: {label_path}"

        # 4. 加载图像（获取尺寸）
        img = cv2.imread(ir_path)
        if img is None:
            return None, None, f"无法读取图像: {ir_path}"
        H, W = img.shape[:2]

        # 5. 加载标定
        try:
            with open(os.path.join(calib_dir, 'intrinsics.json')) as f:
                ir_intrinsics = json.load(f)['infrared']
            with open(os.path.join(calib_dir, 'extrinsics.json')) as f:
                ext = json.load(f)
                ir_ext = ext['infrared']
                li_ext = ext['lidar_front']

            fx, fy = ir_intrinsics['focal_length'], ir_intrinsics['focal_length']
            cx, cy = ir_intrinsics['cc_x'], ir_intrinsics['cc_y']
            K_cam = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            T_cam_to_lidar = np.linalg.inv(get_transform_matrix(ir_ext)) @ get_transform_matrix(li_ext)
        except Exception as e:
            return None, None, f"加载标定失败: {e}"

        # 6. 加载并过滤LiDAR点云
        try:
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        except:
            points = np.zeros((0, 4), dtype=np.float32)

        # 过滤点云
        points_filtered = filter_lidar_points(points, LIDAR_FILTER_CONFIG)

        # 投影到图像
        pixels, _ = project_lidar_to_image(points_filtered, K_cam, T_cam_to_lidar, img.shape)

        if verbose:
            print(f"  原始点云数: {len(points)}")
            print(f"  过滤后点云数: {len(points_filtered)}")
            print(f"  投影到图像内的有效点数: {len(pixels)}")

        # 7. 加载标注框
        yolo_boxes = parse_label_file(label_path)
        boxes = yolo_to_pixel_boxes(yolo_boxes, W, H)

        if verbose:
            print(f"  标注框数量: {len(boxes)}")

        # 8. 分类
        label, stats = classify_image(pixels, boxes, W, H, verbose=verbose)

        return label, stats, None

    except Exception as e:
        import traceback
        return None, None, f"处理出错: {traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description='自动分类打标签脚本')
    parser.add_argument('--test_image', type=str, default=None,
                        help='单图测试模式，例如: --test_image 00_43.png')
    parser.add_argument('--dataset_base', type=str,
                        default='/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset',
                        help='数据集根目录')
    parser.add_argument('--select_dir', type=str,
                        default='dataset/select',
                        help='待处理图像列表目录')
    parser.add_argument('--output_dir', type=str,
                        default='dataset/select-view',
                        help='输出目录')

    args = parser.parse_args()

    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    select_dir = os.path.join(project_root, args.select_dir)
    output_dir = os.path.join(project_root, args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # 单图测试模式
    if args.test_image:
        print(f"=== 单图测试模式: {args.test_image} ===")
        label, stats, error = process_single_image(args.test_image, args.dataset_base, verbose=True)

        if error:
            print(f"错误: {error}")
            return

        print(f"\n最终标签: {label}")
        return

    # 批量处理模式
    print(f"=== 批量处理模式 ===")
    print(f"待处理图像目录: {select_dir}")
    print(f"输出目录: {output_dir}")

    # 获取图像列表
    if not os.path.exists(select_dir):
        print(f"错误: 目录不存在: {select_dir}")
        return

    image_files = sorted([f for f in os.listdir(select_dir) if f.endswith('.png')])
    print(f"找到 {len(image_files)} 张图像")

    # 处理所有图像
    results = []
    errors = []

    # 用于统计分析
    all_point_counts = []
    all_box_areas = []
    all_fill_ratios = []

    for image_name in tqdm(image_files, desc="处理中"):
        label, stats, error = process_single_image(image_name, args.dataset_base, verbose=False)

        if error:
            errors.append((image_name, error))
            continue

        results.append((image_name, label))

        # 收集统计数据
        if stats:
            all_point_counts.append(stats['total_points'])
            for box_stat in stats['box_stats']:
                all_box_areas.append(box_stat['area'])
                all_fill_ratios.append(box_stat['fill_ratio'])

    # 保存结果
    output_file = os.path.join(output_dir, 'selection_summary_new.txt')
    with open(output_file, 'w') as f:
        for image_name, label in results:
            f.write(f"{image_name} | {label}\n")

    print(f"\n结果已保存到: {output_file}")
    print(f"成功处理: {len(results)} 张")
    print(f"失败: {len(errors)} 张")

    # 打印错误日志
    if errors:
        error_log = os.path.join(output_dir, 'errors.log')
        with open(error_log, 'w') as f:
            for img, err in errors:
                f.write(f"{img}: {err}\n")
        print(f"错误日志已保存到: {error_log}")

    # 统计各标签数量
    label_counts = defaultdict(int)
    for _, label in results:
        label_counts[label] += 1

    print("\n分类统计:")
    for label in [0, 1, 2, 3]:
        count = label_counts[label]
        pct = count / len(results) * 100 if len(results) > 0 else 0
        print(f"  标签{label}: {count} 张 ({pct:.1f}%)")

    # 生成可视化分析图
    if len(all_point_counts) > 0:
        print("\n生成分析图...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 子图1: 全图有效点云数量分布
        axes[0].hist(all_point_counts, bins=50, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Point Count')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Total Point Counts')
        axes[0].grid(True, alpha=0.3)

        # 子图2: 标注框面积分布
        axes[1].hist(all_box_areas, bins=50, color='lightcoral', edgecolor='black')
        axes[1].set_xlabel('Box Area (pixels)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Bounding Box Areas')
        axes[1].grid(True, alpha=0.3)

        # 子图3: 标注框内点云占比分布
        axes[2].hist([r * 100 for r in all_fill_ratios], bins=50, color='lightgreen', edgecolor='black')
        axes[2].set_xlabel('Fill Ratio (%)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Distribution of Point Fill Ratios in Boxes')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(output_dir, 'analysis_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"分析图已保存到: {plot_path}")


if __name__ == "__main__":
    main()
