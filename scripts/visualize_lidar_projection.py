#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云投影到红外图像可视化工具

功能：
- 读取 Pohang-Canal-all 数据集
- 应用 LiDAR 点云过滤（深度、高度、强度）
- 将过滤后的 LiDAR 点云投影到红外图像上
- 使用深度信息为点着色（viridis colormap）
- 保存可视化结果

过滤规则：
- 深度范围：3.0 - 150.0 米（移除自车和远处噪声）
- 高度范围：-160.0 - 50.0 米（保留地面点，过滤天空伪影）
- 强度过滤：最小强度 5.0（移除弱反射噪声）

数据结构：
dataset/Pohang-Canal-all/
├── images/        # 8-bit 归一化红外图像
├── lidar_roi/     # 过滤并投影后的点云
├── calibration/   # 标定参数
└── split_data/    # train.txt, test.txt
"""

import os
import json
import cv2
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
import argparse

# ==================== 配置 ====================
DATASET_ROOT = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all'
OUTPUT_DIR = os.path.join(DATASET_ROOT, 'vis_lidar_projection')

# LiDAR 点云过滤配置
FILTER_CONFIG = {
    'min_depth': 3.0,        # 最小深度 (m) - 过滤自车
    'max_depth': 150.0,      # 最大深度 (m) - 移除远处噪声
    'min_height': -160.0,    # 最小高度 (m) - 保留地面点
    'max_height': 50.0,      # 最大高度 (m) - 过滤天空伪影
    'use_intensity_filter': True,
    'min_intensity': 5.0,
    'filter_zero_intensity': True,
}
# ==============================================

def get_transform_matrix(extrinsics):
    """构建 4x4 变换矩阵"""
    quat = extrinsics['quaternion']
    trans = extrinsics['translation']

    x, y, z, w = quat
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
    T[:3, 3] = trans
    return T

def filter_lidar_points(points, config=FILTER_CONFIG):
    """
    过滤 LiDAR 点云以移除噪声和自车

    Args:
        points: (N, 4) numpy array [x, y, z, intensity]
        config: 过滤配置字典

    Returns:
        filtered_points: 过滤后的点云
    """
    original_count = points.shape[0]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    intensity = points[:, 3]

    # 计算距离
    distances = np.sqrt(x**2 + y**2 + z**2)

    # 初始化掩码
    mask = np.ones(original_count, dtype=bool)

    # 1. 深度过滤
    mask &= (distances >= config['min_depth']) & (distances <= config['max_depth'])

    # 2. 高度过滤
    mask &= (y >= config['min_height']) & (y <= config['max_height'])

    # 3. 强度过滤
    if config.get('use_intensity_filter', False):
        if config.get('filter_zero_intensity', False):
            mask &= intensity > config['min_intensity']
        else:
            mask &= intensity >= config['min_intensity']

    filtered_points = points[mask]
    return filtered_points

def project_lidar_to_image(lidar_points, K_cam, T_cam_to_lidar, img_shape):
    """
    投影 LiDAR 点到图像平面

    Returns:
        pixels: (N, 2) 像素坐标 [u, v]
        depths: (N,) 深度值
    """
    xyz_points = lidar_points[:, :3]

    # 齐次坐标
    ones = np.ones((xyz_points.shape[0], 1))
    homo = np.hstack([xyz_points, ones])

    # 应用外参 (LiDAR -> Camera)
    P_cam = (T_cam_to_lidar @ homo.T).T

    # 深度过滤 (Z > 0.1)
    z_coords = P_cam[:, 2]
    valid_mask = z_coords >= 0.1
    P_cam_valid = P_cam[valid_mask]

    if P_cam_valid.shape[0] == 0:
        return np.array([]).reshape(0, 2).astype(int), np.array([])

    # 应用内参
    xyz_cam = P_cam_valid[:, :3]
    P_img = (K_cam @ xyz_cam.T).T

    # 透视除法
    Z = P_img[:, 2]
    u = P_img[:, 0] / Z
    v = P_img[:, 1] / Z

    # 图像边界过滤
    img_height, img_width = img_shape[:2]
    valid_u = (u >= 0) & (u < img_width)
    valid_v = (v >= 0) & (v < img_height)
    valid = valid_u & valid_v

    u_final = u[valid].astype(int)
    v_final = v[valid].astype(int)
    Z_final = Z[valid]

    pixels = np.column_stack([u_final, v_final])
    depths = Z_final

    return pixels, depths

def color_points_by_depth(depths, vmin=1.0, vmax=80.0):
    """根据深度为点着色（使用 viridis colormap）"""
    if len(depths) == 0:
        return np.array([]).reshape(0, 3)

    clipped = np.clip(depths, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)

    # 使用 matplotlib colormap
    viridis = cm.get_cmap('viridis')
    rgba = viridis(normalized)
    rgb = rgba[:, :3]
    bgr = rgb[:, ::-1]  # RGB -> BGR for OpenCV
    bgr_255 = (bgr * 255).astype(np.uint8)

    return bgr_255

def load_calibration(calib_dir):
    """加载标定参数"""
    with open(os.path.join(calib_dir, 'intrinsics.json')) as f:
        ir_intrinsics = json.load(f)['infrared']

    with open(os.path.join(calib_dir, 'extrinsics.json')) as f:
        extrinsics = json.load(f)
        ir_extrinsics = extrinsics['infrared']
        lidar_extrinsics = extrinsics['lidar_front']

    # 内参矩阵
    fx = fy = ir_intrinsics['focal_length']
    cx, cy = ir_intrinsics['cc_x'], ir_intrinsics['cc_y']
    K_cam = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # 外参变换
    T_cam_to_base = get_transform_matrix(ir_extrinsics)
    T_lidar_to_base = get_transform_matrix(lidar_extrinsics)
    T_cam_to_lidar = np.linalg.inv(T_cam_to_base) @ T_lidar_to_base

    return K_cam, T_cam_to_lidar

def visualize_frames(frame_ids, dataset_root, output_dir, K_cam, T_cam_to_lidar):
    """可视化指定帧"""
    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(dataset_root, 'images')
    lidar_dir = os.path.join(dataset_root, 'lidar_roi')

    success_count = 0

    for frame_id in tqdm(frame_ids, desc="生成可视化"):
        img_path = os.path.join(images_dir, frame_id + '.png')
        lidar_path = os.path.join(lidar_dir, frame_id + '.bin')

        if not os.path.exists(img_path) or not os.path.exists(lidar_path):
            continue

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 读取点云
        try:
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        except:
            continue

        if len(points) == 0:
            # 没有点云，保存原图
            cv2.imwrite(os.path.join(output_dir, frame_id + '.png'), img)
            continue

        # 过滤点云（移除噪声和自车）
        points_filtered = filter_lidar_points(points)

        if len(points_filtered) == 0:
            # 过滤后没有点云，保存原图
            cv2.imwrite(os.path.join(output_dir, frame_id + '.png'), img)
            continue

        # 投影
        pixels, depths = project_lidar_to_image(points_filtered, K_cam, T_cam_to_lidar, img.shape)

        if len(pixels) == 0:
            # 没有投影点，保存原图
            cv2.imwrite(os.path.join(output_dir, frame_id + '.png'), img)
            continue

        # 着色
        colors = color_points_by_depth(depths)

        # 绘制
        vis_img = img.copy()
        for (u, v), color in zip(pixels, colors):
            cv2.circle(vis_img, (u, v), 2, (int(color[0]), int(color[1]), int(color[2])), -1)

        # 保存
        cv2.imwrite(os.path.join(output_dir, frame_id + '.png'), vis_img)
        success_count += 1

    return success_count

def main():
    parser = argparse.ArgumentParser(description='点云投影到红外图像可视化')
    parser.add_argument('--dataset_root', type=str, default=DATASET_ROOT,
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='输出目录')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'all'],
                        help='可视化哪个划分 (train/test/all)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='可视化样本数量（None表示全部）')

    args = parser.parse_args()

    print("=" * 80)
    print("🎨 点云投影到红外图像可视化")
    print("=" * 80)
    print(f"数据集: {args.dataset_root}")
    print(f"输出: {args.output_dir}")
    print(f"划分: {args.split}")
    print("=" * 80)

    # 1. 加载标定
    print("\n>>> 步骤 1/3: 加载标定参数")
    calib_dir = os.path.join(args.dataset_root, 'calibration')
    K_cam, T_cam_to_lidar = load_calibration(calib_dir)
    print("✅ 标定参数已加载")

    # 2. 加载数据列表
    print("\n>>> 步骤 2/3: 加载数据列表")
    split_dir = os.path.join(args.dataset_root, 'split_data')

    frame_ids = []
    if args.split in ['train', 'all']:
        train_file = os.path.join(split_dir, 'train.txt')
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                frame_ids.extend([line.strip() for line in f if line.strip()])

    if args.split in ['test', 'all']:
        test_file = os.path.join(split_dir, 'test.txt')
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                frame_ids.extend([line.strip() for line in f if line.strip()])

    if not frame_ids:
        print("❌ 未找到数据")
        return

    # 限制样本数量
    if args.num_samples is not None:
        frame_ids = frame_ids[:args.num_samples]

    print(f"✅ 加载 {len(frame_ids)} 帧")

    # 3. 可视化
    print("\n>>> 步骤 3/3: 生成可视化")
    success_count = visualize_frames(frame_ids, args.dataset_root, args.output_dir, K_cam, T_cam_to_lidar)

    print(f"\n✅ 完成！成功生成 {success_count} 张可视化图像")
    print(f"📁 输出目录: {args.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
