#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整数据集准备脚本
将原始的22,183张红外图像和对应的LiDAR点云数据整理到训练数据集格式

功能：
1. 时间戳同步（红外和LiDAR）
2. 点云过滤和投影
3. 生成深度图
4. 数据集划分（训练/测试 8:2）
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

# ================= 配置区域 =================
# 1. 原始数据路径
RAW_DATA_ROOT = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00'
RAW_IR_DIR = os.path.join(RAW_DATA_ROOT, 'infrared/images')
RAW_LIDAR_DIR = os.path.join(RAW_DATA_ROOT, 'lidar_front/points')
RAW_IR_TIME = os.path.join(RAW_DATA_ROOT, 'infrared/timestamp.txt')
RAW_LIDAR_TIME = os.path.join(RAW_DATA_ROOT, 'lidar_front/timestamp.txt')
CALIB_DIR = os.path.join(RAW_DATA_ROOT, 'calibration')

# 2. 新数据集目标路径
TARGET_ROOT = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all'

# 3. 点云过滤配置
FILTER_CONFIG = {
    'min_depth': 3.0,        # 最小深度(m) - 过滤自车
    'max_depth': 150.0,      # 最大深度(m) - 移除远距离噪声
    'min_height': -160.0,    # 最小高度(m) - 保留地面点
    'max_height': 50.0,      # 最大高度(m) - 过滤天空伪影
}

# ================= 核心函数（从 project_lidar_to_ir_minimal.py 复用）=================

def filter_lidar_points(points, config=FILTER_CONFIG):
    """
    过滤LiDAR点云，移除噪声和自车点
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    distances = np.sqrt(x**2 + y**2 + z**2)

    mask = np.ones(len(points), dtype=bool)

    # 深度过滤
    mask &= (distances >= config['min_depth']) & (distances <= config['max_depth'])

    # 高度过滤
    mask &= (y >= config['min_height']) & (y <= config['max_height'])

    return points[mask]

def get_rotation_matrix_from_quaternion(quat):
    """
    将四元数 [x, y, z, w] 转换为 3x3 旋转矩阵
    """
    x, y, z, w = quat
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)
    return R

def get_transform_matrix(extrinsics):
    """
    从外参字典构建4x4变换矩阵（四元数 + 平移）
    """
    quat = extrinsics['quaternion']
    trans = extrinsics['translation']
    R = get_rotation_matrix_from_quaternion(quat)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T

def project_lidar_to_image(lidar_points, K_cam, T_cam_to_lidar, img_shape):
    """
    将LiDAR点云投影到图像平面
    返回：有效点的索引
    """
    if len(lidar_points) == 0:
        return np.array([], dtype=int)

    # 1. 提取XYZ
    xyz_points = lidar_points[:, :3]

    # 2. 齐次坐标
    ones_column = np.ones((xyz_points.shape[0], 1))
    homogeneous_points = np.hstack([xyz_points, ones_column])

    # 3. 应用外参（LiDAR -> Camera）
    P_cam_transposed = T_cam_to_lidar @ homogeneous_points.T
    P_cam = P_cam_transposed.T

    # 4. 深度过滤（Z > 0.1）
    z_coords = P_cam[:, 2]
    valid_depth_mask = z_coords >= 0.1
    valid_depth_indices = np.where(valid_depth_mask)[0]

    P_cam_filtered = P_cam[valid_depth_mask]

    if P_cam_filtered.shape[0] == 0:
        return np.array([], dtype=int)

    # 5. 应用内参
    xyz_cam = P_cam_filtered[:, :3]
    P_img_homo_transposed = K_cam @ xyz_cam.T
    P_img_homo = P_img_homo_transposed.T

    # 6. 透视除法
    Z_values = P_img_homo[:, 2]
    u_coords = P_img_homo[:, 0] / Z_values
    v_coords = P_img_homo[:, 1] / Z_values

    # 7. 图像边界过滤
    img_height, img_width = img_shape[:2]
    valid_u_mask = (u_coords >= 0) & (u_coords < img_width)
    valid_v_mask = (v_coords >= 0) & (v_coords < img_height)
    valid_boundary_mask = valid_u_mask & valid_v_mask

    # 映射回原始索引
    final_indices = valid_depth_indices[valid_boundary_mask]

    return final_indices

def generate_depth_map(points_in_fov, K_cam, T_cam_to_lidar, img_shape):
    """
    生成深度图（.npy格式）
    """
    H, W = img_shape[:2]
    depth_map = np.zeros((H, W), dtype=np.float32)

    if len(points_in_fov) == 0:
        return depth_map

    # 投影点云到相机坐标系
    ones = np.ones((points_in_fov.shape[0], 1))
    pts_homo = np.hstack([points_in_fov[:, :3], ones])
    pts_cam = (T_cam_to_lidar @ pts_homo.T).T

    # 应用内参
    pts_img_homo = (K_cam @ pts_cam[:, :3].T).T

    u = (pts_img_homo[:, 0] / pts_cam[:, 2]).astype(int)
    v = (pts_img_homo[:, 1] / pts_cam[:, 2]).astype(int)
    z = pts_cam[:, 2]

    # 确保索引在边界内
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # 填充深度图（如果有重叠点，后面的覆盖前面的）
    depth_map[v[valid], u[valid]] = z[valid]

    return depth_map

def load_timestamps(timestamp_path):
    """
    加载时间戳文件
    格式：timestamp filename
    """
    if not os.path.exists(timestamp_path):
        print(f"Error: 时间戳文件未找到: {timestamp_path}")
        return pd.DataFrame()

    try:
        # 尝试用制表符分隔，然后是空格
        df = pd.read_csv(timestamp_path, sep='\t', header=None,
                        names=['timestamp', 'filename'], dtype={'filename': str})
        if len(df.columns) != 2 or df['timestamp'].dtype == object:
            df = pd.read_csv(timestamp_path, delim_whitespace=True, header=None,
                           names=['timestamp', 'filename'], dtype={'filename': str})

        df['timestamp'] = df['timestamp'].astype(float)
        return df
    except Exception as e:
        print(f"Error 读取时间戳 {timestamp_path}: {e}")
        return pd.DataFrame()

def main():
    print("=" * 80)
    print("完整数据集准备脚本")
    print("=" * 80)

    # 1. 创建目录结构
    print("\n>>> 步骤 1/6: 创建目录结构...")
    subdirs = ['images', 'lidar_roi', 'depth_maps', 'calibration', 'split_data']
    for d in subdirs:
        os.makedirs(os.path.join(TARGET_ROOT, d), exist_ok=True)
    print(f"✅ 目标目录: {TARGET_ROOT}")

    # 2. 复制标定文件
    print("\n>>> 步骤 2/6: 复制标定文件...")
    for f in ['intrinsics.json', 'extrinsics.json']:
        src = os.path.join(CALIB_DIR, f)
        dst = os.path.join(TARGET_ROOT, 'calibration', f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✅ 已复制: {f}")
        else:
            print(f"❌ 错误: 标定文件未找到: {f}")
            return

    # 加载标定矩阵
    with open(os.path.join(TARGET_ROOT, 'calibration/intrinsics.json')) as f:
        ir_intrinsics = json.load(f)['infrared']
    with open(os.path.join(TARGET_ROOT, 'calibration/extrinsics.json')) as f:
        ext = json.load(f)

    fx = ir_intrinsics['focal_length']
    fy = ir_intrinsics['focal_length']
    cx = ir_intrinsics['cc_x']
    cy = ir_intrinsics['cc_y']

    K_cam = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    T_cam_to_base = get_transform_matrix(ext['infrared'])
    T_lidar_to_base = get_transform_matrix(ext['lidar_front'])
    T_cam_to_lidar = np.linalg.inv(T_cam_to_base) @ T_lidar_to_base

    print(f"✅ 标定矩阵加载完成")

    # 3. 同步时间戳
    print("\n>>> 步骤 3/6: 同步时间戳...")
    ir_df = load_timestamps(RAW_IR_TIME)
    lidar_df = load_timestamps(RAW_LIDAR_TIME)

    if ir_df.empty or lidar_df.empty:
        print("❌ 错误: 时间戳加载失败")
        return

    ir_df = ir_df.sort_values('timestamp')
    lidar_df = lidar_df.sort_values('timestamp')

    # 使用 merge_asof 进行最近邻匹配
    merged_df = pd.merge_asof(
        ir_df.rename(columns={'filename': 'filename_ir'}),
        lidar_df.rename(columns={'filename': 'filename_lidar'}),
        on='timestamp',
        direction='nearest',
        tolerance=0.2  # 200ms 容差
    )

    # 移除未匹配的帧
    merged_df = merged_df.dropna(subset=['filename_lidar'])

    print(f"✅ 同步完成: 匹配到 {len(merged_df)} 对数据")
    print(f"   红外总帧数: {len(ir_df)}")
    print(f"   LiDAR总帧数: {len(lidar_df)}")
    print(f"   匹配成功率: {len(merged_df)/len(ir_df)*100:.1f}%")

    # 4. 获取图像尺寸（读取第一张图像）
    print("\n>>> 步骤 4/6: 检测图像尺寸...")
    sample_ir_name = str(merged_df.iloc[0]['filename_ir'])
    if not sample_ir_name.endswith('.png'):
        sample_ir_name += '.png'
    sample_img_path = os.path.join(RAW_IR_DIR, sample_ir_name)

    if not os.path.exists(sample_img_path):
        print(f"❌ 错误: 无法读取样本图像: {sample_img_path}")
        return

    sample_img = cv2.imread(sample_img_path)
    IMG_SHAPE = sample_img.shape
    print(f"✅ 图像尺寸: {IMG_SHAPE[1]}x{IMG_SHAPE[0]}")

    # 5. 批处理循环
    print("\n>>> 步骤 5/6: 处理数据（软链接图像 + 过滤LiDAR + 生成深度图）...")
    valid_frames = []  # 记录成功处理的帧名

    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="处理进度"):
        ir_name = str(row['filename_ir'])
        lidar_name = str(row['filename_lidar'])

        # 确保文件名有扩展名
        if not ir_name.endswith('.png'):
            ir_name_with_ext = ir_name + '.png'
        else:
            ir_name_with_ext = ir_name
            ir_name = ir_name[:-4]  # 去掉.png用于命名

        if not lidar_name.endswith('.bin'):
            lidar_name_with_ext = lidar_name + '.bin'
        else:
            lidar_name_with_ext = lidar_name
            lidar_name = lidar_name[:-4]

        src_img_path = os.path.join(RAW_IR_DIR, ir_name_with_ext)
        src_lidar_path = os.path.join(RAW_LIDAR_DIR, lidar_name_with_ext)

        # 检查源文件是否存在
        if not os.path.exists(src_img_path) or not os.path.exists(src_lidar_path):
            continue

        # A. 软链接红外图像（节省空间）
        dst_img_path = os.path.join(TARGET_ROOT, 'images', ir_name_with_ext)
        if not os.path.exists(dst_img_path):
            try:
                os.symlink(src_img_path, dst_img_path)
            except Exception as e:
                # 如果软链接失败（比如Windows系统），则复制文件
                shutil.copy2(src_img_path, dst_img_path)

        # B. 处理LiDAR
        try:
            points = np.fromfile(src_lidar_path, dtype=np.float32).reshape(-1, 4)
        except Exception as e:
            continue

        # 过滤点云
        points_filtered = filter_lidar_points(points)

        if len(points_filtered) == 0:
            # 如果过滤后没有点，仍然保留（深度图为全黑）
            points_in_fov = np.array([]).reshape(0, 4)
        else:
            # 投影并获取FOV内的点
            valid_indices = project_lidar_to_image(points_filtered, K_cam, T_cam_to_lidar, IMG_SHAPE)
            points_in_fov = points_filtered[valid_indices]

        # 保存LiDAR ROI (.bin) - 使用IR的文件名
        dst_lidar_path = os.path.join(TARGET_ROOT, 'lidar_roi', ir_name + '.bin')
        points_in_fov.tofile(dst_lidar_path)

        # C. 生成深度图 (.npy)
        depth_map = generate_depth_map(points_in_fov, K_cam, T_cam_to_lidar, IMG_SHAPE)
        dst_depth_path = os.path.join(TARGET_ROOT, 'depth_maps', ir_name + '.npy')
        np.save(dst_depth_path, depth_map)

        valid_frames.append(ir_name)

    print(f"\n✅ 数据处理完成: 成功处理 {len(valid_frames)} 帧")

    # 6. 生成数据集划分（时间顺序 8:2）
    print("\n>>> 步骤 6/6: 生成数据集划分（时间顺序 80/20）...")

    # 确保按文件名排序（即时间顺序）
    valid_frames.sort()

    split_idx = int(len(valid_frames) * 0.8)
    train_list = valid_frames[:split_idx]
    test_list = valid_frames[split_idx:]

    # 保存训练集列表
    with open(os.path.join(TARGET_ROOT, 'split_data/train.txt'), 'w') as f:
        f.write('\n'.join(train_list))

    # 保存测试集列表
    with open(os.path.join(TARGET_ROOT, 'split_data/test.txt'), 'w') as f:
        f.write('\n'.join(test_list))

    print(f"✅ 训练集: {len(train_list)} 张图像")
    print(f"✅ 测试集: {len(test_list)} 张图像")

    # 打印总结
    print("\n" + "=" * 80)
    print("✅ 数据集准备完成！")
    print("=" * 80)
    print(f"数据集位置: {TARGET_ROOT}")
    print(f"总帧数: {len(valid_frames)}")
    print(f"训练集: {len(train_list)} (80%)")
    print(f"测试集: {len(test_list)} (20%)")
    print("\n目录结构:")
    print(f"  {TARGET_ROOT}/")
    print(f"    ├── images/          # 红外图像（软链接）")
    print(f"    ├── lidar_roi/       # 过滤后的点云（.bin）")
    print(f"    ├── depth_maps/      # 深度图（.npy）")
    print(f"    ├── calibration/     # 标定文件")
    print(f"    └── split_data/      # 数据集划分")
    print(f"        ├── train.txt")
    print(f"        └── test.txt")
    print("=" * 80)

if __name__ == "__main__":
    main()
