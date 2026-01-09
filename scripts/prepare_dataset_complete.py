#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pohang-Canal-all 数据集一键式完整准备脚本

功能（一键完成所有步骤）:
1. 时间戳同步（红外 ↔ LiDAR）
2. 16-bit → 8-bit 图像归一化 ✨
3. LiDAR 点云过滤和投影
4. 深度图生成 (.npy)
5. YOLO 格式掩码生成 ✨
6. 自动过滤无标签数据 ✨
7. 数据集划分 (train 80% / test 20%)

优势：
- 一次运行，生成所有训练所需数据
- 自动处理 16-bit 红外图像归一化
- 智能过滤无标签数据，节省训练时间
- 完整的数据验证和统计
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# ==================== 配置区域 ====================
# 1. 路径配置
RAW_IR_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/infrared/images'
RAW_LIDAR_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/lidar_front/points'
IR_TIMESTAMP = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/infrared/timestamp.txt'
LIDAR_TIMESTAMP = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/lidar_front/timestamp.txt'
CALIB_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/calibration'
YOLO_LABELS_DIR = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/00/all/00_inf_labels'

TARGET_ROOT = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all'

# 2. 处理参数
FILTER_CONFIG = {
    'min_depth': 3.0,
    'max_depth': 150.0,
    'min_height': -160.0,
    'max_height': 50.0,
    'use_intensity_filter': True,
    'min_intensity': 5.0,
}

NORMALIZE_METHOD = 'percentile'  # 'minmax', 'percentile', 'clahe'
TARGET_CLASS_IDS = [0]  # YOLO class_id for boat
TRAIN_RATIO = 0.8
TIMESTAMP_TOLERANCE = 0.2  # 200ms
# ==================================================

def normalize_infrared_image(image_path, method='percentile'):
    """归一化红外图像（16-bit → 8-bit）"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if img.dtype == np.uint8 and img.mean() >= 10:
            return img

        img_float = img.astype(np.float32)

        if method == 'minmax':
            img_min, img_max = img_float.min(), img_float.max()
            if img_max == img_min:
                return np.zeros_like(img_float, dtype=np.uint8)
            normalized = ((img_float - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

        elif method == 'percentile':
            vmin = np.percentile(img_float, 2)
            vmax = np.percentile(img_float, 98)
            if vmax == vmin:
                return np.zeros_like(img_float, dtype=np.uint8)
            img_clipped = np.clip(img_float, vmin, vmax)
            normalized = ((img_clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)

        elif method == 'clahe':
            vmin = np.percentile(img_float, 2)
            vmax = np.percentile(img_float, 98)
            if vmax == vmin:
                return np.zeros_like(img_float, dtype=np.uint8)
            img_clipped = np.clip(img_float, vmin, vmax)
            img_norm = ((img_clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(img_norm)

        return normalized
    except Exception as e:
        print(f"Error normalizing {image_path}: {e}")
        return None

def filter_lidar_points(points, config=FILTER_CONFIG):
    """过滤 LiDAR 点云"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    intensity = points[:, 3]
    distances = np.sqrt(x**2 + y**2 + z**2)

    mask = np.ones(points.shape[0], dtype=bool)
    mask &= (distances >= config['min_depth']) & (distances <= config['max_depth'])
    mask &= (y >= config['min_height']) & (y <= config['max_height'])

    if config.get('use_intensity_filter', False):
        if config.get('filter_zero_intensity', False):
            mask &= intensity > config['min_intensity']
        else:
            mask &= intensity >= config['min_intensity']

    return points[mask]

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

def project_lidar_to_fov(lidar_points, K_cam, T_cam_to_lidar, img_shape):
    """投影 LiDAR 点到图像视野"""
    xyz_points = lidar_points[:, :3]
    ones_column = np.ones((xyz_points.shape[0], 1))
    homogeneous_points = np.hstack([xyz_points, ones_column])

    P_cam_transposed = T_cam_to_lidar @ homogeneous_points.T
    P_cam = P_cam_transposed.T

    z_coords = P_cam[:, 2]
    valid_depth_mask = z_coords >= 0.1
    valid_depth_indices = np.where(valid_depth_mask)[0]
    P_cam_filtered = P_cam[valid_depth_mask]

    if P_cam_filtered.shape[0] == 0:
        return np.array([], dtype=int)

    xyz_cam = P_cam_filtered[:, :3]
    P_img_homo_transposed = K_cam @ xyz_cam.T
    P_img_homo = P_img_homo_transposed.T

    Z_values = P_img_homo[:, 2]
    u_coords = P_img_homo[:, 0] / Z_values
    v_coords = P_img_homo[:, 1] / Z_values

    img_height, img_width = img_shape[:2]
    valid_u_mask = (u_coords >= 0) & (u_coords < img_width)
    valid_v_mask = (v_coords >= 0) & (v_coords < img_height)
    valid_boundary_mask = valid_u_mask & valid_v_mask

    final_indices = valid_depth_indices[valid_boundary_mask]
    return final_indices

def generate_depth_map(points_in_fov, K_cam, T_cam_to_lidar, img_shape):
    """生成深度图"""
    H, W = img_shape[:2]
    depth_map = np.zeros((H, W), dtype=np.float32)

    if len(points_in_fov) == 0:
        return depth_map

    xyz_points = points_in_fov[:, :3]
    ones = np.ones((xyz_points.shape[0], 1))
    homo = np.hstack([xyz_points, ones])
    P_cam = (T_cam_to_lidar @ homo.T).T

    z_coords = P_cam[:, 2]
    valid_mask = z_coords >= 0.1
    P_cam_valid = P_cam[valid_mask]

    if P_cam_valid.shape[0] == 0:
        return depth_map

    xyz_cam = P_cam_valid[:, :3]
    P_img = (K_cam @ xyz_cam.T).T
    Z = P_img[:, 2]
    u = (P_img[:, 0] / Z).astype(int)
    v = (P_img[:, 1] / Z).astype(int)

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid, v_valid, Z_valid = u[valid], v[valid], Z[valid]

    for uu, vv, zz in zip(u_valid, v_valid, Z_valid):
        if depth_map[vv, uu] == 0 or zz < depth_map[vv, uu]:
            depth_map[vv, uu] = zz

    return depth_map

def yolo_to_mask(yolo_path, img_height, img_width, target_class_ids):
    """将 YOLO 标签转换为二值掩码"""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    if not os.path.exists(yolo_path):
        return mask

    try:
        with open(yolo_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            if class_id not in target_class_ids:
                continue

            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            center_x_px = int(center_x * img_width)
            center_y_px = int(center_y * img_height)
            width_px = int(width * img_width)
            height_px = int(height * img_height)

            x1 = max(0, int(center_x_px - width_px / 2))
            y1 = max(0, int(center_y_px - height_px / 2))
            x2 = min(img_width - 1, int(center_x_px + width_px / 2))
            y2 = min(img_height - 1, int(center_y_px + height_px / 2))

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        return mask
    except Exception as e:
        print(f"Error processing {yolo_path}: {e}")
        return mask

def load_timestamps(timestamp_path):
    """加载时间戳文件"""
    if not os.path.exists(timestamp_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(timestamp_path, sep='\t', header=None, names=['timestamp', 'filename'], dtype={'filename': str})
        if len(df.columns) != 2 or df['timestamp'].dtype == object:
            df = pd.read_csv(timestamp_path, delim_whitespace=True, header=None, names=['timestamp', 'filename'], dtype={'filename': str})

        df['timestamp'] = df['timestamp'].astype(float)
        return df
    except Exception as e:
        print(f"Error reading {timestamp_path}: {e}")
        return pd.DataFrame()

def main():
    print("=" * 80)
    print("🚀 Pohang-Canal-all 数据集一键式完整准备")
    print("=" * 80)
    print(f"目标目录: {TARGET_ROOT}")
    print(f"归一化方法: {NORMALIZE_METHOD}")
    print(f"目标类别: {TARGET_CLASS_IDS}")
    print("=" * 80)

    # 创建目录
    os.makedirs(TARGET_ROOT, exist_ok=True)
    for subdir in ['images', 'lidar_roi', 'depth_maps', 'masks', 'calibration', 'split_data']:
        os.makedirs(os.path.join(TARGET_ROOT, subdir), exist_ok=True)

    # 1. 复制标定文件
    print("\n>>> 步骤 1/7: 复制标定文件")
    for calib_file in ['intrinsics.json', 'extrinsics.json']:
        src = os.path.join(CALIB_DIR, calib_file)
        dst = os.path.join(TARGET_ROOT, 'calibration', calib_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print("✅ 标定文件已复制")

    # 2. 加载标定参数
    print("\n>>> 步骤 2/7: 加载标定参数")
    with open(os.path.join(CALIB_DIR, 'intrinsics.json')) as f:
        ir_intrinsics = json.load(f)['infrared']
    with open(os.path.join(CALIB_DIR, 'extrinsics.json')) as f:
        extrinsics_data = json.load(f)
        ir_extrinsics = extrinsics_data['infrared']
        lidar_extrinsics = extrinsics_data['lidar_front']

    fx = fy = ir_intrinsics['focal_length']
    cx, cy = ir_intrinsics['cc_x'], ir_intrinsics['cc_y']
    K_cam = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    T_cam_to_base = get_transform_matrix(ir_extrinsics)
    T_lidar_to_base = get_transform_matrix(lidar_extrinsics)
    T_cam_to_lidar = np.linalg.inv(T_cam_to_base) @ T_lidar_to_base
    print("✅ 标定参数已加载")

    # 3. 时间戳同步
    print("\n>>> 步骤 3/7: 时间戳同步")
    ir_df = load_timestamps(IR_TIMESTAMP)
    lidar_df = load_timestamps(LIDAR_TIMESTAMP)

    if ir_df.empty or lidar_df.empty:
        print("❌ 时间戳文件加载失败")
        return

    ir_df = ir_df.sort_values('timestamp').rename(columns={'filename': 'ir_filename'})
    lidar_df = lidar_df.sort_values('timestamp').rename(columns={'filename': 'lidar_filename'})

    merged_df = pd.merge_asof(ir_df, lidar_df, on='timestamp', direction='nearest', tolerance=TIMESTAMP_TOLERANCE)
    merged_df = merged_df.dropna(subset=['lidar_filename'])

    print(f"✅ 同步完成: {len(merged_df)} 对")

    # 4. 处理数据（归一化、LiDAR、深度图）
    print("\n>>> 步骤 4/7: 处理图像和点云")

    processed_frames = []
    img_shape = None
    first_norm_stats = None

    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="处理数据"):
        ir_name = str(row['ir_filename'])
        lidar_name = str(row['lidar_filename'])

        if not ir_name.endswith('.png'):
            ir_name += '.png'
        if not lidar_name.endswith('.bin'):
            lidar_name += '.bin'

        frame_id = ir_name[:-4]

        src_img = os.path.join(RAW_IR_DIR, ir_name)
        src_lidar = os.path.join(RAW_LIDAR_DIR, lidar_name)

        if not os.path.exists(src_img) or not os.path.exists(src_lidar):
            continue

        # A. 归一化图像
        dst_img = os.path.join(TARGET_ROOT, 'images', ir_name)
        if not os.path.exists(dst_img):
            normalized = normalize_infrared_image(src_img, method=NORMALIZE_METHOD)
            if normalized is None:
                continue
            cv2.imwrite(dst_img, normalized)

            if first_norm_stats is None:
                original = cv2.imread(src_img, cv2.IMREAD_UNCHANGED)
                first_norm_stats = {
                    'original_dtype': str(original.dtype),
                    'original_range': (int(original.min()), int(original.max())),
                    'normalized_range': (int(normalized.min()), int(normalized.max()))
                }

            if img_shape is None:
                img_shape = normalized.shape
        else:
            # 文件已存在，读取以获取 img_shape
            if img_shape is None:
                existing_img = cv2.imread(dst_img)
                if existing_img is not None:
                    img_shape = existing_img.shape

        # B. 处理 LiDAR
        try:
            points = np.fromfile(src_lidar, dtype=np.float32).reshape(-1, 4)
        except:
            continue

        points_filtered = filter_lidar_points(points)
        if len(points_filtered) == 0:
            points_in_fov = np.array([]).reshape(0, 4)
        else:
            fov_indices = project_lidar_to_fov(points_filtered, K_cam, T_cam_to_lidar, img_shape)
            points_in_fov = points_filtered[fov_indices] if len(fov_indices) > 0 else np.array([]).reshape(0, 4)

        # 保存 LiDAR
        lidar_out = os.path.join(TARGET_ROOT, 'lidar_roi', lidar_name)
        points_in_fov.tofile(lidar_out)

        # C. 生成深度图
        depth_map = generate_depth_map(points_in_fov, K_cam, T_cam_to_lidar, img_shape)
        depth_out = os.path.join(TARGET_ROOT, 'depth_maps', frame_id + '.npy')
        np.save(depth_out, depth_map)

        processed_frames.append(frame_id)

    print(f"\n✅ 处理完成: {len(processed_frames)} 帧")
    if first_norm_stats:
        print(f"   图像归一化: {first_norm_stats['original_dtype']} {first_norm_stats['original_range']} → uint8 {first_norm_stats['normalized_range']}")

    # 5. 生成 YOLO 掩码
    print("\n>>> 步骤 5/7: 生成 YOLO 掩码")

    if img_shape is None:
        first_img = os.path.join(TARGET_ROOT, 'images', processed_frames[0] + '.png')
        img = cv2.imread(first_img)
        img_shape = img.shape

    img_height, img_width = img_shape[:2]

    masks_generated = 0
    for frame_id in tqdm(processed_frames, desc="生成掩码"):
        yolo_path = os.path.join(YOLO_LABELS_DIR, frame_id + '.txt')
        mask_path = os.path.join(TARGET_ROOT, 'masks', frame_id + '.png')

        mask = yolo_to_mask(yolo_path, img_height, img_width, TARGET_CLASS_IDS)
        cv2.imwrite(mask_path, mask)
        masks_generated += 1

    print(f"✅ 掩码生成完成: {masks_generated} 个")

    # 6. 过滤无标签数据
    print("\n>>> 步骤 6/7: 过滤无标签数据")

    valid_labels = set()
    for fname in os.listdir(YOLO_LABELS_DIR):
        if fname.endswith('.txt'):
            valid_labels.add(fname[:-4])

    valid_frames = [f for f in processed_frames if f in valid_labels]
    filtered_count = len(processed_frames) - len(valid_frames)

    print(f"✅ 过滤完成: 保留 {len(valid_frames)} 帧，过滤 {filtered_count} 帧")

    # 7. 数据集划分
    print("\n>>> 步骤 7/7: 数据集划分 (80/20)")

    valid_frames.sort()
    split_idx = int(len(valid_frames) * TRAIN_RATIO)
    train_list = valid_frames[:split_idx]
    test_list = valid_frames[split_idx:]

    train_file = os.path.join(TARGET_ROOT, 'split_data', 'train.txt')
    test_file = os.path.join(TARGET_ROOT, 'split_data', 'test.txt')

    with open(train_file, 'w') as f:
        f.write('\n'.join(train_list))
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_list))

    print(f"✅ 训练集: {len(train_list)} 张")
    print(f"✅ 测试集: {len(test_list)} 张")

    # 最终验证
    print("\n" + "=" * 80)
    print("📊 数据集准备完成！")
    print("=" * 80)
    print(f"📁 数据集位置: {TARGET_ROOT}")
    print(f"  ├── images/      {len(os.listdir(os.path.join(TARGET_ROOT, 'images')))} 张 (8-bit 归一化)")
    print(f"  ├── lidar_roi/   {len(os.listdir(os.path.join(TARGET_ROOT, 'lidar_roi')))} 个")
    print(f"  ├── depth_maps/  {len(os.listdir(os.path.join(TARGET_ROOT, 'depth_maps')))} 个")
    print(f"  ├── masks/       {len(os.listdir(os.path.join(TARGET_ROOT, 'masks')))} 个")
    print(f"  └── split_data/")
    print(f"      ├── train.txt  {len(train_list)} 张")
    print(f"      └── test.txt   {len(test_list)} 张")
    print("=" * 80)
    print("\n✅ 可以开始训练了！")
    print("   ./scripts/run_Phase3_improved_v3.sh")
    print("=" * 80)

if __name__ == "__main__":
    main()
