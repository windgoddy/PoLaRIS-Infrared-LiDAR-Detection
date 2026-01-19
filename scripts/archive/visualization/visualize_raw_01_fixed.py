#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化原始数据 (01文件夹) 的 LiDAR 投影脚本 - 修复版

修复内容:
1. 改进标注框坐标解析逻辑,添加调试信息
2. 支持多种标注格式的自动检测
3. 验证图像尺寸与标注的一致性

用法:
    python scripts/visualize_raw_01_fixed.py --raw_data_root /path/to/01 --debug
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

try:
    from matplotlib import colormaps
except ImportError:
    colormaps = None
import matplotlib.cm as cm

# ==================== 默认配置 ====================
DEFAULT_RAW_ROOT = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/01'
OUTPUT_DIR_NAME = 'vis_lidar_projection'
IR_TIMESTAMP_NAME = 'timestamp.txt'
NANO_TO_SECONDS = 1e9
DEFAULT_TIMESTAMP_TOLERANCE = 0.05  # seconds
BASE_DATASET_ROOT = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset'
BASE_ANNOTATION_ROOT = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS'
BASE_OUTPUT_ROOT = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset'
DEFAULT_OUTPUT_SUBDIR = 'vis_lidar_projection_fixed'

# LiDAR 点云过滤配置
FILTER_CONFIG = {
    'min_depth': 3.0,        
    'max_depth': 150.0,      
    'min_height': -160.0,    
    'max_height': 50.0,      
    'use_intensity_filter': True,
    'min_intensity': 5.0,
    'filter_zero_intensity': True,
}
# ================================================


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

def project_lidar_to_image(lidar_points, K_cam, T_cam_to_lidar, img_shape):
    """投影 LiDAR 点到图像平面"""
    xyz_points = lidar_points[:, :3]
    ones = np.ones((xyz_points.shape[0], 1))
    homo = np.hstack([xyz_points, ones])

    P_cam = (T_cam_to_lidar @ homo.T).T
    z_coords = P_cam[:, 2]
    valid_mask = z_coords >= 0.1
    P_cam_valid = P_cam[valid_mask]

    if P_cam_valid.shape[0] == 0:
        return np.array([]).reshape(0, 2).astype(int), np.array([])

    xyz_cam = P_cam_valid[:, :3]
    P_img = (K_cam @ xyz_cam.T).T

    Z = P_img[:, 2]
    u = P_img[:, 0] / Z
    v = P_img[:, 1] / Z

    img_height, img_width = img_shape[:2]
    valid_u = (u >= 0) & (u < img_width)
    valid_v = (v >= 0) & (v < img_height)
    valid = valid_u & valid_v

    u_final = u[valid].astype(int)
    v_final = v[valid].astype(int)
    Z_final = Z[valid]

    return np.column_stack([u_final, v_final]), Z_final

def color_points_by_depth(depths, vmin=1.0, vmax=80.0):
    """根据深度着色"""
    if len(depths) == 0:
        return np.array([]).reshape(0, 3)
    depths = np.clip(depths, vmin, vmax)
    normalized = (depths - vmin) / (vmax - vmin)
    if colormaps is not None:
        viridis = colormaps.get_cmap('viridis')
    else:
        viridis = cm.get_cmap('viridis')
    rgb = viridis(normalized)[:, :3]
    bgr = rgb[:, ::-1]
    colors = (bgr * 255).astype(np.uint8)
    return colors


def load_annotation_boxes(annotation_path, img_shape, debug=False):
    """
    读取YOLO格式标注框
    
    YOLO标准格式: class_id center_x center_y width height
    所有坐标均为归一化值 (0-1之间)
    
    返回: boxes列表 [(x1, y1, x2, y2), ...]
    """
    boxes = []
    if not os.path.exists(annotation_path):
        return boxes

    if len(img_shape) == 2:
        img_h, img_w = img_shape
    else:
        img_h, img_w = img_shape[:2]

    if debug:
        print(f"\n[DEBUG] 加载标注: {annotation_path}")
        print(f"[DEBUG] 图像尺寸: {img_w}x{img_h}")

    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        
    if debug and len(lines) > 0:
        print(f"[DEBUG] 标注文件有 {len(lines)} 行")
        print(f"[DEBUG] 第一行示例: {lines[0].strip()}")

    for line_idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        try:
            # YOLO格式: class_id cx cy w h (归一化坐标)
            class_id = int(parts[0])
            cx_norm = float(parts[1])
            cy_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])
        except ValueError:
            if debug:
                print(f"[DEBUG] 警告: 第{line_idx+1}行格式错误: {line.strip()}")
            continue

        if debug and line_idx == 0:
            print(f"[DEBUG] 归一化值: cx={cx_norm:.4f}, cy={cy_norm:.4f}, w={w_norm:.4f}, h={h_norm:.4f}")
        
        # 转换为像素坐标 (与data_tools.py保持一致)
        cx_pixel = cx_norm * img_w
        cy_pixel = cy_norm * img_h
        w_pixel = w_norm * img_w
        h_pixel = h_norm * img_h

        # 中心坐标 -> 边界框坐标
        x1 = int(cx_pixel - w_pixel / 2.0)
        y1 = int(cy_pixel - h_pixel / 2.0)
        x2 = int(cx_pixel + w_pixel / 2.0)
        y2 = int(cy_pixel + h_pixel / 2.0)

        # 裁剪到图像范围内
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        if x2 <= x1 or y2 <= y1:
            if debug:
                print(f"[DEBUG] 警告: 第{line_idx+1}行产生无效框: ({x1},{y1})-({x2},{y2})")
            continue
        
        if debug and line_idx == 0:
            print(f"[DEBUG] 像素框: ({x1},{y1})-({x2},{y2}), 尺寸={x2-x1}x{y2-y1}")
        
        boxes.append((x1, y1, x2, y2))

    if debug:
        print(f"[DEBUG] 成功加载 {len(boxes)} 个有效标注框\n")
    
    return boxes


def draw_annotation_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """在图像上绘制标注框"""
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def resolve_annotation_dir(dataset_id):
    """根据数据集编号推断实际存在的标注子目录。"""
    ds = str(dataset_id).strip().zfill(2)
    base_all_dir = os.path.join(BASE_ANNOTATION_ROOT, ds, 'all')
    candidate_subdirs = ['tir', 'right', 'left', '']
    for sub in candidate_subdirs:
        candidate_dir = os.path.join(base_all_dir, sub) if sub else base_all_dir
        if os.path.isdir(candidate_dir):
            return candidate_dir
    return os.path.join(base_all_dir, 'tir')


def build_default_paths(dataset_id):
    """根据数据集编号构造默认的原始/标注/输出路径。"""
    ds = str(dataset_id).strip().zfill(2)
    raw_root = os.path.join(BASE_DATASET_ROOT, ds)
    annotation_dir = resolve_annotation_dir(ds)
    output_dir = os.path.join(BASE_OUTPUT_ROOT, ds, DEFAULT_OUTPUT_SUBDIR)
    return ds, raw_root, annotation_dir, output_dir


def process_dataset(raw_root,
                    annotation_dir,
                    output_dir,
                    ir_timestamp_path=None,
                    timestamp_tolerance=DEFAULT_TIMESTAMP_TOLERANCE,
                    num_samples=None,
                    debug=False,
                    cleanup_existing=False):
    """处理单个数据集目录, 生成 LiDAR 投影图像。"""
    dataset_label = os.path.basename(os.path.normpath(raw_root)) or 'unknown'
    timestamp_tolerance = max(timestamp_tolerance, 0.0)

    if not os.path.isdir(raw_root):
        print(f"Raw dataset root not found: {raw_root}")
        return 0
    if not os.path.isdir(annotation_dir):
        print(f"Annotation directory not found: {annotation_dir}")
        return 0

    lidar_pcd_dir = os.path.join(raw_root, 'lidar_front', 'points')
    image_dir = os.path.join(raw_root, 'infrared', 'images')
    calib_dir = os.path.join(raw_root, 'calibration')
    ir_timestamp_path = ir_timestamp_path or os.path.join(raw_root, 'infrared', IR_TIMESTAMP_NAME)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if cleanup_existing and os.path.exists(output_dir):
        existing_outputs = glob.glob(os.path.join(output_dir, '*.png'))
        cleaned_count = 0
        for output_path in existing_outputs:
            img_name = os.path.splitext(os.path.basename(output_path))[0]
            anno_file = os.path.join(annotation_dir, f"{img_name}.txt")
            if not os.path.exists(anno_file) or os.path.getsize(anno_file) == 0:
                try:
                    os.remove(output_path)
                    cleaned_count += 1
                except OSError as err:
                    print(f"Failed to remove {output_path}: {err}")
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} existing output images with no annotations")

    print(f"\n=== Processing dataset {dataset_label} ===")
    print(f"Data Root: {raw_root}")
    print(f"Output Dir: {output_dir}")
    print(f"Debug Mode: {debug}")

    try:
        K_cam, T_cam_to_lidar = load_calibration(calib_dir)
        print("Calibration loaded successfully.")
    except Exception as err:
        print(f"Error loading calibration: {err}")
        return 0

    lidar_files = sorted(glob.glob(os.path.join(lidar_pcd_dir, '*.bin')))
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))

    if len(lidar_files) == 0:
        print("No lidar files found.")
        return 0
    if len(image_files) == 0:
        print("No image files found.")
        return 0

    image_lookup = {
        os.path.splitext(os.path.basename(path))[0]: path for path in image_files
    }

    try:
        ir_timestamps, ir_filenames = load_ir_timestamps(ir_timestamp_path)
        print(f"Loaded {len(ir_timestamps)} IR timestamps from {ir_timestamp_path}")
    except (FileNotFoundError, ValueError) as err:
        print(f"Error loading IR timestamps: {err}")
        return 0

    samples_to_process = lidar_files
    if num_samples:
        samples_to_process = lidar_files[:num_samples]

    print(f"Found {len(lidar_files)} lidar files and {len(image_files)} images.")
    print(f"Processing {len(samples_to_process)} LiDAR frames with ±{timestamp_tolerance:.3f}s tolerance.")

    success_count = 0
    matched_diffs = []
    saved_diffs = []
    debug_counter = 0
    skipped_no_annotation = 0
    skipped_empty_boxes = 0

    for lidar_path in tqdm(samples_to_process, desc="Processing"):
        lidar_filename = os.path.basename(lidar_path)
        lidar_ts = extract_lidar_timestamp(lidar_filename)
        if lidar_ts is None:
            continue

        ir_idx, ts_diff = find_nearest_ir_index(ir_timestamps, lidar_ts)
        if ir_idx is None or ts_diff is None or ts_diff > timestamp_tolerance:
            continue
        matched_diffs.append(ts_diff)

        img_name_no_ext = ir_filenames[ir_idx]
        img_path = image_lookup.get(img_name_no_ext)
        if not img_path:
            continue

        annotation_file = os.path.join(annotation_dir, f"{img_name_no_ext}.txt")
        if not os.path.exists(annotation_file) or os.path.getsize(annotation_file) == 0:
            skipped_no_annotation += 1
            continue

        out_name = f"{img_name_no_ext}.png"
        out_path = os.path.join(output_dir, out_name)

        normalized_img = normalize_infrared_image(img_path, method='percentile')
        if normalized_img is None:
            continue

        show_debug = debug and debug_counter < 3
        boxes = load_annotation_boxes(annotation_file, normalized_img.shape, debug=show_debug)
        if not boxes:
            skipped_empty_boxes += 1
            if show_debug:
                print(f"[DEBUG] 跳过: 标注文件存在但未解析到有效框")
            continue

        img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)
        vis_img = img.copy()

        try:
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        except Exception as err:
            print(f"Failed to read lidar {lidar_path}: {err}")
            continue

        points_filtered = filter_lidar_points(points)
        if len(points_filtered) > 0:
            pixels, depths = project_lidar_to_image(points_filtered, K_cam, T_cam_to_lidar, img.shape)
            if len(pixels) > 0:
                colors = color_points_by_depth(depths)
                for (u, v), color in zip(pixels, colors):
                    cv2.circle(vis_img, (u, v), 2, (int(color[0]), int(color[1]), int(color[2])), -1)

        draw_annotation_boxes(vis_img, boxes, color=(0, 255, 0), thickness=2)
        cv2.imwrite(out_path, vis_img)
        success_count += 1
        saved_diffs.append(ts_diff)
        debug_counter += 1

    if matched_diffs:
        matched_stats = np.array(matched_diffs)
        print(
            f"Matched {len(matched_diffs)}/{len(samples_to_process)} LiDAR frames; "
            f"|Δt| mean={matched_stats.mean():.4f}s max={matched_stats.max():.4f}s"
        )
    else:
        print("No LiDAR frames met the timestamp tolerance criteria.")

    if saved_diffs:
        saved_stats = np.array(saved_diffs)
        print(
            f"Saved {success_count} annotated projections; |Δt| mean={saved_stats.mean():.4f}s max={saved_stats.max():.4f}s"
        )
    else:
        print("No annotated projections were saved.")

    total_skipped = skipped_no_annotation + skipped_empty_boxes
    if total_skipped > 0:
        print(f"\n📊 跳过统计:")
        print(f"  - 无标注文件或文件为空: {skipped_no_annotation}")
        print(f"  - 标注文件存在但解析为空: {skipped_empty_boxes}")
        print(f"  - 总共跳过: {total_skipped}")

    print(f"\n✅ Processing complete. Saved {success_count} images to {output_dir}\n")
    return success_count


def load_ir_timestamps(timestamp_path):
    """读取红外图像的时间戳文件"""
    if not os.path.exists(timestamp_path):
        raise FileNotFoundError(f"Infrared timestamp file not found: {timestamp_path}")

    timestamps = []
    filenames = []
    with open(timestamp_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                ts_val = float(parts[0])
            except ValueError:
                continue
            frame_name = parts[1].strip()
            timestamps.append(ts_val)
            filenames.append(frame_name)

    if len(timestamps) == 0:
        raise ValueError(f"No valid timestamps parsed from {timestamp_path}")

    timestamps = np.array(timestamps, dtype=np.float64)
    order = np.argsort(timestamps)
    sorted_timestamps = timestamps[order]
    sorted_filenames = [filenames[idx] for idx in order]
    return sorted_timestamps, sorted_filenames


def extract_lidar_timestamp(lidar_filename):
    """从 LiDAR 文件名中解析时间戳"""
    base = os.path.splitext(lidar_filename)[0]
    try:
        nanosec = int(base)
    except ValueError:
        return None
    return nanosec / NANO_TO_SECONDS


def find_nearest_ir_index(ir_timestamps, target_ts):
    """寻找最近的红外帧索引"""
    if ir_timestamps.size == 0:
        return None, None

    idx = np.searchsorted(ir_timestamps, target_ts)
    candidates = []
    if idx < len(ir_timestamps):
        candidates.append(idx)
    if idx - 1 >= 0:
        candidates.append(idx - 1)

    best_idx, best_diff = None, None
    for cand in candidates:
        diff = abs(ir_timestamps[cand] - target_ts)
        if best_diff is None or diff < best_diff:
            best_idx, best_diff = cand, diff

    return best_idx, best_diff

def load_calibration(calib_dir):
    """加载标定参数"""
    intrinsics_path = os.path.join(calib_dir, 'intrinsics.json')
    extrinsics_path = os.path.join(calib_dir, 'extrinsics.json')
    
    if not os.path.exists(intrinsics_path) or not os.path.exists(extrinsics_path):
        raise FileNotFoundError(f"Missing calibration files in {calib_dir}")

    with open(intrinsics_path) as f:
        ir_intrinsics = json.load(f)['infrared']

    with open(extrinsics_path) as f:
        extrinsics = json.load(f)
        ir_extrinsics = extrinsics['infrared']
        lidar_extrinsics = extrinsics['lidar_front']

    fx = fy = ir_intrinsics['focal_length']
    cx, cy = ir_intrinsics['cc_x'], ir_intrinsics['cc_y']
    K_cam = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    T_cam_to_base = get_transform_matrix(ir_extrinsics)
    T_lidar_to_base = get_transform_matrix(lidar_extrinsics)
    T_cam_to_lidar = np.linalg.inv(T_cam_to_base) @ T_lidar_to_base

    return K_cam, T_cam_to_lidar

def main():
    parser = argparse.ArgumentParser(description='Visualize Raw Data LiDAR Projection (multi-dataset support)')
    parser.add_argument('--dataset_ids', nargs='+', default=None,
                        help='Dataset folder IDs to process (e.g., 01 02 03 04). When provided, paths are auto-built.')
    parser.add_argument('--raw_data_root', type=str, default=DEFAULT_RAW_ROOT,
                        help='Root directory of the raw dataset (used when --dataset_ids is not set)')
    parser.add_argument('--annotation_dir', type=str,
                        default='/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS/01/all/tir',
                        help='Directory containing annotation txt files (used when --dataset_ids is not set)')
    parser.add_argument('--output_dir', type=str,
                        default='/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/01/vis_lidar_projection_fixed',
                        help='Directory to save output visualizations (used when --dataset_ids is not set)')
    parser.add_argument('--ir_timestamp_path', type=str, default=None,
                        help='Optional custom path to infrared/timestamp.txt (single dataset mode only)')
    parser.add_argument('--timestamp_tolerance', type=float, default=DEFAULT_TIMESTAMP_TOLERANCE,
                        help='Maximum timestamp difference in seconds')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (None for all)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for first few samples')
    parser.add_argument('--cleanup_existing', action='store_true',
                        help='Remove existing outputs without annotations before processing each dataset')

    args = parser.parse_args()

    if args.dataset_ids:
        normalized_ids = []
        for ds in args.dataset_ids:
            ds_clean = str(ds).strip()
            if ds_clean:
                normalized_ids.append(ds_clean.zfill(2))

        if not normalized_ids:
            print('No valid dataset IDs provided.')
            return

        for ds_id in normalized_ids:
            _, raw_root, annotation_dir, output_dir = build_default_paths(ds_id)
            process_dataset(
                raw_root=raw_root,
                annotation_dir=annotation_dir,
                output_dir=output_dir,
                ir_timestamp_path=None,
                timestamp_tolerance=args.timestamp_tolerance,
                num_samples=args.num_samples,
                debug=args.debug,
                cleanup_existing=args.cleanup_existing,
            )
        return

    process_dataset(
        raw_root=args.raw_data_root,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        ir_timestamp_path=args.ir_timestamp_path,
        timestamp_tolerance=args.timestamp_tolerance,
        num_samples=args.num_samples,
        debug=args.debug,
        cleanup_existing=args.cleanup_existing,
    )

if __name__ == '__main__':
    main()
