import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from scipy import interpolate

# --- Reuse functions from project_lidar_to_ir_minimal.py ---
FILTER_CONFIG = {
    'min_depth': 3.0,
    'max_depth': 120.0,
    'min_height': -160.0,
    'max_height': 10.0,
    'use_intensity_filter': True,
    'min_intensity': 5.0,
    'filter_zero_intensity': True,
}

def filter_lidar_points(points, config=FILTER_CONFIG):
    """
    Filter LiDAR points to remove noise and ego-vehicle.
    """
    original_count = points.shape[0]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    intensity = points[:, 3]

    distances = np.sqrt(x**2 + y**2 + z**2)

    mask = np.ones(original_count, dtype=bool)

    # 1. Depth filter
    mask &= (distances >= config['min_depth']) & (distances <= config['max_depth'])

    # 2. Height filter
    mask &= (y >= config['min_height']) & (y <= config['max_height'])

    # 3. Intensity filter
    if config.get('use_intensity_filter', False):
        if config.get('filter_zero_intensity', False):
            mask &= intensity > config['min_intensity']
        else:
            mask &= intensity >= config['min_intensity']

    filtered_points = points[mask]
    return filtered_points

def get_rotation_matrix_from_quaternion(quat):
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
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
    Construct 4x4 transformation matrix from extrinsics dict (quaternion + translation).
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
    Project LiDAR points to image plane.
    Returns pixels, depths, and indices of valid points in the input array.
    """
    # 1. Extract XYZ
    xyz_points = lidar_points[:, :3]

    # 2. Homogeneous coordinates
    ones_column = np.ones((xyz_points.shape[0], 1))
    homogeneous_points = np.hstack([xyz_points, ones_column])

    # 3. Apply Extrinsics (LiDAR -> Camera)
    P_cam_transposed = T_cam_to_lidar @ homogeneous_points.T
    P_cam = P_cam_transposed.T

    # 4. Filter by Depth (Z > 0.1)
    z_coords = P_cam[:, 2]
    valid_depth_mask = z_coords >= 0.1
    valid_depth_indices = np.where(valid_depth_mask)[0]

    P_cam_filtered = P_cam[valid_depth_mask]

    if P_cam_filtered.shape[0] == 0:
        return np.array([]).reshape(0, 2).astype(int), np.array([]), np.array([], dtype=int)

    # 5. Apply Intrinsics
    xyz_cam = P_cam_filtered[:, :3]
    P_img_homo_transposed = K_cam @ xyz_cam.T
    P_img_homo = P_img_homo_transposed.T

    # 6. Perspective Division
    Z_values = P_img_homo[:, 2]
    u_coords = P_img_homo[:, 0] / Z_values
    v_coords = P_img_homo[:, 1] / Z_values

    # 7. Filter by Image Bounds
    img_height, img_width = img_shape[:2]
    valid_u_mask = (u_coords >= 0) & (u_coords < img_width)
    valid_v_mask = (v_coords >= 0) & (v_coords < img_height)
    valid_boundary_mask = valid_u_mask & valid_v_mask

    u_final = u_coords[valid_boundary_mask]
    v_final = v_coords[valid_boundary_mask]
    Z_final = Z_values[valid_boundary_mask]

    pixels = np.column_stack([u_final.astype(int), v_final.astype(int)])
    depths = Z_final.astype(float)

    final_indices = valid_depth_indices[valid_boundary_mask]

    return pixels, depths, final_indices

def load_timestamps(timestamp_path):
    """
    Load timestamps from file.
    Format: timestamp filename
    """
    if not os.path.exists(timestamp_path):
        print(f"Error: Timestamp file not found: {timestamp_path}")
        return pd.DataFrame()

    try:
        # Try reading with tab separator first, then whitespace
        df = pd.read_csv(timestamp_path, sep='\t', header=None, names=['timestamp', 'filename'], dtype={'filename': str})
        if len(df.columns) != 2 or df['timestamp'].dtype == object:
             df = pd.read_csv(timestamp_path, delim_whitespace=True, header=None, names=['timestamp', 'filename'], dtype={'filename': str})

        df['timestamp'] = df['timestamp'].astype(float)
        return df
    except Exception as e:
        print(f"Error reading timestamps {timestamp_path}: {e}")
        return pd.DataFrame()

# --- Core Algorithm: Generate Validity Mask ---

def generate_validity_mask(pixels, img_shape, blur_kernel_size=21):
    """
    Generate LiDAR validity mask using column-wise scan and fill algorithm.

    Physical Principle:
    - LiDAR scans from near (image bottom) to far (image middle)
    - For each column, if a laser point is detected at height v,
      then the entire region from v to image bottom is in LiDAR coverage

    Args:
        pixels: Projected pixel coordinates (N x 2, [u, v])
        img_shape: Image shape (H, W) or (H, W, C)
        blur_kernel_size: Gaussian blur kernel size (must be odd)

    Returns:
        validity_mask: Blurred validity mask (H x W, float32, range 0-1)
    """
    H, W = img_shape[:2]

    # Step A: Extract Top Boundary for Each Column
    # Find the topmost (minimum v) point in each column
    top_boundary = np.full(W, -1, dtype=np.float32)  # -1 indicates no point in column

    for u, v in pixels:
        if 0 <= u < W:
            if top_boundary[u] == -1:
                top_boundary[u] = v
            else:
                top_boundary[u] = min(top_boundary[u], v)  # Keep the topmost point

    # Step B: Interpolate Missing Columns
    # Many columns may be empty due to sparse LiDAR data
    valid_cols = np.where(top_boundary != -1)[0]

    if len(valid_cols) == 0:
        # No valid points, return empty mask
        print("Warning: No valid LiDAR points found in image")
        return np.zeros((H, W), dtype=np.float32)

    if len(valid_cols) == 1:
        # Only one valid column, extend horizontally
        top_boundary[:] = top_boundary[valid_cols[0]]
    else:
        # Interpolate using linear interpolation
        valid_boundaries = top_boundary[valid_cols]

        # Create interpolation function
        interp_func = interpolate.interp1d(
            valid_cols,
            valid_boundaries,
            kind='linear',
            fill_value='extrapolate'
        )

        # Fill all columns
        all_cols = np.arange(W)
        top_boundary = interp_func(all_cols)

        # Clamp to valid range [0, H-1]
        top_boundary = np.clip(top_boundary, 0, H - 1)

    # Step C: Fill Mask from Top Boundary to Bottom
    # Everything from top_boundary[u] to image bottom is LiDAR coverage
    mask = np.zeros((H, W), dtype=np.uint8)

    for u in range(W):
        v_start = int(np.round(top_boundary[u]))
        mask[v_start:H, u] = 255

    # Step D: Apply Gaussian Blur for Smooth Transition
    if blur_kernel_size > 0:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1  # Ensure odd kernel size

        mask_blurred = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

        # Normalize to [0, 1]
        validity_mask = mask_blurred.astype(np.float32) / 255.0
    else:
        validity_mask = mask.astype(np.float32) / 255.0

    return validity_mask

def create_visualization(ir_img, validity_mask, alpha=0.4):
    """
    Create visualization overlay: IR image + semi-transparent red mask.

    Args:
        ir_img: IR image (H x W x 3, BGR)
        validity_mask: Validity mask (H x W, float32, range 0-1)
        alpha: Transparency of the mask overlay

    Returns:
        vis_img: Visualization image (H x W x 3, BGR)
    """
    # Convert IR to color if grayscale
    if len(ir_img.shape) == 2:
        ir_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)

    # Create red mask overlay
    H, W = validity_mask.shape
    red_overlay = np.zeros((H, W, 3), dtype=np.uint8)
    red_overlay[:, :, 2] = (validity_mask * 255).astype(np.uint8)  # Red channel (BGR format)

    # Blend with original image
    vis_img = ir_img.copy()
    mask_indices = validity_mask > 0.1  # Only overlay where mask is significant
    vis_img[mask_indices] = cv2.addWeighted(
        ir_img[mask_indices],
        1 - alpha,
        red_overlay[mask_indices],
        alpha,
        0
    )

    return vis_img

def main():
    # --- Configuration ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Paths - Using same structure as project_lidar_to_ir_minimal.py
    dataset_base = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00'

    ir_timestamp_path = os.path.join(dataset_base, 'infrared/timestamp.txt')
    lidar_timestamp_path = os.path.join(dataset_base, 'lidar_front/timestamp.txt')

    local_images_dir = os.path.join(project_root, 'dataset/Pohang-Canal/images')
    local_lidar_dir = os.path.join(project_root, 'dataset/Pohang-Canal/LiDAR')

    # Output directories
    output_mask_dir = os.path.join(project_root, 'dataset/Pohang-Canal/validity_masks')
    output_vis_dir = os.path.join(project_root, 'dataset/Pohang-Canal/validity_masks_vis')

    calib_dir = os.path.join(dataset_base, 'calibration')

    print("=" * 60)
    print("LiDAR Validity Mask Generation")
    print("=" * 60)
    print(f"IR Timestamps: {ir_timestamp_path}")
    print(f"LiDAR Timestamps: {lidar_timestamp_path}")
    print(f"Output Mask Dir: {output_mask_dir}")
    print(f"Output Vis Dir: {output_vis_dir}")
    print("=" * 60)

    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)

    # --- Load Calibration ---
    print("Loading calibration...")
    intrinsics_path = os.path.join(calib_dir, 'intrinsics.json')
    extrinsics_path = os.path.join(calib_dir, 'extrinsics.json')

    if not os.path.exists(intrinsics_path) or not os.path.exists(extrinsics_path):
        print("Error: Calibration files not found.")
        return

    with open(intrinsics_path) as f:
        intrinsics_data = json.load(f)
        ir_intrinsics = intrinsics_data['infrared']

    with open(extrinsics_path) as f:
        extrinsics_data = json.load(f)
        ir_extrinsics = extrinsics_data['infrared']
        lidar_extrinsics = extrinsics_data['lidar_front']

    # Construct Camera Matrix (K)
    fx = ir_intrinsics['focal_length']
    fy = ir_intrinsics['focal_length']
    cx = ir_intrinsics['cc_x']
    cy = ir_intrinsics['cc_y']

    K_cam = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # Construct Transformation Matrices
    T_cam_to_base = get_transform_matrix(ir_extrinsics)
    T_lidar_to_base = get_transform_matrix(lidar_extrinsics)
    T_cam_to_lidar = np.linalg.inv(T_cam_to_base) @ T_lidar_to_base

    print(f"Camera Matrix K:\n{K_cam}")
    print(f"T_cam_to_lidar:\n{T_cam_to_lidar}")

    # --- Synchronization ---
    print("\nSynchronizing data...")
    ir_df = load_timestamps(ir_timestamp_path)
    lidar_df = load_timestamps(lidar_timestamp_path)

    if ir_df.empty or lidar_df.empty:
        print("Failed to load timestamps.")
        return

    ir_df = ir_df.sort_values('timestamp')
    lidar_df = lidar_df.sort_values('timestamp')

    ir_df = ir_df.rename(columns={'filename': 'ir_filename'})
    lidar_df = lidar_df.rename(columns={'filename': 'lidar_filename'})

    # Merge: Query=IR, Key=LiDAR
    merged_df = pd.merge_asof(
        ir_df,
        lidar_df,
        on='timestamp',
        direction='nearest',
        tolerance=0.2
    )

    merged_df = merged_df.dropna(subset=['lidar_filename'])

    print(f"Synchronization complete. Found {len(merged_df)} matches out of {len(ir_df)} IR frames.")

    # --- Processing Loop ---
    process_count = 0
    success_count = 0

    local_image_names = set(os.listdir(local_images_dir))
    print(f"Found {len(local_image_names)} local images in {local_images_dir}")
    print("\nGenerating validity masks...")
    print("=" * 60)

    for idx, row in merged_df.iterrows():
        ir_fname = str(row['ir_filename'])
        lidar_fname = str(row['lidar_filename'])

        if not ir_fname.endswith('.png'): ir_fname += '.png'
        if not lidar_fname.endswith('.bin'): lidar_fname += '.bin'

        # Filter: Only process if IR image exists locally
        if ir_fname not in local_image_names:
            continue

        # Construct paths
        ir_path = os.path.join(local_images_dir, ir_fname)
        lidar_path = os.path.join(local_lidar_dir, lidar_fname)

        # Fallback to original dataset
        if not os.path.exists(lidar_path):
             lidar_path = os.path.join(dataset_base, 'lidar_front/points', lidar_fname)

        if not os.path.exists(ir_path):
            continue

        if not os.path.exists(lidar_path):
            continue

        if process_count % 100 == 0:
            print(f"Processing Frame {process_count}: IR={ir_fname} <-> LiDAR={lidar_fname}")

        # Load Data
        img = cv2.imread(ir_path)
        if img is None:
            continue

        try:
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        except Exception as e:
            print(f"Failed to read LiDAR: {lidar_path} ({e})")
            continue

        # Filter Points
        points_filtered = filter_lidar_points(points)

        if points_filtered.shape[0] == 0:
            continue

        # Project
        pixels, depths, valid_indices = project_lidar_to_image(
            points_filtered, K_cam, T_cam_to_lidar, img.shape
        )

        if len(pixels) == 0:
            continue

        # Generate Validity Mask
        validity_mask = generate_validity_mask(pixels, img.shape, blur_kernel_size=21)

        # Save Mask (as PNG, 0-255 range)
        mask_fname = ir_fname  # Same filename as IR image
        mask_path = os.path.join(output_mask_dir, mask_fname)
        mask_uint8 = (validity_mask * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_uint8)

        # Visualization: Save every 100 frames
        if process_count % 100 == 0:
            vis_img = create_visualization(img, validity_mask, alpha=0.4)
            vis_path = os.path.join(output_vis_dir, mask_fname)
            cv2.imwrite(vis_path, vis_img)
            print(f"  -> Saved visualization: {mask_fname}")

        success_count += 1
        process_count += 1

    print("=" * 60)
    print(f"Done. Processed {process_count} frames, successfully generated {success_count} masks.")
    print(f"Masks saved to: {output_mask_dir}")
    print(f"Visualizations saved to: {output_vis_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
