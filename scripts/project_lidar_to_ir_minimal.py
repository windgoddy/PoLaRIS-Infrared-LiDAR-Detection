import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.cm

# --- Filter Configuration ---
FILTER_CONFIG = {
    'min_depth': 3.0,        # Minimum depth (m) - filter ego vehicle
    'max_depth': 120.0,      # Maximum depth (m) - remove distant noise
    'min_height': -160.0,    # Minimum height (m) - keep ground points
    'max_height': 10.0,      # Maximum height (m) - filter sky artifacts
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
    """
    # 1. Extract XYZ
    xyz_points = lidar_points[:, :3]
    
    # 2. Homogeneous coordinates
    ones_column = np.ones((xyz_points.shape[0], 1))
    homogeneous_points = np.hstack([xyz_points, ones_column])
    
    # 3. Apply Extrinsics (LiDAR -> Camera)
    # P_cam = T_cam_to_lidar @ P_lidar.T
    P_cam_transposed = T_cam_to_lidar @ homogeneous_points.T
    P_cam = P_cam_transposed.T
    
    # 4. Filter by Depth (Z > 0.1)
    z_coords = P_cam[:, 2]
    valid_depth_mask = z_coords >= 0.1
    P_cam_filtered = P_cam[valid_depth_mask]
    
    if P_cam_filtered.shape[0] == 0:
        return np.array([]).reshape(0, 2).astype(int), np.array([])
    
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
    
    return pixels, depths

def color_points_by_depth(depths, vmin=1.0, vmax=80.0):
    """
    Colorize points based on depth using viridis colormap.
    """
    if len(depths) == 0:
        return np.array([]).reshape(0, 3)
    
    clipped_depths = np.clip(depths, vmin, vmax)
    normalized_depths = (clipped_depths - vmin) / (vmax - vmin)
    
    # Use matplotlib.colormaps if available (newer matplotlib), else fallback
    try:
        viridis_colormap = matplotlib.colormaps['viridis']
    except AttributeError:
        viridis_colormap = matplotlib.cm.get_cmap('viridis')
        
    rgba_colors = viridis_colormap(normalized_depths)
    
    rgb_colors = rgba_colors[:, :3]
    bgr_colors = rgb_colors[:, ::-1] # RGB -> BGR
    bgr_colors_255 = (bgr_colors * 255).astype(np.uint8)
    
    return bgr_colors_255

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
        # Force filename to be string to preserve leading zeros
        df = pd.read_csv(timestamp_path, sep='\t', header=None, names=['timestamp', 'filename'], dtype={'filename': str})
        if len(df.columns) != 2 or df['timestamp'].dtype == object:
             df = pd.read_csv(timestamp_path, delim_whitespace=True, header=None, names=['timestamp', 'filename'], dtype={'filename': str})
        
        df['timestamp'] = df['timestamp'].astype(float)
        return df
    except Exception as e:
        print(f"Error reading timestamps {timestamp_path}: {e}")
        return pd.DataFrame()

def main():
    # --- Configuration ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths
    # Note: Using the original dataset paths for timestamps and raw data to ensure correctness
    dataset_base = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00'
    
    ir_timestamp_path = os.path.join(dataset_base, 'infrared/timestamp.txt')
    lidar_timestamp_path = os.path.join(dataset_base, 'lidar_front/timestamp.txt')
    
    # We will read images from the user's project folder if they exist there, otherwise from original
    # But for synchronization we need the original filenames/timestamps.
    # The user's project folder: dataset/Pohang-Canal/images
    # The user's project folder: dataset/Pohang-Canal/LiDAR (contains .bin files copied earlier)
    
    local_images_dir = os.path.join(project_root, 'dataset/Pohang-Canal/images')
    local_lidar_dir = os.path.join(project_root, 'dataset/Pohang-Canal/LiDAR')
    output_dir = os.path.join(project_root, 'dataset/Pohang-Canal/vis_lidar_on_ir_synced')
    
    calib_dir = os.path.join(dataset_base, 'calibration')
    
    print(f"IR Timestamps: {ir_timestamp_path}")
    print(f"LiDAR Timestamps: {lidar_timestamp_path}")
    print(f"Output Dir: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    print(f"T_cam_to_lidar:\n{T_cam_to_lidar}")
    
    # --- Synchronization ---
    print("Synchronizing data...")
    ir_df = load_timestamps(ir_timestamp_path)
    lidar_df = load_timestamps(lidar_timestamp_path)
    
    if ir_df.empty or lidar_df.empty:
        print("Failed to load timestamps.")
        return
        
    ir_df = ir_df.sort_values('timestamp')
    lidar_df = lidar_df.sort_values('timestamp')
    
    # Rename columns for merge
    ir_df = ir_df.rename(columns={'filename': 'ir_filename'})
    lidar_df = lidar_df.rename(columns={'filename': 'lidar_filename'})
    
    # Merge: Query=IR, Key=LiDAR
    # We want to find the closest LiDAR frame for each IR frame
    merged_df = pd.merge_asof(
        ir_df,
        lidar_df,
        on='timestamp',
        direction='nearest',
        tolerance=0.2 # 200ms tolerance
    )
    
    # Filter out unmatched rows
    merged_df = merged_df.dropna(subset=['lidar_filename'])
    
    print(f"Synchronization complete. Found {len(merged_df)} matches out of {len(ir_df)} IR frames.")
    
    # --- Processing Loop ---
    process_count = 0
    
    # Get list of local images to filter processing
    local_image_names = set(os.listdir(local_images_dir))
    print(f"Found {len(local_image_names)} local images in {local_images_dir}")
    
    # Iterate over synchronized frames
    for idx, row in merged_df.iterrows():
        ir_fname = str(row['ir_filename'])
        lidar_fname = str(row['lidar_filename'])
        
        # Ensure filenames have extensions if missing in timestamp file
        # Note: timestamp.txt has '008881', we need '008881.png'
        # But wait, if 'ir_filename' is read as float/int by pandas, it might lose leading zeros?
        # Let's force string conversion and padding if needed, but '008881' is string in txt.
        
        if not ir_fname.endswith('.png'): ir_fname += '.png'
        if not lidar_fname.endswith('.bin'): lidar_fname += '.bin'
        
        # Filter: Only process if the IR image exists in our local project folder
        if ir_fname not in local_image_names:
            continue

        # Construct paths
        ir_path = os.path.join(local_images_dir, ir_fname)
        lidar_path = os.path.join(local_lidar_dir, lidar_fname)
        
        # Fallback to original dataset paths if local LiDAR not found
        # (We assume IR must be local based on user request, but LiDAR might need fallback if not copied yet)
        if not os.path.exists(lidar_path):
             lidar_path = os.path.join(dataset_base, 'lidar_front/points', lidar_fname)
             
        if not os.path.exists(ir_path):
            print(f"Error: Local IR image missing despite check: {ir_path}")
            continue
            
        if not os.path.exists(lidar_path):
            # print(f"Skipping {ir_fname}: LiDAR file not found at {lidar_path}")
            continue
            
        if process_count % 100 == 0:
            print(f"Processing Pair {process_count}: IR={ir_fname} <-> LiDAR={lidar_fname}")
            
        # Load Data
        img = cv2.imread(ir_path)
        if img is None: 
            print(f"Failed to read image: {ir_path}")
            continue
            
        try:
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        except Exception as e:
            print(f"Failed to read LiDAR: {lidar_path} ({e})")
            continue
            
        # Filter Points
        points_filtered = filter_lidar_points(points)
        
        if points_filtered.shape[0] == 0:
            print(f"Warning: No points after filtering for {ir_fname}")
            continue
            
        # Project
        pixels, depths = project_lidar_to_image(points_filtered, K_cam, T_cam_to_lidar, img.shape)
        
        if len(pixels) > 0:
            colors = color_points_by_depth(depths)
            vis_img = img.copy()
            for (u, v), color in zip(pixels, colors):
                cv2.circle(vis_img, (u, v), 1, (int(color[0]), int(color[1]), int(color[2])), -1)
            
            cv2.imwrite(os.path.join(output_dir, ir_fname), vis_img)
        else:
            # print(f"No points projected for {ir_fname}")
            pass
        
        process_count += 1
        # Removed limit to process all local images
        # if process_count >= 20:
        #    break
            
    print(f"Done. Processed {process_count} frames. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
