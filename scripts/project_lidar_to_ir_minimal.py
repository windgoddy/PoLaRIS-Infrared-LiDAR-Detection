import os
import sys
import json
import cv2
import numpy as np
import glob
import matplotlib.cm

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
    
    viridis_colormap = matplotlib.cm.get_cmap('viridis')
    rgba_colors = viridis_colormap(normalized_depths)
    
    rgb_colors = rgba_colors[:, :3]
    bgr_colors = rgb_colors[:, ::-1] # RGB -> BGR
    bgr_colors_255 = (bgr_colors * 255).astype(np.uint8)
    
    return bgr_colors_255

def main():
    # --- Configuration ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input/Output Paths
    images_dir = os.path.join(project_root, 'dataset/Pohang-Canal/images')
    lidar_dir = os.path.join(project_root, 'dataset/Pohang-Canal/LiDAR')
    output_dir = os.path.join(project_root, 'dataset/Pohang-Canal/vis_lidar_on_ir')
    
    # Calibration Path (Hardcoded as per user request)
    calib_dir = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/calibration'
    
    print(f"Images Dir: {images_dir}")
    print(f"LiDAR Dir: {lidar_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Calibration Dir: {calib_dir}")
    
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
        if 'infrared' not in intrinsics_data:
            print("Error: 'infrared' key not found in intrinsics.json")
            return
        ir_intrinsics = intrinsics_data['infrared']

    with open(extrinsics_path) as f:
        extrinsics_data = json.load(f)
        if 'infrared' not in extrinsics_data or 'lidar_front' not in extrinsics_data:
            print("Error: 'infrared' or 'lidar_front' not found in extrinsics.json")
            return
        ir_extrinsics = extrinsics_data['infrared']
        lidar_extrinsics = extrinsics_data['lidar_front']

    # Construct Camera Matrix (K)
    fx = ir_intrinsics['focal_length']
    fy = ir_intrinsics['focal_length'] # Assuming square pixels if only one focal length provided
    cx = ir_intrinsics['cc_x']
    cy = ir_intrinsics['cc_y']
    
    K_cam = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print(f"Camera Matrix K:\n{K_cam}")
    
    # Construct Transformation Matrices
    T_cam_to_base = get_transform_matrix(ir_extrinsics)
    T_lidar_to_base = get_transform_matrix(lidar_extrinsics)
    
    # T_cam_to_lidar = inv(T_cam_to_base) @ T_lidar_to_base
    # This transforms a point from LiDAR frame to Camera frame
    T_cam_to_lidar = np.linalg.inv(T_cam_to_base) @ T_lidar_to_base
    
    print(f"T_cam_to_lidar:\n{T_cam_to_lidar}")
    
    # --- Process Files ---
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    print(f"Found {len(image_files)} images.")
    
    # Process a subset for verification (e.g., first 20)
    # Remove the slice [:20] to process all
    process_count = 0
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        lidar_name = os.path.splitext(img_name)[0] + '.bin'
        lidar_path = os.path.join(lidar_dir, lidar_name)
        
        if not os.path.exists(lidar_path):
            # print(f"Skipping {img_name}: LiDAR file not found.")
            continue
            
        if process_count % 100 == 0:
            print(f"Processing {img_name}...")
        
        # Load Image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image: {img_path}")
            continue
            
        # Load LiDAR
        try:
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        except Exception as e:
            print(f"Error reading LiDAR {lidar_path}: {e}")
            continue
            
        # Project
        pixels, depths = project_lidar_to_image(points, K_cam, T_cam_to_lidar, img.shape)
        
        if len(pixels) > 0:
            # Colorize
            colors = color_points_by_depth(depths)
            
            # Draw
            vis_img = img.copy()
            for (u, v), color in zip(pixels, colors):
                cv2.circle(vis_img, (u, v), 1, (int(color[0]), int(color[1]), int(color[2])), -1)
                
            # Save
            cv2.imwrite(os.path.join(output_dir, img_name), vis_img)
        else:
            print(f"Warning: No points projected for {img_name}")
            
        process_count += 1
        if process_count >= 20: # Limit to 20 for the "minimal runnable" request to be quick
             break

    print(f"Done. Processed {process_count} frames. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
