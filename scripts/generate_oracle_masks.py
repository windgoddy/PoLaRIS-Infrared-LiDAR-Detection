import os
import sys
import json
import cv2
import numpy as np
from scipy import interpolate
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

# --- Reuse functions from project_lidar_to_ir_minimal.py ---
FILTER_CONFIG = {
    'min_depth': 3.0,
    'max_depth': 150.0,      # Updated from 120.0
    'min_height': -160.0,
    'max_height': 50.0,      # Updated from 10.0
    'use_intensity_filter': True,
    'min_intensity': 5.0,
    'filter_zero_intensity': True,
}

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

# --- Core Fusion Algorithm ---

def generate_mask_from_fusion(lidar_pixels, gt_mask_img, img_shape,
                               blur_kernel_size=21, smooth_window=5):
    """
    Generate oracle validity mask by fusing LiDAR geometry and GT semantics.

    Algorithm: Geometry Envelope Fill
    1. Combine LiDAR projected pixels and GT mask white pixels
    2. Extract upper envelope (topmost point per column)
    3. Interpolate and smooth envelope curve
    4. Fill from envelope to image bottom

    Physical Meaning:
    - If LiDAR detected a point at row v → region below is valid
    - If human annotated a target at row v → region below is valid
    - Union gives us the "farthest confirmed perception boundary"

    Args:
        lidar_pixels: Projected LiDAR pixel coordinates (N x 2, [u, v])
        gt_mask_img: GT mask image (H x W, 0 or 255)
        img_shape: Image shape (H, W) or (H, W, C)
        blur_kernel_size: Gaussian blur kernel size for soft edges
        smooth_window: Moving average window size for envelope smoothing

    Returns:
        oracle_mask: Oracle validity mask (H x W, uint8, 0-255)
    """
    H, W = img_shape[:2]

    # Step A: Union of LiDAR and GT pixels
    # Extract LiDAR pixels
    P_lidar = lidar_pixels  # Already (N x 2, [u, v])

    # Extract GT mask white pixels (255 = annotated target)
    gt_binary = (gt_mask_img > 127).astype(np.uint8)  # Threshold to binary
    P_gt_coords = np.argwhere(gt_binary > 0)  # Returns [row, col] = [v, u]
    P_gt = P_gt_coords[:, [1, 0]]  # Convert to [u, v] format

    # Union: P_union = P_lidar ∪ P_gt
    if len(P_lidar) > 0 and len(P_gt) > 0:
        P_union = np.vstack([P_lidar, P_gt])
    elif len(P_lidar) > 0:
        P_union = P_lidar
    elif len(P_gt) > 0:
        P_union = P_gt
    else:
        # No valid points, return empty mask
        return np.zeros((H, W), dtype=np.uint8)

    # Step B: Extract Upper Envelope
    # For each column u, find the topmost (minimum v) point
    top_boundary = np.full(W, -1, dtype=np.float32)  # -1 = no point in column

    for u, v in P_union:
        if 0 <= u < W:
            if top_boundary[u] == -1:
                top_boundary[u] = v
            else:
                top_boundary[u] = min(top_boundary[u], v)  # Keep topmost

    # Step C: Interpolate and Smooth
    valid_cols = np.where(top_boundary != -1)[0]

    if len(valid_cols) == 0:
        # No valid points
        return np.zeros((H, W), dtype=np.uint8)

    if len(valid_cols) == 1:
        # Only one valid column, extend horizontally
        top_boundary[:] = top_boundary[valid_cols[0]]
    else:
        # Linear interpolation to fill gaps
        valid_boundaries = top_boundary[valid_cols]

        interp_func = interpolate.interp1d(
            valid_cols,
            valid_boundaries,
            kind='linear',
            fill_value='extrapolate'
        )

        all_cols = np.arange(W)
        top_boundary = interp_func(all_cols)

        # Apply 1D smoothing (moving average) for natural envelope
        if smooth_window > 1:
            # Use uniform filter for moving average
            top_boundary = uniform_filter1d(top_boundary, size=smooth_window, mode='nearest')

        # Clamp to valid range [0, H-1]
        top_boundary = np.clip(top_boundary, 0, H - 1)

    # Step D: Fill from envelope to bottom
    # Physical meaning: Everything below the envelope is in perception range
    mask = np.zeros((H, W), dtype=np.uint8)

    for u in range(W):
        v_start = int(np.round(top_boundary[u]))
        mask[v_start:H, u] = 255

    # Post-processing: Gaussian blur for soft gating (confidence decay)
    if blur_kernel_size > 0:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1  # Ensure odd

        mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

    return mask

def main():
    # --- Configuration ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Input paths
    lidar_roi_dir = os.path.join(project_root, 'dataset/Pohang-Canal/lidar_roi')
    gt_mask_dir = '/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal/masks'

    # Calibration (from original dataset)
    dataset_base = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00'
    calib_dir = os.path.join(dataset_base, 'calibration')

    # Output path
    output_mask_dir = os.path.join(project_root, 'dataset/Pohang-Canal/oracle_masks')

    print("=" * 70)
    print("Oracle Validity Mask Generation".center(70))
    print("Fusion: LiDAR Geometry + GT Semantics".center(70))
    print("=" * 70)
    print(f"LiDAR ROI Dir:   {lidar_roi_dir}")
    print(f"GT Mask Dir:     {gt_mask_dir}")
    print(f"Calibration Dir: {calib_dir}")
    print(f"Output Dir:      {output_mask_dir}")
    print("=" * 70)

    os.makedirs(output_mask_dir, exist_ok=True)

    # --- Load Calibration ---
    print("\nLoading calibration...")
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

    # --- Get File Lists ---
    print("\nScanning files...")

    if not os.path.exists(lidar_roi_dir):
        print(f"Error: LiDAR ROI directory not found: {lidar_roi_dir}")
        return

    if not os.path.exists(gt_mask_dir):
        print(f"Error: GT mask directory not found: {gt_mask_dir}")
        return

    lidar_files = sorted([f for f in os.listdir(lidar_roi_dir) if f.endswith('.bin')])
    gt_mask_files = sorted([f for f in os.listdir(gt_mask_dir) if f.endswith('.png')])

    print(f"Found {len(lidar_files)} LiDAR files")
    print(f"Found {len(gt_mask_files)} GT mask files")

    # Match files by filename (assuming same naming convention)
    # Convert lidar filename to image filename: 008881.bin -> 008881.png
    gt_mask_set = set(gt_mask_files)
    common_files = []

    for lidar_fname in lidar_files:
        img_fname = lidar_fname.replace('.bin', '.png')
        if img_fname in gt_mask_set:
            common_files.append((lidar_fname, img_fname))

    print(f"Found {len(common_files)} matching pairs")

    if len(common_files) == 0:
        print("Warning: No matching file pairs found!")
        print("LiDAR files have .bin extension, GT masks have .png extension")
        print("Please verify the file naming convention matches.")
        return

    # --- Processing Loop ---
    print("\nGenerating oracle masks...")
    print("=" * 70)

    success_count = 0
    error_count = 0

    # Determine image shape from first GT mask
    sample_gt = cv2.imread(os.path.join(gt_mask_dir, common_files[0][1]), cv2.IMREAD_GRAYSCALE)
    if sample_gt is None:
        print("Error: Cannot read sample GT mask to determine image size")
        return
    img_shape = sample_gt.shape
    print(f"Image shape: {img_shape[0]}H x {img_shape[1]}W")
    print()

    for lidar_fname, gt_fname in tqdm(common_files, desc="Processing", unit="frame"):
        try:
            # Construct paths
            lidar_path = os.path.join(lidar_roi_dir, lidar_fname)
            gt_path = os.path.join(gt_mask_dir, gt_fname)

            # Load LiDAR points
            if not os.path.exists(lidar_path):
                error_count += 1
                continue

            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

            if points.shape[0] == 0:
                error_count += 1
                continue

            # Load GT mask
            if not os.path.exists(gt_path):
                error_count += 1
                continue

            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                error_count += 1
                continue

            # Project LiDAR to image
            pixels, depths, valid_indices = project_lidar_to_image(
                points, K_cam, T_cam_to_lidar, gt_mask.shape
            )

            # Generate oracle mask (fusion)
            oracle_mask = generate_mask_from_fusion(
                pixels, gt_mask, gt_mask.shape,
                blur_kernel_size=21,
                smooth_window=5
            )

            # Save
            output_path = os.path.join(output_mask_dir, gt_fname)
            cv2.imwrite(output_path, oracle_mask)

            success_count += 1

        except Exception as e:
            tqdm.write(f"Error processing {lidar_fname}: {e}")
            error_count += 1
            continue

    print()
    print("=" * 70)
    print(f"Processing Complete!")
    print(f"Successfully processed: {success_count}/{len(common_files)}")
    print(f"Errors: {error_count}")
    print(f"Oracle masks saved to: {output_mask_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
