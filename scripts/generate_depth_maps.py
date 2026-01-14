import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================
# 配置参数
# ==========================================
CONFIG = {
    # 深度图保存格式
    'save_format': '.npy', # 使用 numpy 格式保存浮点深度
}

# LiDAR 点云过滤配置（与可视化保持一致）
FILTER_CONFIG = {
    'min_depth': 3.0,        # 最小深度 (m) - 过滤自车
    'max_depth': 150.0,      # 最大深度 (m) - 移除远处噪声
    'min_height': -160.0,    # 最小高度 (m) - 保留地面点
    'max_height': 50.0,      # 最大高度 (m) - 过滤天空伪影
    'use_intensity_filter': True,
    'min_intensity': 5.0,
    'filter_zero_intensity': True,
}

# ==========================================
# 工具函数 (复用)
# ==========================================
def filter_lidar_points(points, config=FILTER_CONFIG):
    """
    过滤 LiDAR 点云以移除噪声和自车

    Args:
        points: (N, 4) numpy array [x, y, z, intensity]
        config: 过滤配置字典

    Returns:
        filtered_points: 过滤后的点云
    """
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
    q = extrinsics['quaternion']
    t = extrinsics['translation']
    x, y, z, w = q
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0: x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
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
    if len(lidar_points) == 0:
        return np.zeros(img_shape, dtype=np.float32)

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
        return np.zeros(img_shape, dtype=np.float32)
        
    # 4. 投影
    pts_img_homo = (K_cam @ pts_cam[:, :3].T).T
    u = pts_img_homo[:, 0] / pts_img_homo[:, 2]
    v = pts_img_homo[:, 1] / pts_img_homo[:, 2]
    
    # 5. 图像边界过滤
    H, W = img_shape
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    u = u[valid_uv].astype(int)
    v = v[valid_uv].astype(int)
    depths = pts_cam[valid_z][valid_uv, 2]
    
    # 6. 生成深度图
    # 注意：如果多个点投影到同一个像素，取最近的还是最远的？
    # 通常取最近的（遮挡关系）。但这里是稀疏点，重叠概率低。
    # 我们使用 min 策略
    depth_map = np.zeros((H, W), dtype=np.float32)
    
    # 简单的赋值 (后面的覆盖前面的)
    # 为了正确处理重叠，可以先排序或者用 grid 逻辑，但对于稀疏点云直接赋值通常足够
    depth_map[v, u] = depths
    
    return depth_map

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, 'dataset/Pohang-Canal-all')
    
    images_dir = os.path.join(dataset_dir, 'images')
    lidar_roi_dir = os.path.join(dataset_dir, 'lidar_roi')
    output_dir = os.path.join(dataset_dir, 'depth_maps')
    
    # 标定文件路径
    calib_dir = os.path.join(dataset_dir, '00/calibration')
    if not os.path.exists(calib_dir):
        calib_dir = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/calibration'
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating Depth Maps...")
    print(f"Input LiDAR: {lidar_roi_dir}")
    print(f"Output: {output_dir}")
    
    # 加载标定
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
        print(f"Error loading calibration: {e}")
        return

    # 处理循环
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    for fname in tqdm(image_files):
        # 确定图像尺寸
        if 'H' not in locals():
            img = cv2.imread(os.path.join(images_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            H, W = img.shape
            
        lidar_fname = fname.replace('.png', '.bin')
        lidar_path = os.path.join(lidar_roi_dir, lidar_fname)
        save_path = os.path.join(output_dir, fname.replace('.png', '.npy'))
        
        if os.path.exists(lidar_path):
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            # 应用点云过滤（与可视化保持一致）
            points_filtered = filter_lidar_points(points, FILTER_CONFIG)
            depth_map = project_lidar_to_image(points_filtered, K_cam, T_cam_to_lidar, (H, W))
        else:
            depth_map = np.zeros((H, W), dtype=np.float32)
            
        np.save(save_path, depth_map)

if __name__ == "__main__":
    main()
