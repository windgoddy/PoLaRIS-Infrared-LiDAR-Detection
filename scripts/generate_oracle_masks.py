import os
import sys
import json
import cv2
import numpy as np
from scipy import interpolate
from scipy.spatial import cKDTree
from tqdm import tqdm

# ==========================================
# 配置参数 (Configuration)
# ==========================================
CONFIG = {
    # 1. 预处理
    'sor_nb_neighbors': 20,       # 统计滤波邻居数
    'sor_std_ratio': 2.0,         # 统计滤波标准差倍数
    
    # 2. 高斯掩码参数
    'sigma_factor': 3.0,          # 无点时：中心高斯 sigma = w / 3.0
    'lidar_sigma_factor': 4.0,    # 有点时：LiDAR点高斯 sigma = min(w,h) / 4.0 (更紧致)
    'lidar_dilation_radius': 3,   # 有点时：先膨胀点云的半径，使稀疏点连成片
    'box_gaussian_weight': 0.6,   # 混合模式下，GT框中心高斯的权重 (0.0~1.0)
                                  # 0.6 意味着没有点云覆盖的船体区域，置信度最高为 0.6
    
    # 3. 背景过滤
    'bg_neighbor_radius': 2,      # 定义邻域半径 (2对应 5x5 窗口)
    'bg_min_neighbors': 2,        # 邻域内至少要有几个邻居才算有效背景
    'bg_suppression_radius': 5    # 背景点在 Mask 上挖洞的半径 (像素)
}

# ==========================================
# 核心工具函数
# ==========================================

def get_transform_matrix(extrinsics):
    """从外参字典构建 4x4 变换矩阵"""
    q = extrinsics['quaternion']
    t = extrinsics['translation']
    
    # Quaternion to Rotation Matrix
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

def remove_outliers(points):
    """统计离群点去除 (Statistical Outlier Removal)"""
    if len(points) < CONFIG['sor_nb_neighbors'] + 1:
        return points
    
    xyz = points[:, :3]
    tree = cKDTree(xyz)
    # 查询最近的 k 个邻居
    dists, _ = tree.query(xyz, k=CONFIG['sor_nb_neighbors'] + 1)
    # 计算平均距离 (排除自身)
    mean_dists = np.mean(dists[:, 1:], axis=1)
    
    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    
    # 阈值过滤
    threshold = global_mean + CONFIG['sor_std_ratio'] * global_std
    mask = mean_dists < threshold
    return points[mask]

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

def generate_hybrid_confidence_map(shape, boxes, lidar_pixels):
    """
    生成混合置信度热力图
    - 如果框内有 LiDAR 点：以点为中心向外扩散 (Distance Transform + Gaussian)
    - 如果框内无 LiDAR 点：以框中心为中心向外扩散 (Center Gaussian)
    
    shape: (H, W)
    boxes: list of [x, y, w, h]
    lidar_pixels: (N, 2) 全局 LiDAR 投影点
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    H, W = shape
    
    for (x, y, w, h) in boxes:
        # 确保坐标在图像内
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        
        # 实际绘制区域的宽高
        roi_w = x2 - x1
        roi_h = y2 - y1
        
        if roi_w <= 0 or roi_h <= 0:
            continue
            
        # 无论是否有 LiDAR 点，都先生成一个基础的“框中心高斯”
        # 这代表了 GT 框提供的先验知识：框内大概率是船
        # 网格坐标 (相对于 ROI 左上角)
        xx, yy = np.meshgrid(np.arange(roi_w), np.arange(roi_h))
        
        cx = roi_w / 2.0
        cy = roi_h / 2.0
        
        sigma_x = roi_w / CONFIG['sigma_factor']
        sigma_y = roi_h / CONFIG['sigma_factor']
        sigma_x = max(sigma_x, 1.0)
        sigma_y = max(sigma_y, 1.0)
        
        base_gaussian = np.exp(-((xx - cx)**2 / (2 * sigma_x**2) + (yy - cy)**2 / (2 * sigma_y**2)))
        
        # 查找落入该框内的 LiDAR 点
        if len(lidar_pixels) > 0:
            in_box = (lidar_pixels[:, 0] >= x1) & (lidar_pixels[:, 0] < x2) & \
                     (lidar_pixels[:, 1] >= y1) & (lidar_pixels[:, 1] < y2)
            box_pts = lidar_pixels[in_box]
        else:
            box_pts = np.zeros((0, 2))
            
        if len(box_pts) > 0:
            # === 策略 A: 归一化距离场插值 (Normalized Distance Field) ===
            # 目标：从凸包(1.0) 平滑过渡到 框边缘(0.0)
            
            # 1. 构建“源”掩码 (Source Mask) - LiDAR 凸包
            # 255 = 背景, 0 = 源 (凸包内部)
            source_mask = np.ones((roi_h, roi_w), dtype=np.uint8) * 255
            
            # 映射到局部坐标
            loc_pts = box_pts - np.array([x1, y1])
            loc_pts = loc_pts.astype(np.int32)
            loc_pts[:, 0] = np.clip(loc_pts[:, 0], 0, roi_w - 1)
            loc_pts[:, 1] = np.clip(loc_pts[:, 1], 0, roi_h - 1)
            
            # 绘制凸包作为源
            if len(loc_pts) >= 3:
                hull = cv2.convexHull(loc_pts)
                cv2.fillConvexPoly(source_mask, hull, 0)
            elif len(loc_pts) == 2:
                cv2.line(source_mask, tuple(loc_pts[0]), tuple(loc_pts[1]), 0, thickness=CONFIG['lidar_dilation_radius']*2)
            else:
                cv2.circle(source_mask, tuple(loc_pts[0]), CONFIG['lidar_dilation_radius'], 0, -1)
                
            # 2. 构建“汇”掩码 (Sink Mask) - 框边缘
            # 255 = 背景, 0 = 汇 (框边缘)
            sink_mask = np.ones((roi_h, roi_w), dtype=np.uint8) * 255
            # 将边缘设为 0
            sink_mask[0, :] = 0
            sink_mask[-1, :] = 0
            sink_mask[:, 0] = 0
            sink_mask[:, -1] = 0
            
            # 3. 计算距离场
            # d_source: 到最近凸包点的距离
            d_source = cv2.distanceTransform(source_mask, cv2.DIST_L2, 5)
            # d_sink: 到最近框边缘的距离
            d_sink = cv2.distanceTransform(sink_mask, cv2.DIST_L2, 5)
            
            # 4. 归一化插值
            # Value = d_sink / (d_source + d_sink + epsilon)
            # 在凸包内: d_source=0 -> Value=1
            # 在边缘上: d_sink=0 -> Value=0
            denominator = d_source + d_sink + 1e-5
            conf_roi = d_sink / denominator
            
            # 可选：对结果进行高斯平滑，消除距离变换的棱角
            # conf_roi = cv2.GaussianBlur(conf_roi, (5, 5), 0)
            
        else:
            # === 策略 B: 纯框模式 (Box Only) ===
            # 只有基础高斯，满权重
            conf_roi = base_gaussian
        
        # 叠加 (取最大值)
        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], conf_roi)
        
    return heatmap

# ==========================================
# 核心处理逻辑 (Single Frame)
# ==========================================

def process_frame(ir_path, mask_path, lidar_path, K_cam, T_cam_to_lidar, output_dir, vis_dir):
    fname = os.path.basename(ir_path)
    
    # --- 1. 读取数据 ---
    img = cv2.imread(ir_path) # BGR
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or gt_mask is None:
        return
    
    H, W = gt_mask.shape
    
    # 读取清洗后的 LiDAR (float32, N*4)
    if os.path.exists(lidar_path):
        try:
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        except:
            points = np.zeros((0, 4), dtype=np.float32)
    else:
        points = np.zeros((0, 4), dtype=np.float32)
        
    # --- 2. 预处理 ---
    points_clean = remove_outliers(points)
    pixels, _ = project_lidar_to_image(points_clean, K_cam, T_cam_to_lidar, img.shape)
    
    # --- 3. 生成高斯置信度图 (Gaussian Confidence Map) ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
    
    boxes = []
    for i in range(1, num_labels): # Skip background
        x, y, w, h, area = stats[i]
        boxes.append([x, y, w, h])
        
    # 生成 float32 的热力图 [0.0, 1.0]
    # 使用新的混合生成函数，传入 pixels
    confidence_map = generate_hybrid_confidence_map((H, W), boxes, pixels)
    
    # --- 4. 背景级鲁棒抑制 (Background Suppression) ---
    # 利用 LiDAR 点作为“绝对背景”的证据
    # 只有 GT 框外的点才算背景
    
    if len(pixels) > 0:
        # 排除落在 GT 框内的点
        is_bg_pixel = gt_mask[pixels[:, 1], pixels[:, 0]] == 0
        bg_pixels = pixels[is_bg_pixel]
        
        if len(bg_pixels) > 0:
            # 绘制背景 Hit Map
            bg_hit_map = np.zeros((H, W), dtype=np.uint8)
            bg_hit_map[bg_pixels[:, 1], bg_pixels[:, 0]] = 1
            
            # 孤立点剔除: 检查 5x5 邻域
            kernel = np.ones((5, 5), dtype=np.uint8)
            neighbor_count = cv2.filter2D(bg_hit_map, -1, kernel)
            
            # 只有邻居数 >= 设定值的才保留
            valid_bg = neighbor_count >= (CONFIG['bg_min_neighbors'] + 1)
            
            # 提取有效背景点坐标
            valid_bg_y, valid_bg_x = np.where((bg_hit_map == 1) & valid_bg)
            
            # 在置信度图上“挖洞”
            # 对于每个确认的背景点，将其周围强制设为 0
            # 这是一个很强的约束：LiDAR 说这里是背景，那这里一定不能是目标
            if len(valid_bg_x) > 0:
                for bx, by in zip(valid_bg_x, valid_bg_y):
                    cv2.circle(confidence_map, (bx, by), CONFIG['bg_suppression_radius'], 0.0, -1)
            
            # 记录被剔除的点用于可视化 (Blue)
            removed_bg = (bg_hit_map == 1) & (~valid_bg)
    else:
        removed_bg = np.zeros((H, W), dtype=bool)

    # --- 5. 保存结果 ---
    # 将 [0.0, 1.0] 映射到 [0, 255]
    final_mask = (confidence_map * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, fname), final_mask)
    
    # --- 6. 可视化 (Visualization) ---
    if True: 
        vis_img = img.copy()
        
        # 1. 画绿色 GT 框
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 1)
        
        # 2. 画红色 Heatmap (半透明)
        # 使用伪彩色映射更直观
        heatmap_color = cv2.applyColorMap(final_mask, cv2.COLORMAP_JET)
        vis_img = cv2.addWeighted(vis_img, 0.6, heatmap_color, 0.4, 0)
        
        # 3. 画蓝色 被剔除的背景点 (Debug)
        if 'removed_bg' in locals() and np.any(removed_bg):
            vis_img[removed_bg] = [255, 0, 0] # BGR Blue
            
        # 4. 画黄色 确认的背景点 (Debug - 那些挖洞的点)
        # if 'valid_bg_x' in locals() and len(valid_bg_x) > 0:
        #     for bx, by in zip(valid_bg_x, valid_bg_y):
        #         vis_img[by, bx] = [0, 255, 255] # Yellow
            
        cv2.imwrite(os.path.join(vis_dir, fname), vis_img)
        
    # print(f"Frame {fname}: {frame_stats}")

def main():
    # --- 路径配置 (请修改这里) ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, 'dataset/Pohang-Canal')
    
    images_dir = os.path.join(dataset_dir, 'images')
    masks_dir = os.path.join(dataset_dir, 'masks')
    lidar_roi_dir = os.path.join(dataset_dir, 'lidar_roi')
    
    # 标定文件路径
    calib_dir = os.path.join(dataset_dir, '00/calibration')
    
    # 自动回退标定路径逻辑
    if not os.path.exists(calib_dir):
        # 尝试备用路径
        calib_dir = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/calibration'
        if not os.path.exists(calib_dir):
            print("Error: Calibration directory not found.")
            return

    output_dir = os.path.join(dataset_dir, 'oracle_masks')
    vis_dir = os.path.join(dataset_dir, 'oracle_vis')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Reading Images from: {images_dir}")
    print(f"Reading LiDAR from: {lidar_roi_dir}")
    print(f"Saving Output to: {output_dir}")
    
    # --- 加载标定 ---
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

    # --- 主循环 ---
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    print(f"Found {len(image_files)} images.")
    
    for fname in tqdm(image_files):
        ir_path = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, fname)
        
        # 关键修改：直接同名替换
        lidar_fname = fname.replace('.png', '.bin')
        lidar_path = os.path.join(lidar_roi_dir, lidar_fname)
        
        if not os.path.exists(mask_path):
            continue
            
        process_frame(ir_path, mask_path, lidar_path, K_cam, T_cam_to_lidar, output_dir, vis_dir)

if __name__ == "__main__":
    main()