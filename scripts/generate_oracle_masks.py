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

    # 2. 形状先验流 (Shape Stream - 保底置信度)
    # 作用：提供物体存在的"保底"置信度
    # 建议：0.3-0.4。太高会导致全图橙色，掩盖LiDAR特征；太低导致漏检。
    'shape_base_weight': 0.3,

    # 3. LiDAR 点云流 (LiDAR Stream - 物理证据)
    # 作用：提供物理实锤的高置信度
    # 建议：0.7-0.8。与 Shape 叠加后应达到 1.0。
    'lidar_weight': 0.7,

    # LiDAR 形状生成参数
    'point_dilation_radius': 6,   # 适度膨胀，让稀疏点连接
    'morphology_kernel_size': 9,  # 闭运算核，填补空隙
    'lidar_gaussian_scale': 0.25, # 高斯模糊尺度 (ROI尺寸的百分比)

    # 3.5. 渐变控制
    # 作用：控制无点云框的渐变平缓程度
    # gradient_power: 幂指数，越小过渡区域越大
    # - 1.0: 线性渐变
    # - 0.8: 推荐值（过渡区域适中）
    # - 0.5: 平方根渐变（过渡区域更大）
    'gradient_power': 0.8,  # 幂指数：越小过渡区域越大

    # 4. 背景抑制
    'bg_neighbor_radius': 2,      # 定义邻域半径 (2对应 5x5 窗口)
    'bg_min_neighbors': 2,        # 邻域内至少要有几个邻居才算有效背景
    'bg_suppression_radius': 5    # 背景点在 Mask 上挖洞的半径 (像素)
}

# ==========================================
# 核心工具函数
# ==========================================

# LiDAR 点云过滤配置（与训练时保持一致）
LIDAR_FILTER_CONFIG = {
    'min_depth': 3.0,
    'max_depth': 150.0,
    'min_height': -160.0,
    'max_height': 50.0,
    'use_intensity_filter': True,
    'min_intensity': 5.0,
    'filter_zero_intensity': True,
}

def filter_lidar_points(points, config=LIDAR_FILTER_CONFIG):
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

def generate_hybrid_confidence_map(shape, boxes, lidar_pixels, gt_mask):
    """
    生成混合置信度热力图 - 严格按照渐变规则
    规则：
    1. 所有框都有从1→0的渐变
    2. 有点云的框：点云区域=1，到框边缘逐渐→0
    3. 无点云的框：从框中心=1，到框边缘逐渐→0
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    H, W = shape

    for (x, y, w, h) in boxes:
        # 1. 边界处理
        x1, y1 = int(np.clip(x, 0, W)), int(np.clip(y, 0, H))
        x2, y2 = int(np.clip(x + w, 0, W)), int(np.clip(y + h, 0, H))
        roi_w, roi_h = x2 - x1, y2 - y1
        if roi_w <= 0 or roi_h <= 0: continue

        # 2. 提取 ROI GT Mask
        roi_gt_mask = gt_mask[y1:y2, x1:x2]
        roi_gt_bool = roi_gt_mask > 0
        if not np.any(roi_gt_bool): continue

        # 3. 筛选框内点云
        box_pts = np.array([])
        if len(lidar_pixels) > 0:
            in_box = (lidar_pixels[:, 0] >= x1) & (lidar_pixels[:, 0] < x2) & \
                     (lidar_pixels[:, 1] >= y1) & (lidar_pixels[:, 1] < y2)
            box_pts = lidar_pixels[in_box]

        has_lidar = len(box_pts) > 0

        # ===== 分支1: 有点云的框 =====
        if has_lidar:
            # 步骤1: 生成点云掩码（连接稀疏点）
            loc_pts = (box_pts - np.array([x1, y1])).astype(np.int32)
            point_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            
            for pt in loc_pts:
                px = np.clip(pt[0], 0, roi_w - 1)
                py = np.clip(pt[1], 0, roi_h - 1)
                cv2.circle(point_mask, (px, py), CONFIG['point_dilation_radius'], 255, -1)
            
            # 闭运算填补空隙
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (CONFIG['morphology_kernel_size'], CONFIG['morphology_kernel_size']))
            point_mask = cv2.morphologyEx(point_mask, cv2.MORPH_CLOSE, kernel)
            point_mask_bool = point_mask > 0

            # 步骤2: 生成渐变
            conf_roi = np.zeros((roi_h, roi_w), dtype=np.float32)
            
            # 2.1 点云区域直接设为1.0
            conf_roi[point_mask_bool] = 1.0
            
            # 2.2 非点云区域：计算到点云边缘的距离，做反向归一化
            # 创建一个反向掩码（GT内但不在点云掩码内的区域）
            non_lidar_region = roi_gt_bool & (~point_mask_bool)
            
            if np.any(non_lidar_region):
                # 对点云掩码做距离变换，得到到最近点云边缘的距离
                # 注意：这里要对整个ROI做距离变换，而不是只对GT区域
                dist_to_lidar = cv2.distanceTransform((~point_mask_bool).astype(np.uint8), cv2.DIST_L2, 5)
                
                # 对非点云区域，距离越大，置信度越低
                # 归一化：最远距离→0，紧邻点云边缘→接近1
                if dist_to_lidar[non_lidar_region].max() > 0:
                    max_dist = dist_to_lidar[non_lidar_region].max()
                    # 反向映射：距离0→1.0，距离max→0.0
                    conf_roi[non_lidar_region] = 1.0 - (dist_to_lidar[non_lidar_region] / max_dist)

        # ===== 分支2: 无点云的框 =====
        else:
            # 策略：使用GT mask的距离变换生成渐变（形状骨架方案）
            # 优势：自动适应mask的实际形状（圆形、L型、弓形等）
            # 距离变换会找到到最近边缘的距离，形成自然的形状骨架

            # 使用距离变换：计算每个像素到GT mask边缘的距离
            dist_transform = cv2.distanceTransform(roi_gt_mask, cv2.DIST_L2, 5)
            
            # 检查距离变换是否有效（处理GT mask填满bbox的情况）
            # 当GT mask填满整个bbox时，没有边缘，距离变换会返回异常值
            if dist_transform.max() > 0 and dist_transform.max() < 10000:  # 正常范围
                # 归一化并应用幂函数调整渐变曲线
                if dist_transform.max() > 0:
                    # 先归一化到[0,1]：边缘=0，形状骨架=1
                    normalized = dist_transform / dist_transform.max()

                    # 应用幂函数：x^α (α < 1 使渐变更平缓)
                    conf_roi = np.power(normalized, CONFIG['gradient_power'])
                else:
                    conf_roi = np.ones((roi_h, roi_w), dtype=np.float32)
            else:
                # Fallback: GT mask填满bbox，使用bbox边缘距离
                yy, xx = np.ogrid[:roi_h, :roi_w]
                dist_to_left = xx
                dist_to_right = roi_w - 1 - xx
                dist_to_top = yy
                dist_to_bottom = roi_h - 1 - yy
                
                dist_to_bbox_edge = np.minimum(np.minimum(dist_to_left, dist_to_right),
                                               np.minimum(dist_to_top, dist_to_bottom)).astype(np.float32)
                
                if dist_to_bbox_edge.max() > 0:
                    normalized = dist_to_bbox_edge / dist_to_bbox_edge.max()
                    conf_roi = np.power(normalized, CONFIG['gradient_power'])
                else:
                    conf_roi = np.ones((roi_h, roi_w), dtype=np.float32)

        # 4. 强制约束在GT区域内
        conf_roi = conf_roi * roi_gt_bool.astype(np.float32)

        # 5. 叠加到全局热力图
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
    # 2.1 应用 LiDAR 过滤（与训练时保持一致）
    points_filtered = filter_lidar_points(points, LIDAR_FILTER_CONFIG)
    # 2.2 统计离群点去除
    points_clean = remove_outliers(points_filtered)
    # 2.3 投影到图像
    pixels, _ = project_lidar_to_image(points_clean, K_cam, T_cam_to_lidar, img.shape)
    
    # --- 3. 生成高斯置信度图 (Gaussian Confidence Map) ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
    
    boxes = []
    for i in range(1, num_labels): # Skip background
        x, y, w, h, area = stats[i]
        boxes.append([x, y, w, h])
        
    # 生成 float32 的热力图 [0.0, 1.0]
    # 使用新的混合生成函数，传入 pixels 和 gt_mask（用于裁剪L型等非凸形状）
    confidence_map = generate_hybrid_confidence_map((H, W), boxes, pixels, gt_mask)
    
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
    dataset_dir = os.path.join(project_root, 'dataset/Pohang-Canal-all')
    
    images_dir = os.path.join(dataset_dir, 'images')
    masks_dir = os.path.join(dataset_dir, 'masks')
    lidar_roi_dir = os.path.join(dataset_dir, 'lidar_roi')
    
    # 标定文件路径
    calib_dir = os.path.join(dataset_dir, 'calibration')
    
    # 自动回退标定路径逻辑 (注释掉以防止错误读取)
    # if not os.path.exists(calib_dir):
    #     # 尝试备用路径
    #     calib_dir = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/calibration'
    #     if not os.path.exists(calib_dir):
    #         print("Error: Calibration directory not found.")
    #         return

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