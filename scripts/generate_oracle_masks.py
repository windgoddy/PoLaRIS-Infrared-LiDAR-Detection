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

    # --- 自适应矩形分解 (Adaptive Rectangular Decomposition) ---
    # 作用:解决不规则形状(L型、U型)的重心偏移问题
    'use_rect_decomposition': True,  # 是否启用矩形分解算法
    'max_rect_iterations': 5,        # 最多分解为几个矩形
    'min_rect_area': 10,             # 忽略小于此面积的矩形碎片
    'rect_gradient_power': 0.5,      # 每个子矩形的渐变幂指数
    'gaussian_blur_size': 15,         # 矩形拼接后的高斯模糊核大小

    # 4. 背景抑制
    'bg_neighbor_radius': 2,      # 定义邻域半径 (2对应 5x5 窗口)
    'bg_min_neighbors': 2,        # 邻域内至少要有几个邻居才算有效背景
    'bg_suppression_radius': 5,   # 背景点在 Mask 上挖洞的半径 (像素)

    # 5. 纯视觉目标（无点云）的软标签机制
    'visual_max_confidence': 0.6,  # 无点云框的最大置信度（建议0.5-0.7）

    # 6. 纹理平滑度过滤（框内区域细化）
    'gradient_threshold': 8,     # 纹理梯度阈值（8-bit: 3-10, 16-bit: 100-500）
    'texture_dilation_kernel': 4,  # 形态学膨胀核大小（连接碎片纹理）
    'min_refined_area': 20,        # 细化后的最小有效面积（像素数）
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

def get_largest_rectangle_in_mask(mask):
    """
    在二值掩码中找到最大的内切矩形
    使用直方图最大矩形法 (Largest Rectangle in Histogram)
    
    Args:
        mask: 二值掩码 (numpy array, 值为0或1)
    
    Returns:
        (x, y, w, h): 矩形坐标和尺寸,如果未找到则返回 (0, 0, 0, 0)
    """
    if mask.sum() == 0:
        return (0, 0, 0, 0)
    
    h, w = mask.shape
    max_area = 0
    best_rect = (0, 0, 0, 0)
    
    # 构建累积直方图
    heights = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            if mask[i, j] > 0:
                heights[i, j] = heights[i-1, j] + 1 if i > 0 else 1
    
    # 对每一行应用直方图最大矩形算法
    for i in range(h):
        hist = heights[i, :]
        area, rect_info = _largest_rectangle_in_histogram(hist)
        if area > max_area:
            max_area = area
            x, width = rect_info
            # 计算矩形的顶部y坐标
            rect_height = hist[x]
            y = i - rect_height + 1
            best_rect = (x, y, width, rect_height)
    
    return best_rect

def _largest_rectangle_in_histogram(heights):
    """
    直方图中的最大矩形面积(单调栈算法)
    
    Args:
        heights: 一维数组,表示直方图的高度
    
    Returns:
        (max_area, (x, width)): 最大面积和矩形的起始位置及宽度
    """
    stack = []
    max_area = 0
    best_pos = (0, 0)
    
    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            area = height * (i - idx)
            if area > max_area:
                max_area = area
                best_pos = (idx, i - idx)
            start = idx
        stack.append((start, h))
    
    # 处理栈中剩余元素
    for idx, height in stack:
        area = height * (len(heights) - idx)
        if area > max_area:
            max_area = area
            best_pos = (idx, len(heights) - idx)
    
    return max_area, best_pos

def generate_gradient_for_rect(rect_shape, gradient_power=0.5):
    """
    为单个矩形生成从中心到边缘的渐变热力图
    
    Args:
        rect_shape: (h, w) 矩形的高度和宽度
        gradient_power: 幂函数指数,控制渐变平滑度
    
    Returns:
        gradient_map: 渐变热力图,范围 [0, 1]
    """
    h, w = rect_shape
    
    # 创建坐标网格
    yy, xx = np.ogrid[:h, :w]
    
    # 计算到各边的距离
    dist_to_left = xx
    dist_to_right = w - 1 - xx
    dist_to_top = yy
    dist_to_bottom = h - 1 - yy
    
    # 到最近边缘的距离
    dist_to_edge = np.minimum(np.minimum(dist_to_left, dist_to_right),
                              np.minimum(dist_to_top, dist_to_bottom)).astype(np.float32)
    
    # 归一化并应用幂函数
    if dist_to_edge.max() > 0:
        normalized = dist_to_edge / dist_to_edge.max()
        gradient_map = np.power(normalized, gradient_power)
    else:
        gradient_map = np.ones((h, w), dtype=np.float32)
    
    return gradient_map

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
    depths = pts_cam[valid_uv, 2]
    
    pixels = np.column_stack([u.astype(int), v.astype(int)])
    return pixels, depths

def generate_hybrid_confidence_map(shape, boxes, lidar_pixels, gt_mask, image, verbose=False):
    """
    生成混合置信度热力图 - 严格按照渐变规则 + 软标签机制
    规则：
    1. 有点云的框：点云区域=1.0，到框边缘逐渐→0
    2. 无点云的框：
       - 先进行纹理检查，过滤平滑区域（天空/海面）
       - 通过纹理检查后，从框中心=visual_max_confidence（如0.6），到框边缘逐渐→0

    Args:
        shape: 图像尺寸 (H, W)
        boxes: 标注框列表 [(x, y, w, h), ...]
        lidar_pixels: 投影到图像上的点云坐标 [(u, v), ...]
        gt_mask: Ground Truth 掩码
        image: 红外图像（原始位深），用于纹理分析
        verbose: 是否输出调试信息
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    H, W = shape

    # 确保 image 是灰度图（用于计算纹理标准差）
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    
    # 检测图像位深，为16-bit数据调整阈值
    is_16bit = image_gray.dtype == np.uint16
    bit_scale_factor = 64.0 if is_16bit else 1.0  # 16-bit时放大阈值
    
    if verbose and is_16bit:
        print(f"  ℹ️  检测到16-bit图像，纹理阈值放大系数: {bit_scale_factor}x")

    for idx, (x, y, w, h) in enumerate(boxes, 1):
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

        # 调试输出
        if verbose:
            print(f"  标注框 {idx}: 位置=({x1},{y1})-({x2},{y2}), 尺寸={roi_w}×{roi_h}, 点云数={len(box_pts)}, 状态={'✅有点云' if has_lidar else '❌无点云'}")

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
            # 【新增】框内区域细化 (ROI Refinement)
            # 目标：去除框内的平滑背景（天空/海面），只在有纹理的物体核心区域生成渐变

            # 步骤1: 提取红外图像ROI并计算纹理梯度图
            roi_ir = image_gray[y1:y2, x1:x2]

            # 使用 Laplacian 算子计算梯度（对边缘和纹理敏感，对平滑区域响应为0）
            # 根据数据类型选择正确的深度参数
            if roi_ir.dtype == np.uint16:
                # 16-bit 图像，直接使用 CV_64F 避免溢出
                laplacian = cv2.Laplacian(roi_ir, cv2.CV_64F)
            else:
                # 8-bit 图像，也使用 CV_64F 以保持一致性
                laplacian = cv2.Laplacian(roi_ir, cv2.CV_64F)
            grad_map = np.abs(laplacian)

            if verbose:
                print(f"    → 梯度计算: 最大值={grad_map.max():.2f}, 平均值={grad_map.mean():.2f}")

            # 步骤2: 二值化得到纹理掩码（有纹理=1，平滑=0）
            # 自适应阈值：16-bit图像使用放大后的阈值
            adaptive_threshold = CONFIG['gradient_threshold'] * bit_scale_factor
            
            if verbose:
                print(f"    → 自适应阈值={adaptive_threshold:.2f} (原始阈值={CONFIG['gradient_threshold']} × 缩放={bit_scale_factor})")
            
            _, texture_mask = cv2.threshold(
                grad_map,
                adaptive_threshold,
                255,
                cv2.THRESH_BINARY
            )
            texture_mask = texture_mask.astype(np.uint8)

            if verbose:
                print(f"    → 二值化阈值={CONFIG['gradient_threshold']}, 纹理像素数={np.sum(texture_mask > 0)}")

            # 步骤3: 形态学膨胀，连接碎片纹理并填补物体内部空洞
            kernel_size = CONFIG['texture_dilation_kernel']
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            texture_mask = cv2.dilate(texture_mask, kernel, iterations=1)

            if verbose:
                print(f"    → 形态学膨胀后: 纹理像素数={np.sum(texture_mask > 0)}")

            # 步骤4: 将纹理掩码转为布尔型，并与GT mask求交集
            # 这样可以精确地只保留"既在GT内，又有纹理"的区域
            texture_mask_bool = texture_mask > 0
            refined_mask = roi_gt_bool & texture_mask_bool  # 交集

            refined_area = np.sum(refined_mask)

            if verbose:
                print(f"    → 纹理掩码与GT交集: 有效像素数={refined_area}")

            # 步骤5: 检查细化后的区域是否太小（可能是噪点）
            if refined_area < CONFIG['min_refined_area']:
                if verbose:
                    print(f"    → ⚠️ 细化后面积{refined_area}<阈值{CONFIG['min_refined_area']}，判定为噪点，跳过该框")
                continue

            if not np.any(refined_mask):
                # 全是平滑背景，跳过该框
                if verbose:
                    print(f"    → ⚠️ 未检测到有效纹理区域，判定为纯背景（天空/海面），跳过该框")
                continue

            if verbose:
                print(f"    → ✅ 区域细化完成，在细化区域内生成渐变（最大置信度={CONFIG['visual_max_confidence']}）")

            # 步骤6: 在整个ROI内生成渐变，但最后用refined_mask精确裁剪
            # 初始化整个ROI的置信度图（全0）
            conf_roi = np.zeros((roi_h, roi_w), dtype=np.float32)

            # 根据配置选择生成策略（在整个ROI内操作，但用refined_mask作为目标区域）
            if CONFIG['use_rect_decomposition']:
                # 策略1: 自适应矩形分解（在refined_mask区域内操作）
                if verbose:
                    print(f"    → 使用矩形分解算法（基于纹理掩码）")

                # 使用refined_mask作为输入
                temp_mask = refined_mask.astype(np.uint8)

                # 迭代分解
                for iter_idx in range(CONFIG['max_rect_iterations']):
                    # 1. 找到当前最大的内切矩形
                    rx, ry, rw, rh = get_largest_rectangle_in_mask(temp_mask)
                    rect_area = rw * rh

                    if rect_area < CONFIG['min_rect_area']:
                        if verbose:
                            print(f"       迭代{iter_idx+1}: 矩形面积{rect_area}<阈值，停止")
                        break

                    if verbose:
                        print(f"       迭代{iter_idx+1}: 发现矩形 ({rx},{ry}) 尺寸={rw}×{rh}")

                    # 2. 为该矩形生成局部渐变
                    local_gradient = generate_gradient_for_rect((rh, rw), CONFIG['rect_gradient_power'])

                    # 3. 融合到全局热力图
                    conf_roi[ry:ry+rh, rx:rx+rw] = np.maximum(
                        conf_roi[ry:ry+rh, rx:rx+rw],
                        local_gradient
                    )

                    # 4. 从临时mask中移除已处理的矩形
                    temp_mask[ry:ry+rh, rx:rx+rw] = 0

                    if temp_mask.sum() == 0:
                        if verbose:
                            print(f"       迭代{iter_idx+1}: 所有区域已处理完毕")
                        break

                # 5. 后处理: 高斯模糊
                if CONFIG['gaussian_blur_size'] > 0:
                    ksize = CONFIG['gaussian_blur_size']
                    if ksize % 2 == 0:
                        ksize += 1
                    conf_roi = cv2.GaussianBlur(conf_roi, (ksize, ksize), 0)

                    # 重新归一化
                    if conf_roi.max() > 0:
                        conf_roi = conf_roi / conf_roi.max()

                if verbose:
                    print(f"    → 矩形分解完成")

            else:
                # 策略2: 距离变换（在refined_mask区域内操作）
                if verbose:
                    print(f"    → 使用距离变换算法（基于纹理掩码）")

                # 对refined_mask做距离变换
                dist_transform = cv2.distanceTransform(refined_mask.astype(np.uint8), cv2.DIST_L2, 5)

                if dist_transform.max() > 0 and dist_transform.max() < 10000:
                    # 归一化并应用幂函数
                    normalized = dist_transform / dist_transform.max()
                    conf_roi = np.power(normalized, CONFIG['gradient_power'])

                    if verbose:
                        print(f"    → 距离变换: max={dist_transform.max():.2f}, 幂函数α={CONFIG['gradient_power']}")
                else:
                    # Fallback: 使用到边缘的距离
                    if verbose:
                        print(f"    → 使用边缘距离作为fallback")

                    yy, xx = np.ogrid[:roi_h, :roi_w]
                    dist_to_left = xx
                    dist_to_right = roi_w - 1 - xx
                    dist_to_top = yy
                    dist_to_bottom = roi_h - 1 - yy

                    dist_to_edge = np.minimum(np.minimum(dist_to_left, dist_to_right),
                                             np.minimum(dist_to_top, dist_to_bottom)).astype(np.float32)

                    if dist_to_edge.max() > 0:
                        normalized = dist_to_edge / dist_to_edge.max()
                        conf_roi = np.power(normalized, CONFIG['gradient_power'])
                    else:
                        conf_roi = np.ones((roi_h, roi_w), dtype=np.float32)

                if verbose:
                    print(f"    → 距离变换完成")

            # 【关键】步骤7: 用纹理掩码精确裁剪，去除所有平滑背景（包括侧边）
            # 只保留refined_mask为True的区域
            conf_roi = conf_roi * refined_mask.astype(np.float32)

            if verbose:
                print(f"    → 纹理掩码裁剪后: 非零像素数={np.sum(conf_roi > 0)}, 最大值={conf_roi.max():.2f}")

            # 【新增】应用软标签系数 (Soft Label)
            # 无点云框的置信度上限降到 visual_max_confidence (如0.6)
            # 避免网络过拟合纯视觉目标
            if verbose:
                print(f"    → 应用软标签前: 最大值={conf_roi.max():.2f}")

            conf_roi = conf_roi * CONFIG['visual_max_confidence']

            if verbose:
                print(f"    → 应用软标签后（×{CONFIG['visual_max_confidence']}）: 最大值={conf_roi.max():.2f}")

        # 4. 强制约束在GT区域内
        conf_roi = conf_roi * roi_gt_bool.astype(np.float32)

        if verbose:
            print(f"    → GT裁剪后最大值={conf_roi.max():.2f}, 平均值={conf_roi[roi_gt_bool].mean():.2f}")

        # 5. 叠加到全局热力图
        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], conf_roi)

    return heatmap

# ==========================================
# 核心处理逻辑 (Single Frame)
# ==========================================

def process_frame(ir_path, mask_path, lidar_path, K_cam, T_cam_to_lidar, output_dir, vis_dir):
    fname = os.path.basename(ir_path)
    
    # --- 1. 读取数据 ---
    # 使用 IMREAD_UNCHANGED 读取原始位深（支持16-bit红外图像）
    img = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Failed to load: {ir_path}")
        return
    
    # 保存原始图像用于纹理计算
    img_raw = img.copy()
    
    # 为可视化生成8-bit版本
    if img.ndim == 2:  # 单通道灰度图
        # 归一化到8-bit用于可视化
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:  # 彩色图
        # 如果是16-bit彩色图，也需要归一化
        if img.dtype == np.uint16:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if gt_mask is None:
        print(f"Failed to load: {mask_path}")
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
    # 使用新的混合生成函数，传入 pixels、gt_mask 和 img_raw（保留16-bit信息用于纹理分析）
    confidence_map = generate_hybrid_confidence_map((H, W), boxes, pixels, gt_mask, img_raw)
    
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
    dataset_dir = os.path.join(project_root, 'dataset/select')
    
    images_dir = os.path.join(dataset_dir, 'images')
    masks_dir = os.path.join(dataset_dir, 'masks')
    lidar_roi_dir = os.path.join(dataset_dir, 'lidar_roi')
    
    # 多序列标定文件路径（根据文件名前缀选择）
    calib_base_dir = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset'
    
    output_dir = os.path.join(dataset_dir, 'oracle_masks')
    vis_dir = os.path.join(dataset_dir, 'oracle_vis')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Reading Images from: {images_dir}")
    print(f"Reading LiDAR from: {lidar_roi_dir}")
    print(f"Saving Output to: {output_dir}")
    
    # --- 预加载所有序列的标定文件 ---
    calibrations = {}  # {sequence_id: (K_cam, T_cam_to_lidar)}
    
    for seq_id in ['00', '01', '03']:
        calib_dir = os.path.join(calib_base_dir, seq_id, 'calibration')
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
            
            calibrations[seq_id] = (K_cam, T_cam_to_lidar)
            print(f"✓ Loaded calibration for sequence {seq_id}")
        except Exception as e:
            print(f"✗ Error loading calibration for sequence {seq_id}: {e}")
    
    if not calibrations:
        print("Error: No calibration files loaded!")
        return

    # --- 主循环 ---
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    print(f"Found {len(image_files)} images.")
    
    for fname in tqdm(image_files):
        # 从文件名提取序列ID（如 00_43.png -> 00）
        seq_id = fname.split('_')[0]
        
        if seq_id not in calibrations:
            print(f"Warning: No calibration found for sequence {seq_id}, skipping {fname}")
            continue
        
        K_cam, T_cam_to_lidar = calibrations[seq_id]
        
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