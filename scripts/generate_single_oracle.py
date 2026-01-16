#!/usr/bin/env python3
"""
单张图像 Oracle Mask 生成脚本 (优化版)
优化目标：生成具有平滑梯度、层次分明的置信度图
融合策略：加权求和 (Weighted Sum) 代替 最大值 (Maximum)

使用方法：
python scripts/generate_single_oracle.py --image_id 009043 --dataset dataset/Pohang-Canal-all
"""

import os
import sys
import json
import cv2
import numpy as np
from scipy.spatial import cKDTree
import argparse

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# 1. 核心配置参数 (集中管理)
# ==========================================
CONFIG = {
    # --- 预处理 ---
    'sor_nb_neighbors': 20,
    'sor_std_ratio': 2.0,

    # --- 形状先验流 (Shape Stream) ---
    # 作用：提供物体存在的"保底"置信度
    # 建议：0.3-0.4。太高会导致全图橙色，掩盖LiDAR特征；太低导致漏检。
    'shape_base_weight': 0.3, 

    # --- LiDAR 点云流 (LiDAR Stream) ---
    # 作用：提供物理实锤的高置信度
    # 建议：0.7-0.8。与 Shape 叠加后应达到 1.0。
    'lidar_weight': 0.7,
    
    # LiDAR 形状生成参数
    'point_dilation_radius': 6,   # 适度膨胀，让稀疏点连接
    'morphology_kernel_size': 9,  # 闭运算核，填补空隙
    'lidar_gaussian_scale': 0.25, # 高斯模糊尺度 (ROI尺寸的百分比)

    # --- 渐变控制 ---
    # 作用：控制无点云框的渐变平缓程度
    # gradient_power: 幂指数，越小过渡区域越大
    # - 1.0: 线性渐变
    # - 0.5: 平方根渐变（过渡区域更大，推荐）
    # - 0.3: 更平缓（过渡区域非常大）
    'gradient_power': 0.8,
    # --- 自适应矩形分解 (Adaptive Rectangular Decomposition) ---
    # 作用:解决不规则形状(L型、U型)的重心偏移问题
    'use_rect_decomposition': True,  # 是否启用矩形分解算法
    'max_rect_iterations': 5,        # 最多分解为几个矩形
    'min_rect_area': 10,             # 忽略小于此面积的矩形碎片
    'rect_gradient_power': 0.5,      # 每个子矩形的渐变幂指数
    'gaussian_blur_size': 15,         # 矩形拼接后的高斯模糊核大小
    # --- 背景抑制 ---
    'bg_neighbor_radius': 2,
    'bg_min_neighbors': 2,
    'bg_suppression_radius': 5,

    # --- 纯视觉目标（无点云）的软标签机制 ---
    # 作用：降低无点云支持目标的置信度上限，避免网络过拟合误标
    'visual_max_confidence': 0.6,  # 无点云框的最大置信度（建议0.5-0.7）

    # --- 纹理平滑度过滤（框内区域细化）---
    # 作用：识别并去除框内的平滑区域（天空/海面），只在有纹理的物体核心区域生成渐变
    # gradient_threshold: 梯度阈值，用于区分有纹理区域和平滑区域
    # - 8-bit图像(0-255): 建议 3.0-10.0
    # - 16-bit图像: 建议 100-500
    'gradient_threshold': 8,  # 纹理梯度阈值（Laplacian响应）
    'texture_dilation_kernel': 4,  # 形态学膨胀核大小（连接碎片纹理）
    'min_refined_area': 20,  # 细化后的最小有效面积（像素数）
}

# LiDAR 点云过滤配置
LIDAR_FILTER_CONFIG = {
    'min_depth': 3.0,
    'max_depth': 150.0,
    'min_height': -160.0,
    'max_height': 50.0,
    'use_intensity_filter': True,
    'min_intensity': 5.0,
    'filter_zero_intensity': True,
}

# ==========================================
# 2. 工具函数
# ==========================================

def filter_lidar_points(points, config=LIDAR_FILTER_CONFIG):
    """过滤 LiDAR 点云"""
    if len(points) == 0: return points
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    
    depth = np.sqrt(x**2 + y**2 + z**2)
    depth_mask = (depth >= config['min_depth']) & (depth <= config['max_depth'])
    height_mask = (z >= config['min_height']) & (z <= config['max_height'])
    
    if config['use_intensity_filter']:
        intensity_mask = intensity >= config['min_intensity']
        if config['filter_zero_intensity']:
            intensity_mask = intensity_mask & (intensity > 0)
    else:
        intensity_mask = np.ones(len(points), dtype=bool)

    return points[depth_mask & height_mask & intensity_mask]

def get_transform_matrix(extrinsics):
    """构建变换矩阵"""
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

def remove_outliers(points):
    """统计离群点去除"""
    if len(points) < CONFIG['sor_nb_neighbors'] + 1: return points
    xyz = points[:, :3]
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=CONFIG['sor_nb_neighbors'] + 1)
    mean_dists = np.mean(dists[:, 1:], axis=1)
    threshold = np.mean(mean_dists) + CONFIG['sor_std_ratio'] * np.std(mean_dists)
    return points[mean_dists < threshold]

def project_lidar_to_image(lidar_points, K_cam, T_cam_to_lidar, img_shape):
    """LiDAR 投影"""
    if len(lidar_points) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)
    
    xyz_points = lidar_points[:, :3]
    pts_homo = np.hstack([xyz_points, np.ones((xyz_points.shape[0], 1))])
    pts_cam = (T_cam_to_lidar @ pts_homo.T).T
    
    valid_z = pts_cam[:, 2] >= 0.1
    pts_cam = pts_cam[valid_z]
    if len(pts_cam) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)

    pts_img_homo = (K_cam @ pts_cam[:, :3].T).T
    u = pts_img_homo[:, 0] / pts_img_homo[:, 2]
    v = pts_img_homo[:, 1] / pts_img_homo[:, 2]

    H, W = img_shape[:2]
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    pixels = np.column_stack([u[valid_uv].astype(int), v[valid_uv].astype(int)])
    return pixels, pts_cam[valid_z][valid_uv, 2]

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

# ==========================================
# 3. 核心生成逻辑 (优化版)
# ==========================================

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
        image: 红外图像，用于纹理分析 (需要灰度图或彩色图的单通道)
        verbose: 是否输出调试信息
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    H, W = shape

    # 确保 image 是灰度图（用于计算纹理标准差）
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

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
            laplacian = cv2.Laplacian(roi_ir, cv2.CV_64F)
            grad_map = np.abs(laplacian)

            if verbose:
                print(f"    → 梯度计算: 最大值={grad_map.max():.2f}, 平均值={grad_map.mean():.2f}")

            # 步骤2: 二值化得到纹理掩码（有纹理=1，平滑=0）
            _, texture_mask = cv2.threshold(
                grad_map,
                CONFIG['gradient_threshold'],
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
# 4. 主流程
# ==========================================

def process_single_image(image_id, dataset_dir, output_base_dir=None):
    if output_base_dir is None: output_base_dir = dataset_dir

    # 路径构建
    ir_path = os.path.join(dataset_dir, 'images', f'{image_id}.png')
    mask_path = os.path.join(dataset_dir, 'masks', f'{image_id}.png')
    lidar_path = os.path.join(dataset_dir, 'lidar_roi', f'{image_id}.bin')
    calib_dir = os.path.join(dataset_dir, 'calibration')
    
    output_dir = os.path.join(output_base_dir, 'oracle_masks')
    vis_dir = os.path.join(output_base_dir, 'oracle_vis')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Processing: {image_id} ...")

    # 1. 检查文件
    if not os.path.exists(ir_path) or not os.path.exists(mask_path):
        print("❌ Image or Mask not found.")
        return False

    # 2. 加载数据
    img = cv2.imread(ir_path)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    H, W = gt_mask.shape
    
    # 加载点云 (容错)
    try:
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    except:
        points = np.zeros((0, 4), dtype=np.float32)

    # 加载标定
    try:
        with open(os.path.join(calib_dir, 'intrinsics.json')) as f:
            ir_intrinsics = json.load(f)['infrared']
        with open(os.path.join(calib_dir, 'extrinsics.json')) as f:
            ext = json.load(f)
            K_cam = np.array([[ir_intrinsics['focal_length'], 0, ir_intrinsics['cc_x']], 
                              [0, ir_intrinsics['focal_length'], ir_intrinsics['cc_y']], 
                              [0, 0, 1]], dtype=np.float32)
            T_cam_to_lidar = np.linalg.inv(get_transform_matrix(ext['infrared'])) @ get_transform_matrix(ext['lidar_front'])
    except:
        print("❌ Calibration failed.")
        return False

    # 3. 处理点云
    pts_clean = remove_outliers(filter_lidar_points(points))
    pixels, _ = project_lidar_to_image(pts_clean, K_cam, T_cam_to_lidar, img.shape)

    # 4. 生成 Oracle Mask
    num, _, stats, _ = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
    boxes = [stats[i][:4] for i in range(1, num)] # x,y,w,h

    print(f"\n📊 图像 {image_id} 的标注框点云统计：")
    print(f"总点云数: {len(pixels)}")
    print(f"总标注框数: {len(boxes)}")

    confidence_map = generate_hybrid_confidence_map((H, W), boxes, pixels, gt_mask, img, verbose=True)

    # 5. 背景抑制 (挖洞)
    if len(pixels) > 0:
        bg_pixels = pixels[gt_mask[pixels[:, 1], pixels[:, 0]] == 0]
        if len(bg_pixels) > 0:
            bg_hit_map = np.zeros((H, W), dtype=np.uint8)
            bg_hit_map[bg_pixels[:, 1], bg_pixels[:, 0]] = 1
            kernel = np.ones((5, 5), dtype=np.uint8)
            # 只有当周围点数足够多才视为可靠背景
            valid_bg_y, valid_bg_x = np.where((bg_hit_map == 1) & (cv2.filter2D(bg_hit_map, -1, kernel) >= CONFIG['bg_min_neighbors'] + 1))
            for bx, by in zip(valid_bg_x, valid_bg_y):
                cv2.circle(confidence_map, (bx, by), CONFIG['bg_suppression_radius'], 0.0, -1)

    # 6. 保存与可视化
    cv2.imwrite(os.path.join(output_dir, f'{image_id}.png'), (confidence_map * 255).astype(np.uint8))
    
    vis_img = cv2.addWeighted(img, 0.6, cv2.applyColorMap((confidence_map * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.4, 0)
    cv2.drawContours(vis_img, cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(vis_dir, f'{image_id}.png'), vis_img)
    
    print(f"✅ Done. Vis saved to {os.path.join(vis_dir, f'{image_id}.png')}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_id', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='dataset/Pohang-Canal-all')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, args.dataset)
    process_single_image(args.image_id, dataset_dir, os.path.join(project_root, args.output) if args.output else dataset_dir)

if __name__ == "__main__":
    main()