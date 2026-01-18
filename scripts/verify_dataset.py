"""
验证数据集质量：可视化红外图像、LiDAR投影、Oracle Mask和GT Mask
检查对齐情况和软标签效果
"""

import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# ==========================================
# LiDAR 投影相关函数
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

def project_lidar_to_image(lidar_points, K_cam, T_cam_to_lidar, img_shape):
    """将 LiDAR 点投影到图像平面"""
    if len(lidar_points) == 0:
        return np.zeros((0, 2), dtype=int)

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
        return np.zeros((0, 2), dtype=int)
        
    # 4. 投影
    pts_img_homo = (K_cam @ pts_cam[:, :3].T).T
    u = pts_img_homo[:, 0] / pts_img_homo[:, 2]
    v = pts_img_homo[:, 1] / pts_img_homo[:, 2]
    
    # 5. 图像边界过滤
    H, W = img_shape[:2]
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    u = u[valid_uv]
    v = v[valid_uv]
    
    pixels = np.column_stack([u.astype(int), v.astype(int)])
    return pixels

# ==========================================
# 主验证函数
# ==========================================

def load_calibrations(calib_base_dir):
    """加载所有序列的标定文件"""
    calibrations = {}
    
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
    
    return calibrations

def visualize_sample(fname, dataset_dir, calibrations, save_dir=None):
    """可视化单个样本"""
    # 提取序列ID
    seq_id = fname.split('_')[0]
    
    if seq_id not in calibrations:
        print(f"Warning: No calibration for sequence {seq_id}, skipping {fname}")
        return
    
    K_cam, T_cam_to_lidar = calibrations[seq_id]
    
    # 读取文件
    img_path = os.path.join(dataset_dir, 'images', fname)
    mask_path = os.path.join(dataset_dir, 'masks', fname)
    oracle_mask_path = os.path.join(dataset_dir, 'oracle_masks', fname)
    lidar_path = os.path.join(dataset_dir, 'lidar_roi', fname.replace('.png', '.bin'))
    
    # 1. 读取16-bit红外图像
    img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        print(f"Failed to load image: {img_path}")
        return
    
    # 归一化到0-255用于显示
    if img_raw.dtype == np.uint16:
        img_display = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_display = img_raw.copy()
    
    # 转为RGB用于matplotlib显示
    if len(img_display.shape) == 2:
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    
    # 2. 读取GT Mask
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"Failed to load GT mask: {mask_path}")
        return
    
    # 3. 读取Oracle Mask
    oracle_mask = cv2.imread(oracle_mask_path, cv2.IMREAD_GRAYSCALE)
    if oracle_mask is None:
        print(f"Failed to load Oracle mask: {oracle_mask_path}")
        return
    
    # 4. 读取LiDAR点云
    if os.path.exists(lidar_path):
        try:
            lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        except:
            lidar_points = np.zeros((0, 4), dtype=np.float32)
    else:
        lidar_points = np.zeros((0, 4), dtype=np.float32)
    
    # 5. 投影LiDAR点到图像
    pixels = project_lidar_to_image(lidar_points, K_cam, T_cam_to_lidar, img_rgb.shape)
    
    # ==========================================
    # 可视化
    # ==========================================
    
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig)
    
    # 左图：红外原图 + LiDAR点投影（红色点）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    if len(pixels) > 0:
        ax1.scatter(pixels[:, 0], pixels[:, 1], c='red', s=1, alpha=0.6)
    ax1.set_title(f'{fname}\nInfrared + LiDAR Points ({len(pixels)} pts)', fontsize=10)
    ax1.axis('off')
    
    # 中图：Oracle Mask 热力图
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(oracle_mask, cmap='jet', vmin=0, vmax=255)
    ax2.set_title(f'Oracle Mask\nMax={oracle_mask.max()}, Mean={oracle_mask[oracle_mask>0].mean():.1f}', fontsize=10)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # 右图：Mask叠加在原图上（检查对齐）
    ax3 = fig.add_subplot(gs[0, 2])
    # 将oracle mask转为彩色热力图
    oracle_color = cv2.applyColorMap(oracle_mask, cv2.COLORMAP_JET)
    oracle_color_rgb = cv2.cvtColor(oracle_color, cv2.COLOR_BGR2RGB)
    # 叠加
    overlay = cv2.addWeighted(img_rgb, 0.6, oracle_color_rgb, 0.4, 0)
    # 绘制GT轮廓
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    ax3.imshow(overlay)
    ax3.set_title('Oracle Mask Overlay + GT Contour (green)', fontsize=10)
    ax3.axis('off')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, fname.replace('.png', '_verify.png'))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()
    
    # 打印统计信息
    oracle_max = oracle_mask.max()
    oracle_mean = oracle_mask[oracle_mask > 0].mean() if np.any(oracle_mask > 0) else 0
    print(f"  Oracle Mask Max: {oracle_max} ({oracle_max/255.0:.2f})")
    print(f"  Oracle Mask Mean (non-zero): {oracle_mean:.2f}")
    print(f"  LiDAR Points: {len(pixels)}")
    print(f"  GT Mask Non-zero: {np.sum(gt_mask > 0)}")
    print()

def main():
    # 配置路径
    dataset_dir = os.path.join(project_root, 'dataset/select')
    train_list_path = os.path.join(dataset_dir, 'split_data/train.txt')
    calib_base_dir = '/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset'
    
    # 可视化输出目录（可选）
    save_dir = os.path.join(dataset_dir, 'verification_samples')
    
    print("=" * 60)
    print("Dataset Verification Tool")
    print("=" * 60)
    
    # 加载标定文件
    print("\n[1/3] Loading calibration files...")
    calibrations = load_calibrations(calib_base_dir)
    
    if not calibrations:
        print("Error: No calibration files loaded!")
        return
    
    # 读取训练集列表
    print("\n[2/3] Reading train.txt...")
    if not os.path.exists(train_list_path):
        print(f"Error: {train_list_path} not found!")
        return
    
    with open(train_list_path, 'r') as f:
        train_files = [line.strip() for line in f if line.strip()]
    
    # 确保文件名有 .png 扩展名
    train_files = [f if f.endswith('.png') else f + '.png' for f in train_files]
    
    # 取前10个
    sample_files = train_files[:10]
    print(f"Loaded {len(train_files)} training samples, visualizing first {len(sample_files)}")
    
    # 可视化每个样本
    print("\n[3/3] Visualizing samples...")
    print("-" * 60)
    
    for i, fname in enumerate(sample_files, 1):
        print(f"[{i}/{len(sample_files)}] Processing: {fname}")
        visualize_sample(fname, dataset_dir, calibrations, save_dir)
    
    print("=" * 60)
    print(f"Verification complete! Results saved to: {save_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
