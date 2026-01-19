import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# ================= 配置 =================
# 目标文件夹 (会被覆盖！)
TARGET_DIR = 'dataset/Pohang-Canal-all/images'
# =======================================

def convert_and_overwrite(img_path):
    # 1. 读取 16位图像
    img_16 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if img_16 is None:
        print(f"⚠️ 无法读取: {img_path}")
        return
    
    # 检查是否已经是 8位 (避免重复处理)
    if img_16.dtype == np.uint8:
        # print(f"跳过: {img_path} 已经是 uint8")
        return

    # 2. 归一化逻辑
    min_val = np.min(img_16)
    max_val = np.max(img_16)
    
    if max_val == min_val:
        img_8 = np.zeros(img_16.shape, dtype=np.uint8)
    else:
        # 线性拉伸
        img_norm = (img_16.astype(np.float32) - min_val) / (max_val - min_val)
        img_8 = (img_norm * 255).astype(np.uint8)
        
    # 3. 覆盖保存
    # cv2.imwrite 默认保存为 8-bit png
    cv2.imwrite(img_path, img_8)

def main():
    print(f"🚀 开始批量处理: {TARGET_DIR}")
    
    # 获取所有 png 图片
    images = glob.glob(os.path.join(TARGET_DIR, "*.png"))
    print(f"发现 {len(images)} 张图片")
    
    if len(images) == 0:
        print("未找到图片，请检查路径。")
        return

    # 使用 tqdm 显示进度条
    for img_path in tqdm(images):
        convert_and_overwrite(img_path)
        
    print("\n✅ 所有图片处理完成！已全部转换为 8-bit 可视化格式。")

if __name__ == '__main__':
    main()