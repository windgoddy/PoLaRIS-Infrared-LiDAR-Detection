import os
import json
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

def json_to_mask(json_dir, out_dir, label_name='boat'):
    """
    将 LabelMe 的 JSON 文件转换为二值掩码图像
    """
    os.makedirs(out_dir, exist_ok=True)
    json_files = glob(os.path.join(json_dir, '*.json'))
    
    print(f"Found {len(json_files)} json files. Converting to masks...")
    
    for json_path in tqdm(json_files):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 获取图像尺寸
        h = data['imageHeight']
        w = data['imageWidth']
        
        # 创建黑色背景
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 遍历所有形状
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            shape_type = shape['shape_type']
            
            # 只处理目标类别 (或者处理所有非背景类别)
            # 这里假设标签是 'boat' 或者 'target' 或者 '0' (YOLO class)
            if label == label_name or label == 'target' or label == '0':
                points = np.array(points, dtype=np.int32)
                
                if shape_type == 'polygon':
                    cv2.fillPoly(mask, [points], 255)
                elif shape_type == 'rectangle':
                    # 如果您保留了矩形框（不建议），也可以转，但建议精标只用 polygon
                    cv2.rectangle(mask, tuple(points[0]), tuple(points[1]), 255, -1)
                elif shape_type == 'circle':
                    # 简单的圆处理
                    center = tuple(points[0])
                    edge = tuple(points[1])
                    radius = int(np.linalg.norm(np.array(center) - np.array(edge)))
                    cv2.circle(mask, center, radius, 255, -1)

        # 保存掩码
        # 文件名保持一致，后缀改为 .png
        filename = os.path.basename(json_path).replace('.json', '.png')
        out_path = os.path.join(out_dir, filename)
        cv2.imwrite(out_path, mask)

if __name__ == "__main__":
    # 配置路径
    # 注意：您的 JSON 文件似乎保存在 labels_yolo 文件夹中
    JSON_DIR = 'dataset/Pohang-Canal/golden_set/labels_yolo'
    MASK_OUT_DIR = 'dataset/Pohang-Canal/golden_set/masks'
    
    # 执行转换
    # 请确保 LabelMe 中使用的标签名称与这里的 label_name 一致
    json_to_mask(JSON_DIR, MASK_OUT_DIR, label_name='boat')
