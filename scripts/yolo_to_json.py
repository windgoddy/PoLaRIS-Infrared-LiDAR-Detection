import os, json, cv2
from glob import glob

def convert_yolo_to_json(img_dir, label_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_files = glob(os.path.join(img_dir, '*.png'))
    
    for img_p in img_files:
        img_id = os.path.basename(img_p).replace('.png', '')
        img = cv2.imread(img_p)
        h, w = img.shape[:2]
        
        shapes = []
        txt_p = os.path.join(label_dir, img_id + '.txt')
        if os.path.exists(txt_p):
            with open(txt_p, 'r') as f:
                for line in f:
                    cls, x, y, bw, bh = map(float, line.split())
                    if int(cls) != 0: continue # 只处理船只
                    
                    # 还原坐标并处理边缘越界 (Clipping)
                    x1 = max(0, (x - bw/2) * w)
                    y1 = max(0, (y - bh/2) * h)
                    x2 = min(w, (x + bw/2) * w)
                    y2 = min(h, (y + bh/2) * h)
                    
                    shapes.append({
                        "label": "boat",
                        "points": [[x1, y1], [x2, y2]],
                        "shape_type": "rectangle"
                    })
        
        data = {
            "imagePath": img_id + ".png",
            "imageData": None,
            "imageHeight": h, "imageWidth": w,
            "shapes": shapes
        }
        with open(os.path.join(out_dir, img_id + '.json'), 'w') as f:
            json.dump(data, f, indent=2)

# 执行
convert_yolo_to_json('dataset/Pohang-Canal/golden_set/images', 'dataset/Pohang-Canal/golden_set/labels_yolo', 'dataset/Pohang-Canal/golden_set/json_prep')