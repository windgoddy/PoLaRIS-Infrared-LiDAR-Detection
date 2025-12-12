import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# 导入模型和数据加载器
from model.model_Phase3 import MS_CAFNet
from model.utils import TestSetLoader, load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Phase 3 Visualization')
    parser.add_argument('--dataset_dir', type=str, default='dataset/Pohang-Canal', help='Dataset root')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained .pth.tar file')
    parser.add_argument('--save_dir', type=str, default='vis_phase3_results', help='Directory to save results')
    parser.add_argument('--num_vis', type=int, default=50, help='Number of images to visualize')
    return parser.parse_args()

def apply_heatmap(img, mask, colormap=cv2.COLORMAP_JET):
    """将置信度图(Mask)作为热力图叠加到原图(Img)上"""
    # 确保 mask 是 uint8 格式
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # 生成热力图
    heatmap = cv2.applyColorMap(mask, colormap)
    
    # 确保 img 是 3 通道 BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 叠加
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

def main():
    args = parse_args()
    
    # 1. 准备目录
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[Info] Results will be saved to: {args.save_dir}")

    # 2. 加载数据
    # 读取 test.txt
    _, val_img_ids, _ = load_dataset('dataset', 'Pohang-Canal', '50_50')
    
    # 初始化 DataLoader (in_channels=2 会自动读取 IR + Depth)
    testset = TestSetLoader(
        args.dataset_dir, 
        img_id=val_img_ids, 
        base_size=256, 
        crop_size=256, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ]),
        suffix='.png', 
        in_channels=2 # 关键：双通道输入
    )
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    # 3. 加载模型
    print(f"[Info] Loading model MS_CAFNet...")
    model = MS_CAFNet(num_classes=1, input_channels=2).cuda()
    
    # 加载权重
    if os.path.isfile(args.weights):
        print(f"[Info] Loading weights from: {args.weights}")
        checkpoint = torch.load(args.weights)
        # 兼容性处理：如果权重里包含 'state_dict' 键，则提取它；否则直接加载
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    else:
        print(f"[Error] Weights file not found: {args.weights}")
        return

    model.eval()

    # 4. 推理与可视化
    print(f"[Info] Visualizing first {args.num_vis} images...")
    
    with torch.no_grad():
        for i, (input_tensor, gt_mask) in enumerate(tqdm(test_loader)):
            if i >= args.num_vis:
                break
            
            img_id = val_img_ids[i]
            input_tensor = input_tensor.cuda()
            
            # 前向传播
            # MS_CAFNet 返回 (output, pred_conf)
            output, pred_conf = model(input_tensor)
            
            # --- 后处理数据 ---
            
            # A. 原始红外图 (从 Input 的第0通道提取)
            # 反归一化并转为 numpy (0-255)
            ir_img = input_tensor[0, 0, :, :].cpu().numpy()
            # 简单的 min-max 归一化用于显示
            ir_img = (ir_img - ir_img.min()) / (ir_img.max() - ir_img.min() + 1e-5)
            ir_img = (ir_img * 255).astype(np.uint8)
            
            # B. GT Mask
            gt_img = gt_mask[0, 0, :, :].numpy()
            gt_img = (gt_img * 255).astype(np.uint8)
            
            # C. 预测分割结果 (Prediction)
            pred_map = torch.sigmoid(output[0, 0, :, :]).cpu().numpy()
            pred_img = (pred_map > 0.5).astype(np.uint8) * 255
            
            # D. 置信度图 (Confidence Map) - 您的核心创新！
            # ConfidenceNet 输出本身已经是 Sigmoid 过的 (0-1)，或者如果最后一层没加Sigmoid这里加
            # 查看 model_Phase3 代码，ConfidenceNet 最后一层是 Sigmoid，所以直接取值
            conf_map = pred_conf[0, 0, :, :].cpu().numpy()
            # 转为热力图可视化 (红=高置信度, 蓝=低置信度)
            conf_vis = apply_heatmap(ir_img, conf_map)

            # --- 拼接结果图 ---
            # 格式: [红外原图] | [GT] | [预测分割] | [置信度热力图]
            
            # 转为 3 通道以便拼接
            ir_bgr = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
            gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
            pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
            
            # 在图片上写字
            cv2.putText(ir_bgr, "Input IR", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(gt_bgr, "GT Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(pred_bgr, "Prediction", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(conf_vis, "Confidence Map", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 水平拼接
            row1 = np.hstack([ir_bgr, gt_bgr])
            row2 = np.hstack([pred_bgr, conf_vis])
            final_vis = np.vstack([row1, row2]) # 2x2 网格布局
            
            # 保存
            save_path = os.path.join(args.save_dir, f"{img_id}_vis.png")
            cv2.imwrite(save_path, final_vis)

    print(f"[Success] Visualization done! Check folder: {args.save_dir}")

if __name__ == "__main__":
    main()