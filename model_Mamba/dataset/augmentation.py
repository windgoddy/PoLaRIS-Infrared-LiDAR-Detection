"""
强化数据增强模块 - 针对红外小目标检测
================================================

参考顶会论文的数据增强策略：
- DNANet (TMM 2023)
- ACM (CVPR 2022)
- UIUNet (TGRS 2022)

关键改进：
1. 垂直翻转 + 90度旋转
2. 多尺度训练（Scale augmentation）
3. Gamma/对比度调整（红外图像关键）
4. 高斯噪声/模糊（提升鲁棒性）

Author: PoLaRIS Team
Date: 2026-02-26
"""

import torch
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms.functional as TF


class InfraredAugmentation:
    """
    专门针对红外小目标检测的数据增强

    参考：DNANet, ACM, UIUNet 的增强策略

    Args:
        mode: 'train' 或 'test'
        use_all: 是否使用全部增强（默认True）
    """
    def __init__(self, mode='train', use_all=True):
        self.mode = mode
        self.use_all = use_all

    def __call__(self, img, mask, depth=None):
        """
        应用数据增强

        Args:
            img: PIL Image (IR图像)
            mask: PIL Image (mask)
            depth: PIL Image (LiDAR深度图，可选)

        Returns:
            img, mask, depth (增强后)
        """
        if self.mode != 'train':
            return img, mask, depth

        # Tier 1: 几何变换（必备）
        img, mask, depth = self.random_horizontal_flip(img, mask, depth)
        img, mask, depth = self.random_vertical_flip(img, mask, depth)
        img, mask, depth = self.random_rotate_90(img, mask, depth)
        img, mask, depth = self.random_scale_crop(img, mask, depth)

        if self.use_all:
            # Tier 2: 颜色/对比度增强（红外关键）
            img = self.random_gamma(img)
            img = self.random_contrast(img)
            img = self.random_brightness(img)

            # Tier 3: 噪声/模糊（鲁棒性）
            img = self.random_gaussian_noise(img)
            img = self.random_blur(img)

        return img, mask, depth

    # ========== 几何变换 ==========

    def random_horizontal_flip(self, img, mask, depth):
        """水平翻转（标准操作）"""
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if depth is not None:
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, depth

    def random_vertical_flip(self, img, mask, depth):
        """垂直翻转（DNANet用了这个！）"""
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if depth is not None:
                depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask, depth

    def random_rotate_90(self, img, mask, depth):
        """
        90度旋转（DNANet关键技巧）

        随机选择 90/180/270 度旋转
        """
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            img = img.rotate(angle, expand=False, resample=Image.BILINEAR)
            mask = mask.rotate(angle, expand=False, resample=Image.NEAREST)
            if depth is not None:
                depth = depth.rotate(angle, expand=False, resample=Image.BILINEAR)
        return img, mask, depth

    def random_scale_crop(self, img, mask, depth, scale_range=(0.8, 1.2)):
        """
        多尺度训练（关键！）

        随机缩放图像，然后裁剪/填充回原始大小
        这强制模型学习多尺度特征
        """
        if random.random() < 0.5:
            w, h = img.size
            scale = random.uniform(*scale_range)
            new_w, new_h = int(w * scale), int(h * scale)

            # 缩放
            img = img.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)
            if depth is not None:
                depth = depth.resize((new_w, new_h), Image.BILINEAR)

            # 裁剪或填充回原始大小
            if scale > 1.0:
                # 放大：随机裁剪
                x = random.randint(0, max(0, new_w - w))
                y = random.randint(0, max(0, new_h - h))
                img = img.crop((x, y, x + w, y + h))
                mask = mask.crop((x, y, x + w, y + h))
                if depth is not None:
                    depth = depth.crop((x, y, x + w, y + h))
            else:
                # 缩小：填充
                img = img.resize((w, h), Image.BILINEAR)
                mask = mask.resize((w, h), Image.NEAREST)
                if depth is not None:
                    depth = depth.resize((w, h), Image.BILINEAR)

        return img, mask, depth

    # ========== 颜色/对比度增强 ==========

    def random_gamma(self, img, gamma_range=(0.7, 1.3)):
        """
        Gamma 校正（红外图像常用）

        模拟不同的传感器增益
        """
        if random.random() < 0.3:
            gamma = random.uniform(*gamma_range)
            img_np = np.array(img, dtype=np.float32)

            # 归一化到 [0, 1]
            img_min, img_max = img_np.min(), img_np.max()
            if img_max > img_min:
                img_norm = (img_np - img_min) / (img_max - img_min)
                img_gamma = np.power(img_norm, gamma)
                img_np = img_gamma * (img_max - img_min) + img_min

            img = Image.fromarray(np.uint8(np.clip(img_np, 0, 255)))
        return img

    def random_contrast(self, img, contrast_range=(0.7, 1.3)):
        """对比度调整"""
        if random.random() < 0.3:
            factor = random.uniform(*contrast_range)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        return img

    def random_brightness(self, img, brightness_range=(0.8, 1.2)):
        """亮度调整"""
        if random.random() < 0.3:
            factor = random.uniform(*brightness_range)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        return img

    # ========== 噪声/模糊 ==========

    def random_gaussian_noise(self, img, std_range=(5, 15)):
        """
        高斯噪声（模拟传感器噪声）

        红外相机在低光照下噪声明显
        """
        if random.random() < 0.2:
            std = random.uniform(*std_range)
            img_np = np.array(img, dtype=np.float32)
            noise = np.random.randn(*img_np.shape) * std
            img_np = img_np + noise
            img = Image.fromarray(np.uint8(np.clip(img_np, 0, 255)))
        return img

    def random_blur(self, img, radius_range=(0.5, 1.5)):
        """
        高斯模糊（模拟不同成像质量）

        模拟失焦、运动模糊等
        """
        if random.random() < 0.2:
            radius = random.uniform(*radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class TestTimeAugmentation:
    """
    测试时增强（TTA）- 8-way augmentation

    通过多次增强+预测+平均来提升性能

    参考：DNANet, ACM 等顶会论文都用了 TTA
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def predict(self, img, lidar=None):
        """
        8-way TTA 预测

        Args:
            img: (B, C, H, W) tensor
            lidar: (B, 1, H, W) tensor (可选)

        Returns:
            pred: (B, 1, H, W) 平均预测
        """
        predictions = []

        # 1. Original
        pred = self.model(img, lidar)
        predictions.append(pred)

        # 2. Horizontal flip
        img_hflip = torch.flip(img, dims=[-1])
        lidar_hflip = torch.flip(lidar, dims=[-1]) if lidar is not None else None
        pred_hflip = self.model(img_hflip, lidar_hflip)
        pred_hflip = torch.flip(pred_hflip, dims=[-1])
        predictions.append(pred_hflip)

        # 3. Vertical flip
        img_vflip = torch.flip(img, dims=[-2])
        lidar_vflip = torch.flip(lidar, dims=[-2]) if lidar is not None else None
        pred_vflip = self.model(img_vflip, lidar_vflip)
        pred_vflip = torch.flip(pred_vflip, dims=[-2])
        predictions.append(pred_vflip)

        # 4. Both flips
        img_hvflip = torch.flip(img, dims=[-2, -1])
        lidar_hvflip = torch.flip(lidar, dims=[-2, -1]) if lidar is not None else None
        pred_hvflip = self.model(img_hvflip, lidar_hvflip)
        pred_hvflip = torch.flip(pred_hvflip, dims=[-2, -1])
        predictions.append(pred_hvflip)

        # 5-7. Rotations (90, 180, 270)
        for k in [1, 2, 3]:
            img_rot = torch.rot90(img, k, dims=[-2, -1])
            lidar_rot = torch.rot90(lidar, k, dims=[-2, -1]) if lidar is not None else None
            pred_rot = self.model(img_rot, lidar_rot)
            pred_rot = torch.rot90(pred_rot, -k, dims=[-2, -1])
            predictions.append(pred_rot)

        # Average all predictions
        final_pred = torch.stack(predictions).mean(dim=0)

        return final_pred


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("测试数据增强模块")
    print("=" * 60)

    # 创建测试图像
    test_img = Image.new('L', (256, 256), color=128)
    test_mask = Image.new('L', (256, 256), color=0)
    test_depth = Image.new('L', (256, 256), color=64)

    # 测试增强
    aug = InfraredAugmentation(mode='train', use_all=True)

    print("\n测试 10 次增强...")
    for i in range(10):
        img_aug, mask_aug, depth_aug = aug(
            test_img.copy(),
            test_mask.copy(),
            test_depth.copy()
        )
        print(f"  增强 {i+1}: img={img_aug.size}, mask={mask_aug.size}, depth={depth_aug.size}")

    print("\n✅ 数据增强模块测试通过！")
    print("=" * 60)
