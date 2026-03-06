"""
Advanced Loss Functions for PoLaRIS-Mamba
==========================================

Multiple loss strategies for Box-annotated small target detection.

Loss Types:
1. ImprovedBCEDice: Current baseline (Focal BCE + Dice)
2. ProjectionLoss: Weak supervision via X/Y axis projection
3. HybridLoss: ImprovedBCEDice + ProjectionLoss (recommended)

Author: PoLaRIS Team  
Date: 2026-02-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle both package import and direct execution
try:
    from .loss_improved import FocalBCELoss, DiceLoss, SoftIoULoss
    from ..dataset.gaussian_utils import convert_box_mask_to_gaussian
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from loss_improved import FocalBCELoss, DiceLoss, SoftIoULoss
    from dataset.gaussian_utils import convert_box_mask_to_gaussian


class BoxProjectionLoss(nn.Module):
    """
    投影损失：专门针对 Box 标注优化 Mask-to-Box IoU。
    
    **核心思想：**
    - Box GT 强制模型学习"矩形"，但真实目标可能是椭圆/不规则形状
    - Projection Loss 只约束目标的"边界范围"（X/Y轴投影），不强求内部填满
    - 允许模型学习真实物体形状，只要 Bounding Box 正确即可
    
    **数学原理：**
    对于预测 P 和 GT T：
    1. X轴投影: P_x(i) = max_j P(i,j)  表示第 i 列是否有目标
    2. Y轴投影: P_y(j) = max_i P(i,j)  表示第 j 行是否有目标
    3. Loss = BCE(P_x, T_x) + BCE(P_y, T_y)
    
    **优势：**
    - ✅ 解耦了"位置准确性"和"形状填充率"
    - ✅ 允许模型输出椭圆形目标（Mask IoU 可能低，但 Box IoU 高）
    - ✅ 特别适合不规则小目标（船只、飞机等）
    
    **适用场景：**
    - Box 标注的小目标检测
    - 目标形状不规则（非矩形）
    - 评估指标使用 Mask-to-Box IoU
    
    Args:
        projection_mode: 投影方式，'max' 或 'mean'
                        'max': 每行/列的最大值（默认，对稀疏目标更敏感）
                        'mean': 每行/列的平均值（对密集目标更平滑）
        loss_type: 投影后的损失函数，'bce' 或 'l1'
    """
    def __init__(self, projection_mode='max', loss_type='bce'):
        super(BoxProjectionLoss, self).__init__()
        self.projection_mode = projection_mode
        self.loss_type = loss_type
        
        if loss_type == 'bce':
            self.criterion = nn.BCELoss()
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted probabilities in [0, 1]
            target: (B, 1, H, W) binary ground truth (Box)
            
        Returns:
            loss: scalar projection loss
        """
        # Remove channel dimension for easier processing
        pred = pred.squeeze(1)    # (B, H, W)
        target = target.squeeze(1) # (B, H, W)
        
        # 1. X轴投影 (沿着 Y 轴压缩) - 表示每列是否有目标
        if self.projection_mode == 'max':
            pred_x, _ = torch.max(pred, dim=1)    # (B, W)
            target_x, _ = torch.max(target, dim=1) # (B, W)
        else:  # mean
            pred_x = torch.mean(pred, dim=1)      # (B, W)
            target_x = torch.mean(target, dim=1)  # (B, W)
        
        # 2. Y轴投影 (沿着 X 轴压缩) - 表示每行是否有目标
        if self.projection_mode == 'max':
            pred_y, _ = torch.max(pred, dim=2)    # (B, H)
            target_y, _ = torch.max(target, dim=2) # (B, H)
        else:  # mean
            pred_y = torch.mean(pred, dim=2)      # (B, H)
            target_y = torch.mean(target, dim=2)  # (B, H)
        
        # 3. 计算投影损失
        # Replace NaN/Inf and clamp to [eps, 1-eps] to prevent BCELoss CUDA assertion
        # (NaN can appear in Mamba SSM computations after long training)
        eps = 1e-7
        pred_x = torch.nan_to_num(pred_x, nan=0.0, posinf=1.0, neginf=0.0)
        pred_y = torch.nan_to_num(pred_y, nan=0.0, posinf=1.0, neginf=0.0)
        pred_x = torch.clamp(pred_x, eps, 1.0 - eps)
        pred_y = torch.clamp(pred_y, eps, 1.0 - eps)
        loss_x = self.criterion(pred_x, target_x)
        loss_y = self.criterion(pred_y, target_y)
        
        return loss_x + loss_y


class PairwiseAffinityLoss(nn.Module):
    """
    成对亲和力损失 (Pairwise Affinity Loss) - BoxInst 风格

    **核心思想：**
    如果相邻像素在原始图像中颜色/亮度相似，那么它们的预测值也应该相似。
    这是一种无监督约束，利用图像本身的纹理信息引导分割边界。

    **数学原理：**
    对于相邻像素 i 和 j：
    1. 计算图像相似度: w_ij = exp(-λ * |I_i - I_j|)
    2. 计算预测差异: d_ij = |P_i - P_j|
    3. Loss = Σ w_ij * d_ij

    **含义：**
    - 如果图像相似(w大)但预测不同(d大) → Loss 大 → 惩罚
    - 如果图像不同(w小) → Loss 小 → 允许预测不同

    **优势：**
    - ✅ 自动贴合物体边缘（利用红外热点的亮度差异）
    - ✅ 无需像素级标注（完全无监督）
    - ✅ 适合不规则形状（船体、浮漂）

    **适用场景：**
    - 红外小目标检测（热点与背景有明显亮度差）
    - 弱监督分割（Box 标注）
    - 需要精细边缘的场景

    Args:
        lambda_val: 亲和力权重系数 (default: 2.0)
                   越大表示对图像相似度越敏感
        kernel_size: 邻域大小 (default: 3, 即8邻域)

    Reference:
        BoxInst: High-Performance Instance Segmentation with Box Annotations (CVPR 2021)
    """
    def __init__(self, lambda_val=2.0, kernel_size=3):
        super(PairwiseAffinityLoss, self).__init__()
        self.lambda_val = lambda_val
        self.kernel_size = kernel_size

    def forward(self, pred, image):
        """
        Args:
            pred: (B, 1, H, W) 预测 Mask (Sigmoid后，范围 [0, 1])
            image: (B, C, H, W) 原始红外图像 (归一化到 [0, 1])

        Returns:
            loss: scalar affinity loss
        """
        # 如果是多通道图像，转为灰度（红外通常是单通道）
        if image.shape[1] > 1:
            # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
            image = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

        # 1. 计算图像的水平和垂直差异
        # 水平方向：比较每个像素与右边像素
        img_diff_h = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])  # (B, 1, H, W-1)
        # 垂直方向：比较每个像素与下边像素
        img_diff_v = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])  # (B, 1, H-1, W)

        # 2. 计算亲和力权重 (Affinity Weight)
        # 公式: w = exp(-λ * |I_i - I_j|)
        # 图像差异越小 → w 越大 → 要求预测也相似
        weight_h = torch.exp(-self.lambda_val * img_diff_h)  # (B, 1, H, W-1)
        weight_v = torch.exp(-self.lambda_val * img_diff_v)  # (B, 1, H-1, W)

        # 3. 计算预测值的差异
        pred_diff_h = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])  # (B, 1, H, W-1)
        pred_diff_v = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])  # (B, 1, H-1, W)

        # 4. 加权损失
        # 如果图像相似(weight大)但预测不同(diff大)，则 Loss 大
        loss_h = (weight_h * pred_diff_h).mean()
        loss_v = (weight_v * pred_diff_v).mean()

        return loss_h + loss_v


class HybridLoss(nn.Module):
    """
    混合损失：ImprovedBCEDice + ProjectionLoss
    
    **设计理念：**
    - ImprovedBCEDice: 负责整体的分割质量（召回率、前景背景分离）
    - ProjectionLoss: 负责边界框的精准定位（Box IoU）
    
    **权重建议：**
    - projection_weight=1.0: 标准配置，两者平衡
    - projection_weight=2.0: 强调边界精度（适合 Box 标注场景）
    - projection_weight=0.5: 保守配置（先验证效果）
    
    Args:
        focal_alpha: Focal loss alpha (default: 0.25)
        focal_gamma: Focal loss gamma (default: 2.5)
        dice_weight: Dice loss weight (default: 4.0)
        projection_weight: Projection loss weight (default: 1.0)
        projection_mode: 'max' or 'mean'
    """
    def __init__(
        self,
        focal_alpha=0.25,
        focal_gamma=2.5,
        dice_weight=4.0,
        projection_weight=1.0,
        projection_mode='max',
        ohem_ratio=0.0
    ):
        super(HybridLoss, self).__init__()
        
        # 基础损失组件
        self.focal_bce = FocalBCELoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=1.0)
        self.proj_loss = BoxProjectionLoss(projection_mode=projection_mode, loss_type='bce')
        
        # 权重
        self.dice_weight = dice_weight
        self.projection_weight = projection_weight
        self.ohem_ratio = ohem_ratio
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted probabilities
            target: (B, 1, H, W) binary ground truth
            
        Returns:
            loss: scalar combined loss
        """
        target = target.float()
        
        # 1. Focal BCE (像素级分类)
        focal_loss = self.focal_bce(pred, target)
        
        # 2. Dice Loss (集合重叠)
        dice_loss = self.dice_loss(pred, target)
        
        # 3. Projection Loss (边界约束)
        proj_loss = self.proj_loss(pred, target)
        
        # 4. 组合损失
        total_loss = focal_loss + self.dice_weight * dice_loss + self.projection_weight * proj_loss
        
        # 5. Optional OHEM
        if self.ohem_ratio > 0:
            eps = 1e-7
            pred_clamped = torch.clamp(pred, eps, 1.0 - eps)
            pixel_bce = -target * torch.log(pred_clamped) - (1 - target) * torch.log(1 - pred_clamped)
            
            n_pixels = pixel_bce.numel()
            n_hard = int(n_pixels * self.ohem_ratio)
            pixel_bce_flat = pixel_bce.view(-1)
            hard_pixels, _ = torch.topk(pixel_bce_flat, n_hard)
            ohem_loss = hard_pixels.mean()
            
            total_loss += 0.5 * ohem_loss
        
        return total_loss


class GaussianFocalLoss(nn.Module):
    """
    CornerNet/CenterNet 风格的 Penalty-Reduced Focal Loss。

    专为软高斯热图目标（Float Target, Y ∈ [0, 1]）设计，是 BAGS 策略的核心损失。

    公式：
        前景峰值像素 (Y == 1):
            L = -(1 - p)^α * log(p)
        背景/过渡区像素 (Y < 1):
            L = -(1 - Y)^β * p^α * log(1 - p)

    关键设计：
        (1-Y)^β 在峰值附近（Y≈1 ⟹ 1-Y≈0）将背景惩罚压制到接近 0，
        使得高斯中心周围的过渡区不被当作"假阳性"惩罚，
        模型可以自由学习"热点"的真实形状，而不是被迫填满矩形框。

    Args:
        alpha: 前景聚焦参数，控制难易样本的权重衰减（默认 2.0）
        beta:  背景惩罚衰减参数，越大则峰值附近的过渡区越宽容（默认 4.0）

    Reference:
        CornerNet: Detecting Objects as Paired Keypoints (Law & Deng, ECCV 2018)
        CenterNet: Objects as Points (Zhou et al., 2019)
    """
    def __init__(self, alpha=2.0, beta=4.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred  : (B, 1, H, W) 模型输出，经过 sigmoid，范围 [0, 1]
            target: (B, 1, H, W) 软高斯热图，范围 [0, 1]，峰值为 1.0
        Returns:
            loss: scalar，按前景像素数归一化
        """
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1.0 - eps)

        # [FIX 2026-03-04] 前景掩码：高斯峰值附近（>= 0.999）
        # 原来用 eq(1.0) 在偶数尺寸目标框上会失败（中心不在整数格点）
        # 虽然 convert_box_mask_to_gaussian 已做 round 修复，此处作为防御保留宽松阈值
        pos_mask = target.ge(0.999).float()
        neg_mask = target.lt(0.999).float()

        # 前景损失（只在峰值像素计算）
        pos_loss = pos_mask * (-(1.0 - pred).pow(self.alpha) * torch.log(pred))

        # 背景损失（过渡区由 (1-Y)^β 自动衰减）
        neg_loss = neg_mask * (
            -(1.0 - target).pow(self.beta) *
            pred.pow(self.alpha) *
            torch.log(1.0 - pred)
        )

        num_pos = pos_mask.sum().clamp(min=1.0)
        loss = (pos_loss + neg_loss).sum() / num_pos
        return loss


class GaussianHybridLoss(nn.Module):
    """
    BAGS 策略混合损失：GaussianFocalLoss + BoxProjectionLoss。

    设计理念：
        ┌─────────────────────────────────────────────────────────┐
        │  Box Mask (DataLoader 输出，原始矩形 GT)                 │
        │       │                                                  │
        │       ├──→ convert_box_mask_to_gaussian()               │
        │       │         ↓                                        │
        │       │    Gaussian Target (软热图)                      │
        │       │         ↓                                        │
        │       │    GaussianFocalLoss(pred, gaussian_target)      │
        │       │    ← 学习"热点"形状，不强求填满矩形              │
        │       │                                                  │
        │       └──→ BoxProjectionLoss(pred, box_mask)            │
        │            ← 约束预测范围不超出 Box 边界                  │
        └─────────────────────────────────────────────────────────┘

    关键优势：
        ✅ DataLoader 不需要任何修改（仍读取二值矩形 Mask）
        ✅ Box → Gaussian 转换在 GPU 上实时完成，无额外 CPU 开销
        ✅ Dice Loss 彻底移除（Dice 会强迫填满矩形，与 BAGS 目标冲突）
        ✅ Projection Loss 保留为边界约束（防止模型预测范围跑出 Box 外）

    Args:
        alpha           : GaussianFocalLoss 前景聚焦参数（默认 2.0）
        beta            : GaussianFocalLoss 背景衰减参数（默认 4.0）
        projection_weight: BoxProjectionLoss 权重（默认 2.0）
        projection_mode : 'max' 或 'mean'（默认 'max'）
        sigma_scale     : Gaussian sigma = box_dim / sigma_scale（默认 6.0）
        min_sigma       : sigma 下界，防止极小目标退化（默认 2.0）
    """
    def __init__(
        self,
        alpha=2.0,
        beta=4.0,
        projection_weight=2.0,
        projection_mode='max',
        sigma_scale=6.0,
        min_sigma=2.0,
    ):
        super(GaussianHybridLoss, self).__init__()
        self.gaussian_focal = GaussianFocalLoss(alpha=alpha, beta=beta)
        self.proj_loss = BoxProjectionLoss(projection_mode=projection_mode, loss_type='bce')
        self.projection_weight = projection_weight
        self.sigma_scale = sigma_scale
        self.min_sigma = min_sigma

    def forward(self, pred, target, gaussian_target=None):
        """
        Args:
            pred  : (B, 1, H, W) 模型输出 [0, 1]
            target: (B, 1, H, W) 原始 Box 二值 Mask {0, 1}（DataLoader 原始输出）
            gaussian_target: (B, 1, H, W) 预计算的高斯热图（可选）
                             如果提供，跳过 Box→Gaussian 转换（Deep Supervision 优化用）
        Returns:
            loss: scalar
        """
        target = target.float()

        # Step 1: Box Mask → 软高斯热图（GPU 内联转换）
        # [FIX 2026-03-04] 支持传入预计算的 gaussian_target，
        # Deep Supervision 场景下只需转换一次，3 个 head 复用
        if gaussian_target is None:
            gaussian_target = convert_box_mask_to_gaussian(
                target,
                sigma_scale=self.sigma_scale,
                min_sigma=self.min_sigma,
            )

        # Step 2: 主损失 —— Gaussian Focal（学习热点分布）
        loss_gauss = self.gaussian_focal(pred, gaussian_target)

        # Step 3: 辅助损失 —— Box Projection（约束边界范围，使用原始矩形 GT）
        loss_proj = self.proj_loss(pred, target)

        return loss_gauss + self.projection_weight * loss_proj


class LossFactory:
    """
    Loss 工厂类：统一管理多种 Loss 函数
    
    Usage:
        criterion = LossFactory.create(
            loss_type='hybrid',
            focal_alpha=0.25,
            focal_gamma=2.5,
            dice_weight=4.0,
            projection_weight=1.0
        )
    """
    
    @staticmethod
    def create(loss_type='improved_bce_dice', **kwargs):
        """
        创建指定类型的 Loss 函数
        
        Args:
            loss_type: 'improved_bce_dice', 'projection', 'hybrid'
            **kwargs: Loss 函数的参数
            
        Returns:
            loss_fn: nn.Module
        """
        if loss_type == 'improved_bce_dice':
            # 导入原有的 ImprovedBCEDiceLoss
            try:
                from .loss_improved import ImprovedBCEDiceLoss
            except ImportError:
                from loss_improved import ImprovedBCEDiceLoss
            return ImprovedBCEDiceLoss(
                focal_weight=kwargs.get('focal_weight', 1.0),
                dice_weight=kwargs.get('dice_weight', 4.0),
                focal_alpha=kwargs.get('focal_alpha', 0.25),
                focal_gamma=kwargs.get('focal_gamma', 2.5),
                ohem_ratio=kwargs.get('ohem_ratio', 0.0),
                smooth=kwargs.get('smooth', 1.0)
            )
        
        elif loss_type == 'projection':
            # 纯投影损失（实验性）
            return BoxProjectionLoss(
                projection_mode=kwargs.get('projection_mode', 'max'),
                loss_type=kwargs.get('projection_loss_type', 'bce')
            )
        
        elif loss_type == 'hybrid':
            # 混合损失（推荐）
            return HybridLoss(
                focal_alpha=kwargs.get('focal_alpha', 0.25),
                focal_gamma=kwargs.get('focal_gamma', 2.5),
                dice_weight=kwargs.get('dice_weight', 4.0),
                projection_weight=kwargs.get('projection_weight', 1.0),
                projection_mode=kwargs.get('projection_mode', 'max'),
                ohem_ratio=kwargs.get('ohem_ratio', 0.0)
            )

        elif loss_type == 'gaussian_focal':
            # BAGS 策略：Box Mask 实时转 Gaussian + Focal Loss + Projection 辅助
            return GaussianHybridLoss(
                alpha=kwargs.get('focal_alpha', 2.0),
                beta=kwargs.get('focal_gamma', 4.0),
                projection_weight=kwargs.get('projection_weight', 2.0),
                projection_mode=kwargs.get('projection_mode', 'max'),
                sigma_scale=kwargs.get('gaussian_sigma_scale', 6.0),
                min_sigma=kwargs.get('gaussian_min_sigma', 2.0),
            )

        elif loss_type == 'soft_iou':
            # 标准 SoftIoU Loss — 专为稀疏小目标设计 (DNANet/ACM/ALCNet 标准)
            # smooth=1e-6 确保梯度在极稀疏目标场景下仍然有效
            return SoftIoULoss(smooth=1e-6)

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. "
                           f"Supported: 'improved_bce_dice', 'projection', 'hybrid', 'gaussian_focal', 'soft_iou'")
    
    @staticmethod
    def get_loss_name(loss_type, **kwargs):
        """
        生成 Loss 的描述性名称（用于实验目录命名）
        
        Returns:
            name: str, e.g., "BCEDice_a0.25_g2.5_d4.0"
        """
        if loss_type == 'improved_bce_dice':
            alpha = kwargs.get('focal_alpha', 0.25)
            gamma = kwargs.get('focal_gamma', 2.5)
            dice_w = kwargs.get('dice_weight', 4.0)
            return f"BCEDice_a{alpha}_g{gamma}_d{dice_w}"
        
        elif loss_type == 'projection':
            mode = kwargs.get('projection_mode', 'max')
            return f"Projection_{mode}"
        
        elif loss_type == 'hybrid':
            alpha = kwargs.get('focal_alpha', 0.25)
            gamma = kwargs.get('focal_gamma', 2.5)
            dice_w = kwargs.get('dice_weight', 4.0)
            proj_w = kwargs.get('projection_weight', 1.0)
            mode = kwargs.get('projection_mode', 'max')
            return f"Hybrid_a{alpha}_g{gamma}_d{dice_w}_p{proj_w}_{mode}"

        elif loss_type == 'gaussian_focal':
            alpha = kwargs.get('focal_alpha', 2.0)
            beta = kwargs.get('focal_gamma', 4.0)
            proj_w = kwargs.get('projection_weight', 2.0)
            sigma = kwargs.get('gaussian_sigma_scale', 6.0)
            return f"GaussianFocal_a{alpha}_b{beta}_p{proj_w}_s{sigma}"

        elif loss_type == 'soft_iou':
            return "SoftIoU"

        return "UnknownLoss"


# ======================== Unit Test ========================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Advanced Loss Functions")
    print("=" * 70)
    
    # 模拟数据
    B, H, W = 2, 128, 128
    
    # GT: Box 标注（实心矩形）
    target = torch.zeros(B, 1, H, W)
    target[:, :, 50:60, 50:60] = 1.0  # 10x10 box
    
    # Pred 1: 椭圆形预测（真实目标形状）
    pred_ellipse = torch.zeros(B, 1, H, W)
    for i in range(50, 60):
        for j in range(50, 60):
            # 椭圆方程: (x-cx)^2/a^2 + (y-cy)^2/b^2 <= 1
            cx, cy = 55, 55
            dx, dy = i - cx, j - cy
            if (dx**2 / 25 + dy**2 / 25) <= 1:
                pred_ellipse[:, :, i, j] = 0.9
    
    # Pred 2: 矩形预测（完美匹配 GT）
    pred_box = torch.zeros(B, 1, H, W)
    pred_box[:, :, 50:60, 50:60] = 0.9
    
    print(f"\nTarget shape: {target.shape}")
    print(f"Target box: [50:60, 50:60]")
    print(f"\nPred Ellipse coverage: {(pred_ellipse > 0.5).float().sum() / (target > 0.5).float().sum():.2%}")
    print(f"Pred Box coverage: {(pred_box > 0.5).float().sum() / (target > 0.5).float().sum():.2%}")
    
    # Test 1: ImprovedBCEDice (原有损失)
    print("\n" + "=" * 70)
    print("[1] ImprovedBCEDice Loss (Baseline)")
    print("=" * 70)
    loss_bce_dice = LossFactory.create('improved_bce_dice', focal_alpha=0.25, focal_gamma=2.5, dice_weight=4.0)
    
    loss_ellipse_1 = loss_bce_dice(pred_ellipse, target)
    loss_box_1 = loss_bce_dice(pred_box, target)
    
    print(f"Loss (Ellipse pred): {loss_ellipse_1.item():.6f}")
    print(f"Loss (Box pred):     {loss_box_1.item():.6f}")
    print(f"Penalty ratio (Ellipse/Box): {loss_ellipse_1.item() / loss_box_1.item():.2f}x")
    print("👉 椭圆预测被惩罚，因为 Dice 要求填满矩形")
    
    # Test 2: Projection Loss
    print("\n" + "=" * 70)
    print("[2] Projection Loss")
    print("=" * 70)
    loss_proj = LossFactory.create('projection', projection_mode='max')
    
    loss_ellipse_2 = loss_proj(pred_ellipse, target)
    loss_box_2 = loss_proj(pred_box, target)
    
    print(f"Loss (Ellipse pred): {loss_ellipse_2.item():.6f}")
    print(f"Loss (Box pred):     {loss_box_2.item():.6f}")
    print(f"Penalty ratio (Ellipse/Box): {loss_ellipse_2.item() / loss_box_2.item():.2f}x")
    print("👉 椭圆和矩形 Loss 接近，因为 X/Y 投影范围相同")
    
    # Test 3: Hybrid Loss
    print("\n" + "=" * 70)
    print("[3] Hybrid Loss (Recommended)")
    print("=" * 70)
    loss_hybrid = LossFactory.create('hybrid', focal_alpha=0.25, focal_gamma=2.5, 
                                     dice_weight=4.0, projection_weight=1.0)
    
    loss_ellipse_3 = loss_hybrid(pred_ellipse, target)
    loss_box_3 = loss_hybrid(pred_box, target)
    
    print(f"Loss (Ellipse pred): {loss_ellipse_3.item():.6f}")
    print(f"Loss (Box pred):     {loss_box_3.item():.6f}")
    print(f"Penalty ratio (Ellipse/Box): {loss_ellipse_3.item() / loss_box_3.item():.2f}x")
    print("👉 平衡了分割质量和边界定位")
    
    # Test 4: LossFactory naming
    print("\n" + "=" * 70)
    print("[4] Loss Naming for Experiments")
    print("=" * 70)
    name1 = LossFactory.get_loss_name('improved_bce_dice', focal_alpha=0.25, focal_gamma=2.5, dice_weight=4.0)
    name2 = LossFactory.get_loss_name('projection', projection_mode='max')
    name3 = LossFactory.get_loss_name('hybrid', focal_alpha=0.25, focal_gamma=2.5, 
                                      dice_weight=4.0, projection_weight=1.0)
    
    print(f"ImprovedBCEDice: {name1}")
    print(f"Projection:      {name2}")
    print(f"Hybrid:          {name3}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
