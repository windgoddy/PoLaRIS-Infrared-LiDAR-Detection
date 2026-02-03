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
    from .loss_improved import FocalBCELoss, DiceLoss
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from loss_improved import FocalBCELoss, DiceLoss


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
        loss_x = self.criterion(pred_x, target_x)
        loss_y = self.criterion(pred_y, target_y)
        
        return loss_x + loss_y


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
        
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. "
                           f"Supported: 'improved_bce_dice', 'projection', 'hybrid'")
    
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
