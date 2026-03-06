"""
PGA-Mamba: Prior-Guided Attention Mamba UNet++
===============================================

架构重构 (2026-03-06):
  旧方案: prior_scale 标量 * identity (只放大目标，不抑制背景；初始为 0 等于无效)
  新方案: PriorGuidedAttention (PGA) 模块，解决三个核心问题:
    1. 梯度阻断: 用加法注入取代 max() 硬选择，learned_attn 在所有区域都能获得梯度
    2. 背景抑制: x * (alpha + (1-alpha) * final_attn)，alpha 可学习从 1→0
    3. 浅层污染: shallow CNN 输出也注入 prior_256/128，防止 skip 带回杂波

物理先验金字塔 (完整版):
  prior_full(256) → MaxPool→ prior_128 → MaxPool→ prior_64 → MaxPool→ prior_32 → MaxPool→ prior_16
  全程 MaxPool (非 AvgPool)，保留稀疏目标 SNR 峰值

Author: PoLaRIS Team
Date: 2026-03-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_Mamba.core.ss2d_components import VSSBlock


# ============================================================
# 1. 物理先验生成器 (无可学习参数)
# ============================================================

class PhysicalSNRPrior(nn.Module):
    """
    在原始输入分辨率上计算局部信噪比 (SNR)，无可学习参数。

    公式: SNR = ReLU(I - mu) / (sigma + eps)
    其中 mu, sigma 由 15x15 AvgPool 估计，保持"局部"统计的物理意义。
    输出经 per-image min-max 归一化至 [0, 1]。

    为什么在输入分辨率计算:
      - 每个像素对应真实热辐射强度，局部统计有物理意义
      - 深层特征 (16x16) 的均值/方差反映的是抽象语义，而非热辐射
    """
    def __init__(self, kernel_size: int = 15):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) 单通道红外图像 (原始辐射值)
        Returns:
            snr_norm: (B, 1, H, W) 归一化 SNR，值域 [0, 1]
        """
        mu     = self.avg_pool(x)
        sq_avg = self.avg_pool(x ** 2)
        var    = torch.clamp(sq_avg - mu ** 2, min=1e-6)
        sigma  = torch.sqrt(var)
        snr    = F.relu(x - mu) / (sigma + 1e-4)

        # Per-image min-max 归一化
        B, C, H, W = snr.shape
        snr_flat = snr.view(B, C, -1)
        snr_min  = snr_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        snr_max  = snr_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        return (snr - snr_min) / (snr_max - snr_min + 1e-8)


# ============================================================
# 2. 先验引导注意力模块 (PriorGuidedAttention)
# ============================================================

class PriorGuidedAttention(nn.Module):
    """
    先验引导注意力 (PGA): 用物理 SNR 先验和语义特征联合生成注意力图。

    核心设计决策:
      [A] 注意力生成: final_attn = sigmoid(learned_logit + prior_weight * prior)
          - 加法注入，梯度在 learned_logit 和 prior 两个分支都能流动
          - 不使用 max()，避免"梯度阻断"问题

      [B] 特征调制: x_out = x * (alpha + (1-alpha) * final_attn)
          - alpha ∈ [0, 1]，可学习，初始化为 1.0 (纯 identity，保守启动)
          - alpha→0: 完全受 final_attn 调制 (背景被压制)
          - alpha→1: 特征原样通过 (梯度安全的 fallback)
          - background(attn≈0): x * alpha → 被抑制
          - target(attn≈1):   x * 1.0  → 完整保留

      [C] 残差保护: MambaBlockWrapper 中的 identity 残差不经过 PGA
          - 确保梯度在 Mamba 被抑制时仍能流回浅层
    """
    def __init__(self, ch: int):
        super().__init__()
        mid_ch = max(ch // 4, 4)  # 轻量: 通道数压缩到 1/4，至少 4

        # 注意力生成器: concat(feat[ch], prior[1]) → logit[1]
        # 用 1x1 conv 保持空间结构，参数量极少
        self.attn_conv = nn.Sequential(
            nn.Conv2d(ch + 1, mid_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True),
        )

        # prior_weight: 物理先验在注意力 logit 中的加权系数
        # 初始化为 1.0: 先验与学习注意力各占 1 份权重
        self.prior_weight = nn.Parameter(torch.ones(1))

        # alpha: 背景抑制强度控制
        # 初始化为 1.0 (保守): x_out ≈ x，随训练逐渐学习降低 alpha
        self.alpha = nn.Parameter(torch.ones(1))

        # 初始化 attn_conv 最后一层接近零输出
        # 确保训练初期注意力主要由物理先验驱动，而不是随机卷积
        nn.init.constant_(self.attn_conv[-1].bias, 0.0)
        nn.init.normal_(self.attn_conv[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, C, H, W) 特征图
            prior: (B, 1, H, W) 物理 SNR 先验，值域 [0, 1]
        Returns:
            x_out: (B, C, H, W) 调制后的特征
        """
        # Step 1: 生成学习注意力 logit
        fused        = torch.cat([x, prior], dim=1)   # (B, C+1, H, W)
        learned_logit = self.attn_conv(fused)          # (B, 1, H, W) 原始 logit

        # Step 2: 加法融合物理先验，生成最终注意力图
        # final_attn ∈ (0, 1)，高 SNR 区域趋近于 1，低 SNR 区域趋近于 0
        final_attn = torch.sigmoid(learned_logit + self.prior_weight * prior)

        # Step 3: 可控软抑制
        # alpha clamp 到 [0, 1] 防止参数越界 (替代 sigmoid 保留梯度线性区)
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        x_out = x * (alpha + (1.0 - alpha) * final_attn)

        return x_out


# ============================================================
# 3. Mamba Block Wrapper (集成 PGA)
# ============================================================

class MambaBlockWrapper(nn.Module):
    """
    将 VSSBlock 包装为 UNet++ 兼容格式，并集成 PriorGuidedAttention。

    数据流:
        x (B, in_ch, H, W)
          → proj_in → identity (B, out_ch, H, W)   ← 残差路径
          → PGA(identity, prior) → x_in             ← 调制路径 (prior 只影响此路径)
          → permute → VSSBlock → permute
          → spatial_restore (depthwise 3x3)
          → identity + mamba_out                    ← 残差不受 prior 影响，梯度稳定

    注: prior 不进入残差路径。当 prior 使某些区域被抑制时，
        残差仍保证信号不会彻底消失，避免 SSM 状态崩溃。
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pga     = PriorGuidedAttention(out_ch)
        self.mamba   = VSSBlock(hidden_dim=out_ch, drop_path=0.1)
        # depthwise conv: 桥接 Mamba 全局特征 与 CNN 局部特征
        self.spatial_restore = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch)

    def forward(self, x: torch.Tensor, spatial_prior=None) -> torch.Tensor:
        """
        Args:
            x            : (B, in_ch, H, W)
            spatial_prior: (B, 1, H, W) 物理 SNR 先验 [0,1]，或 None
        Returns:
            out: (B, out_ch, H, W)
        """
        identity = self.proj_in(x)                    # (B, out_ch, H, W)

        # PGA 调制 (只在先验存在时激活)
        if spatial_prior is not None:
            x_in = self.pga(identity, spatial_prior)  # (B, out_ch, H, W)
        else:
            x_in = identity

        # Mamba 路径
        B, C, H, W = x_in.shape
        x_mamba = x_in.permute(0, 2, 3, 1)           # (B, H, W, C)
        x_mamba = self.mamba(x_mamba)                 # (B, H, W, C) 纯 SSM
        x_mamba = x_mamba.permute(0, 3, 1, 2)        # (B, C, H, W)
        x_mamba = self.spatial_restore(x_mamba)       # 局部特征恢复

        return identity + x_mamba                     # 残差不经过 PGA


# ============================================================
# 4. 主模型: MambaUNetPlusPlus
# ============================================================

class MambaUNetPlusPlus(nn.Module):
    """
    PGA-Mamba UNet++: 物理先验引导的 Mamba-UNet++ 混合架构

    核心改动 (2026-03-06):
      1. 完整先验金字塔: prior_256/128/64/32/16 (全程 MaxPool)
      2. 浅层 CNN 注入: x0_0, x1_0 用 prior 软抑制背景 (floor=0.5)
      3. 深层 Mamba 注入: 使用 PGA 而非旧的 prior_scale 标量

    Args:
        num_classes   : 输出类别数 (默认 1，二值分割)
        input_channels: 输入通道数 (1 for IR-only, 3 for pseudo-RGB)
        nb_filter     : 各层通道数，默认 [16, 32, 64, 128, 256] (与 DNANet 相同)
        deep_supervision: 是否使用深度监督 (训练时输出多个预测)
        use_lidar     : 是否启用物理先验注入 (True=SNR/LiDAR prior)
        shallow_floor : 浅层 CNN 先验门控的下限 [0,1]，越小背景抑制越强
                        默认 0.5: background 保留至少 50% 信号
    """

    def __init__(self, num_classes=1, input_channels=3, nb_filter=None,
                 deep_supervision=False, use_lidar=False, shallow_floor=0.5):
        super().__init__()
        if nb_filter is None:
            nb_filter = [16, 32, 64, 128, 256]

        self.pool             = nn.MaxPool2d(2, 2)
        self.up               = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deep_supervision = deep_supervision
        self.use_lidar        = use_lidar
        self.shallow_floor    = shallow_floor  # 浅层先验门控下限

        # PhysicalSNRPrior: 无可学习参数，纯物理计算
        if self.use_lidar:
            self.snr_prior = PhysicalSNRPrior(kernel_size=15)

        # ===== Encoder =====
        # 浅层: CNN (局部纹理/边缘提取)
        self.conv0_0 = self._make_cnn_block(input_channels, nb_filter[0])  # 256x256, ch=16
        self.conv1_0 = self._make_cnn_block(nb_filter[0], nb_filter[1])    # 128x128, ch=32

        # 深层: Mamba + PGA (全局建模 + 先验引导)
        self.conv2_0 = self._make_mamba_block(nb_filter[1], nb_filter[2])  # 64x64,  ch=64
        self.conv3_0 = self._make_mamba_block(nb_filter[2], nb_filter[3])  # 32x32,  ch=128
        self.conv4_0 = self._make_mamba_block(nb_filter[3], nb_filter[4])  # 16x16,  ch=256

        # ===== Decoder (UNet++ 密集跳连) =====
        # Level 0 (256x256, ch=16)
        self.conv0_1 = self._make_cnn_block(nb_filter[0] + nb_filter[1],         nb_filter[0])
        self.conv0_2 = self._make_cnn_block(nb_filter[0] * 2 + nb_filter[1],     nb_filter[0])
        self.conv0_3 = self._make_cnn_block(nb_filter[0] * 3 + nb_filter[1],     nb_filter[0])
        self.conv0_4 = self._make_cnn_block(nb_filter[0] * 4 + nb_filter[1],     nb_filter[0])

        # Level 1 (128x128, ch=32)
        self.conv1_1 = self._make_cnn_block(nb_filter[1] + nb_filter[2],         nb_filter[1])
        self.conv1_2 = self._make_cnn_block(nb_filter[1] * 2 + nb_filter[2],     nb_filter[1])
        self.conv1_3 = self._make_cnn_block(nb_filter[1] * 3 + nb_filter[2],     nb_filter[1])

        # Level 2 (64x64, ch=64) — Mamba
        self.conv2_1 = self._make_mamba_block(nb_filter[2] + nb_filter[3],       nb_filter[2])
        self.conv2_2 = self._make_mamba_block(nb_filter[2] * 2 + nb_filter[3],   nb_filter[2])

        # Level 3 (32x32, ch=128) — Mamba
        self.conv3_1 = self._make_mamba_block(nb_filter[3] + nb_filter[4],       nb_filter[3])

        # ===== 输出头 =====
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    # ----------------------------------------------------------
    # 工厂方法
    # ----------------------------------------------------------
    def _make_cnn_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """
        CNN Block with GroupNorm (小 batch 下比 BatchNorm 更稳定).
        8 groups for ch<64, 16 groups for ch>=64.
        """
        num_groups = 8 if out_ch < 64 else 16
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def _make_mamba_block(self, in_ch: int, out_ch: int) -> MambaBlockWrapper:
        return MambaBlockWrapper(in_ch, out_ch)

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor, lidar=None) -> torch.Tensor:
        """
        Args:
            x    : (B, C, H, W) 红外图像
            lidar: (B, 1, H, W) LiDAR 深度图 (可选)
                   若为 None 或全零 → 自动使用 SNR 先验

        Returns:
            deep_supervision=True : [output4, output3, output2, output1] (主输出在首位)
            deep_supervision=False: (B, 1, H, W) 概率图
        """

        # ===== 物理先验金字塔 =====
        # 在 no_grad 中计算，切断梯度，节省显存
        # 全程 MaxPool 确保小目标 SNR 峰值在缩小的网格中存活
        if self.use_lidar:
            with torch.no_grad():
                lidar_is_real = (lidar is not None and
                                 lidar.abs().max().item() > 1e-6)
                if lidar_is_real:
                    # LiDAR 模式: 用归一化深度图作空间先验
                    prior_full = lidar.clamp(min=0)
                    B, C, H, W = prior_full.shape
                    p_max = prior_full.view(B, C, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    prior_full = prior_full / (p_max + 1e-8)
                else:
                    # SNR 模式: 从原始 IR 图像计算物理 SNR
                    prior_full = self.snr_prior(x[:, :1])       # (B, 1, H, W)

                # 完整先验金字塔 (MaxPool 每次缩小 2x, 保留峰值)
                prior_256 = prior_full                                            # (B,1,H,  W  )
                prior_128 = F.max_pool2d(prior_full, kernel_size=2,  stride=2)   # (B,1,H/2,W/2)
                prior_64  = F.max_pool2d(prior_full, kernel_size=4,  stride=4)   # (B,1,H/4,W/4)
                prior_32  = F.max_pool2d(prior_full, kernel_size=8,  stride=8)   # (B,1,H/8,W/8)
                prior_16  = F.max_pool2d(prior_full, kernel_size=16, stride=16)  # (B,1,H/16,W/16)
        else:
            prior_256 = prior_128 = prior_64 = prior_32 = prior_16 = None

        # ===== Encoder =====

        # 浅层 CNN (局部特征提取)
        x0_0 = self.conv0_0(x)                        # (B, 16, H,   W  )
        # 浅层先验软门控: 防止近岸杂波特征通过 skip 污染解码器
        # floor=shallow_floor 保留背景最低信号量，避免完全屏蔽 (防 State Collapse)
        if prior_256 is not None:
            x0_0 = x0_0 * (self.shallow_floor + (1.0 - self.shallow_floor) * prior_256)

        x1_0 = self.conv1_0(self.pool(x0_0))          # (B, 32, H/2, W/2)
        if prior_128 is not None:
            x1_0 = x1_0 * (self.shallow_floor + (1.0 - self.shallow_floor) * prior_128)

        # 深层 Mamba + PGA (全局建模 + 先验引导注意力)
        x2_0 = self.conv2_0(self.pool(x1_0), prior_64)   # (B, 64,  H/4, W/4)
        x3_0 = self.conv3_0(self.pool(x2_0), prior_32)   # (B, 128, H/8, W/8)
        x4_0 = self.conv4_0(self.pool(x3_0), prior_16)   # (B, 256, H/16,W/16)

        # ===== Decoder (UNet++ 密集连接) =====
        # 规则: X^{i,j} = Conv([X^{i,0..j-1}, Up(X^{i+1,j-1})])

        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1), prior_64)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1), prior_32)

        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1), prior_64)

        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        # Column 4 (最终解码特征)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # ===== 输出 =====
        if self.deep_supervision:
            output1 = torch.sigmoid(self.final1(x0_1))
            output2 = torch.sigmoid(self.final2(x0_2))
            output3 = torch.sigmoid(self.final3(x0_3))
            output4 = torch.sigmoid(self.final4(x0_4))
            return [output4, output3, output2, output1]   # 主输出在首位
        else:
            return torch.sigmoid(self.final(x0_4))


# ============================================================
# 5. 工厂函数
# ============================================================

def mamba_unetplusplus(in_channels=3, num_classes=1, deep_supervision=False,
                       use_lidar=False, shallow_floor=0.5):
    """
    创建 PGA-Mamba UNet++ 实例。

    Args:
        in_channels    : IR 输入通道数 (1=单通道, 3=伪RGB)
        num_classes    : 输出类别数 (1=二值分割)
        deep_supervision: 启用深度监督 (推荐小目标任务)
        use_lidar      : 启用物理先验注入 (True=SNR/LiDAR prior)
        shallow_floor  : 浅层 CNN 先验门控下限 [0,1]，默认 0.5

    Returns:
        model: MambaUNetPlusPlus 实例

    Example:
        >>> model = mamba_unetplusplus(in_channels=1, num_classes=1,
        ...                            deep_supervision=True, use_lidar=True)
        >>> ir    = torch.randn(4, 1, 256, 256)
        >>> lidar = torch.randn(4, 1, 256, 256)
        >>> out   = model(ir, lidar)  # (4, 1, 256, 256)
    """
    return MambaUNetPlusPlus(
        num_classes=num_classes,
        input_channels=in_channels,
        nb_filter=[16, 32, 64, 128, 256],
        deep_supervision=deep_supervision,
        use_lidar=use_lidar,
        shallow_floor=shallow_floor,
    )


# ============================================================
# 6. 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing PGA-Mamba UNet++ (Prior-Guided Attention)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Test 1: use_lidar=False (baseline, no prior) ---
    print("\n[1] Baseline (no prior, 3-ch pseudo-RGB)...")
    model = mamba_unetplusplus(in_channels=3, num_classes=1, use_lidar=False).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"  Input : {x.shape}")
    print(f"  Output: {out.shape}, range [{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == (2, 1, 256, 256)

    # --- Test 2: use_lidar=True + SNR prior (no real LiDAR) ---
    print("\n[2] SNR Prior mode (use_lidar=True, lidar=None)...")
    model_snr = mamba_unetplusplus(in_channels=1, num_classes=1, use_lidar=True).to(device)
    x_ir = torch.randn(2, 1, 256, 256).to(device)
    with torch.no_grad():
        out_snr = model_snr(x_ir, lidar=None)
    print(f"  Input : {x_ir.shape}")
    print(f"  Output: {out_snr.shape}, range [{out_snr.min():.3f}, {out_snr.max():.3f}]")
    assert out_snr.shape == (2, 1, 256, 256)

    # --- Test 3: use_lidar=True + real LiDAR ---
    print("\n[3] LiDAR Prior mode (use_lidar=True, lidar provided)...")
    lidar = torch.rand(2, 1, 256, 256).to(device)  # LiDAR depth in [0,1]
    with torch.no_grad():
        out_lidar = model_snr(x_ir, lidar=lidar)
    print(f"  Input : {x_ir.shape}, LiDAR: {lidar.shape}")
    print(f"  Output: {out_lidar.shape}, range [{out_lidar.min():.3f}, {out_lidar.max():.3f}]")

    # --- Test 4: Deep supervision ---
    print("\n[4] Deep supervision mode...")
    model_ds = mamba_unetplusplus(in_channels=1, num_classes=1,
                                   deep_supervision=True, use_lidar=True).to(device)
    model_ds.train()
    out_ds = model_ds(x_ir, lidar=lidar)
    assert isinstance(out_ds, list) and len(out_ds) == 4
    print(f"  Deep supervision outputs: {len(out_ds)}")
    for i, o in enumerate(out_ds):
        print(f"    output[{i}]: {o.shape}")

    # --- Test 5: PriorGuidedAttention gradient flow ---
    print("\n[5] Gradient flow check for PriorGuidedAttention...")
    model_grad = mamba_unetplusplus(in_channels=1, num_classes=1, use_lidar=True).to(device)
    x_grad = torch.randn(2, 1, 256, 256, requires_grad=False).to(device)
    lidar_grad = torch.rand(2, 1, 256, 256).to(device)
    out_grad = model_grad(x_grad, lidar=lidar_grad)
    loss = out_grad.mean()
    loss.backward()
    # Check PGA parameters have gradients
    for name, module in model_grad.named_modules():
        if isinstance(module, PriorGuidedAttention):
            assert module.prior_weight.grad is not None, f"{name}.prior_weight has no grad!"
            assert module.alpha.grad is not None, f"{name}.alpha has no grad!"
    print("  All PGA parameters received gradients.")

    # --- Parameter count ---
    total_params = sum(p.numel() for p in model_snr.parameters())
    pga_params   = sum(p.numel() for m in model_snr.modules()
                       if isinstance(m, PriorGuidedAttention) for p in m.parameters())
    print(f"\n[Summary]")
    print(f"  Total params  : {total_params:,}")
    print(f"  PGA params    : {pga_params:,} ({100*pga_params/total_params:.1f}% overhead)")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
