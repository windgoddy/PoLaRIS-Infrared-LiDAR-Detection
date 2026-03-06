"""
DNANet-SNR: DNANet + Physical SNR Prior Injection
==================================================

在 DNANet 编码器前三层注入无参数物理 SNR 先验，无需修改解码器和跳跃连接。

核心创新点:
  PhysicalSNRPrior  - 闭合式局部信噪比计算 (无可学习参数)
                      SNR = ReLU(I - mu) / (sigma + 1e-4)
  SNRPriorGate      - 轻量 1x1 卷积注意力门
                      output = x + x * sigmoid(W * [x, prior])
  注入位置          - x0_0 (256x256), x1_0 (128x128), x2_0 (64x64)
  先验金字塔        - MaxPool 下采样，保留稀疏小目标的 SNR 峰值

参数量新增:
  SNRPriorGate x3:  (nb_filter[0]+1)*nb_filter[0]
                  + (nb_filter[1]+1)*nb_filter[1]
                  + (nb_filter[2]+1)*nb_filter[2]
  对于 channel_size=three: (16+1)*16 + (32+1)*32 + (64+1)*64 = 272+1056+4160 = 5488 个参数
  相比 DNANet 总量几乎可以忽略不计。

Author: PoLaRIS Team
Date: 2026-03-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_DNANet import Res_CBAM_block


# ============================================================
# 1. 物理 SNR 先验生成器（无可学习参数）
# ============================================================

class PhysicalSNRPrior(nn.Module):
    """
    局部信噪比计算器，完全基于物理公式，无可学习参数。

    公式: SNR = ReLU(I - mu) / (sigma + eps)
      mu    = 局部均值 (15x15 AvgPool)
      sigma = 局部标准差 (由二阶矩推导)

    输入支持单通道或多通道（多通道先取均值转灰度）。
    输出经 per-image min-max 归一化至 [0, 1]。
    """
    def __init__(self, kernel_size: int = 15):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)，单通道或 RGB 红外图像
        Returns:
            snr_norm: (B, 1, H, W)，归一化 SNR 先验图，值域 [0, 1]
        """
        # 多通道取均值转灰度（红外 RGB 三通道通常为同一通道的副本）
        if x.shape[1] > 1:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x

        mu     = self.avg_pool(x_gray)
        sq_avg = self.avg_pool(x_gray ** 2)
        var    = torch.clamp(sq_avg - mu ** 2, min=1e-6)
        sigma  = torch.sqrt(var)
        snr    = F.relu(x_gray - mu) / (sigma + 1e-4)

        # Per-image min-max 归一化
        B, C, H, W = snr.shape
        snr_flat = snr.view(B, C, -1)
        snr_min  = snr_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        snr_max  = snr_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        return (snr - snr_min) / (snr_max - snr_min + 1e-8)


# ============================================================
# 2. 轻量 SNR 先验注入门
# ============================================================

class SNRPriorGate(nn.Module):
    """
    将 SNR 先验图融合进特征图，放大高 SNR 区域的响应。

    设计原则:
      - 1x1 卷积 + BN + Sigmoid 生成注意力权重（与 DNANet 保持 BN 风格一致）
      - 残差连接: output = x + x * attn
        当 attn→0（低 SNR 区域），output ≈ x（不退化）
        当 attn→1（高 SNR 区域），output ≈ 2x（强化目标特征）
      - 极少参数: (ch+1) * ch 个权重
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C, H, W) 特征图
            prior: (B, 1, H, W) SNR 先验，值域 [0, 1]，与 x 空间分辨率一致
        Returns:
            (B, C, H, W) SNR 增强后的特征
        """
        attn = self.gate(torch.cat([x, prior], dim=1))
        return x + x * attn


# ============================================================
# 3. DNANet-SNR 主体
# ============================================================

class DNANet_SNR(nn.Module):
    """
    DNANet with Physical SNR Prior Gates.

    相对于原始 DNANet 的唯一改动:
      1. __init__ 新增 snr_prior, gate0, gate1, gate2
      2. forward 在 x0_0/x1_0/x2_0 后插入对应 gate

    解码器、跳跃连接、输出头完全不变。
    """
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,
                 deep_supervision=False):
        super(DNANet_SNR, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        # === 编码器（与原始 DNANet 完全一致）===
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        # === 解码器（与原始 DNANet 完全一致）===
        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],
                                        nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],
                                        nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],
                                        nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2] + nb_filter[0],
                                        nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3] + nb_filter[1],
                                        nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2] + nb_filter[0],
                                        nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])
        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        # === 输出头（与原始 DNANet 完全一致）===
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        # === SNR 先验注入模块（新增）===
        self.snr_prior = PhysicalSNRPrior(kernel_size=15)
        self.gate0 = SNRPriorGate(nb_filter[0])   # 256x256 级别
        self.gate1 = SNRPriorGate(nb_filter[1])   # 128x128 级别
        self.gate2 = SNRPriorGate(nb_filter[2])   # 64x64  级别

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = [block(input_channels, output_channels)]
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        # === SNR 先验金字塔（MaxPool 保留稀疏峰值）===
        prior_256 = self.snr_prior(input)
        prior_128 = F.max_pool2d(prior_256, kernel_size=2, stride=2)
        prior_64  = F.max_pool2d(prior_256, kernel_size=4, stride=4)

        # === 编码器（前三层注入 SNR 先验）===
        x0_0 = self.gate0(self.conv0_0(input),          prior_256)
        x1_0 = self.gate1(self.conv1_0(self.pool(x0_0)), prior_128)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.gate2(self.conv2_0(self.pool(x1_0)), prior_64)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),
                       self.up_8 (self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)),
                       self.up   (self.conv0_1_1x1(x1_3)),
                       x0_4], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(Final_x0_4)
            return output
