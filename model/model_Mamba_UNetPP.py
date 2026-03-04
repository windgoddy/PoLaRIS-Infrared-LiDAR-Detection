"""
Mamba-UNet++: Combining Mamba with Dense Skip Connections
==========================================================

Architecture:
- Encoder: Mamba blocks (long-range dependency)
- Decoder: Dense nested structure (UNet++ style)
- Skip connections: Multi-scale feature fusion

Key Innovation:
- Mamba for global context
- UNet++ for local details and small targets
- Best of both worlds

Author: PoLaRIS Team
Date: 2026-03-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Mamba block from existing code
from model_Mamba.core.polaris_mamba import VSSBlock

class MambaUNetPlusPlus(nn.Module):
    """
    Mamba-UNet++: Hybrid architecture for small target detection.

    Key differences from pure Mamba:
    1. Dense skip connections (from UNet++)
    2. Multi-scale feature fusion
    3. Hybrid blocks (CNN shallow + Mamba deep)

    Key differences from DNANet:
    1. Mamba blocks for global modeling
    2. Linear complexity vs quadratic (Transformer)
    """

    def __init__(self, num_classes=1, input_channels=3, nb_filter=[16, 32, 64, 128, 256],
                 deep_supervision=False, use_lidar=False):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deep_supervision = deep_supervision
        self.use_lidar = use_lidar   # [LiDAR] enable SS2D gating

        # ===== Encoder (保留CNN特征提取能力) =====
        # Shallow layers: CNN blocks (for local features)
        self.conv0_0 = self._make_cnn_block(input_channels, nb_filter[0])
        self.conv1_0 = self._make_cnn_block(nb_filter[0], nb_filter[1])

        # Deep layers: Mamba blocks (for global context)
        self.conv2_0 = self._make_mamba_block(nb_filter[1], nb_filter[2], use_lidar)
        self.conv3_0 = self._make_mamba_block(nb_filter[2], nb_filter[3], use_lidar)
        self.conv4_0 = self._make_mamba_block(nb_filter[3], nb_filter[4], use_lidar)

        # ===== Decoder with Dense Connections (UNet++ style) =====
        # Level 0 (shallowest, 256x256)
        self.conv0_1 = self._make_cnn_block(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv0_2 = self._make_cnn_block(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv0_3 = self._make_cnn_block(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv0_4 = self._make_cnn_block(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        # Level 1 (128x128)
        self.conv1_1 = self._make_cnn_block(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv1_2 = self._make_cnn_block(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv1_3 = self._make_cnn_block(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        # Level 2 (64x64)
        self.conv2_1 = self._make_mamba_block(nb_filter[2] + nb_filter[3], nb_filter[2], use_lidar)
        self.conv2_2 = self._make_mamba_block(nb_filter[2]*2 + nb_filter[3], nb_filter[2], use_lidar)

        # Level 3 (32x32)
        self.conv3_1 = self._make_mamba_block(nb_filter[3] + nb_filter[4], nb_filter[3], use_lidar)

        # Final output layers
        # [FIX 2026-03-02] Deep Supervision: multiple output heads for UNet++ columns
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_cnn_block(self, in_ch, out_ch):
        """
        CNN block with GroupNorm (Better for small batch size)

        [FIX 2026-03-03] Replaced BatchNorm with GroupNorm
        - BatchNorm is unstable with small batch size (4)
        - GroupNorm is independent of batch size
        - Standard solution for medical/small-target detection
        """
        # GroupNorm: 8 groups for channels < 64, 16 groups for channels >= 64
        # Ensures each group has at least 2-4 channels
        num_groups = 8 if out_ch < 64 else 16

        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),  # ✅ Replaced BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),  # ✅ Replaced BatchNorm
            nn.ReLU(inplace=True),
        )

    def _make_mamba_block(self, in_ch, out_ch, use_lidar=False):
        """
        Mamba block wrapper for UNet integration.

        Note: VSSBlock expects (B, H, W, C) format,
        so we need to add permute operations.
        """
        return MambaBlockWrapper(in_ch, out_ch, use_lidar_gate=use_lidar)

    def forward(self, x, lidar=None):
        """
        Forward pass with dense skip connections.

        Args:
            x: (B, C, H, W) input image (1-ch IR when use_lidar=True, 3-ch otherwise)
            lidar: (B, 1, H, W) LiDAR depth map (optional; used only when use_lidar=True)

        Returns:
            If deep_supervision=True:
                [output1, output2, output3, output4]: List of predictions from each UNet++ column
            Else:
                output: (B, 1, H, W) final prediction map
        """
        # ===== Encoder =====
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))

        # [LiDAR] Pre-downsample depth map to match Mamba stage spatial resolutions.
        # Input x is (B, C, H, W); after successive 2x pools: H/4, H/8, H/16.
        if lidar is not None and self.use_lidar:
            lidar_f4  = F.max_pool2d(lidar, kernel_size=4,  stride=4)   # (B,1,H/4, W/4)
            lidar_f8  = F.max_pool2d(lidar, kernel_size=8,  stride=8)   # (B,1,H/8, W/8)
            lidar_f16 = F.max_pool2d(lidar, kernel_size=16, stride=16)  # (B,1,H/16,W/16)
        else:
            lidar_f4 = lidar_f8 = lidar_f16 = None

        x2_0 = self.conv2_0(self.pool(x1_0), lidar_f4)
        x3_0 = self.conv3_0(self.pool(x2_0), lidar_f8)
        x4_0 = self.conv4_0(self.pool(x3_0), lidar_f16)

        # ===== Decoder with Dense Connections =====
        # UNet++ 规则：X^{i,j} = Conv([X^{i,0}, ..., X^{i,j-1}, Upsample(X^{i+1,j-1})])
        # 只连接同一深度的之前节点 + 下一深度的上采样节点

        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1), lidar_f4)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1), lidar_f8)

        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1), lidar_f4)

        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        # Column 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # [FIX 2026-03-02] Deep Supervision: return multiple outputs
        if self.deep_supervision:
            # Apply final conv + sigmoid to each column
            output1 = torch.sigmoid(self.final1(x0_1))
            output2 = torch.sigmoid(self.final2(x0_2))
            output3 = torch.sigmoid(self.final3(x0_3))
            output4 = torch.sigmoid(self.final4(x0_4))
            return [output4, output3, output2, output1]  # Main output first
        else:
            # Standard mode: only return final output
            output = self.final(x0_4)
            # Apply sigmoid to convert logits to probabilities [0, 1]
            # Required by loss functions (FocalBCE, Dice, BoxProjection)
            output = torch.sigmoid(output)
            return output


class MambaBlockWrapper(nn.Module):
    """
    Wrapper to make VSSBlock compatible with UNet structure.

    Handles:
    1. Channel conversion (in_ch → out_ch)
    2. Format conversion (BCHW ↔ BHWC)
    3. [FIX 2026-03-02] Spatial restore: bridge the semantic gap between Mamba and CNN
    4. [LiDAR] Optional LiDAR gating via SS2D lidar_gate_conv
    """
    def __init__(self, in_ch, out_ch, use_lidar_gate=False):
        super().__init__()

        # Project channels if needed
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # Mamba block (expects BHWC format)
        # VSSBlock already contains internal LayerNorm; no extra norm needed here
        self.mamba = VSSBlock(
            hidden_dim=out_ch,
            drop_path=0.1,
            use_lidar_gate=use_lidar_gate,  # [LiDAR] controlled by caller
        )

        # [FIX 2026-03-02] Spatial restore layer
        # Use depthwise convolution to convert global Mamba features back to local-friendly features
        # This bridges the semantic gap between Mamba's global context and CNN's local features
        self.spatial_restore = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch)

    def forward(self, x, lidar=None):
        """
        Args:
            x: (B, C, H, W)
            lidar: (B, 1, H, W) LiDAR depth at this stage's resolution (optional)
        Returns:
            out: (B, out_ch, H, W)
        """
        # Channel projection
        identity = self.proj_in(x)  # (B, out_ch, H, W) - save for residual

        # Convert to Mamba format
        B, C, H, W = identity.shape
        x_mamba = identity.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Mamba block (global context modeling)
        # [LiDAR] Pass lidar_feat so SS2D can apply gating if use_lidar_gate=True
        x_mamba = self.mamba(x_mamba, lidar)  # (B, H, W, C)

        # Convert back
        x_mamba = x_mamba.permute(0, 3, 1, 2)  # (B, C, H, W)

        # [FIX 2026-03-02] Restore spatial details with depthwise conv
        x_mamba = self.spatial_restore(x_mamba)

        # [FIX 2026-03-02] Add residual connection for stability
        return identity + x_mamba


# ===== Factory function for easy instantiation =====
def mamba_unetplusplus(in_channels=3, num_classes=1, deep_supervision=False, use_lidar=False):
    """
    Create Mamba-UNet++ model.

    Args:
        in_channels: Number of input channels for the IR branch
                     (1 when using PoLaRIS loader with in_channels=2 / LiDAR split off,
                      3 when using 8-bit RGB-replicated loader without LiDAR)
        num_classes: Number of output classes (1 for binary segmentation)
        deep_supervision: Enable deep supervision (recommended for small targets)
        use_lidar: Enable SS2D LiDAR gating in all Mamba blocks

    Returns:
        model: MambaUNetPlusPlus instance

    Example:
        >>> model = mamba_unetplusplus(in_channels=1, num_classes=1,
        ...                            deep_supervision=True, use_lidar=True)
        >>> ir = torch.randn(4, 1, 256, 256)
        >>> lidar = torch.randn(4, 1, 256, 256)
        >>> out = model(ir, lidar)
    """
    return MambaUNetPlusPlus(
        num_classes=num_classes,
        input_channels=in_channels,
        nb_filter=[16, 32, 64, 128, 256],  # Same as DNANet
        deep_supervision=deep_supervision,
        use_lidar=use_lidar,
    )


if __name__ == "__main__":
    # Test the model
    print("="*70)
    print("Testing Mamba-UNet++")
    print("="*70)

    model = mamba_unetplusplus(in_channels=3, num_classes=1)

    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n✅ Model test passed!")
