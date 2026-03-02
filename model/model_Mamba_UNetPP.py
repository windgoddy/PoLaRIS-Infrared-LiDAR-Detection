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

    def __init__(self, num_classes=1, input_channels=3, nb_filter=[16, 32, 64, 128, 256]):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ===== Encoder (保留CNN特征提取能力) =====
        # Shallow layers: CNN blocks (for local features)
        self.conv0_0 = self._make_cnn_block(input_channels, nb_filter[0])
        self.conv1_0 = self._make_cnn_block(nb_filter[0], nb_filter[1])

        # Deep layers: Mamba blocks (for global context)
        self.conv2_0 = self._make_mamba_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = self._make_mamba_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = self._make_mamba_block(nb_filter[3], nb_filter[4])

        # ===== Decoder with Dense Connections (UNet++ style) =====
        # Level 0
        self.conv0_1 = self._make_cnn_block(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv0_2 = self._make_cnn_block(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv0_3 = self._make_cnn_block(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv0_4 = self._make_cnn_block(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        # Level 1
        self.conv1_1 = self._make_cnn_block(nb_filter[1] + nb_filter[2] + nb_filter[0], nb_filter[1])
        self.conv1_2 = self._make_cnn_block(nb_filter[1]*2 + nb_filter[2] + nb_filter[0], nb_filter[1])
        self.conv1_3 = self._make_cnn_block(nb_filter[1]*3 + nb_filter[2] + nb_filter[0], nb_filter[1])

        # Level 2
        self.conv2_1 = self._make_mamba_block(nb_filter[2] + nb_filter[3] + nb_filter[1], nb_filter[2])
        self.conv2_2 = self._make_mamba_block(nb_filter[2]*2 + nb_filter[3] + nb_filter[1], nb_filter[2])

        # Level 3
        self.conv3_1 = self._make_mamba_block(nb_filter[3] + nb_filter[4] + nb_filter[2], nb_filter[3])

        # Final output
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_cnn_block(self, in_ch, out_ch):
        """Standard CNN block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _make_mamba_block(self, in_ch, out_ch):
        """
        Mamba block wrapper for UNet integration.

        Note: VSSBlock expects (B, H, W, C) format,
        so we need to add permute operations.
        """
        return MambaBlockWrapper(in_ch, out_ch)

    def forward(self, x):
        """
        Forward pass with dense skip connections.

        Args:
            x: (B, C, H, W) input image

        Returns:
            output: (B, 1, H, W) prediction map
        """
        # ===== Encoder =====
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # ===== Decoder with Dense Connections =====
        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), x0_1], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), x1_1], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), x2_1], 1))

        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), x0_2], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), x1_2], 1))

        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), x0_3], 1))

        # Column 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # Final prediction
        output = self.final(x0_4)
        return output


class MambaBlockWrapper(nn.Module):
    """
    Wrapper to make VSSBlock compatible with UNet structure.

    Handles:
    1. Channel conversion (in_ch → out_ch)
    2. Format conversion (BCHW ↔ BHWC)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # Project channels if needed
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # Mamba block (expects BHWC format)
        self.mamba = VSSBlock(
            hidden_dim=out_ch,
            drop_path=0.1,
            use_lidar_gate=False,  # Disable LiDAR for now
        )

        # Batch norm for stability
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, out_ch, H, W)
        """
        # Channel projection
        x = self.proj_in(x)  # (B, out_ch, H, W)

        # Convert to Mamba format
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Mamba block
        x = self.mamba(x)  # (B, H, W, C)

        # Convert back
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.norm(x)

        return x


# ===== Factory function for easy instantiation =====
def mamba_unetplusplus(in_channels=3, num_classes=1):
    """
    Create Mamba-UNet++ model.

    Args:
        in_channels: Number of input channels (1=IR, 2=IR+Depth, 3=RGB)
        num_classes: Number of output classes (1 for binary segmentation)

    Returns:
        model: MambaUNetPlusPlus instance

    Example:
        >>> model = mamba_unetplusplus(in_channels=3, num_classes=1)
        >>> x = torch.randn(4, 3, 256, 256)
        >>> out = model(x)
        >>> print(out.shape)  # (4, 1, 256, 256)
    """
    return MambaUNetPlusPlus(
        num_classes=num_classes,
        input_channels=in_channels,
        nb_filter=[16, 32, 64, 128, 256],  # Same as DNANet
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
