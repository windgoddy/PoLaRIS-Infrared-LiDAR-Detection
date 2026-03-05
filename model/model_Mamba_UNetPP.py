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
from model_Mamba.core.ss2d_components import VSSBlock


class PhysicalSNRPrior(nn.Module):
    """
    Computes local SNR from raw IR image ONCE at input resolution (256×256).

    Physical reasoning:
      - At input resolution every pixel = real thermal radiance.
      - 15×15 AvgPool covers only ~0.3% of the 256×256 image → genuine LOCAL stats.
      - SNR high → isolated bright target on calm sea     → spatial prior ≈ 1.
      - SNR low  → coastline clutter (high local variance) → spatial prior ≈ 0.

    The [0,1] map is MaxPool-downsampled to each Mamba stage resolution and
    injected BEFORE (not inside) the SSM, preserving physical meaning at all scales.
    No learnable parameters — pure closed-form physics.
    """
    def __init__(self, kernel_size: int = 15):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) single-channel IR image (raw radiance)
        Returns:
            snr_norm: (B, 1, H, W) in [0, 1]
        """
        mu     = self.avg_pool(x)
        sq_avg = self.avg_pool(x ** 2)
        var    = torch.clamp(sq_avg - mu ** 2, min=1e-6)
        sigma  = torch.sqrt(var)
        snr    = F.relu(x - mu) / (sigma + 1e-4)

        # Per-image min-max normalise → [0, 1]
        B, C, H, W = snr.shape
        snr_flat = snr.view(B, C, -1)
        snr_min  = snr_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        snr_max  = snr_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        return (snr - snr_min) / (snr_max - snr_min + 1e-8)

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
        self.use_lidar = use_lidar   # enable physical prior injection

        # PhysicalSNRPrior: no learnable parameters; computes closed-form SNR map
        # at input resolution. Only instantiated when use_lidar=True.
        if self.use_lidar:
            self.snr_prior = PhysicalSNRPrior(kernel_size=15)

        # ===== Encoder (CNN for local, Mamba for global) =====
        # Shallow layers: CNN blocks (local texture, edge)
        self.conv0_0 = self._make_cnn_block(input_channels, nb_filter[0])
        self.conv1_0 = self._make_cnn_block(nb_filter[0], nb_filter[1])

        # Deep layers: Mamba blocks (global context + prior-guided)
        self.conv2_0 = self._make_mamba_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = self._make_mamba_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = self._make_mamba_block(nb_filter[3], nb_filter[4])

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
        self.conv2_1 = self._make_mamba_block(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv2_2 = self._make_mamba_block(nb_filter[2]*2 + nb_filter[3], nb_filter[2])

        # Level 3 (32x32)
        self.conv3_1 = self._make_mamba_block(nb_filter[3] + nb_filter[4], nb_filter[3])

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

    def _make_mamba_block(self, in_ch, out_ch):
        """Mamba block wrapper for UNet integration."""
        return MambaBlockWrapper(in_ch, out_ch)

    def forward(self, x, lidar=None):
        """
        Forward pass with dense skip connections and physical prior injection.

        Args:
            x    : (B, C, H, W) IR image (1-ch)
            lidar: (B, 1, H, W) LiDAR depth (Exp C) or None/zeros (Exp B → SNR prior)

        Returns:
            deep_supervision=True : [output4, output3, output2, output1] (main first)
            deep_supervision=False: (B, 1, H, W) final prediction
        """
        # ===== Physical Prior Pyramid =====
        # Computed ONCE at input resolution; no semantic contamination.
        # PhysicalSNRPrior has no learnable params — safe inside no_grad.
        if self.use_lidar:
            with torch.no_grad():
                lidar_is_real = (lidar is not None and
                                 lidar.abs().max().item() > 1e-6)
                if lidar_is_real:
                    # Exp C: normalised LiDAR depth as spatial prior
                    prior_full = lidar.clamp(min=0)
                    B, C, H, W = prior_full.shape
                    p_max = prior_full.view(B, C, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    prior_full = prior_full / (p_max + 1e-8)
                else:
                    # Exp B: Physical SNR from raw IR (kernel=15 → 0.3% of 256² area)
                    prior_full = self.snr_prior(x[:, :1])  # (B, 1, H, W)

                # MaxPool downsample to Mamba stage resolutions (preserves peaks)
                prior_64 = F.max_pool2d(prior_full, kernel_size=4,  stride=4)   # (B,1,64,64)
                prior_32 = F.max_pool2d(prior_full, kernel_size=8,  stride=8)   # (B,1,32,32)
                prior_16 = F.max_pool2d(prior_full, kernel_size=16, stride=16)  # (B,1,16,16)
        else:
            prior_64 = prior_32 = prior_16 = None

        # ===== Encoder =====
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))

        x2_0 = self.conv2_0(self.pool(x1_0), prior_64)
        x3_0 = self.conv3_0(self.pool(x2_0), prior_32)
        x4_0 = self.conv4_0(self.pool(x3_0), prior_16)

        # ===== Decoder with Dense Connections =====
        # UNet++ 规则：X^{i,j} = Conv([X^{i,0}, ..., X^{i,j-1}, Upsample(X^{i+1,j-1})])
        # 只连接同一深度的之前节点 + 下一深度的上采样节点

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
    Wrapper making VSSBlock compatible with the UNet++ structure.

    Physical SNR Prior Architecture (2026-03-05):
    1. Channel projection    (in_ch → out_ch)
    2. Spatial prior injection BEFORE Mamba:
           x_in = identity * (1 + prior_scale * spatial_prior)
       - prior_scale: learnable scalar, init=0 → pure identity at epoch 0
       - spatial_prior: downsampled PhysicalSNRPrior (or LiDAR depth), in [0,1]
       - Ships (high local SNR) get amplified into Mamba; coastline suppressed.
       - Prior computed at INPUT resolution — physical meaning preserved at all depths.
    3. VSSBlock   (pure Mamba, no internal gating)
    4. spatial_restore depthwise conv (bridges Mamba global ↔ CNN local)
    5. Residual connection (identity WITHOUT prior, stable gradient flow)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.proj_in = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # Trust scalar for physical prior.
        # Init=0 → identity.  Model learns how much to rely on the prior at each scale.
        # Gradient flows correctly: d(loss)/d(prior_scale) uses prior as a fixed constant.
        self.prior_scale = nn.Parameter(torch.zeros(1))

        self.mamba = VSSBlock(hidden_dim=out_ch, drop_path=0.1)
        self.spatial_restore = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch)

    def forward(self, x, spatial_prior=None):
        """
        Args:
            x             : (B, C, H, W)
            spatial_prior : (B, 1, H, W) downsampled physical prior in [0,1], or None
        Returns:
            out: (B, out_ch, H, W)
        """
        identity = self.proj_in(x)                       # (B, out_ch, H, W) — residual

        # Inject prior into Mamba input path (not the residual path)
        x_in = identity
        if spatial_prior is not None:
            x_in = identity * (1.0 + self.prior_scale * spatial_prior)

        B, C, H, W = x_in.shape
        x_mamba = x_in.permute(0, 2, 3, 1)              # (B, H, W, C)
        x_mamba = self.mamba(x_mamba)                    # (B, H, W, C)  — pure Mamba
        x_mamba = x_mamba.permute(0, 3, 1, 2)           # (B, C, H, W)
        x_mamba = self.spatial_restore(x_mamba)          # local feature restoration

        return identity + x_mamba                        # residual skips the prior path


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
