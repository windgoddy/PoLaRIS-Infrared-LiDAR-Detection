"""
PoLaRIS-Mamba with Progressive Decoder (V2 Architecture)
=========================================================

Key Difference from MultiScale Version:
- OLD: Downsample all features to Stage4 resolution → Fuse → Upsample 32x
- NEW: Progressive upsampling with U-Net style skip connections

Architecture (with patch_size=2):
    Stage4 (H/16, W/16, 512)
      ↓ Upsample 2x + Decoder Block
    → Fuse with Stage3 (H/8, W/8, 256)
      ↓ Upsample 2x + Decoder Block
    → Fuse with Stage2 (H/4, W/4, 128)
      ↓ Upsample 2x + Decoder Block
    → Fuse with Stage1 (H/2, W/2, 64)
      ↓ Final Head + 2x Upsample
    → Output (H, W, 1)

Benefits:
1. Preserves spatial details from shallow layers
2. No aggressive downsampling of high-res features
3. Each decoder stage refines features at native resolution
4. Better physical realism (ship contours, masts, etc.)

Expected Improvements:
- Box IoU: 0.574 → 0.70+
- Better visual quality (sharp edges, no blur)
- Faster convergence (better gradient flow)

Author: PoLaRIS Team
Date: 2026-02-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ss2d_components import VSSBlock


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with configurable patch size."""
    def __init__(self, in_channels=1, embed_dim=64, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H//P, W//P)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer (Downsampling)."""
    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        # Pad if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            B, H, W, C = x.shape

        # Split into 4 groups
        x0 = x[:, 0::2, 0::2, :]  # (B, H//2, W//2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H//2, W//2, 4C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H//2, W//2, out_dim)

        return x


class MambaStage(nn.Module):
    """A Mamba encoder stage, consisting of multiple VSSBlocks."""
    def __init__(self, dim, depth, drop_path=0., use_lidar_gate=True):
        super().__init__()

        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i],
                use_lidar_gate=use_lidar_gate,
            )
            for i in range(depth)
        ])

    def forward(self, x, lidar_feat=None):
        """
        Args:
            x: (B, H, W, C)
            lidar_feat: (B, 1, H, W) LiDAR feature (optional)
        Returns:
            out: (B, H, W, C)
        """
        for block in self.blocks:
            x = block(x, lidar_feat)
        return x


class LiDARDownsampler(nn.Module):
    """Downsamples LiDAR depth map using MaxPool."""
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, lidar):
        if self.scale_factor == 1:
            return lidar

        lidar_down = F.max_pool2d(
            lidar,
            kernel_size=self.scale_factor,
            stride=self.scale_factor,
        )
        return lidar_down


class ProgressiveDecoderBlock(nn.Module):
    """
    Progressive Decoder Block for upsampling and feature fusion.

    Process:
    1. Upsample deep features 2x (bilinear)
    2. Concat with skip connection from encoder
    3. 1x1 conv to reduce channels
    4. Mamba block to refine features
    """
    def __init__(self, deep_dim, skip_dim, out_dim, use_lidar_gate=True):
        super().__init__()

        # Channel reduction after concatenation
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(deep_dim + skip_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        # Mamba refinement block
        self.refine_block = VSSBlock(
            hidden_dim=out_dim,
            drop_path=0.1,
            use_lidar_gate=use_lidar_gate,
        )

    def forward(self, deep_feat, skip_feat, lidar_feat=None):
        """
        Args:
            deep_feat: (B, C_deep, H, W) from deeper stage
            skip_feat: (B, C_skip, 2H, 2W) from encoder skip connection
            lidar_feat: (B, 1, 2H, 2W) optional LiDAR feature
        Returns:
            out: (B, C_out, 2H, 2W)
        """
        # Upsample deep features
        B, C, H, W = deep_feat.shape
        deep_up = F.interpolate(
            deep_feat,
            size=(H * 2, W * 2),
            mode='bilinear',
            align_corners=False
        )

        # Concatenate with skip connection
        fused = torch.cat([deep_up, skip_feat], dim=1)

        # Channel reduction
        fused = self.fusion_conv(fused)  # (B, out_dim, 2H, 2W)

        # Convert to (B, H, W, C) for Mamba
        fused = fused.permute(0, 2, 3, 1).contiguous()

        # Mamba refinement
        refined = self.refine_block(fused, lidar_feat)

        # Convert back to (B, C, H, W)
        refined = refined.permute(0, 3, 1, 2).contiguous()

        return refined


class ProgressiveHead(nn.Module):
    """
    Final prediction head with lightweight refinement.

    Only needs 4x upsampling (not 32x like old version).
    """
    def __init__(self, in_dim, upscale_factor=4):
        super().__init__()
        self.upscale_factor = upscale_factor

        # Lightweight refinement
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim // 2, in_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim // 4),
            nn.ReLU(inplace=True),
        )

        # Final prediction
        self.conv_out = nn.Conv2d(in_dim // 4, 1, kernel_size=1, bias=True)

        # Initialize bias for stable training
        # NOTE: Changed from -2.0 to 0.0 to improve output confidence
        # Previous bias=-2.0 caused sigmoid(logits) to be too low (~0.12 initially)
        # This led to 50%+ samples requiring threshold=0.1 for best IoU
        nn.init.constant_(self.conv_out.bias, 0.0)

        # Upsampling
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor,
            mode='bilinear',
            align_corners=False,
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H/4, W/4) feature map
        Returns:
            output: (B, 1, H, W) heatmap in [0, 1]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        logits = self.conv_out(x)
        logits = self.upsample(logits)
        output = torch.sigmoid(logits)
        return output


class PoLaRIS_Mamba_Progressive(nn.Module):
    """
    PoLaRIS-Mamba with Progressive Decoder.

    Key Design:
    1. Encoder: 4-stage Mamba (same as before)
    2. Decoder: Progressive upsampling with U-Net style fusion
    3. Each decoder stage: Upsample → Concat → Conv → Mamba
    4. Final head: Only 4x upsampling (not 32x)

    Args:
        in_channels: Number of input channels (1 for IR)
        num_classes: Number of output classes (1 for binary)
        embed_dim: Base embedding dimension (default: 64 for tiny)
        depths: Number of blocks in each stage (default: [2, 2, 4, 2])
        drop_path_rate: Stochastic depth rate (default: 0.1)
        use_lidar: Whether to use LiDAR gating (default: True)
        patch_size: Patch size for embedding (default: 2)
        use_deep_supervision: Auxiliary heads for training (default: False)
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        embed_dim=64,
        depths=[2, 2, 4, 2],
        drop_path_rate=0.1,
        use_lidar=True,
        patch_size=2,  # Changed from 4 to 2
        use_deep_supervision=False,
    ):
        super().__init__()

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.use_lidar = use_lidar
        self.patch_size = patch_size
        self.use_deep_supervision = use_deep_supervision

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build encoder stages
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.lidar_downsamplers = nn.ModuleList()

        dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        # For tiny model with patch_size=2: dims = [64, 128, 256, 512]

        for i_stage in range(self.num_stages):
            dim = dims[i_stage]
            stage_dpr = dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])]

            # Mamba stage
            stage = MambaStage(
                dim=dim,
                depth=depths[i_stage],
                drop_path=stage_dpr,
                use_lidar_gate=use_lidar,
            )
            self.stages.append(stage)

            # Downsampler (except last stage)
            if i_stage < self.num_stages - 1:
                downsampler = PatchMerging(dim=dim, out_dim=dims[i_stage + 1])
                self.downsamplers.append(downsampler)
            else:
                self.downsamplers.append(nn.Identity())

            # LiDAR downsampler
            lidar_scale = patch_size * (2 ** i_stage)
            lidar_downsampler = LiDARDownsampler(scale_factor=lidar_scale)
            self.lidar_downsamplers.append(lidar_downsampler)

        # Progressive Decoder
        # Decode: Stage4 → Stage3 → Stage2 → Stage1
        self.decoder_blocks = nn.ModuleList()

        # Decoder 3: Stage4 → Stage3
        self.decoder_blocks.append(
            ProgressiveDecoderBlock(
                deep_dim=dims[3],      # 512
                skip_dim=dims[2],      # 256
                out_dim=dims[2],       # 256
                use_lidar_gate=use_lidar,
            )
        )

        # Decoder 2: Stage3 → Stage2
        self.decoder_blocks.append(
            ProgressiveDecoderBlock(
                deep_dim=dims[2],      # 256
                skip_dim=dims[1],      # 128
                out_dim=dims[1],       # 128
                use_lidar_gate=use_lidar,
            )
        )

        # Decoder 1: Stage2 → Stage1
        self.decoder_blocks.append(
            ProgressiveDecoderBlock(
                deep_dim=dims[1],      # 128
                skip_dim=dims[0],      # 64
                out_dim=dims[0],       # 64
                use_lidar_gate=use_lidar,
            )
        )

        # Final prediction head
        self.head = ProgressiveHead(
            in_dim=dims[0],  # 64
            upscale_factor=patch_size  # 2 (not 32!)
        )

        # Optional: Deep supervision (auxiliary heads)
        if self.use_deep_supervision:
            self.aux_head_d2 = ProgressiveHead(in_dim=dims[1], upscale_factor=patch_size * 2)
            self.aux_head_d3 = ProgressiveHead(in_dim=dims[2], upscale_factor=patch_size * 4)

        # Initialize weights
        self.apply(self._init_weights)
        self._init_prediction_heads()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_prediction_heads(self):
        """Initialize prediction head biases."""
        # NOTE: Changed from -2.0 to 0.0 (2026-02-08)
        # Reason: bias=-2.0 caused sigmoid(logits)~0.12, leading to 50%+ samples
        # requiring threshold=0.1 for best IoU. This severely limits model performance.
        if hasattr(self.head, 'conv_out'):
            nn.init.constant_(self.head.conv_out.bias, 0.0)

        if self.use_deep_supervision:
            if hasattr(self, 'aux_head_d2') and hasattr(self.aux_head_d2, 'conv_out'):
                nn.init.constant_(self.aux_head_d2.conv_out.bias, 0.0)
            if hasattr(self, 'aux_head_d3') and hasattr(self.aux_head_d3, 'conv_out'):
                nn.init.constant_(self.aux_head_d3.conv_out.bias, 0.0)

        print("✅ [Progressive] Prediction heads initialized with bias=0.0")

    def forward(self, ir_img, lidar_img=None):
        """
        Args:
            ir_img: (B, 1, H, W) infrared image
            lidar_img: (B, 1, H, W) LiDAR depth map (optional)

        Returns:
            If training with deep supervision:
                [main_output, aux_output_d2, aux_output_d3]
            Else:
                main_output (B, 1, H, W) in [0, 1]
        """
        B, C, H, W = ir_img.shape

        # Patch embedding
        x = self.patch_embed(ir_img)  # (B, H/2, W/2, 64) with patch_size=2

        # Encoder: save features for skip connections
        # encoder_feats[0]: (B, 64, H/2, W/2) - Stage1
        # encoder_feats[1]: (B, 128, H/4, W/4) - Stage2
        # encoder_feats[2]: (B, 256, H/8, W/8) - Stage3
        # encoder_feats[3]: (B, 512, H/16, W/16) - Stage4
        encoder_feats = []

        for i_stage in range(self.num_stages):
            # Downsample LiDAR to match current stage resolution
            if self.use_lidar and lidar_img is not None:
                lidar_feat = self.lidar_downsamplers[i_stage](lidar_img)
            else:
                lidar_feat = None

            # Mamba stage
            x = self.stages[i_stage](x, lidar_feat)

            # Save features (convert to B, C, H, W)
            encoder_feats.append(x.permute(0, 3, 1, 2).contiguous())

            # Downsampling (2x downsample for next stage)
            if i_stage < self.num_stages - 1:
                x = self.downsamplers[i_stage](x)

        # Progressive Decoder
        # Start from Stage4 (deepest)
        # With patch_size=2: encoder_feats[3] is H/16 resolution
        decoder_feat = encoder_feats[3]  # (B, 512, H/16, W/16)

        # Decode Stage4 → Stage3 (output: H/8)
        lidar_s3 = self.lidar_downsamplers[2](lidar_img) if (self.use_lidar and lidar_img is not None) else None
        decoder_feat = self.decoder_blocks[0](
            decoder_feat,
            encoder_feats[2],  # Skip from Stage3
            lidar_s3
        )
        feat_d3 = decoder_feat  # Save for aux head

        # Decode Stage3 → Stage2 (output: H/4)
        lidar_s2 = self.lidar_downsamplers[1](lidar_img) if (self.use_lidar and lidar_img is not None) else None
        decoder_feat = self.decoder_blocks[1](
            decoder_feat,
            encoder_feats[1],  # Skip from Stage2 (H/4)
            lidar_s2
        )
        feat_d2 = decoder_feat  # (B, 128, H/4, W/4) - Save for aux head

        # Decode Stage2 → Stage1 (output: H/2)
        lidar_s1 = self.lidar_downsamplers[0](lidar_img) if (self.use_lidar and lidar_img is not None) else None
        decoder_feat = self.decoder_blocks[2](
            decoder_feat,
            encoder_feats[0],  # Skip from Stage1 (H/2)
            lidar_s1
        )
        feat_d1 = decoder_feat  # (B, 64, H/2, W/2) - Final decoder output

        # Main prediction
        main_output = self.head(feat_d1)  # (B, 1, H, W)

        # Deep supervision
        if self.training and self.use_deep_supervision:
            aux_output_d2 = self.aux_head_d2(feat_d2)
            aux_output_d3 = self.aux_head_d3(feat_d3)
            return [main_output, aux_output_d2, aux_output_d3]

        return main_output


# ======================== Model Variants ========================

def polaris_mamba_tiny_progressive(use_lidar=True, use_deep_supervision=False):
    """
    Tiny model with Progressive Decoder.

    Parameters: ~11.5M (similar to multiscale)
    Expected Box IoU: 0.70+ (vs 0.574 for multiscale)
    """
    return PoLaRIS_Mamba_Progressive(
        embed_dim=64,
        depths=[2, 2, 4, 2],
        drop_path_rate=0.1,
        use_lidar=use_lidar,
        patch_size=2,  # Key difference: 2 instead of 4
        use_deep_supervision=use_deep_supervision,
    )


def polaris_mamba_small_progressive(use_lidar=True, use_deep_supervision=False):
    """Small model with Progressive Decoder."""
    return PoLaRIS_Mamba_Progressive(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        drop_path_rate=0.2,
        use_lidar=use_lidar,
        patch_size=2,
        use_deep_supervision=use_deep_supervision,
    )


# ======================== Unit Test ========================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing PoLaRIS_Mamba_Progressive Model")
    print("=" * 70)

    print("\n[1] Testing polaris_mamba_tiny_progressive...")
    model = polaris_mamba_tiny_progressive(use_lidar=True, use_deep_supervision=False)
    model.eval()

    ir = torch.randn(2, 1, 512, 512)
    lidar = torch.randn(2, 1, 512, 512)

    with torch.no_grad():
        output = model(ir, lidar)

    print(f"  ✓ Input: {ir.shape}")
    print(f"  ✓ Output: {output.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parameters: {num_params / 1e6:.2f}M")

    print("\n[2] Testing with deep supervision...")
    model.train()
    model.use_deep_supervision = True
    outputs = model(ir, lidar)

    if isinstance(outputs, list):
        print(f"  ✓ Deep supervision outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"    - Output {i}: {out.shape}")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
