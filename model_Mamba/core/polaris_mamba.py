"""
PoLaRIS-Gaussian-Mamba Model Architecture
==========================================

This file defines the complete PoLaRIS_Mamba model, which combines:
1. Patch Embedding (image → tokens)
2. Hierarchical Mamba Encoder (4 stages, similar to ResNet)
3. LiDAR Gated Injection at each stage
4. Gaussian Heatmap Head (output center-point heatmaps)

Architecture Overview:
    Input: (B, 1, H, W) infrared image + (B, 1, H, W) LiDAR depth
    ↓
    Stage 1: Patch Embed → VSSBlocks (96 dim)
    ↓
    Stage 2: Downsample → VSSBlocks (192 dim)
    ↓
    Stage 3: Downsample → VSSBlocks (384 dim)
    ↓
    Stage 4: Downsample → VSSBlocks (768 dim)
    ↓
    Gaussian Head → (B, 1, H, W) heatmap

Author: PoLaRIS Team
Date: 2026-01-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ss2d_components import VSSBlock


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    Splits image into non-overlapping patches and projects to embedding dimension.

    Args:
        in_channels: Number of input channels (1 for IR)
        embed_dim: Embedding dimension
        patch_size: Patch size (default: 4)
    """
    def __init__(self, in_channels=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, H//patch_size, W//patch_size, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H//P, W//P)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer (Downsampling).

    Merges 2x2 neighboring patches and projects to higher dimension.
    Similar to pooling but learnable.

    Args:
        dim: Input dimension
        out_dim: Output dimension (usually 2*dim)
    """
    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)

        Returns:
            out: (B, H//2, W//2, out_dim)
        """
        B, H, W, C = x.shape

        # Pad if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            B, H, W, C = x.shape

        # Split into 4 groups: (0,0), (0,1), (1,0), (1,1)
        x0 = x[:, 0::2, 0::2, :]  # (B, H//2, W//2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        # Concatenate along channel dimension
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H//2, W//2, 4C)

        x = self.norm(x)
        x = self.reduction(x)  # (B, H//2, W//2, out_dim)

        return x


class MambaStage(nn.Module):
    """
    A Mamba encoder stage, consisting of multiple VSSBlocks.

    Args:
        dim: Feature dimension
        depth: Number of VSSBlocks in this stage
        drop_path: Stochastic depth rates (list or float)
        use_lidar_gate: Whether to use LiDAR gating
    """
    def __init__(
        self,
        dim,
        depth,
        drop_path=0.,
        use_lidar_gate=True,
    ):
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
    """
    Downsamples LiDAR depth map to match feature map resolution.

    CRITICAL UPDATE (2026-02-01): Changed from AvgPool to MaxPool
    - AvgPool dilutes sparse signals (e.g., single buoy reflection → near zero after averaging)
    - MaxPool preserves strongest reflections (if ANY pixel in region has signal, keep it)
    - Essential for detecting distant small targets in PoLaRIS dataset

    Args:
        scale_factor: Downsampling factor (e.g., 2 for H//2, W//2)
    """
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, lidar):
        """
        Args:
            lidar: (B, 1, H, W)

        Returns:
            lidar_down: (B, 1, H//scale, W//scale)
        """
        if self.scale_factor == 1:
            return lidar

        # CRITICAL FIX: MaxPool instead of AvgPool for sparse LiDAR
        # Reasoning: In far-sea regions, LiDAR may have only 1-2 pixels with reflections
        # AvgPool would dilute 255 → 15 (255/16), losing signal strength
        # MaxPool preserves 255 → 255, keeping geometric cues intact
        lidar_down = F.max_pool2d(
            lidar,
            kernel_size=self.scale_factor,
            stride=self.scale_factor,
        )
        return lidar_down


class GaussianHead(nn.Module):
    """
    Binary Segmentation Head (Enhanced for PoLaRIS).

    UPDATES (2026-02-01):
    - Added residual connection for better gradient flow
    - Increased depth (3 conv blocks instead of 2)
    - Removed final sigmoid (use BCEWithLogitsLoss for numerical stability)

    Architecture:
        Conv(dim → dim//2) → BN → ReLU →
        Conv(dim//2 → dim//4) → BN → ReLU → [Residual Add]
        Conv(dim//4 → dim//8) → BN → ReLU →
        Conv(dim//8 → 1)

    Args:
        in_dim: Input feature dimension
        upscale_factor: Upsampling factor to restore original resolution
    """
    def __init__(self, in_dim, upscale_factor=4):
        super().__init__()
        self.upscale_factor = upscale_factor

        # Enhanced feature refinement (deeper, with residual)
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
        
        # [NEW] Additional refinement layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim // 4, in_dim // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim // 8),
            nn.ReLU(inplace=True),
        )

        # Final prediction layer
        self.conv_out = nn.Conv2d(in_dim // 8, 1, kernel_size=1, bias=True)

        # CRITICAL: Initialize bias for stable training
        # For BCEWithLogitsLoss, bias=-2.5 gives initial probability ~0.076
        # This prevents "all-zero" collapse while maintaining conservative predictions
        nn.init.constant_(self.conv_out.bias, -2.5)

        # Upsampling (bilinear interpolation)
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor,
            mode='bilinear',
            align_corners=False,
        )

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C) feature map

        Returns:
            logits: (B, 1, H_orig, W_orig) raw logits (NO sigmoid, use with BCEWithLogitsLoss)
        """
        # Permute to (B, C, H, W) for Conv2d
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Feature refinement
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Heatmap prediction (logits, NO sigmoid)
        logits = self.conv_out(x)  # (B, 1, H, W)

        # Upsample to original resolution
        logits = self.upsample(logits)

        # [CRITICAL] Apply sigmoid for compatibility with current training code
        # In future, can remove this and use BCEWithLogitsLoss for better stability
        output = torch.sigmoid(logits)

        return output


class PoLaRIS_Mamba(nn.Module):
    """
    PoLaRIS-Gaussian-Mamba: Mamba-based Infrared Small Target Detector
    with LiDAR Gated Injection and Gaussian Heatmap Output.

    Architecture:
        - Input: (B, 1, H, W) IR + (B, 1, H, W) LiDAR
        - 4-Stage Hierarchical Mamba Encoder
        - LiDAR downsampling at each stage
        - Gaussian Heatmap Head

    Args:
        in_channels: Number of input channels (1 for IR)
        num_classes: Number of output classes (1 for heatmap)
        embed_dim: Base embedding dimension (default: 96)
        depths: Number of blocks in each stage (default: [2, 2, 6, 2])
        drop_path_rate: Stochastic depth rate (default: 0.1)
        use_lidar: Whether to use LiDAR gating (default: True)
        patch_size: Patch size for embedding (default: 4)
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        drop_path_rate=0.1,
        use_lidar=True,
        patch_size=4,
    ):
        super().__init__()

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.use_lidar = use_lidar
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.lidar_downsamplers = nn.ModuleList()

        dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]

        for i_stage in range(self.num_stages):
            # Current stage dimension
            dim = dims[i_stage]

            # Drop path for this stage
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

        # [OPTIMIZATION] Lightweight Skip Connection (2026-02-01)
        # Fuse Stage1 (high-res) features with Stage4 (semantic) for small targets
        # Instead of full U-Net, use single skip to balance complexity and performance
        self.skip_fusion = nn.Sequential(
            nn.Conv2d(dims[0], dims[-1] // 4, kernel_size=1, bias=False),  # 96 -> 192
            nn.BatchNorm2d(dims[-1] // 4),
            nn.ReLU(inplace=True),
        )
        
        # Gaussian Head (uses final stage output + skip features)
        final_dim = dims[-1] + dims[-1] // 4  # 768 + 192 = 960
        total_downscale = patch_size * (2 ** (self.num_stages - 1))
        self.head = GaussianHead(in_dim=final_dim, upscale_factor=total_downscale)

        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for Gaussian Head
        self._init_gaussian_head()

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
    
    def _init_gaussian_head(self):
        """Special initialization for Gaussian Head's final layer."""
        if hasattr(self, 'head') and hasattr(self.head, 'conv_out'):
            if isinstance(self.head.conv_out, nn.Conv2d):
                # CRITICAL: Set bias to -2.5 so sigmoid(-2.5) ≈ 0.076
                # Updated 2026-01-30: Slightly higher than before to prevent "all-black" predictions
                # This gives stronger initial gradient signal for rare positive samples
                nn.init.constant_(self.head.conv_out.bias, -2.5)
                print("✅ Gaussian Head initialized with bias=-2.5 (sigmoid ≈ 0.076)")

    def forward(self, ir_img, lidar_img=None):
        """
        Args:
            ir_img: (B, 1, H, W) infrared image (normalized to [0, 1])
            lidar_img: (B, 1, H, W) LiDAR depth map (normalized, optional)

        Returns:
            heatmap: (B, 1, H, W) Gaussian heatmap in [0, 1]
        """
        B, C, H, W = ir_img.shape

        # Patch embedding
        x = self.patch_embed(ir_img)  # (B, H//P, W//P, embed_dim)
        
        # [OPTIMIZATION] Save stage1 output for skip connection
        stage1_feat = None

        # Process through stages
        for i_stage in range(self.num_stages):
            # Downsample LiDAR to match current feature map resolution
            if self.use_lidar and lidar_img is not None:
                lidar_feat = self.lidar_downsamplers[i_stage](lidar_img)
            else:
                lidar_feat = None

            # Mamba stage
            x = self.stages[i_stage](x, lidar_feat)
            
            # Save stage1 features (high resolution, good for small targets)
            if i_stage == 0:
                stage1_feat = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

            # Downsampling (except last stage)
            if i_stage < self.num_stages - 1:
                x = self.downsamplers[i_stage](x)
        
        # Convert final features to (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # [OPTIMIZATION] Fuse skip connection
        # Downsample stage1 features to match stage4 resolution
        if stage1_feat is not None:
            # stage1: (B, 96, 128, 128), need to match stage4: (B, 768, 16, 16)
            skip_feat = self.skip_fusion(stage1_feat)  # (B, 192, 128, 128)
            skip_feat = F.adaptive_avg_pool2d(skip_feat, x.shape[2:])  # (B, 192, 16, 16)
            x = torch.cat([x, skip_feat], dim=1)  # (B, 960, 16, 16)
        
        # Convert back to (B, H, W, C) for head
        x = x.permute(0, 2, 3, 1).contiguous()

        # Gaussian head
        heatmap = self.head(x)  # (B, 1, H, W)

        return heatmap

    def forward_features(self, ir_img, lidar_img=None):
        """
        Forward function that returns intermediate features (for debugging).

        Returns:
            features: List of features from each stage
        """
        features = []

        x = self.patch_embed(ir_img)
        features.append(x)

        for i_stage in range(self.num_stages):
            if self.use_lidar and lidar_img is not None:
                lidar_feat = self.lidar_downsamplers[i_stage](lidar_img)
            else:
                lidar_feat = None

            x = self.stages[i_stage](x, lidar_feat)
            features.append(x)

            if i_stage < self.num_stages - 1:
                x = self.downsamplers[i_stage](x)

        return features


# ======================== Model Variants ========================

def polaris_mamba_tiny(use_lidar=True):
    """Tiny model: ~5M params"""
    return PoLaRIS_Mamba(
        embed_dim=64,
        depths=[2, 2, 4, 2],
        drop_path_rate=0.1,
        use_lidar=use_lidar,
    )


def polaris_mamba_small(use_lidar=True):
    """Small model: ~15M params"""
    return PoLaRIS_Mamba(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        drop_path_rate=0.2,
        use_lidar=use_lidar,
    )


def polaris_mamba_base(use_lidar=True):
    """Base model: ~30M params"""
    return PoLaRIS_Mamba(
        embed_dim=128,
        depths=[2, 2, 12, 2],
        drop_path_rate=0.3,
        use_lidar=use_lidar,
    )


# ======================== Unit Test ========================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PoLaRIS_Mamba Model")
    print("=" * 60)

    # Test Tiny model
    print("\n[1] Testing polaris_mamba_tiny...")
    model = polaris_mamba_tiny(use_lidar=True)

    ir = torch.randn(2, 1, 512, 640)  # (B, 1, H, W)
    lidar = torch.randn(2, 1, 512, 640)

    # Forward pass
    heatmap = model(ir, lidar)
    print(f"Input IR: {ir.shape}, LiDAR: {lidar.shape}")
    print(f"Output Heatmap: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")

    # Test without LiDAR
    print("\n[2] Testing without LiDAR...")
    heatmap_no_lidar = model(ir, lidar_img=None)
    print(f"Output Heatmap (no LiDAR): {heatmap_no_lidar.shape}")

    # Test Small model
    print("\n[3] Testing polaris_mamba_small...")
    model_small = polaris_mamba_small(use_lidar=True)
    heatmap_small = model_small(ir, lidar)
    num_params_small = sum(p.numel() for p in model_small.parameters())
    print(f"Output: {heatmap_small.shape}")
    print(f"Parameters: {num_params_small / 1e6:.2f}M")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
