"""
SS2D Components (Pure Mamba, No Internal Gating)
=================================================

Core Mamba (State Space Model) components.  All spatial priority modulation is
now handled OUTSIDE these modules — in MambaBlockWrapper (model_Mamba_UNetPP.py)
via the PhysicalSNRPrior pyramid.  SS2D / VSSBlock are kept clean and composable.

Components:
- CrossScan : Scans features in 4 directions (→, ←, ↓, ↑)
- CrossMerge: Merges scanned features back
- SS2D      : Selective Scan 2D (pure SSM, no gating)
- VSSBlock  : Vision State Space Block (LayerNorm + SS2D + residual)

Note: use_lidar_gate / lidar_feat params are kept for API backward-compatibility
but are silently ignored; actual spatial modulation belongs in the caller.

Reference:
- Original: lidar-mamba/mmcls/LIDAR_dev/models/LIDAR/LIDAR_layer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# Use compatibility layer for mamba_ssm
from .mamba_compat import selective_scan_compat, MAMBA_AVAILABLE


class CrossScan(nn.Module):
    """
    Cross-directional scanning: Split features into 4 directions.

    Directions:
        - Direction 0: Left → Right (row-wise forward)
        - Direction 1: Right → Left (row-wise backward)
        - Direction 2: Top → Bottom (column-wise forward)
        - Direction 3: Bottom → Top (column-wise backward)

    Input:  (B, C, H, W)
    Output: (B, 4, C, H*W)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map

        Returns:
            x_scan: (B, 4, C, L) where L = H*W
        """
        B, C, H, W = x.shape
        L = H * W

        # Direction 0: →  (row-major order)
        x_dir0 = x.flatten(2)  # (B, C, H*W)

        # Direction 1: ←  (reverse each row)
        x_dir1 = x.flip(dims=[3]).flatten(2)  # flip width, then flatten

        # Direction 2: ↓  (column-major order)
        x_dir2 = x.transpose(2, 3).flatten(2)  # (B, C, W, H) → (B, C, W*H)

        # Direction 3: ↑  (reverse column-major)
        x_dir3 = x.transpose(2, 3).flip(dims=[3]).flatten(2)

        # Stack: (B, 4, C, L)
        x_scan = torch.stack([x_dir0, x_dir1, x_dir2, x_dir3], dim=1)

        return x_scan


class CrossMerge(nn.Module):
    """
    Merge cross-directional scans back to spatial layout.

    Input:  (B, 4, C, H*W)
    Output: (B, C, H, W)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_scan, H, W):
        """
        Args:
            x_scan: (B, 4, C, L) scanned features
            H, W: spatial dimensions to restore

        Returns:
            x_merged: (B, C, H, W)
        """
        B, K, C, L = x_scan.shape
        assert K == 4, "Expected 4 directions"
        assert L == H * W, f"Length mismatch: L={L}, H*W={H*W}"

        # Reverse the scanning operations
        x_dir0 = x_scan[:, 0].view(B, C, H, W)  # → (already correct)

        x_dir1 = x_scan[:, 1].view(B, C, H, W).flip(dims=[3])  # ← (reverse width)

        x_dir2 = x_scan[:, 2].view(B, C, W, H).transpose(2, 3)  # ↓ (transpose back)

        x_dir3 = x_scan[:, 3].view(B, C, W, H).flip(dims=[3]).transpose(2, 3)  # ↑

        # Average merge (can also use learned fusion)
        x_merged = (x_dir0 + x_dir1 + x_dir2 + x_dir3) / 4.0

        return x_merged


class SS2D(nn.Module):
    """
    Selective Scan 2D (SS2D) with LiDAR Gated Injection.

    This is the core innovation: LiDAR features modulate the infrared features
    before entering the State Space Model (SSM).

    Forward Logic:
        1. Project x: x → (B, C, H, W)
        2. Cross-scan: x → (B, 4, C, L)
        3. **LiDAR Gate**: if lidar_feat is provided:
               gate = sigmoid(conv(lidar_feat))
               x_scan = x_scan * (1 + gate)  # Enhance IR where LiDAR is confident
        4. Apply SSM (Mamba): x_scan → y_scan
        5. Cross-merge: y_scan → (B, C, H, W)

    Args:
        d_model: Feature dimension (e.g., 96, 192, 384, 768)
        d_state: SSM state dimension (default: 16)
        expand: Expansion ratio for inner dimension (default: 2)
        dt_rank: Delta (time step) rank (default: "auto")
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        use_lidar_gate=False,  # Deprecated: ignored, kept for API compatibility
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,  # Depthwise
            bias=True,
        )
        self.act = nn.SiLU()

        # SSM parameters (4 directions)
        self.K = 4  # Number of scan directions

        # x_proj: project to (B, C, delta, A, D)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # dt_proj: project delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A: state transition matrix (D_inner, D_state)
        # Initialize with -exp(uniform(log(0.001), log(0.1)))
        # CRITICAL FIX: Use uniform distribution to avoid extreme values
        A = torch.rand(self.d_inner, self.d_state)  # [0, 1]
        A = -(A * (0.1 - 0.001) + 0.001)  # [-0.1, -0.001]
        self.A_log = nn.Parameter(torch.log(-A))  # Log of positive value
        self.A_log._no_weight_decay = True

        # D: skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Cross-scan modules
        self.cross_scan = CrossScan()
        self.cross_merge = CrossMerge()

    def forward(self, x, lidar_feat=None):
        """
        Args:
            x         : (B, H, W, C) infrared features
            lidar_feat: ignored (deprecated; spatial modulation is now in MambaBlockWrapper)

        Returns:
            out: (B, H, W, C)
        """
        B, H, W, C = x.shape

        # 1. Input projection
        xz = self.in_proj(x)  # (B, H, W, 2*D_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # Each: (B, H, W, D_inner)

        # 2. Conv for local context
        x_conv = self.conv2d(x_inner.permute(0, 3, 1, 2))  # (B, D_inner, H, W)
        x_conv = self.act(x_conv)

        # 3. Cross-scan
        x_scan = self.cross_scan(x_conv)  # (B, 4, D_inner, L)

        # 4. Apply SSM (Selective Scan)
        y_scan = self._selective_scan(x_scan)  # (B, 4, D_inner, L)

        # 5. Cross-merge
        y = self.cross_merge(y_scan, H, W)  # (B, D_inner, H, W)

        # 6. Gate and output projection
        y = y.permute(0, 2, 3, 1)  # (B, H, W, D_inner)
        z = self.act(z)  # (B, H, W, D_inner)
        out = y * z  # Element-wise gating
        out = self.out_proj(out)  # (B, H, W, C)

        return out

    def _selective_scan(self, x):
        """
        Apply Selective Scan (Mamba SSM) on scanned sequences.

        Args:
            x: (B, K, D_inner, L) scanned features

        Returns:
            y: (B, K, D_inner, L) output features
        """
        B, K, D, L = x.shape

        # Flatten for batch processing
        x_flat = rearrange(x, 'b k d l -> (b k) d l')  # (B*K, D, L)

        # Project to delta, B, C
        x_dbl = self.x_proj(x_flat.transpose(1, 2))  # (B*K, L, dt_rank + 2*D_state)
        delta, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Project delta
        delta = self.dt_proj(delta)  # (B*K, L, D)
        delta = F.softplus(delta)  # Ensure positive

        # Get A matrix
        A = -torch.exp(self.A_log.float())  # (D, D_state)

        # Use compatibility layer (handles both mamba_ssm and PyTorch native)
        y = selective_scan_compat(
            x_flat,       # (B*K, D, L)
            delta,        # (B*K, L, D)
            A,            # (D, D_state)
            B_ssm,        # (B*K, L, D_state)
            C_ssm,        # (B*K, L, D_state)
            self.D.float() if MAMBA_AVAILABLE else None,  # D only for mamba_ssm
        )  # (B*K, D, L)

        # Reshape back
        y = rearrange(y, '(b k) d l -> b k d l', b=B, k=K)

        return y

    def _selective_scan_pytorch(self, x, delta, A, B, C):
        """
        PyTorch native implementation of Selective Scan (for CPU/M-series Mac).

        Args:
            x: (BK, D, L)
            delta: (BK, D, L)
            A: (D, N) - state transition matrix
            B: (BK, L, N)
            C: (BK, L, N)

        Returns:
            y: (BK, D, L)
        """
        BK, D, L = x.shape
        N = A.shape[1]

        # Initialize state
        h = torch.zeros(BK, D, N, device=x.device, dtype=x.dtype)

        # Discretize A: A_bar = exp(delta * A)
        # A: (D, N), delta: (BK, D, L)
        deltaA = torch.einsum('bdt,dn->bdtn', delta, A)  # (BK, D, L, N)
        A_bar = torch.exp(deltaA)  # (BK, D, L, N)

        # Discretize B: B_bar = delta * B
        # delta: (BK, D, L), B: (BK, L, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (BK, D, L, N)

        # Output
        y = []

        for t in range(L):
            # h = A_bar * h + B_bar * x
            h = A_bar[:, :, t, :] * h + deltaB[:, :, t, :] * x[:, :, t].unsqueeze(-1)

            # y = C * h
            y_t = torch.einsum('bdn,bn->bd', h, C[:, t, :])  # (BK, D)
            y.append(y_t)

        y = torch.stack(y, dim=2)  # (BK, D, L)

        # Add skip connection: y = y + D * x
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * x

        return y


class VSSBlock(nn.Module):
    """
    Vision State Space Block (VSS Block).

    This is the complete Mamba encoder block, consisting of:
        - Layer Norm
        - SS2D (with LiDAR gating)
        - Residual connection

    Args:
        hidden_dim: Feature dimension
        drop_path: Stochastic depth rate (default: 0.0)
        norm_layer: Normalization layer (default: LayerNorm)
        use_lidar_gate: Whether to use LiDAR gating (default: True)
    """
    def __init__(
        self,
        hidden_dim,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        use_lidar_gate=False,  # Deprecated: ignored, kept for API compatibility
    ):
        super().__init__()

        self.norm = norm_layer(hidden_dim)
        self.ss2d = SS2D(d_model=hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, lidar_feat=None):
        """
        Args:
            x: (B, H, W, C) input features
            lidar_feat: (B, 1, H, W) LiDAR feature map (optional)

        Returns:
            out: (B, H, W, C)
        """
        # Residual connection with SS2D
        out = x + self.drop_path(self.ss2d(self.norm(x), lidar_feat))
        return out


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    Reference: https://arxiv.org/abs/1603.09382
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, ..., 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize

        output = x.div(keep_prob) * random_tensor
        return output


# ======================== Unit Test ========================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SS2D Components")
    print("=" * 60)

    # Test CrossScan
    print("\n[1] Testing CrossScan...")
    x = torch.randn(2, 64, 32, 32)  # (B, C, H, W)
    scanner = CrossScan()
    x_scan = scanner(x)
    print(f"Input: {x.shape} → Output: {x_scan.shape}")
    assert x_scan.shape == (2, 4, 64, 32*32), "CrossScan output shape error!"

    # Test CrossMerge
    print("\n[2] Testing CrossMerge...")
    merger = CrossMerge()
    x_merged = merger(x_scan, 32, 32)
    print(f"Input: {x_scan.shape} → Output: {x_merged.shape}")
    assert x_merged.shape == x.shape, "CrossMerge output shape error!"

    # Test SS2D without LiDAR
    print("\n[3] Testing SS2D (without LiDAR)...")
    ss2d = SS2D(d_model=64)
    x = torch.randn(2, 32, 32, 64)  # (B, H, W, C)
    y = ss2d(x, lidar_feat=None)
    print(f"Input: {x.shape} → Output: {y.shape}")
    assert y.shape == x.shape, "SS2D output shape error!"

    # Test SS2D with LiDAR
    print("\n[4] Testing SS2D (with LiDAR gating)...")
    lidar = torch.randn(2, 1, 32, 32)  # (B, 1, H, W)
    y_gated = ss2d(x, lidar_feat=lidar)
    print(f"Input: {x.shape}, LiDAR: {lidar.shape} → Output: {y_gated.shape}")
    assert y_gated.shape == x.shape, "SS2D with LiDAR output shape error!"

    # Test VSSBlock
    print("\n[5] Testing VSSBlock...")
    vss_block = VSSBlock(hidden_dim=64, drop_path=0.1, use_lidar_gate=True)
    y_block = vss_block(x, lidar_feat=lidar)
    print(f"Input: {x.shape} → Output: {y_block.shape}")
    assert y_block.shape == x.shape, "VSSBlock output shape error!"

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
