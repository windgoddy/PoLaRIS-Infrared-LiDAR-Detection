"""
Mamba SSM Compatibility Layer
==============================

This module provides a compatibility layer for different versions of mamba_ssm.
Supports both mamba_ssm 1.2.0+ and PyTorch native fallback.

Author: PoLaRIS Team
Date: 2026-01-30
"""

import torch
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
    print("[INFO] Using mamba_ssm hardware-accelerated selective scan")
except ImportError:
    MAMBA_AVAILABLE = False
    print("[WARNING] mamba_ssm not available. Using PyTorch native fallback (slower).")


def selective_scan_mamba_v1_2(x, delta, A, B, C, D):
    """
    Wrapper for mamba_ssm 1.2.0+ selective_scan_fn.
    
    Args:
        x: Input tensor (B, D, L)
        delta: Delta tensor (B, L, D)
        A: State transition matrix (D, D_state)
        B: Input matrix (B, L, D_state)
        C: Output matrix (B, L, D_state)
        D: Skip connection (D,)
    
    Returns:
        y: Output tensor (B, D, L)
    """
    B_batch, D, L = x.shape
    D_state = A.shape[1]
    
    # Transpose delta to (B, D, L)
    delta_transposed = delta.transpose(1, 2)  # (B, D, L)
    
    # Reshape B and C for mamba_ssm 1.2.0
    # Expected: B: (batch, n_groups, dstate, seqlen)
    # We have: B: (B, L, D_state)
    B_reshaped = B.transpose(1, 2).unsqueeze(1)  # (B, 1, D_state, L)
    C_reshaped = C.transpose(1, 2).unsqueeze(1)  # (B, 1, D_state, L)
    
    # Call mamba_ssm 1.2.0
    y = selective_scan_fn(
        x,                  # u: (batch, dim, seqlen)
        delta_transposed,   # delta: (batch, dim, seqlen)
        A,                  # A: (dim, dstate)
        B_reshaped,         # B: (batch, n_groups, dstate, seqlen)
        C_reshaped,         # C: (batch, n_groups, dstate, seqlen)
        D,                  # D: (dim,)
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    )
    
    return y


def selective_scan_pytorch_native(x, delta, A, B, C):
    """
    PyTorch native implementation of selective scan (fallback).
    
    Args:
        x: Input tensor (B, D, L)
        delta: Delta tensor (B, D, L)
        A: State transition matrix (D, D_state)
        B: Input matrix (B, D_state, L)
        C: Output matrix (B, D_state, L)
    
    Returns:
        y: Output tensor (B, D, L)
    """
    B_batch, D, L = x.shape
    D_state = A.shape[1]
    
    # Expand A for batch processing
    A_expanded = A.unsqueeze(0).expand(B_batch, -1, -1)  # (B, D, D_state)
    
    # Compute deltaA and deltaB
    deltaA = torch.einsum('bdt,dn->bdtn', delta, A)  # (B, D, L, D_state)
    A_bar = torch.exp(deltaA)  # (B, D, L, D_state)
    
    deltaB_u = torch.einsum('bdt,bnt,bdt->bdtn', delta, B, x)  # (B, D, L, D_state)
    
    # Initialize state
    h = torch.zeros(B_batch, D, D_state, device=x.device, dtype=x.dtype)
    
    # Recurrent computation
    ys = []
    for t in range(L):
        h = A_bar[:, :, t] * h + deltaB_u[:, :, t]
        y_t = torch.einsum('bdn,bn->bd', h, C[:, :, t])
        ys.append(y_t)
    
    y = torch.stack(ys, dim=2)  # (B, D, L)
    
    return y


def selective_scan_compat(x, delta, A, B, C, D=None):
    """
    Universal selective scan with automatic version detection.
    
    Args:
        x: Input tensor (B, D, L)
        delta: Delta tensor (B, L, D) or (B, D, L)
        A: State transition matrix (D, D_state)
        B: Input matrix (B, L, D_state) or (B, D_state, L)
        C: Output matrix (B, L, D_state) or (B, D_state, L)
        D: Skip connection (D,), optional
    
    Returns:
        y: Output tensor (B, D, L)
    """
    if MAMBA_AVAILABLE and D is not None:
        # Use hardware-accelerated version
        try:
            return selective_scan_mamba_v1_2(x, delta, A, B, C, D)
        except Exception as e:
            print(f"[WARNING] mamba_ssm failed: {e}")
            print("[INFO] Falling back to PyTorch native implementation")
    
    # Use PyTorch native fallback
    # Normalize input shapes
    if delta.shape[1] != x.shape[1]:  # delta is (B, L, D)
        delta = delta.transpose(1, 2)  # -> (B, D, L)
    
    if B.shape[1] != A.shape[1]:  # B is (B, L, D_state)
        B = B.transpose(1, 2)  # -> (B, D_state, L)
    
    if C.shape[1] != A.shape[1]:  # C is (B, L, D_state)
        C = C.transpose(1, 2)  # -> (B, D_state, L)
    
    return selective_scan_pytorch_native(x, delta, A, B, C)


if __name__ == "__main__":
    # Unit test
    print("\n" + "="*60)
    print("Mamba Compatibility Layer Unit Test")
    print("="*60)
    
    B, D, L, D_state = 2, 64, 100, 16
    
    x = torch.randn(B, D, L).cuda()
    delta = torch.randn(B, L, D).cuda()
    A = -torch.exp(torch.randn(D, D_state)).cuda()
    B_mat = torch.randn(B, L, D_state).cuda()
    C_mat = torch.randn(B, L, D_state).cuda()
    D_vec = torch.randn(D).cuda()
    
    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  delta: {delta.shape}")
    print(f"  A: {A.shape}")
    print(f"  B: {B_mat.shape}")
    print(f"  C: {C_mat.shape}")
    print(f"  D: {D_vec.shape}")
    
    # Test compatibility layer
    try:
        y = selective_scan_compat(x, delta, A, B_mat, C_mat, D_vec)
        print(f"\n✅ Output shape: {y.shape}")
        print(f"✅ Compatibility layer working!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "="*60)
