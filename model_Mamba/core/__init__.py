"""
PoLaRIS-Gaussian-Mamba Core Module
===================================

This module contains the core components of the Mamba architecture.

Components:
- ss2d_components: SS2D and VSSBlock with LiDAR gating
- polaris_mamba: Main model architecture
- loss: Gaussian Focal Loss

Author: PoLaRIS Team
Date: 2026-01-30
"""

from .ss2d_components import SS2D, VSSBlock, CrossScan, CrossMerge
from .polaris_mamba import PoLaRIS_Mamba, GaussianHead, polaris_mamba_tiny, polaris_mamba_small, polaris_mamba_base
from .loss import GaussianFocalLoss, AdaptiveGaussianFocalLoss, AverageMeter

__all__ = [
    'SS2D',
    'VSSBlock',
    'CrossScan',
    'CrossMerge',
    'PoLaRIS_Mamba',
    'GaussianHead',
    'polaris_mamba_tiny',
    'polaris_mamba_small',
    'polaris_mamba_base',
    'GaussianFocalLoss',
    'AdaptiveGaussianFocalLoss',
    'AverageMeter',
]

__version__ = '1.0.0'
