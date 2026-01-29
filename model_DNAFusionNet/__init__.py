"""
DNA-FusionNet Package

DNA-FusionNet: Early Fusion Version for Multi-modal Ship Detection
"""

from .model_DNAFusion_EarlyFusion import (
    DNAFusionNet_EarlyFusion,
    dnafusion_early_resnet18,
    Res_CBAM_block,
    ChannelAttention,
    SpatialAttention
)

__all__ = [
    'DNAFusionNet_EarlyFusion',
    'dnafusion_early_resnet18',
    'Res_CBAM_block',
    'ChannelAttention',
    'SpatialAttention'
]

__version__ = '1.0.0'
__author__ = 'DNA-FusionNet Project'
