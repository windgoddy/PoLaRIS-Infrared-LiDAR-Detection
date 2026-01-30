"""
PoLaRIS-Gaussian-Mamba Project
===============================

A complete Mamba-based architecture for infrared small target detection
with LiDAR gated injection and Gaussian heatmap output.

Directory Structure:
--------------------
model_Mamba/
├── core/               # Core model components
├── dataset/            # Data processing utilities
├── scripts/            # Deployment and training scripts
├── docs/               # Documentation
├── train.py            # Training script
├── visualize.py        # Visualization script
└── README.md           # Main documentation

Quick Start:
------------
1. Environment setup:
   bash model_Mamba/scripts/setup_server.sh

2. Training:
   bash model_Mamba/scripts/train_mamba_server.sh

3. Visualization:
   bash model_Mamba/scripts/visualize_mamba.sh

Documentation:
--------------
- README.md                      - Project overview
- docs/QUICKSTART.md             - 5-minute quick start
- docs/SERVER_DEPLOYMENT.md      - Complete deployment guide
- docs/FILES_SUMMARY.md          - File structure explanation
- docs/CODE_REVIEW_REPORT.md     - Code quality report

Author: PoLaRIS Team
Date: 2026-01-30
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'PoLaRIS Team'

from .core import *
from .dataset import *

__all__ = [
    'SS2D',
    'VSSBlock',
    'PoLaRIS_Mamba',
    'GaussianFocalLoss',
    'generate_gaussian_target',
]
