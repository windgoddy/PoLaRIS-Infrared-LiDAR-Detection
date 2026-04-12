#!/usr/bin/env python3
"""
临时的MMCV ops fallback方案，用于绕过CUDA库兼容性问题
"""

import sys
import importlib.util
from unittest.mock import MagicMock

# 创建一个mock的mmcv.ops模块
class MockMMCVOps:
    def __getattr__(self, name):
        print(f"Warning: Using mock for mmcv.ops.{name}")
        return MagicMock()

# 替换mmcv.ops模块
mock_ops = MockMMCVOps()
sys.modules['mmcv.ops'] = mock_ops

# 为常用的ops函数提供简单的fallback
mock_ops.nms_match = MagicMock(return_value=[])
mock_ops.RoIPool = MagicMock()
mock_ops.point_sample = MagicMock()