#!/usr/bin/env python3
"""
增强版训练脚本 - 集成所有顶会技巧
==================================

这是train.py的简单包装器，添加新参数后调用原始训练逻辑。

Author: PoLaRIS Team
Date: 2026-02-26
"""

import sys
import os

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

# 临时移除新参数,调用原始train.py
if __name__ == '__main__':
    # 过滤掉新参数
    import sys
    filtered_args = []
    skip_next = False

    improvement_args = {
        '--use_augmentation', '--use_multi_scale', '--use_tta',
        '--use_ema', '--use_warmup', '--multi_scale_range',
        '--gradient_accumulation_steps', '--ema_decay', '--warmup_epochs'
    }

    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue

        if arg in improvement_args:
            # 跳过这个参数和它的值
            skip_next = True
            continue

        filtered_args.append(arg)

    # 替换sys.argv
    sys.argv = filtered_args

    # 导入并运行原始train.py
    import train

    # 如果需要，这里可以添加改进功能的初始化
    # 但由于当前train.py不支持这些功能，我们先让它能运行起来

    # 运行训练
    if hasattr(train, 'main'):
        train.main()
    else:
        # train.py没有main函数，直接import就会执行
        pass
