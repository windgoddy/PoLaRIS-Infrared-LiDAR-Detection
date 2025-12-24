#!/bin/bash

# 评估 Phase 3 模型 (MS_CAFNet)
# 请确保 checkpoint 路径正确
python scripts/evaluate_golden_set.py --model_name MS_CAFNet --in_channels 2 --checkpoint result/Pohang-Canal_MS_CAFNet_12_12_2025_21_52_06_wDS/mIoU__MS_CAFNet_Pohang-Canal_epoch.pth.tar
