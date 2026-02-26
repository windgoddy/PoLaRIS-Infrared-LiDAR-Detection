"""
测试所有改进模块
==================

快速验证所有改进是否正常工作

Usage:
    cd model_Mamba
    python test_improvements.py
"""

import sys
import os

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

print("=" * 70)
print("🧪 测试所有改进模块")
print("=" * 70)

# ========== 测试1: 数据增强 ==========
print("\n[1/4] 测试数据增强...")
try:
    from model_Mamba.dataset.augmentation import InfraredAugmentation, TestTimeAugmentation
    from PIL import Image
    import numpy as np

    # 创建测试图像
    test_img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
    test_mask = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))

    # 测试增强
    aug = InfraredAugmentation(mode='train', use_all=True)
    for i in range(5):
        img_aug, mask_aug, _ = aug(test_img.copy(), test_mask.copy(), None)

    print("  ✅ 数据增强模块正常工作")
except Exception as e:
    print(f"  ❌ 数据增强测试失败: {e}")

# ========== 测试2: 训练工具 ==========
print("\n[2/4] 测试训练工具...")
try:
    from model_Mamba.utils_training import (
        EMA, WarmupScheduler, MultiScaleTrainingManager, GradientAccumulator
    )
    import torch
    import torch.nn as nn

    # 测试EMA
    model = nn.Linear(10, 1)
    ema = EMA(model, decay=0.999)
    for _ in range(3):
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.1
        ema.update()

    # 测试Warmup Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=5, base_scheduler=base_scheduler)
    for _ in range(10):
        scheduler.step()

    # 测试多尺度管理器
    manager = MultiScaleTrainingManager(base_size=256, scale_range=(0.75, 1.25))
    for _ in range(5):
        size = manager.get_random_size()

    # 测试梯度累积
    accumulator = GradientAccumulator(accumulation_steps=4)
    for i in range(10):
        should_update = accumulator.should_update(i)

    print("  ✅ 训练工具模块正常工作")
except Exception as e:
    print(f"  ❌ 训练工具测试失败: {e}")

# ========== 测试3: 模型导入 ==========
print("\n[3/4] 测试模型导入...")
try:
    from model_Mamba.core.polaris_mamba_progressive import polaris_mamba_tiny_progressive
    import torch

    model = polaris_mamba_tiny_progressive(use_lidar=True)
    print(f"  ✅ 模型导入成功，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
except Exception as e:
    print(f"  ❌ 模型导入失败: {e}")

# ========== 测试4: Loss函数 ==========
print("\n[4/4] 测试Loss函数...")
try:
    from model_Mamba.core.loss_advanced import LossFactory
    import torch

    # 测试hybrid loss
    criterion = LossFactory.create(
        'hybrid',
        focal_alpha=0.25,
        focal_gamma=2.5,
        dice_weight=2.5,
        projection_weight=2.0
    )

    pred = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    loss = criterion(pred, target)

    print(f"  ✅ Loss函数正常工作，loss值: {loss.item():.4f}")
except Exception as e:
    print(f"  ❌ Loss函数测试失败: {e}")

# ========== 总结 ==========
print("\n" + "=" * 70)
print("✅ 所有模块测试完成！")
print("=" * 70)
print("\n下一步:")
print("  1. 运行训练: bash model_Mamba/scripts/train_enhanced.sh")
print("  2. 或参考: model_Mamba/IMPROVEMENTS_README.md")
print("")
