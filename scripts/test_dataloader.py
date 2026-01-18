#!/usr/bin/env python
"""
测试 PoLaRIS DataLoader 是否正常工作

用法:
    python scripts/test_dataloader.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.utils_lidar import PoLaRISTrainLoader, PoLaRISTestLoader
from model.load_param_data import load_dataset
from torch.utils.data import DataLoader
import torch

def test_dataloader(in_channels=2, normalize_16bit=True):
    """
    测试 DataLoader

    Args:
        in_channels: 1 (仅红外) 或 2 (红外+深度)
        normalize_16bit: True (Min-Max) 或 False (简单缩放)
    """
    print("=" * 60)
    print(f"测试 PoLaRIS DataLoader")
    print(f"  - in_channels: {in_channels}")
    print(f"  - normalize_16bit: {normalize_16bit}")
    print("=" * 60)

    # 加载数据集划分
    try:
        train_ids, val_ids, _ = load_dataset(
            root='dataset',
            dataset='Pohang-Canal',
            split_method='split_data'
        )
        print(f"✅ 数据集加载成功")
        print(f"   - 训练集: {len(train_ids)} 张")
        print(f"   - 验证集: {len(val_ids)} 张")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    # 创建 DataLoader
    try:
        train_dataset = PoLaRISTrainLoader(
            dataset_dir='dataset/Pohang-Canal',
            img_id=train_ids[:5],  # 只测试前 5 张
            base_size=512,
            crop_size=480,
            transform=None,
            suffix='.png',
            normalize_16bit=normalize_16bit,
            in_channels=in_channels
        )
        print(f"✅ TrainLoader 创建成功")
    except Exception as e:
        print(f"❌ TrainLoader 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试加载数据
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        print(f"✅ DataLoader 创建成功")
    except Exception as e:
        print(f"❌ DataLoader 创建失败: {e}")
        return

    # 读取一个 batch
    try:
        for i, batch in enumerate(train_loader):
            print(f"\n📦 Batch {i + 1}:")
            print(f"   - image shape: {batch['image'].shape}")
            print(f"   - image dtype: {batch['image'].dtype}")
            print(f"   - image range: [{batch['image'].min():.4f}, {batch['image'].max():.4f}]")
            print(f"   - mask shape: {batch['mask'].shape}")
            print(f"   - oracle_mask shape: {batch['oracle_mask'].shape}")
            print(f"   - oracle_mask unique values: {torch.unique(batch['oracle_mask'])}")
            print(f"   - lidar points: {[pts.shape for pts in batch['lidar']]}")
            print(f"   - img_id: {batch['img_id']}")
            print(f"   - is_16bit: {batch['is_16bit']}")

            # 验证数据范围
            assert batch['image'].min() >= 0.0 and batch['image'].max() <= 1.0, "图像范围应该在 [0, 1]"
            assert batch['mask'].min() >= 0.0 and batch['mask'].max() <= 1.0, "Mask 范围应该在 [0, 1]"
            assert batch['oracle_mask'].min() >= 0.0 and batch['oracle_mask'].max() <= 1.0, "Oracle mask 范围应该在 [0, 1]"

            # 检查通道数
            if in_channels == 2:
                assert batch['image'].shape[1] == 2, f"in_channels=2 时应该是 2 通道，但是是 {batch['image'].shape[1]}"
                print(f"   ✅ 2通道检查通过 (红外 + 深度)")
            else:
                assert batch['image'].shape[1] == 1, f"in_channels=1 时应该是 1 通道，但是是 {batch['image'].shape[1]}"
                print(f"   ✅ 1通道检查通过 (仅红外)")

            break  # 只测试一个 batch

        print(f"\n✅ 数据加载测试成功！")
        return True

    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 测试 1: 单通道模式 (仅红外)")
    success1 = test_dataloader(in_channels=1, normalize_16bit=True)

    print("\n\n🧪 测试 2: 双通道模式 (红外 + 深度)")
    success2 = test_dataloader(in_channels=2, normalize_16bit=True)

    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 所有测试通过！")
        print("=" * 60)
        print("\n下一步:")
        print("1. 运行训练脚本:")
        print("   ./scripts/run_Phase3_DualGeo_16bit.sh")
        print("\n2. 或者使用旧版 8-bit DataLoader:")
        print("   ./scripts/run_Phase3_DualGeo_8bit.sh")
    else:
        print("❌ 部分测试失败，请检查错误信息")
    print("=" * 60)
