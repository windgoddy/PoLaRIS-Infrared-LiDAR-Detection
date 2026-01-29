"""
DNA-FusionNet 快速测试脚本

验证模型是否能正常创建和运行。

用法：
    python model_DNAFusionNet/test_model.py
"""

import os
import sys

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from model_DNAFusionNet.model_DNAFusion_EarlyFusion import dnafusion_early_resnet18


def test_model_creation():
    """测试模型创建"""
    print("="*80)
    print("测试 1: 模型创建")
    print("="*80)

    try:
        model = dnafusion_early_resnet18(
            num_classes=1,
            input_channels=2,
            deep_supervision=True
        )
        print("✅ 模型创建成功")

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n模型参数:")
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 参数量 (MB): {total_params * 4 / 1024 / 1024:.2f}")

        return True, model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False, None


def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "="*80)
    print("测试 2: 前向传播")
    print("="*80)

    try:
        # 模拟输入：红外 + LiDAR
        batch_size = 2
        height = 512
        width = 512

        dummy_input = torch.randn(batch_size, 2, height, width)
        print(f"输入形状: {dummy_input.shape}")
        print(f"  - 通道 0: 红外图像")
        print(f"  - 通道 1: LiDAR 深度图")

        # 前向传播
        outputs = model(dummy_input)

        print(f"\n输出格式: {type(outputs)}")
        if isinstance(outputs, list):
            print(f"输出数量: {len(outputs)} (Deep Supervision)")
            for i, out in enumerate(outputs, 1):
                print(f"  - Output {i}: {out.shape}")
        else:
            print(f"输出形状: {outputs.shape}")

        print("\n✅ 前向传播成功")
        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_compatibility():
    """测试 GPU 兼容性"""
    print("\n" + "="*80)
    print("测试 3: GPU 兼容性")
    print("="*80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，跳过 GPU 测试")
        return True

    try:
        device = torch.device('cuda:0')
        model = dnafusion_early_resnet18(
            num_classes=1,
            input_channels=2,
            deep_supervision=True
        ).to(device)

        dummy_input = torch.randn(1, 2, 512, 512).to(device)
        outputs = model(dummy_input)

        print(f"✅ GPU 测试成功 (设备: {device})")
        print(f"  - 输入设备: {dummy_input.device}")
        print(f"  - 输出设备: {outputs[0].device if isinstance(outputs, list) else outputs.device}")

        return True
    except Exception as e:
        print(f"❌ GPU 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deep_supervision_off():
    """测试关闭深度监督"""
    print("\n" + "="*80)
    print("测试 4: 深度监督开关")
    print("="*80)

    try:
        # Deep Supervision = True
        model_ds_on = dnafusion_early_resnet18(
            num_classes=1,
            input_channels=2,
            deep_supervision=True
        )

        # Deep Supervision = False
        model_ds_off = dnafusion_early_resnet18(
            num_classes=1,
            input_channels=2,
            deep_supervision=False
        )

        dummy_input = torch.randn(1, 2, 512, 512)

        output_ds_on = model_ds_on(dummy_input)
        output_ds_off = model_ds_off(dummy_input)

        print(f"Deep Supervision = True:")
        print(f"  - 输出类型: {type(output_ds_on)}")
        if isinstance(output_ds_on, list):
            print(f"  - 输出数量: {len(output_ds_on)}")

        print(f"\nDeep Supervision = False:")
        print(f"  - 输出类型: {type(output_ds_off)}")
        print(f"  - 输出形状: {output_ds_off.shape}")

        print("\n✅ 深度监督开关测试成功")
        return True
    except Exception as e:
        print(f"❌ 深度监督开关测试失败: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("DNA-FusionNet Early Fusion - 快速测试")
    print("="*80)
    print()

    results = []

    # 测试 1: 模型创建
    success, model = test_model_creation()
    results.append(("模型创建", success))

    if not success:
        print("\n" + "="*80)
        print("❌ 测试失败：无法创建模型")
        print("="*80)
        return

    # 测试 2: 前向传播
    success = test_forward_pass(model)
    results.append(("前向传播", success))

    # 测试 3: GPU 兼容性
    success = test_gpu_compatibility()
    results.append(("GPU 兼容性", success))

    # 测试 4: 深度监督开关
    success = test_deep_supervision_off()
    results.append(("深度监督开关", success))

    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    all_passed = True
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False

    print()
    if all_passed:
        print("🎉 所有测试通过！模型可以正常使用。")
        print()
        print("下一步：")
        print("  1. 开始训练：")
        print("     python model_DNAFusionNet/train_DNAFusion_EarlyFusion.py --gpu 0 --epochs 50")
        print()
        print("  2. 在服务器上运行：")
        print("     CUDA_VISIBLE_DEVICES=5 python model_DNAFusionNet/train_DNAFusion_EarlyFusion.py \\")
        print("         --dataset Pohang-Canal-3k --split_method 50_50_2k_new --gpu 0 --epochs 50")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")

    print("="*80)


if __name__ == '__main__':
    main()
