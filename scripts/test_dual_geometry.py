#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证 MS_CAFNet 和 MS_CAFNet_DualGeo 两个模型
测试：形状匹配、参数存在性、前向传播
"""

import sys
sys.path.append('.')

import torch
from model.model_Phase3 import MS_CAFNet, MS_CAFNet_DualGeo, VisualStructureExtractor

def test_visual_extractor():
    """测试 VisualStructureExtractor"""
    print("=" * 60)
    print("测试 1: VisualStructureExtractor")
    print("=" * 60)

    extractor = VisualStructureExtractor(in_channels=1)

    # 模拟红外图像 [B, 1, H, W]
    x_ir = torch.randn(2, 1, 512, 640)

    # 前向传播
    structure_map = extractor(x_ir)

    print(f"✅ 输入形状: {x_ir.shape}")
    print(f"✅ 输出形状: {structure_map.shape}")
    print(f"✅ 输出范围: [{structure_map.min():.3f}, {structure_map.max():.3f}]")

    assert structure_map.shape == x_ir.shape, "形状不匹配！"
    assert structure_map.min() >= 0 and structure_map.max() <= 1, "输出范围错误！"

    print("✅ VisualStructureExtractor 测试通过\n")

def test_original_model():
    """测试原始 MS_CAFNet（备份版本）"""
    print("=" * 60)
    print("测试 2: 原始 MS_CAFNet (备份版本)")
    print("=" * 60)

    model = MS_CAFNet(num_classes=1, input_channels=2)

    # 检查不包含新组件
    assert not hasattr(model, 'visual_extractor'), "原始模型不应包含 visual_extractor!"
    assert not hasattr(model, 'gate_lidar'), "原始模型不应包含 gate_lidar!"
    assert not hasattr(model, 'gate_visual'), "原始模型不应包含 gate_visual!"

    print("✅ 原始 MS_CAFNet 不包含新组件")

    # 测试前向传播
    model.eval()
    input_tensor = torch.randn(1, 2, 256, 320)

    with torch.no_grad():
        output, pred_conf = model(input_tensor)

    print(f"✅ 分割输出形状: {output.shape}")
    print(f"✅ 置信度输出形状: {pred_conf.shape}")
    print("✅ 原始 MS_CAFNet 测试通过\n")

def test_dual_geo_model():
    """测试 MS_CAFNet_DualGeo（升级版本）"""
    print("=" * 60)
    print("测试 3: MS_CAFNet_DualGeo (升级版本)")
    print("=" * 60)

    model = MS_CAFNet_DualGeo(num_classes=1, input_channels=2)

    # 检查新增组件
    assert hasattr(model, 'visual_extractor'), "缺少 visual_extractor!"
    assert hasattr(model, 'gate_lidar'), "缺少 gate_lidar!"
    assert hasattr(model, 'gate_visual'), "缺少 gate_visual!"

    print(f"✅ visual_extractor 存在: {type(model.visual_extractor).__name__}")
    print(f"✅ gate_lidar 初始值: {model.gate_lidar.item():.3f}")
    print(f"✅ gate_visual 初始值: {model.gate_visual.item():.3f}")

    # 检查参数是否可学习
    assert model.gate_lidar.requires_grad, "gate_lidar 不可学习！"
    assert model.gate_visual.requires_grad, "gate_visual 不可学习！"

    print("✅ 门控参数可学习")

    # 测试前向传播
    model.eval()
    input_tensor = torch.randn(1, 2, 256, 320)

    with torch.no_grad():
        output, pred_conf = model(input_tensor)

    print(f"✅ 分割输出形状: {output.shape}")
    print(f"✅ 置信度输出形状: {pred_conf.shape}")
    print("✅ MS_CAFNet_DualGeo 测试通过\n")

def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def test_parameter_comparison():
    """对比两个模型的参数量"""
    print("=" * 60)
    print("测试 4: 参数量对比")
    print("=" * 60)

    original = MS_CAFNet(num_classes=1, input_channels=2)
    dual_geo = MS_CAFNet_DualGeo(num_classes=1, input_channels=2)

    total_orig, _ = count_parameters(original)
    total_dual, _ = count_parameters(dual_geo)

    diff = total_dual - total_orig
    percentage = (diff / total_orig) * 100

    print(f"原始 MS_CAFNet 参数量: {total_orig:,} ({total_orig/1e6:.2f}M)")
    print(f"MS_CAFNet_DualGeo 参数量: {total_dual:,} ({total_dual/1e6:.2f}M)")
    print(f"新增参数量: {diff} ({percentage:.4f}%)")

    # 验证新增参数量极小
    assert diff < 200, f"新增参数量过大: {diff}"

    print("✅ 参数量增加在可接受范围 (<200, <0.01%)\n")

def test_dual_geometry_fusion():
    """测试双流融合逻辑"""
    print("=" * 60)
    print("测试 5: 双流几何融合验证")
    print("=" * 60)

    model = MS_CAFNet_DualGeo(num_classes=1, input_channels=2)

    # 场景1: LiDAR 强 (近距离)
    input_lidar_strong = torch.randn(1, 2, 256, 320)
    input_lidar_strong[:, 1, :, :] = torch.rand(1, 1, 256, 320) * 10

    # 场景2: LiDAR 弱 (远距离)
    input_lidar_weak = torch.randn(1, 2, 256, 320)
    input_lidar_weak[:, 1, :, :] = torch.zeros(1, 1, 256, 320)

    model.eval()
    with torch.no_grad():
        out1, conf1 = model(input_lidar_strong)
        out2, conf2 = model(input_lidar_weak)

    print(f"场景1 (LiDAR 强) - 置信度均值: {conf1.mean():.3f}")
    print(f"场景2 (LiDAR 弱) - 置信度均值: {conf2.mean():.3f}")

    # 验证置信度差异
    assert conf1.mean() > conf2.mean(), "置信度逻辑异常！"

    print("✅ 双流融合逻辑正确\n")

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" MS_CAFNet 双模型验证")
    print(" - 原始版本 (MS_CAFNet) - 备份")
    print(" - 双流几何引导版本 (MS_CAFNet_DualGeo) - 实验")
    print("=" * 60 + "\n")

    try:
        test_visual_extractor()
        test_original_model()
        test_dual_geo_model()
        test_parameter_comparison()
        test_dual_geometry_fusion()

        print("=" * 60)
        print("🎉 所有测试通过！两个模型都可以正常使用！")
        print("=" * 60)
        print("\n✅ 模型选择:")
        print("  - 原始版本 (稳定): MS_CAFNet")
        print("  - 实验版本 (创新): MS_CAFNet_DualGeo")
        print("\n✅ 下一步:")
        print("  1. 修改 train_Phase3.py 选择要训练的模型")
        print("  2. 对比两个模型的性能")
        print("  3. 监控 alpha, beta 参数演化\n")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
