#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比 DNANet 和 MS_CAFNet 的模型复杂度
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from model.model_DNANet import DNANet, Res_CBAM_block
from model.model_Phase3 import MS_CAFNet

def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def measure_inference_time(model, input_tensor, warmup=10, iterations=100):
    """测量推理时间"""
    model.eval()
    device = next(model.parameters()).device

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # 计时
    torch.cuda.synchronize() if device.type == 'cuda' else None
    import time
    start = time.time()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()

    avg_time = (end - start) / iterations * 1000  # ms
    return avg_time

if __name__ == '__main__':
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = (1, 2, 512, 640)  # [B, C, H, W]

    nb_filter = [16, 32, 64, 128, 256]
    num_blocks = [1, 1, 1, 1]

    print("=" * 60)
    print("模型复杂度对比")
    print("=" * 60)

    # DNANet
    print("\n【DNANet - Baseline】")
    dnanet = DNANet(
        num_classes=1,
        input_channels=2,
        block=Res_CBAM_block,
        num_blocks=num_blocks,
        nb_filter=nb_filter,
        deep_supervision=False
    ).to(device)

    total_dna, trainable_dna = count_parameters(dnanet)
    print(f"  参数量: {total_dna:,} ({total_dna/1e6:.2f}M)")
    print(f"  可训练: {trainable_dna:,}")

    # MS_CAFNet
    print("\n【MS_CAFNet - Phase3】")
    ms_cafnet = MS_CAFNet(
        num_classes=1,
        input_channels=2
    ).to(device)

    total_ms, trainable_ms = count_parameters(ms_cafnet)
    print(f"  参数量: {total_ms:,} ({total_ms/1e6:.2f}M)")
    print(f"  可训练: {trainable_ms:,}")

    # 对比
    print("\n【参数量对比】")
    diff = total_ms - total_dna
    ratio = (total_ms / total_dna - 1) * 100
    print(f"  MS_CAFNet vs DNANet: {diff:+,} ({ratio:+.2f}%)")

    # 推理速度测试
    print("\n【推理速度测试】")
    print(f"  输入尺寸: {input_size}")

    input_tensor = torch.randn(*input_size).to(device)

    time_dna = measure_inference_time(dnanet, input_tensor, warmup=5, iterations=50)
    print(f"  DNANet 平均推理时间: {time_dna:.2f} ms")

    time_ms = measure_inference_time(ms_cafnet, input_tensor, warmup=5, iterations=50)
    print(f"  MS_CAFNet 平均推理时间: {time_ms:.2f} ms")

    speedup = (time_dna / time_ms - 1) * 100
    print(f"\n  MS_CAFNet 提速: {speedup:+.2f}%")

    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    if total_ms < total_dna:
        print(f"✅ MS_CAFNet 参数量减少 {abs(ratio):.1f}%")
    else:
        print(f"⚠️  MS_CAFNet 参数量增加 {ratio:.1f}%")

    if speedup > 0:
        print(f"✅ MS_CAFNet 推理速度提升 {speedup:.1f}%")
    else:
        print(f"⚠️  MS_CAFNet 推理速度下降 {abs(speedup):.1f}%")

    print("\n💡 速度提升的主要原因:")
    print("  1. 简化的主干网络（9层 vs 15层）")
    print("  2. 减少密集跳跃连接（4个 vs 15个）")
    print("  3. 消除下采样操作（0次 vs 5次）")
    print("  4. 轻量级置信度网络（仅3层）")
    print("  5. 更好的内存访问模式")
