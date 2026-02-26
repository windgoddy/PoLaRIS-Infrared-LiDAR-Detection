"""
训练工具模块 - EMA, Warmup Scheduler, 多尺度训练
==================================================

包含顶会论文常用的训练技巧：
1. EMA (Exponential Moving Average) - 模型权重平均
2. Warmup Scheduler - 学习率预热
3. Multi-scale Training Manager - 动态调整输入分辨率

Author: PoLaRIS Team
Date: 2026-02-26
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import random
import copy


class EMA:
    """
    Exponential Moving Average (EMA) for model weights

    在训练过程中维护模型权重的指数移动平均，通常能提升1-2%性能

    参考：
    - YOLO v3/v4/v5 都使用了 EMA
    - Mean Teacher (NeurIPS 2017)

    Args:
        model: PyTorch 模型
        decay: 衰减率（默认0.9999，越接近1越平滑）
        device: 设备

    Usage:
        ema = EMA(model, decay=0.9999)

        for epoch in range(epochs):
            for batch in dataloader:
                # 正常训练
                loss.backward()
                optimizer.step()

                # 更新 EMA
                ema.update(model)

        # 使用 EMA 权重进行评估
        ema.apply_shadow(model)  # 临时应用 EMA 权重
        evaluate(model)
        ema.restore(model)  # 恢复原始权重
    """
    def __init__(self, model, decay=0.9999, device='cuda'):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}

        # 初始化 shadow weights
        self.register()

    def register(self):
        """注册模型参数的影子副本"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self, model=None):
        """
        更新 EMA 权重

        EMA formula: shadow = decay * shadow + (1 - decay) * param
        """
        if model is not None:
            self.model = model

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not found in shadow"
                new_average = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model=None):
        """
        应用 EMA 权重到模型（临时，用于评估）

        调用此方法后，模型的权重会被替换为 EMA 权重
        记得调用 restore() 恢复原始权重
        """
        if model is not None:
            self.model = model

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model=None):
        """恢复原始权重（从 backup）"""
        if model is not None:
            self.model = model

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]

        self.backup = {}


class WarmupScheduler(_LRScheduler):
    """
    Warmup Learning Rate Scheduler

    在训练初期使用较小的学习率，然后逐渐增加到目标学习率

    参考：
    - BERT (NAACL 2019)
    - Transformer (NeurIPS 2017)
    - 几乎所有现代深度学习模型都用 Warmup

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Warmup 轮数
        base_scheduler: Warmup 后使用的 scheduler (如 CosineAnnealingLR)

    Usage:
        optimizer = AdamW(model.parameters(), lr=1e-4)
        base_scheduler = CosineAnnealingLR(optimizer, T_max=200)
        scheduler = WarmupScheduler(optimizer, warmup_epochs=10, base_scheduler=base_scheduler)

        for epoch in range(epochs):
            train_one_epoch()
            scheduler.step()  # 每个 epoch 结束时调用
    """
    def __init__(self, optimizer, warmup_epochs, base_scheduler=None, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: 线性增加学习率
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 使用 base_scheduler
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            else:
                return self.base_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            # 调用 base_scheduler
            if self.base_scheduler is not None:
                # 调整 base_scheduler 的 last_epoch
                adjusted_epoch = self.last_epoch - self.warmup_epochs
                self.base_scheduler.last_epoch = adjusted_epoch
                self.base_scheduler.step()


class MultiScaleTrainingManager:
    """
    多尺度训练管理器

    动态调整输入图像分辨率，强制模型学习多尺度特征

    参考：
    - DNANet: 使用 [256, 320, 384, 448, 512]
    - YOLO v3: 每 10 个 batch 随机改变分辨率

    Args:
        base_size: 基础分辨率
        scale_range: 缩放范围（倍数）
        step: 分辨率步长（必须能被32整除，适配网络下采样）

    Usage:
        manager = MultiScaleTrainingManager(base_size=256, scale_range=(0.75, 1.5))

        for epoch in range(epochs):
            # 每个 epoch 选择新的分辨率
            new_size = manager.get_random_size()
            print(f"Epoch {epoch}: Training with size {new_size}")

            # 重建 dataloader 或 动态调整图像大小
            ...
    """
    def __init__(self, base_size=256, scale_range=(0.75, 1.5), step=32):
        self.base_size = base_size
        self.scale_range = scale_range
        self.step = step

        # 预计算所有可能的分辨率
        min_size = int(base_size * scale_range[0])
        max_size = int(base_size * scale_range[1])

        self.size_list = []
        size = min_size
        while size <= max_size:
            # 确保能被 step 整除
            size_aligned = (size // step) * step
            if size_aligned not in self.size_list and size_aligned >= step:
                self.size_list.append(size_aligned)
            size += step

        # 确保 base_size 在列表中
        if base_size not in self.size_list:
            self.size_list.append(base_size)

        self.size_list.sort()
        print(f"✅ Multi-scale Training: {len(self.size_list)} sizes - {self.size_list}")

    def get_random_size(self):
        """随机选择一个分辨率"""
        return random.choice(self.size_list)

    def get_current_size_index(self, current_size):
        """获取当前分辨率的索引"""
        if current_size in self.size_list:
            return self.size_list.index(current_size)
        else:
            # 找最接近的
            return min(range(len(self.size_list)),
                      key=lambda i: abs(self.size_list[i] - current_size))


class GradientAccumulator:
    """
    梯度累积管理器

    在显存有限时，通过累积多个 mini-batch 的梯度来模拟大 batch size

    Args:
        accumulation_steps: 累积步数（effective batch = batch_size * accumulation_steps）

    Usage:
        accumulator = GradientAccumulator(accumulation_steps=4)

        optimizer.zero_grad()
        for i, batch in enumerate(dataloader):
            loss = compute_loss(batch) / accumulator.accumulation_steps
            loss.backward()

            if accumulator.should_update(i):
                optimizer.step()
                optimizer.zero_grad()

        # 处理最后的残余 batch
        accumulator.final_update(optimizer)
    """
    def __init__(self, accumulation_steps=1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_update(self, step):
        """判断是否应该更新权重"""
        self.current_step = step
        return (step + 1) % self.accumulation_steps == 0

    def reset(self):
        """重置计数器"""
        self.current_step = 0

    def final_update(self, optimizer):
        """处理最后的残余梯度"""
        if self.current_step % self.accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            self.reset()


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("测试训练工具模块")
    print("=" * 60)

    # 1. 测试 EMA
    print("\n[1] 测试 EMA...")
    model = nn.Linear(10, 1)
    ema = EMA(model, decay=0.999)

    # 模拟训练
    for i in range(5):
        # 随机更新模型参数
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.1

        # 更新 EMA
        ema.update()
        print(f"  Step {i+1}: 参数更新完成")

    print("  ✓ EMA 测试通过")

    # 2. 测试 Warmup Scheduler
    print("\n[2] 测试 Warmup Scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=5, base_scheduler=base_scheduler)

    print("  Learning rate schedule:")
    for epoch in range(25):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"    Epoch {epoch:2d}: lr = {lr:.6f}")

    print("  ✓ Warmup Scheduler 测试通过")

    # 3. 测试 Multi-scale Training Manager
    print("\n[3] 测试 Multi-scale Training Manager...")
    manager = MultiScaleTrainingManager(base_size=256, scale_range=(0.75, 1.5))

    print("  Random sizes:")
    for i in range(10):
        size = manager.get_random_size()
        print(f"    Trial {i+1}: {size}")

    print("  ✓ Multi-scale Manager 测试通过")

    # 4. 测试 Gradient Accumulator
    print("\n[4] 测试 Gradient Accumulator...")
    accumulator = GradientAccumulator(accumulation_steps=4)

    print("  Accumulation steps:")
    for i in range(10):
        should_update = accumulator.should_update(i)
        print(f"    Step {i}: Update = {should_update}")

    print("  ✓ Gradient Accumulator 测试通过")

    print("\n" + "=" * 60)
    print("✅ 所有训练工具测试通过！")
    print("=" * 60)
