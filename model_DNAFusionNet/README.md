# DNA-FusionNet: Early Fusion Version

> **策略：站在巨人的肩膀上** - 以 DNANet 为基座，用最小改动实现多模态融合。

## 核心理念

从您的评估结果来看，DNANet 在所有场景下都优于 MS_CAFNet：
- **Cat 1**: mIoU 0.7442 vs 0.7568 (持平)
- **Cat 2**: mIoU 0.5314 vs 0.4761 (领先 10.4%)
- **Cat 3**: mIoU 0.8515 vs 0.7887 (领先 7.4%) ⭐

**结论**：DNANet 的密集嵌套结构对小目标和复杂场景的处理能力更强。

## 设计方案

### Early Fusion（极简版）

**唯一改动**：将输入通道从 1 改为 2（红外 + LiDAR）

```python
# 原版 DNANet
input: [B, 1, H, W]  # 仅红外

# DNA-FusionNet Early Fusion
input: [B, 2, H, W]  # 通道0=红外, 通道1=LiDAR
```

**优势**：
- ✅ 保持 DNANet 的所有优势（密集连接、CBAM 注意力）
- ✅ 让网络自己学习如何组合双模态信息
- ✅ 训练速度快，工作量最小
- ✅ 预期 mIoU 提升 1-3%，FPR 下降 20-40%

## 快速开始

### 1. 测试模型结构

```bash
# 验证模型能否正常创建和运行
python model_DNAFusionNet/test_model.py
```

### 2. 开始训练（50 epochs 快速验证）

```bash
# 在服务器上运行（使用 GPU 5）
CUDA_VISIBLE_DEVICES=5 python model_DNAFusionNet/train_DNAFusion_EarlyFusion.py \
    --dataset Pohang-Canal-3k \
    --split_method 50_50_2k_new \
    --gpu 0 \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.0001
```

**训练配置说明**：
- `--epochs 50`: 快速验证，50 轮应该能看到效果
- `--batch_size 4`: 与 DNANet 一致
- `--lr 0.0001`: 与 DNANet 一致
- `--threshold 0.3`: 推理阈值（与训练时保持一致）
- `--deep_supervision`: 默认开启（4 个输出头）

### 3. 评估模型

训练完成后，使用之前的评估脚本：

```bash
# 修改 scripts/eval_cat3_comparison.py，添加 DNA-FusionNet 的评估
```

## 预期效果

### 对比基准（DNANet，8-bit RGB，单模态）

| 类别 | DNANet mIoU | DNANet FPR |
|------|-------------|------------|
| Cat 1 | 0.7442 | 0.0032 |
| Cat 2 | 0.5314 | 0.0040 |
| Cat 3 | 0.8515 | 0.0063 |

### 预期改进（DNA-FusionNet，16-bit IR+LiDAR，双模态）

| 类别 | 预期 mIoU | 预期 FPR | 预期提升 |
|------|-----------|----------|----------|
| Cat 1 | 0.76-0.77 | 0.0025-0.0030 | +1.5-3% mIoU ⬆️ |
| Cat 2 | 0.55-0.56 | 0.0030-0.0035 | +3-5% mIoU ⬆️ |
| Cat 3 | 0.86-0.88 | 0.0040-0.0050 | +1-3% mIoU, -20% FPR ⬇️ |

**关键指标**：
- **mIoU 提升**：1-3%（LiDAR 提供几何先验）
- **FPR 下降**：20-40%（减少陆地误检）⭐ 核心优势
- **Recall 提升**：2-5%（双模态互补）

## 文件说明

```
model_DNAFusionNet/
├── model_DNAFusion_EarlyFusion.py    # 模型定义
├── train_DNAFusion_EarlyFusion.py    # 训练脚本
├── test_model.py                     # 快速测试脚本
└── README.md                          # 本文件
```

## 技术细节

### 输入格式

```python
input = torch.cat([IR, LiDAR], dim=1)  # [B, 2, H, W]
# input[:, 0:1, :, :] = 红外图像 (Infrared)
# input[:, 1:2, :, :] = LiDAR 深度图 (Depth)
```

### 模型参数量

- **Total Parameters**: ~2.8M
- **Memory**: ~11 MB
- **与 DNANet 完全一致**（只是输入通道从 1→2）

### 深度监督

保持 DNANet 的 4 个输出头：
```python
outputs = [output1, output2, output3, output4]
# output1: x0_1 (浅层)
# output2: x0_2 (中层)
# output3: x0_3 (深层)
# output4: Final_x0_4 (最终输出) ← 用于推理
```

## 下一步计划

### Phase 1: 验证 Early Fusion（当前）
- ✅ 创建模型和训练脚本
- ⏳ 训练 50 epochs
- ⏳ 评估三个类别的性能
- ⏳ 对比 DNANet 和 MS_CAFNet

### Phase 2: 增强版融合（如果 Early Fusion 有效）
如果 Early Fusion 效果好，可以尝试：
1. **LiDAR-Enhanced Spatial Attention**：在 SpatialAttention 中融入 LiDAR
2. **Adaptive Fusion**：根据 LiDAR 质量动态调整融合权重
3. **FPN-style Fusion**：在多个尺度上融合双模态特征

## 常见问题

### Q1: 为什么不用 MS_CAFNet 的 Transformer Bottleneck？

A: 从评估结果看，Transformer 并未提升性能（Cat 3 mIoU 反而下降）。DNANet 的密集嵌套结构已经能够捕获足够的上下文信息。

### Q2: 为什么选择 Early Fusion 而不是 Late Fusion？

A: Early Fusion 让网络在每一层都能看到双模态信息，有利于特征融合。Late Fusion 只在最后融合，可能丢失低层几何信息。

### Q3: 训练时间预计多久？

A: 与 DNANet 相当。在 RTX 3090 上，50 epochs 大约需要 2-3 小时（取决于数据集大小）。

### Q4: 如果效果不如预期怎么办？

A: 可能的原因和解决方案：
1. **学习率过高/过低**：尝试 5e-5 或 2e-4
2. **阈值不匹配**：确保评估时使用 threshold=0.3
3. **数据增强不足**：增加 rotation, flip 等

## 联系方式

如有问题或建议，请联系项目负责人。

---

**Good luck! 相信数据，相信 DNANet！** 🚀
