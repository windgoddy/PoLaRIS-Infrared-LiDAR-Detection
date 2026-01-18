# PoLaRIS DataLoader 升级总结

## 📋 修改概览

本次升级使您的模型可以：
1. ✅ **支持 16-bit 红外图像**（Min-Max 归一化）
2. ✅ **支持软标签训练**（Oracle Masks: 0.0, 0.6, 1.0）
3. ✅ **兼容旧版 8-bit DataLoader**（通过 .sh 脚本切换）
4. ✅ **正确使用深度图**（depth_maps/*.npy）而非原始点云
5. ✅ **修复了训练循环中的软标签使用**

---

## 🔧 修改文件清单

### 1. 核心文件修改

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `model/utils_lidar.py` | 添加深度图加载支持、修复 LiDAR 翻转注释 | ✅ 完成 |
| `model/parse_args_train.py` | 添加新参数：`use_lidar_dataloader`, `normalize_16bit`, `use_soft_labels` | ✅ 完成 |
| `train_Phase3.py` | 修改训练/测试循环支持新旧 DataLoader 切换 | ✅ 完成 |
| `docs/DataLoader_Guide.md` | 添加快速开始指南和 FAQ | ✅ 完成 |

### 2. 新增文件

| 文件 | 用途 | 状态 |
|------|------|------|
| `scripts/run_Phase3_DualGeo_8bit.sh` | 8-bit 旧版 DataLoader 训练脚本 | ✅ 新增 |
| `scripts/run_Phase3_DualGeo_16bit.sh` | 16-bit 新版 DataLoader 训练脚本（推荐） | ✅ 新增 |
| `scripts/run_Phase3_DualGeo_16bit_IR_only.sh` | 仅红外图像（无深度图）训练脚本 | ✅ 新增 |
| `scripts/test_dataloader.py` | DataLoader 测试脚本 | ✅ 新增 |
| `docs/UPGRADE_SUMMARY.md` | 本文档 | ✅ 新增 |

---

## 🚀 快速开始

### 步骤 1: 验证数据集结构

确保您的数据集包含以下文件：

```bash
dataset/Pohang-Canal/
├── images/              # 红外图像（支持 16-bit）
├── masks/               # GT 标签
├── oracle_masks/        # Oracle 标签（软标签）
├── depth_maps/          # 深度图（.npy 文件，in_channels=2 需要）
└── lidar_roi/           # LiDAR 点云（可选，仅用于生成深度图）
```

**检查命令**：
```bash
# 检查 Oracle Masks
ls dataset/Pohang-Canal/oracle_masks/ | wc -l

# 检查深度图（in_channels=2 需要）
ls dataset/Pohang-Canal/depth_maps/*.npy | wc -l
```

### 步骤 2: 选择训练模式

#### 方式 A: 16-bit + 软标签（推荐）

```bash
chmod +x scripts/run_Phase3_DualGeo_16bit.sh
./scripts/run_Phase3_DualGeo_16bit.sh
```

#### 方式 B: 旧版 8-bit

```bash
chmod +x scripts/run_Phase3_DualGeo_8bit.sh
./scripts/run_Phase3_DualGeo_8bit.sh
```

#### 方式 C: 仅红外（无深度图）

```bash
chmod +x scripts/run_Phase3_DualGeo_16bit_IR_only.sh
./scripts/run_Phase3_DualGeo_16bit_IR_only.sh
```

### 步骤 3: 测试 DataLoader（可选）

```bash
python scripts/test_dataloader.py
```

---

## 🔍 关键修改详解

### 1. `model/utils_lidar.py` - 深度图加载支持

**问题**：原始版本只加载 LiDAR 点云 (.bin)，但模型期望的是深度图 (.npy)

**解决方案**：
- 添加 `_load_depth()` 方法加载 `.npy` 深度图
- 添加 `in_channels` 参数支持单通道/双通道模式
- 修改 `__getitem__()` 返回正确的图像格式：
  - `in_channels=1`: 返回 `(1, H, W)` 红外图像
  - `in_channels=2`: 返回 `(2, H, W)` 红外+深度堆叠

**代码示例**：
```python
if self.in_channels == 2:
    # 加载深度图
    depth = self._load_depth(depth_path)
    # 堆叠 IR + Depth
    img = img / 255.0
    depth = depth / 80.0
    combined = np.stack([img, depth], axis=0)  # (2, H, W)
```

### 2. `train_Phase3.py` - 训练循环修复

**问题 1**：使用硬标签 (`labels`) 而非软标签 (`oracle_masks`) 训练

**修复前**：
```python
loss_seg = SoftIoULoss(pred, labels)  # ❌ 错误
```

**修复后**：
```python
train_target = oracle_masks if self.use_soft_labels else labels
loss_seg = SoftIoULoss(pred, train_target)  # ✅ 正确
```

**问题 2**：模型期望 2 通道输入，而非分离的点云

**修复前**：
```python
pred, pred_conf = self.model(data, lidar_points)  # ❌ 模型不接受 lidar_points
```

**修复后**：
```python
# data 已经是 (B, 2, H, W)，包含 IR + Depth
pred, pred_conf = self.model(data)  # ✅ 正确
```

### 3. 新增命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_lidar_dataloader` | False | 是否使用新 DataLoader |
| `--normalize_16bit` | True | 16-bit 归一化方式 |
| `--use_soft_labels` | True | 是否使用软标签 |

**使用示例**：
```bash
python train_Phase3.py \
    --use_lidar_dataloader True \
    --normalize_16bit True \
    --use_soft_labels True \
    --in_channels 2
```

---

## ⚠️ 重要注意事项

### 1. 关于 16-bit 精度损失

当前实现流程：
```
16-bit (0-65535) → Min-Max归一化 → uint8 (0-255) → 数据增强 → float32 (0-1)
```

**影响**：
- 灰度级从 65536 降到 256（损失 99.6%）
- 原因：PIL 的数据增强需要 uint8 格式

**如何优化**（如果需要）：
1. 使用 `torchvision.transforms` 代替 PIL（支持 float32）
2. 跳过 uint8 转换，直接在 float32 上做增强

### 2. 关于软标签的 Loss 值

使用软标签后，Loss 值会**变小**，这是**正常现象**：

```python
# 硬标签：target ∈ {0, 1}
loss_hard = BCE(pred, [0, 1, 1, 0])  # 较大

# 软标签：target ∈ [0.0, 1.0]
loss_soft = BCE(pred, [0, 0.6, 1.0, 0])  # 较小（0.6 降低了 loss）
```

**解决方案**（如果收敛慢）：
```python
# 方式 1: 提高正样本权重
criterion = CombinedSoftLoss(pos_weight=torch.tensor([2.0]))

# 方式 2: 增大 loss 系数
loss = criterion(pred, target) * 2.0
```

### 3. 关于 LiDAR 点云

**重要**：当前实现中，LiDAR 点云 (.bin) **仅用于生成深度图**，训练时使用的是**预生成的深度图** (.npy)。

如果您需要在训练时动态使用点云，需要：
1. 在模型中添加点云投影模块
2. 修改 DataLoader 返回点云而非深度图
3. 修改模型 `forward()` 接受点云输入

---

## 📊 对比表：旧版 vs 新版

| 特性 | 旧版 DataLoader | 新版 DataLoader |
|------|----------------|----------------|
| 16-bit 支持 | ❌ | ✅ Min-Max 归一化 |
| 软标签支持 | ❌ 二值化 | ✅ 保留 float 值 |
| 深度图加载 | ✅ | ✅ |
| LiDAR 点云 | ❌ | ✅ 加载但不用于训练 |
| 返回格式 | Tuple | Dict（更清晰） |
| 数据增强 | ✅ | ✅ |
| Oracle Mask | ❌ | ✅ 作为训练 target |

---

## 🧪 测试建议

### 1. 对比实验

建议运行以下对比实验：

| 实验 | 脚本 | 目的 |
|------|------|------|
| Baseline | `run_Phase3_DualGeo_8bit.sh` | 8-bit + 硬标签 |
| Test 1 | `run_Phase3_DualGeo_16bit.sh` | 16-bit + 软标签 |
| Test 2 | `run_Phase3_DualGeo_16bit_IR_only.sh` | 16-bit + 无深度图 |

### 2. 监控指标

- **训练 Loss**：软标签会让 Loss 变小（正常）
- **验证 IoU**：应该提升（16-bit + 软标签的优势）
- **收敛速度**：可能稍慢，可以增加 `pos_weight`

### 3. 调试技巧

如果遇到问题，运行测试脚本：
```bash
python scripts/test_dataloader.py
```

检查输出：
- ✅ `image shape`: `(B, 2, H, W)` 或 `(B, 1, H, W)`
- ✅ `image range`: `[0.0, 1.0]`
- ✅ `oracle_mask unique values`: `tensor([0.0000, 0.6000, 1.0000])`

---

## 📚 参考文档

- **详细使用指南**：[docs/DataLoader_Guide.md](DataLoader_Guide.md)
- **代码实现**：[model/utils_lidar.py](../model/utils_lidar.py)
- **训练脚本示例**：[scripts/](../scripts/)

---

## ❓ 常见问题

### Q1: 为什么软标签是 0.6 而不是 0.5？

**A**: 设计选择：
- **1.0**: 有 LiDAR 验证的目标（强监督）
- **0.6**: 无 LiDAR 但视觉确认的目标（弱监督，避免过拟合）
- **0.0**: 背景

0.6 > 0.5 表明这些目标确实存在，只是缺少几何验证。

### Q2: 新版 DataLoader 会自动切换 8-bit/16-bit 吗？

**A**: 会！DataLoader 自动检测图像格式：
```python
if img.mode == 'I;16':
    is_16bit = True
    # 16-bit 处理
else:
    # 8-bit 处理
```

### Q3: 如果我的数据集没有深度图怎么办？

**A**: 使用 `in_channels=1` 模式：
```bash
python train_Phase3.py \
    --in_channels 1 \
    --use_lidar_dataloader True
```

### Q4: 旧版脚本还能用吗？

**A**: 完全可以！设置 `--use_lidar_dataloader False` 即可：
```bash
python train_Phase3.py \
    --use_lidar_dataloader False \
    --in_channels 2
```

---

## 🎯 下一步行动

1. **验证数据集**：
   ```bash
   ls dataset/Pohang-Canal/oracle_masks/ | wc -l
   ls dataset/Pohang-Canal/depth_maps/*.npy | wc -l
   ```

2. **测试 DataLoader**：
   ```bash
   python scripts/test_dataloader.py
   ```

3. **运行训练**：
   ```bash
   # 推荐：16-bit + 软标签
   ./scripts/run_Phase3_DualGeo_16bit.sh
   ```

4. **监控结果**：
   - 检查 Loss 曲线
   - 检查 IoU 指标
   - 对比旧版结果

---

## ✅ 验证清单

在运行训练前，请确认：

- [ ] Oracle masks 已生成（`oracle_masks/` 文件夹存在）
- [ ] 深度图已生成（`depth_maps/*.npy` 存在，in_channels=2 时）
- [ ] 测试脚本通过（`python scripts/test_dataloader.py`）
- [ ] 选择了正确的 .sh 脚本
- [ ] GPU 可用（`nvidia-smi` 检查）

---

## 📞 支持

如果遇到问题：
1. 运行 `python scripts/test_dataloader.py` 诊断
2. 检查 `docs/DataLoader_Guide.md` FAQ 部分
3. 查看错误堆栈，定位问题文件

---

**祝训练顺利！🚀**
