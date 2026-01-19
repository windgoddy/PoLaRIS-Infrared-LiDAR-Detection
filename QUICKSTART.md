# 快速开始指南 - PoLaRIS Infrared-LiDAR Detection

## 🚀 一分钟快速开始

### 1. 训练模型

```bash
# 自动模式（推荐）- 根据数据集自动选择
./scripts/train.sh auto --dataset Pohang-Canal-3k  # 自动使用 16-bit
./scripts/train.sh auto --dataset Pohang-Canal     # 自动使用 8-bit

# 手动指定模式
./scripts/train.sh 16bit  # 强制 16-bit 模式
./scripts/train.sh 8bit   # 强制 8-bit 模式

# 自定义参数
./scripts/train.sh auto --dataset Pohang-Canal-3k --gpu 0 --epochs 100
```

**数据集模式规则**:

- `Pohang-Canal-3k` → 自动使用 **16-bit 模式**（Min-Max 归一化 + 软标签）
- 其他数据集 → 自动使用 **8-bit 模式**（旧版 DataLoader）

### 2. 测试 DataLoader

```bash
python scripts/test_dataloader.py
```

### 3. 生成 Oracle Masks

```bash
python scripts/generate_oracle_masks.py --dataset Pohang-Canal
```

---

## 📚 完整工作流程

### 步骤 1: 准备数据集

```bash
# 1. 确保数据集结构正确
dataset/Pohang-Canal/
├── images/       # 红外图像（支持 8-bit 或 16-bit .png）
├── masks/        # GT 标签
└── lidar_roi/    # LiDAR 点云 (.bin)

# 2. 生成深度图（可选，in_channels=2 需要）
python scripts/generate_depth_maps.py --dataset Pohang-Canal

# 3. 生成 Oracle Masks（软标签）
python scripts/generate_oracle_masks.py --dataset Pohang-Canal

# 4. 验证数据集
python scripts/verify_dataset.py --dataset Pohang-Canal
```

### 步骤 2: 选择训练模式

| 模式 | 适用场景 | 命令 |
|------|---------|------|
| **auto** | 自动选择（推荐）：Pohang-Canal-3k 用 16bit，其他用 8bit | `./scripts/train.sh auto --dataset <数据集>` |
| **16bit** | 手动强制 16-bit 红外 + 深度图 + 软标签 | `./scripts/train.sh 16bit` |
| **8bit** | 手动强制 8-bit 红外 + 深度图（旧版） | `./scripts/train.sh 8bit` |
| **16bit-ir** | 仅 16-bit 红外（无深度图） | `./scripts/train.sh 16bit-ir` |
| **baseline1** | DNANet 基准对比 | `./scripts/train.sh baseline1` |
| **baseline2** | DNANet 变体对比 | `./scripts/train.sh baseline2` |

### 步骤 3: 训练模型

```bash
# 推荐：使用自动模式
./scripts/train.sh auto --dataset Pohang-Canal-3k

# 自定义参数
./scripts/train.sh auto \
    --dataset Pohang-Canal-3k \
    --gpu 0 \
    --epochs 200

# 或手动指定模式
./scripts/train.sh 16bit --dataset Pohang-Canal-3k
```

### 步骤 4: 分析结果

```bash
# 分析训练结果
python scripts/analyze_training.py --experiment Phase3_DualGeo_16bit

# 可视化 LiDAR 投影
python scripts/visualize_lidar_projection.py --dataset Pohang-Canal

# 模型复杂度对比
python scripts/compare_model_complexity.py
```

---

## 🧹 整理 Scripts 文件夹

Scripts 文件夹有 **22 个旧文件**可以归档或删除。

### 预览要归档的文件

```bash
./scripts/cleanup.sh
```

### 归档旧文件（安全）

```bash
./scripts/cleanup.sh --archive
```

归档后，文件会移动到 `scripts/archive/`，可随时恢复：

```bash
# 恢复某个文件
cp scripts/archive/training/run_Phase3_DualGeo.sh scripts/
```

### 删除旧文件（危险！）

```bash
./scripts/cleanup.sh --delete
```

---

## 📖 详细文档

- **项目主页**: [README.md](README.md) - 项目概览和快速开始
- **高级使用指南**: [docs/ADVANCED.md](docs/ADVANCED.md) - DataLoader 技术细节和高级配置
- **脚本详细说明**: [docs/SCRIPTS_GUIDE.md](docs/SCRIPTS_GUIDE.md) - 每个脚本的作用、输入输出
- **完整配置文件**: [run_config.sh](run_config.sh) - 所有可用参数和配置模板

---

## 🎯 常用命令速查

### 训练

```bash
# 自动模式（推荐）
./scripts/train.sh auto --dataset Pohang-Canal-3k --gpu 0

# 对比实验
./scripts/train.sh baseline1 --dataset Pohang-Canal --gpu 0
./scripts/train.sh auto --dataset Pohang-Canal-3k --gpu 1
./scripts/train.sh auto --dataset Pohang-Canal --gpu 2
```

### 数据准备

```bash
# 完整数据准备流程
./scripts/run_prepare_dataset.sh

# 单独生成组件
python scripts/generate_depth_maps.py --dataset Pohang-Canal
python scripts/generate_oracle_masks.py --dataset Pohang-Canal
```

### 测试和验证

```bash
# 测试 DataLoader
python scripts/test_dataloader.py

# 验证数据集
python scripts/verify_dataset.py --dataset Pohang-Canal

# 诊断问题
python scripts/diagnose_dataset.py --dataset Pohang-Canal
```

### 可视化

```bash
# LiDAR 投影可视化
python scripts/visualize_lidar_projection.py --dataset Pohang-Canal

# 模型架构可视化
python scripts/visualize_model_architecture.py --model MS_CAFNet_DualGeo
```

---

## ❓ 常见问题

### Q1: 如何切换 GPU？

```bash
./scripts/train.sh 16bit --gpu 3
```

或者手动设置：

```bash
export CUDA_VISIBLE_DEVICES=3
./scripts/train.sh 16bit
```

### Q2: 16-bit 和 8-bit 有什么区别？

| 特性 | 8-bit 模式 | 16-bit 模式 |
|------|-----------|------------|
| 图像支持 | 8-bit .png | 8-bit + 16-bit .png |
| 归一化 | /255 | Min-Max 归一化 |
| 软标签 | ❌ | ✅ (0.0, 0.6, 1.0) |
| DataLoader | 旧版 | 新版 PoLaRIS |

### Q3: 我的旧脚本还能用吗？

可以，但建议迁移到统一入口：

```bash
# 旧方式
./scripts/run_Phase3_DualGeo_16bit.sh

# 新方式（推荐）
./scripts/train.sh 16bit
```

### Q4: 如何添加自定义训练模式？

编辑 `scripts/train.sh`，在 `case $MODE in` 部分添加：

```bash
my-mode)
    echo "🔹 My Custom Mode"
    python train_Phase3.py \
        --experiment_name my_custom_mode \
        --model MS_CAFNet_DualGeo \
        --dataset Pohang-Canal \
        ...
    ;;
```

然后运行：

```bash
./scripts/train.sh my-mode
```

---

## 🎉 整理成果

### 整理前

- ❌ 14 个训练脚本（难以管理）
- ❌ 40+ 个 Python 脚本（功能重复）
- ❌ 没有统一文档

### 整理后

- ✅ 1 个统一训练入口 (`train.sh`)
- ✅ 清晰的文件分类
- ✅ 完整的使用文档
- ✅ 22 个旧文件可归档

---

**开始训练吧！** 🚀

```bash
# Pohang-Canal-3k 数据集（自动使用 16-bit）
./scripts/train.sh auto --dataset Pohang-Canal-3k

# 其他数据集（自动使用 8-bit）
./scripts/train.sh auto --dataset Pohang-Canal
```
