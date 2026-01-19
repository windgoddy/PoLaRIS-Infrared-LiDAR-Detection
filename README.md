# PoLaRIS: Infrared-LiDAR Detection

**P**oint Cloud and **L**iDAR **R**ange **I**mage **S**egmentation for Infrared Small Target Detection

> 基于 DNANet 的红外小目标检测项目，增强支持 LiDAR 点云和 16-bit 红外图像

---

## 项目特点

- ✅ **16-bit 红外图像支持** - Min-Max 归一化，保留更多细节
- ✅ **LiDAR 点云融合** - 结合深度信息提升检测精度
- ✅ **软标签训练** - Oracle Masks（0.0, 0.6, 1.0）提供更灵活的监督
- ✅ **自动模式选择** - 根据数据集自动选择 8-bit/16-bit 训练模式
- ✅ **统一训练脚本** - 一个脚本支持所有训练模式

---

## 快速开始

### 安装依赖

```bash
pip install torch torchvision
pip install numpy pillow opencv-python
```

### 训练模型

```bash
# 自动模式（推荐）- 根据数据集自动选择
./scripts/train.sh auto --dataset Pohang-Canal-3k  # 16-bit 模式
./scripts/train.sh auto --dataset Pohang-Canal     # 8-bit 模式

# 手动指定模式
./scripts/train.sh 16bit --dataset Pohang-Canal-3k --gpu 0 --epochs 200
```

**数据集模式规则**：

- `Pohang-Canal-3k` → 自动使用 **16-bit 模式**（Min-Max 归一化 + 软标签）
- 其他数据集 → 自动使用 **8-bit 模式**（旧版 DataLoader）

### 数据集准备

```bash
# 1. 生成深度图
python scripts/generate_depth_maps.py --dataset Pohang-Canal-3k

# 2. 生成 Oracle Masks
python scripts/generate_oracle_masks.py --dataset Pohang-Canal-3k

# 3. 验证数据集
python scripts/verify_dataset.py --dataset Pohang-Canal-3k

# 4. 测试 DataLoader
python scripts/test_dataloader.py
```

---

## 数据集结构

```
dataset/Pohang-Canal-3k/
├── images/              # 红外图像（支持 8-bit 或 16-bit .png）
├── masks/               # Ground Truth 标签（硬标签）
├── oracle_masks/        # Oracle 标签（软标签：0.0, 0.6, 1.0）
├── depth_maps/          # 深度图（.npy 文件，in_channels=2 需要）
└── lidar_roi/           # LiDAR 点云（.bin 文件，用于生成深度图）
```

---

## 训练模式

| 模式 | 说明 | 适用场景 |
| ---- | ---- | -------- |
| **auto** | 自动选择模式 | 推荐使用，自动根据数据集选择 8bit/16bit |
| **16bit** | 16-bit + 软标签 | Pohang-Canal-3k 数据集（完整 PoLaRIS） |
| **8bit** | 8-bit + 硬标签 | 其他数据集或兼容旧版 |
| **16bit-ir** | 仅红外（无深度） | 消融实验：移除深度图 |
| **baseline1** | DNANet 原始 | 对比基准（DNANet 论文配置） |

---

## 项目结构

```
PoLaRIS-Infrared-LiDAR-Detection/
├── model/
│   ├── utils_lidar.py           # PoLaRIS DataLoader（16-bit + LiDAR）
│   ├── parse_args_train.py      # 训练参数配置
│   └── ...
├── scripts/
│   ├── train.sh                 # 统一训练脚本 ⭐
│   ├── generate_depth_maps.py   # 生成深度图
│   ├── generate_oracle_masks.py # 生成软标签
│   ├── test_dataloader.py       # 测试 DataLoader
│   ├── verify_dataset.py        # 验证数据集
│   └── archive/                 # 归档的旧脚本
├── dataset/                     # 数据集目录
├── train_Phase3.py              # 主训练脚本
├── README.md                    # 本文档
└── QUICKSTART.md                # 快速开始指南 📖
```

---

## 常用命令

### 训练

```bash
# 自动模式（推荐）
./scripts/train.sh auto --dataset Pohang-Canal-3k --gpu 0

# 对比实验
./scripts/train.sh baseline1 --dataset Pohang-Canal --gpu 0
./scripts/train.sh auto --dataset Pohang-Canal-3k --gpu 1
./scripts/train.sh auto --dataset Pohang-Canal --gpu 2
```

### 数据分析

```bash
# 分析训练结果
python scripts/analyze_training.py --experiment Phase3_DualGeo_16bit

# 可视化 LiDAR 投影
python scripts/visualize_lidar_projection.py --dataset Pohang-Canal-3k

# 模型复杂度对比
python scripts/compare_model_complexity.py
```

### 测试和验证

```bash
# 测试 DataLoader
python scripts/test_dataloader.py

# 验证数据集
python scripts/verify_dataset.py --dataset Pohang-Canal-3k

# 诊断问题
python scripts/diagnose_dataset.py --dataset Pohang-Canal-3k
```

---

## 技术细节

### 16-bit 图像处理

```python
# Min-Max 归一化
img_normalized = (img - img_min) / (img_max - img_min) * 255  # → [0, 255]
```

### 软标签（Oracle Masks）

- **1.0** - 有 LiDAR 验证的目标（强监督）
- **0.6** - 无 LiDAR 但视觉确认的目标（弱监督）
- **0.0** - 背景

### LiDAR 数据融合

```python
# 2-通道输入：IR + Depth
if in_channels == 2:
    img = np.stack([ir_channel, depth_channel], axis=0)  # (2, H, W)
```

---

## 文档

- **快速开始指南**: [QUICKSTART.md](QUICKSTART.md) - 详细的使用教程和常见问题
- **高级指南**: [docs/ADVANCED.md](docs/ADVANCED.md) - DataLoader 技术细节和优化
- **脚本说明**: [docs/SCRIPTS_GUIDE.md](docs/SCRIPTS_GUIDE.md) - 所有脚本的详细说明
- **完整配置**: [run_config.sh](run_config.sh) - 所有可用参数和配置模板

---

## 整理成果

### 整理前

- ❌ 14 个训练脚本（难以管理）
- ❌ 43 个 Python/Shell 脚本（功能重复）
- ❌ 没有统一文档

### 整理后

- ✅ 1 个统一训练入口 (`train.sh`)
- ✅ 21 个核心脚本（减少 51%）
- ✅ 完整的使用文档
- ✅ 22 个旧文件已归档到 `scripts/archive/`

---

## 致谢

本项目基于 [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection) 开发。

**DNANet**: Dense Nested Attention Network for Infrared Small Target Detection

- Paper: [IEEE TIP 2023](https://arxiv.org/pdf/2106.00487.pdf)
- Authors: Boyang Li, Chao Xiao, Longguang Wang, Yingqian Wang

感谢以下项目的贡献：

- [ACM](https://github.com/YimianDai/open-acm) by Yimian Dai
- [PSA](https://github.com/jiwoon-ahn/psa) by jiwoon-ahn

---

## 引用

如果您使用了本项目的代码，请引用原始 DNANet 论文：

```bibtex
@article{DNANet,
  title={Dense nested attention network for infrared small target detection},
  author={Li, Boyang and Xiao, Chao and Wang, Longguang and Wang, Yingqian and Lin, Zaiping and Li, Miao and An, Wei and Guo, Yulan},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  volume={32},
  pages={1745-1758},
  publisher={IEEE}
}
```

---

## 许可证

本项目遵循原始 DNANet 项目的许可证。

---

**开始使用吧！** 🚀

```bash
./scripts/train.sh auto --dataset Pohang-Canal-3k
```
