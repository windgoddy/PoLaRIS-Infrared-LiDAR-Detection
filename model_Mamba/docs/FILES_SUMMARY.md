# PoLaRIS-Gaussian-Mamba 文件清单

> 本文档总结所有新增的 Mamba 相关文件及其用途

---

## 📂 文件结构树

```
PoLaRIS-Infrared-LiDAR-Detection/
│
├── 📁 model_Mamba/                          [新增模块 - Mamba架构]
│   ├── __init__.py                           (32行) 包初始化
│   ├── ss2d_components.py                    (600行) SS2D + VSSBlock 核心组件
│   ├── polaris_mamba.py                      (550行) PoLaRIS_Mamba 主模型
│   ├── loss.py                               (300行) Gaussian Focal Loss
│   └── README.md                             (700行) 完整技术文档
│
├── 📁 dataset/
│   └── gaussian_utils.py                     (400行) YOLO → 高斯热力图转换
│
├── 📁 scripts/
│   ├── setup_server.sh                       (150行) 服务器环境配置脚本 ⭐
│   ├── train_mamba_server.sh                 (120行) 一键训练脚本 ⭐
│   └── visualize_mamba.sh                    (90行) 可视化脚本 ⭐
│
├── train_Mamba.py                            (450行) 训练脚本（支持多GPU）
├── vis_mamba.py                              (350行) 可视化脚本
│
├── 📄 MAMBA_QUICKSTART.md                    快速启动指南（服务器版）
├── 📄 SERVER_DEPLOYMENT.md                   服务器部署完整指南 ⭐
└── 📄 MAMBA_FILES_SUMMARY.md                 本文档

总计: ~3600 行代码 + 详细文档
```

---

## 🎯 使用流程（服务器端）

### 阶段1：部署和环境配置

```bash
# 1. 上传代码到服务器
scp -r PoLaRIS-Infrared-LiDAR-Detection user@server:/workspace/
ssh user@server
cd /workspace/PoLaRIS-Infrared-LiDAR-Detection

# 2. 运行环境配置脚本
bash scripts/setup_server.sh
```

**该脚本做什么**：
- ✅ 检查 CUDA/GPU 状态
- ✅ 验证 PyTorch 安装
- ✅ 安装依赖（opencv, tqdm, einops, scipy）
- ✅ 安装 `mamba_ssm` 加速库
- ✅ 运行单元测试

**输出文档**：[scripts/setup_server.sh](scripts/setup_server.sh:1)

---

### 阶段2：数据准备

确保数据集包含以下文件夹：

```
dataset/Pohang-Canal-3k/
├── images/          ✅ 红外图像
├── labels/          ⭐ YOLO格式标注（新需求！）
├── lidar_roi/       ✅ LiDAR点云（可选）
└── 50_50/
    ├── train.txt
    └── test.txt
```

**如果缺少 `labels/` 文件夹**：
- 使用 [dataset/gaussian_utils.py](dataset/gaussian_utils.py:1) 中的工具转换
- 或参考 [SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md:1) "问题3"

---

### 阶段3：训练模型

```bash
# 方法1: 使用一键训练脚本（推荐）
bash scripts/train_mamba_server.sh

# 方法2: 手动命令
python train_Mamba.py \
    --model mamba_small \
    --dataset Pohang-Canal-3k \
    --use_lidar True \
    --gpus "0,1" \
    --train_batch_size 8 \
    --epochs 200
```

**核心文件**：
- [scripts/train_mamba_server.sh](scripts/train_mamba_server.sh:1) - 一键训练脚本
- [train_Mamba.py](train_Mamba.py:1) - Python训练脚本（支持多GPU）

**输出**：
```
result/Pohang-Canal-3k/mamba_small_baseline_server/
├── best_model.pth
├── checkpoint_epoch20.pth
└── train_log.csv
```

---

### 阶段4：可视化结果

```bash
# 方法1: 使用可视化脚本
bash scripts/visualize_mamba.sh

# 方法2: 手动命令
python vis_mamba.py \
    --checkpoint result/.../best_model.pth \
    --model mamba_small \
    --num_samples 20 \
    --output_dir visualizations/
```

**核心文件**：
- [scripts/visualize_mamba.sh](scripts/visualize_mamba.sh:1) - 一键可视化脚本
- [vis_mamba.py](vis_mamba.py:1) - Python可视化脚本

**输出**：
```
visualizations/mamba_small_baseline_server/
├── 000001.png  (4列拼接图: IR | LiDAR | Pred | GT)
├── 000002.png
└── ...
```

---

## 📚 核心模块详解

### 1. SS2D 组件 ([model_Mamba/ss2d_components.py](model_Mamba/ss2d_components.py:1))

**功能**：Vision Mamba 的核心实现，包含 LiDAR 门控注入机制

**关键类**：
- `CrossScan`: 4方向扫描（→ ← ↓ ↑）
- `CrossMerge`: 4方向特征融合
- `SS2D`: Selective Scan 2D **（核心创新：LiDAR门控）**
- `VSSBlock`: Vision State Space Block（编码器基本单元）

**LiDAR 门控逻辑**：
```python
# 在 SS2D.forward() 中
if lidar_feat is not None:
    gate = torch.sigmoid(self.lidar_gate_conv(lidar_feat))  # 0~1
    x_scan = x_scan * (1 + gate)  # 动态增强
    # gate ≈ 1 (LiDAR有效) → 增强 2x
    # gate ≈ 0 (LiDAR缺失) → 保持原样
```

**单元测试**：
```bash
python -m model_Mamba.ss2d_components
```

---

### 2. PoLaRIS_Mamba 模型 ([model_Mamba/polaris_mamba.py](model_Mamba/polaris_mamba.py:1))

**功能**：完整的模型架构定义

**核心类**：
- `PatchEmbed`: 图像 → Patch 嵌入
- `PatchMerging`: 下采样层（类似池化）
- `MambaStage`: 包含多个 VSSBlock 的编码器阶段
- `LiDARDownsampler`: LiDAR 深度图自适应下采样
- `GaussianHead`: 高斯热力图预测头
- `PoLaRIS_Mamba`: 主模型

**模型变体**：
```python
polaris_mamba_tiny()   # ~5M 参数
polaris_mamba_small()  # ~15M 参数 (推荐)
polaris_mamba_base()   # ~30M 参数
```

**前向传播**：
```python
heatmap = model(ir_img, lidar_img)  # (B, 1, H, W) ∈ [0, 1]
```

**单元测试**：
```bash
python -m model_Mamba.polaris_mamba
```

---

### 3. Gaussian Focal Loss ([model_Mamba/loss.py](model_Mamba/loss.py:1))

**功能**：高斯热力图的专用损失函数

**核心类**：
- `GaussianFocalLoss`: 主损失函数（推荐）
- `AdaptiveGaussianFocalLoss`: 自适应权重版本
- `MSEHeatmapLoss`: 简单基线（调试用）
- `AverageMeter`: 损失值统计工具

**损失公式**：
```python
# 正样本 (target == 1)
pos_loss = -((1 - pred)^α) * log(pred)

# 负样本 (target < 1)
neg_weight = (1 - target)^β  # 关键！
neg_loss = -neg_weight * (pred^α) * log(1 - pred)
```

**参数**：
- `alpha=2`: Focal Loss 聚焦参数
- `beta=4`: 高斯衰减权重

**单元测试**：
```bash
python -m model_Mamba.loss
```

---

### 4. 高斯目标生成工具 ([dataset/gaussian_utils.py](dataset/gaussian_utils.py:1))

**功能**：YOLO 标注 → 高斯热力图转换

**核心函数**：
- `gaussian_2d()`: 生成 2D 高斯核
- `gaussian_radius()`: 自适应高斯半径计算
- `draw_gaussian()`: 在热力图上绘制高斯圆
- `generate_gaussian_target()`: 主转换函数
- `load_yolo_labels()`: 加载 YOLO 格式标注

**使用示例**：
```python
from dataset.gaussian_utils import generate_gaussian_target, load_yolo_labels

labels = load_yolo_labels('dataset/.../labels/000001.txt')
heatmap = generate_gaussian_target(labels, img_size=(512, 640))
# heatmap: (512, 640) NumPy array, 值域 [0, 1]
```

**单元测试**：
```bash
python -m dataset.gaussian_utils
```

---

## 🔧 服务器脚本详解

### 1. 环境配置脚本 ([scripts/setup_server.sh](scripts/setup_server.sh:1))

**用途**：一键配置服务器运行环境

**执行流程**：
```
1. 检查 CUDA 环境 (nvidia-smi)
2. 检查 Python 版本 (>= 3.8)
3. 验证 PyTorch + CUDA
4. 安装基础依赖 (opencv, tqdm, einops, scipy)
5. 安装 mamba_ssm 加速库
6. 运行单元测试验证
```

**使用方法**：
```bash
bash scripts/setup_server.sh
```

**输出**：环境诊断报告 + 测试结果

---

### 2. 训练脚本 ([scripts/train_mamba_server.sh](scripts/train_mamba_server.sh:1))

**用途**：一键启动训练（带参数配置）

**可配置参数**：
```bash
# 在脚本开头修改
DATASET="Pohang-Canal-3k"
MODEL="mamba_small"
GPUS="0,1"              # 多GPU配置
BATCH_SIZE=8
EPOCHS=200
USE_LIDAR="True"
EXPERIMENT_NAME="baseline_server"
```

**使用方法**：
```bash
# 1. 编辑配置
vim scripts/train_mamba_server.sh

# 2. 运行训练
bash scripts/train_mamba_server.sh

# 3. 后台运行（tmux）
tmux new -s mamba_train
bash scripts/train_mamba_server.sh
# Ctrl+B -> D 分离
```

**输出**：训练日志 + 模型检查点

---

### 3. 可视化脚本 ([scripts/visualize_mamba.sh](scripts/visualize_mamba.sh:1))

**用途**：一键生成预测结果的可视化图像

**可配置参数**：
```bash
MODEL="mamba_small"
EXPERIMENT_NAME="baseline_server"
NUM_SAMPLES=20          # 可视化样本数
PEAK_THRESHOLD=0.5      # 峰值检测阈值
```

**使用方法**：
```bash
# 1. 编辑配置（可选）
vim scripts/visualize_mamba.sh

# 2. 运行可视化
bash scripts/visualize_mamba.sh

# 3. 下载结果到本地
scp -r user@server:/path/to/visualizations/ ./
```

**输出格式**：
```
每个样本生成一张 4列拼接图:
┌──────────┬──────────┬──────────┬──────────┐
│ IR Image │ LiDAR    │ Pred+    │ GT       │
│          │ Depth    │ Peaks    │ Heatmap  │
└──────────┴──────────┴──────────┴──────────┘
```

---

## 📖 文档层级

### 快速上手（5分钟）

👉 [MAMBA_QUICKSTART.md](MAMBA_QUICKSTART.md:1)

**内容**：
- ✅ 一键部署脚本
- ✅ 3步快速开始
- ✅ 常见问题快速解决

**适合人群**：新手、想快速验证的用户

---

### 服务器部署（20分钟）

👉 [SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md:1)

**内容**：
- ✅ 硬件/软件要求
- ✅ 详细部署步骤
- ✅ 多GPU训练配置
- ✅ 完整的问题排查指南
- ✅ 性能基准数据

**适合人群**：需要在服务器上正式部署的用户

---

### 技术深度文档（60分钟）

👉 [model_Mamba/README.md](model_Mamba/README.md:1)

**内容**：
- ✅ 架构设计详解
- ✅ LiDAR 门控机制原理
- ✅ 高斯热力图技术
- ✅ 与现有模型对比
- ✅ 高级用法和调优

**适合人群**：想深入理解原理、需要定制改进的研究人员

---

## 🎯 核心创新点总结

### 1. LiDAR 门控注入 (Gated Injection)

**传统方法**：通道拼接 `concat([IR, LiDAR])`

**新方法**：门控调制
```python
gate = sigmoid(conv(lidar))  # 0~1
ir_enhanced = ir * (1 + gate)
```

**优势**：
- ✅ 远海无LiDAR时自动降级
- ✅ 近海LiDAR引导特征扫描
- ✅ 门控值反映LiDAR置信度

---

### 2. 高斯热力图输出

**问题**：矩形框标注不准确（~3像素误差）

**解决**：输出中心点热力图
```python
heatmap[cy, cx] = 1.0           # 峰值
heatmap[cy±r, cx±r] = exp(-d²)  # 高斯衰减
```

**优势**：
- ✅ 对标注噪声鲁棒（允许±5像素误差）
- ✅ 自然处理重叠目标
- ✅ 软监督学习

---

### 3. Vision Mamba 架构

**对比**：

| 特性 | CNN | Transformer | **Mamba** |
|------|-----|------------|----------|
| 感受野 | 局部 | 全局 | **全局** |
| 复杂度 | O(n) | O(n²) | **O(n)** |
| 参数量 | 适中 | 大 | **小** |

**Mamba 优势**：
- ✅ 4方向扫描 = 全局感受野
- ✅ 线性复杂度
- ✅ 长序列建模能力强

---

## 📊 性能预期

### 训练速度（512×640，Pohang-Canal-3k）

| 配置 | 吞吐量 | 单epoch | 200 epochs |
|------|-------|---------|-----------|
| 1x 3090, BS=8, mamba_small | ~8 samples/s | 3.8 min | ~12 小时 |
| 2x 4090, BS=16, mamba_small | ~28 samples/s | 1.1 min | ~4 小时 |
| 4x A100, BS=32, mamba_base | ~45 samples/s | 0.7 min | ~2.5 小时 |

### 检测性能（预期）

| 模型 | 参数量 | IoU | Recall | Precision |
|------|--------|-----|--------|-----------|
| DNANet (基准) | 8M | 0.72 | 0.81 | 0.85 |
| MS_CAFNet (Phase3) | 12M | 0.76 | 0.84 | 0.88 |
| **mamba_small (新)** | **15M** | **~0.78** | **~0.86** | **~0.89** |

---

## ✅ 部署完成检查清单

部署完成后，请确认：

- [ ] 所有单元测试通过
- [ ] `mamba_ssm` 成功安装（或看到 fallback 警告）
- [ ] 数据集包含 `images/` 和 `labels/` 文件夹
- [ ] 训练脚本成功启动
- [ ] 多GPU训练时，所有GPU显存均增加
- [ ] 训练日志 `train_log.csv` 正常生成
- [ ] 可视化脚本生成正确的拼接图

---

## 🚀 下一步

### 新手：
1. ✅ 运行 `bash scripts/setup_server.sh`
2. ✅ 准备数据集（确保有 `labels/`）
3. ✅ 运行 `bash scripts/train_mamba_server.sh`
4. ✅ 等待训练完成（200 epochs）
5. ✅ 运行 `bash scripts/visualize_mamba.sh`

### 进阶：
1. 🔬 对比 `use_lidar=True/False` 性能差异
2. 🎛️ 调整 `gaussian_iou` 适配数据集
3. 📊 与 DNANet/MS_CAFNet 进行消融研究

### 专家：
1. 🧪 修改门控函数（尝试其他融合方式）
2. 🏗️ 添加多尺度输出（FPN风格）
3. 🚀 实现 DistributedDataParallel（替换 DataParallel）

---

**完整文档索引**：

- 📄 [MAMBA_QUICKSTART.md](MAMBA_QUICKSTART.md:1) - 5分钟快速开始
- 📄 [SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md:1) - 服务器部署指南
- 📄 [model_Mamba/README.md](model_Mamba/README.md:1) - 技术详解
- 📄 [MAMBA_FILES_SUMMARY.md](MAMBA_FILES_SUMMARY.md:1) - 本文档

祝实验顺利！🎉
