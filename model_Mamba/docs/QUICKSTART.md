# PoLaRIS-Gaussian-Mamba 快速启动指南（服务器版）

> **新架构！基于 Vision Mamba + LiDAR 门控 + 高斯热力图**
>
> **目标环境**: Linux 服务器 + CUDA 11.7+ + PyTorch 2.0+

---

## 🚀 一键部署（推荐）

### 方法1：使用自动化脚本

```bash
# 步骤1: 上传代码到服务器
scp -r PoLaRIS-Infrared-LiDAR-Detection user@server:/path/to/workspace/
ssh user@server

# 步骤2: 进入项目目录
cd /path/to/workspace/PoLaRIS-Infrared-LiDAR-Detection

# 步骤3: 运行一键配置脚本
bash scripts/setup_server.sh
```

**该脚本会自动**：
- ✅ 检查 CUDA 环境
- ✅ 验证 PyTorch + CUDA 安装
- ✅ 安装所有必需依赖
- ✅ 安装 `mamba_ssm` 加速库
- ✅ 运行单元测试验证

**预期输出**：
```
========================================
✅ 环境配置完成！
========================================

📚 下一步：
  1. 准备数据集（确保有 images/ 和 labels/ 文件夹）
  2. 运行训练：
     bash scripts/train_mamba_server.sh
```

---

### 方法2：手动配置

如果自动脚本失败，可以手动执行以下步骤：

```bash
# 1. 创建虚拟环境
conda create -n polaris_mamba python=3.9
conda activate polaris_mamba

# 2. 安装 PyTorch (CUDA 11.8 示例)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装基础依赖
pip install opencv-python pillow tqdm einops scipy

# 4. 安装 mamba_ssm（关键！）
pip install causal-conv1d>=1.1.0
pip install mamba-ssm

# 5. 验证安装
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('✅ mamba_ssm OK')"
```

---

## 🎯 5分钟快速开始

### 1. 验证安装（单元测试）

```bash
cd /path/to/PoLaRIS-Infrared-LiDAR-Detection

# 测试核心组件
python -m model_Mamba.ss2d_components
python -m model_Mamba.polaris_mamba
python -m model_Mamba.loss
python -m dataset.gaussian_utils
```

**预期输出**: `All tests passed! ✓`

**如果看到警告**:
```
[WARNING] mamba_ssm not available. Using PyTorch native fallback (slower).
```
→ 说明 `mamba_ssm` 未成功安装，训练速度会慢 ~3倍（但仍可运行）

---

### 2. 准备数据

**确保你的数据集包含以下文件夹：**

```
dataset/Pohang-Canal-3k/
├── images/          # 红外图像
├── labels/          # ⭐ YOLO格式标注（必需！）
├── lidar_roi/       # LiDAR点云（可选）
└── 50_50/
    ├── train.txt
    └── test.txt
```

**如果没有 `labels/` 文件夹**：

- 选项1：从 `masks/` 转换（需要编写转换脚本）
- 选项2：使用 LabelImg 手动标注 YOLO 格式

**YOLO 格式示例** (`labels/000001.txt`):
```
0 0.512 0.487 0.023 0.031
0 0.678 0.234 0.019 0.028
```
格式：`class_id center_x center_y width height`（归一化坐标）

---

### 3. 训练模型

#### 方法1：使用一键训练脚本（推荐）

```bash
# 编辑配置参数（可选）
vim scripts/train_mamba_server.sh

# 启动训练
bash scripts/train_mamba_server.sh
```

**脚本默认配置**：
- 模型：`mamba_small`（~15M 参数）
- GPU：`0,1`（双卡训练）
- Batch Size：8
- Epochs：200
- LiDAR门控：开启

**多GPU训练**：

编辑 `scripts/train_mamba_server.sh` 中的 `GPUS` 参数：

```bash
GPUS="0,1,2,3"    # 4卡训练
GPUS="0"          # 单卡训练
```

---

#### 方法2：手动命令行训练

**最简单的命令（单卡）：**

```bash
python train_Mamba.py \
    --model mamba_tiny \
    --dataset Pohang-Canal-3k \
    --use_lidar False \
    --epochs 100 \
    --gpus 0
```

**推荐配置（多卡 + LiDAR）：**

```bash
python train_Mamba.py \
    --model mamba_small \
    --dataset Pohang-Canal-3k \
    --use_lidar True \
    --in_channels 1 \
    --epochs 200 \
    --train_batch_size 8 \
    --lr 0.0001 \
    --optimizer AdamW \
    --workers 8 \
    --experiment_name baseline1 \
    --gpus "0,1"          # 多卡用逗号分隔
```

**训练输出**：

```
result/Pohang-Canal-3k/mamba_small_baseline1/
├── best_model.pth          # 最佳模型
├── checkpoint_epoch20.pth  # 周期性检查点
└── train_log.csv           # 训练日志
```

---

### 4. 可视化结果

```bash
python vis_mamba.py \
    --checkpoint result/Pohang-Canal-3k/mamba_small_baseline1/best_model.pth \
    --model mamba_small \
    --dataset Pohang-Canal-3k \
    --num_samples 20 \
    --output_dir visualizations/baseline1
```

生成的可视化包含：
- 原始红外图像
- LiDAR 深度图
- **预测热力图 + 检测峰值**
- Ground Truth 热力图

---

## 🔧 常见问题快速解决

### Q1: `mamba_ssm` 未安装

**A**: 不用担心！代码会自动使用 PyTorch 原生实现（稍慢但可用）。

```
[WARNING] mamba_ssm not available. Using PyTorch native fallback (slower).
```

如果需要加速（仅 Linux + CUDA）：
```bash
pip install mamba-ssm
```

---

### Q2: 缺少 `labels/` 文件夹

**A**: 该架构需要 YOLO 格式标注来生成高斯热力图。如果只有 `masks/`，需要转换：

```python
# 简单转换脚本示例
import cv2
import numpy as np
from glob import glob

for mask_path in glob('dataset/Pohang-Canal-3k/masks/*.png'):
    mask = cv2.imread(mask_path, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_id = mask_path.split('/')[-1].replace('.png', '')
    H, W = mask.shape

    with open(f'dataset/Pohang-Canal-3k/labels/{img_id}.txt', 'w') as f:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = (x + w/2) / W, (y + h/2) / H
            w_norm, h_norm = w / W, h / H
            f.write(f'0 {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}\n')
```

---

### Q3: 内存不足 (OOM)

**A**: 尝试以下方法：

1. 减小 batch size:
   ```bash
   --train_batch_size 2
   ```

2. 降低图像分辨率:
   ```bash
   --crop_size 256
   ```

3. 使用更小的模型:
   ```bash
   --model mamba_tiny
   ```

4. 热力图下采样:
   ```bash
   --heatmap_downscale 4  # 输出 H/4 × W/4 的热力图
   ```

---

### Q4: 训练损失不下降

**A**: 检查以下几点：

1. **可视化高斯目标**：
   ```bash
   python -m dataset.gaussian_utils
   # 查看生成的热力图是否合理
   ```

2. **调整学习率**：
   ```bash
   --lr 0.00001  # 降低学习率
   ```

3. **调整高斯半径**：
   ```bash
   --gaussian_iou 0.5  # 增大高斯圆（目标稀疏时）
   --gaussian_iou 0.9  # 减小高斯圆（目标密集时）
   ```

4. **检查数据标注质量**：
   ```python
   # 可视化几个样本的标注
   import matplotlib.pyplot as plt
   from dataset.gaussian_utils import generate_gaussian_target, load_yolo_labels

   labels = load_yolo_labels('dataset/Pohang-Canal-3k/labels/000001.txt')
   heatmap = generate_gaussian_target(labels, (512, 640))
   plt.imshow(heatmap, cmap='jet')
   plt.show()
   ```

---

## 📊 与现有模型的对比

| 方面 | 旧模型 (DNANet/MS_CAFNet) | 新模型 (PoLaRIS_Mamba) |
|------|-------------------------|----------------------|
| **训练脚本** | `train_Phase3.py` | `train_Mamba.py` |
| **模型定义** | `model/model_Phase3.py` | `model_Mamba/polaris_mamba.py` |
| **数据输入** | 红外 + 深度图（通道拼接） | 红外 + LiDAR门控 |
| **输出形式** | 二值掩码 (0/1) | 高斯热力图 (0~1 连续值) |
| **损失函数** | SoftIoU + BCE | Gaussian Focal Loss |
| **评价指标** | IoU, Precision, Recall | IoU, Precision, Recall (热力图) |
| **适用场景** | 精确边界分割 | 中心点检测（对标注噪声鲁棒） |

**关键优势**：
- ✅ **对标注噪声鲁棒**：高斯热力图允许位置误差
- ✅ **LiDAR 稀疏性友好**：门控机制自动降级
- ✅ **长序列建模**：Mamba 架构天然适合扫描式检测
- ✅ **独立代码库**：不影响现有模型

---

## 🚀 进阶用法

### 1. 使用深度图替代点云

```bash
python train_Mamba.py \
    --model mamba_small \
    --in_channels 2 \
    --use_lidar True
```

数据加载器会自动从 `depth_maps/*.npy` 加载深度图。

---

### 2. 纯红外训练（无 LiDAR）

```bash
python train_Mamba.py \
    --model mamba_small \
    --use_lidar False \
    --in_channels 1
```

模型退化为纯 Vision Mamba，不使用门控机制。

---

### 3. 恢复训练（断点续训）

```python
# 修改 train_Mamba.py 的 Trainer.__init__()
checkpoint = torch.load('result/.../checkpoint_epoch50.pth')
self.net.load_state_dict(checkpoint['model_state_dict'])
self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
args.start_epoch = checkpoint['epoch'] + 1
```

---

### 4. 导出模型为 ONNX（部署）

```python
import torch
from model_Mamba.polaris_mamba import polaris_mamba_small

# 加载模型
model = polaris_mamba_small(use_lidar=True)
checkpoint = torch.load('result/.../best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 导出 ONNX
dummy_ir = torch.randn(1, 1, 512, 640)
dummy_lidar = torch.randn(1, 1, 512, 640)
torch.onnx.export(
    model,
    (dummy_ir, dummy_lidar),
    'polaris_mamba.onnx',
    input_names=['ir_image', 'lidar_depth'],
    output_names=['heatmap'],
    dynamic_axes={'ir_image': {0: 'batch'}, 'lidar_depth': {0: 'batch'}, 'heatmap': {0: 'batch'}},
)
```

---

## 📖 详细文档

完整的技术细节、架构设计和调试指南，请查看：

- [model_Mamba/README.md](model_Mamba/README.md) - 完整文档

---

## 🎓 学习路径

**新手**：
1. 运行单元测试验证环境 ✅
2. 用 `mamba_tiny` 训练 50 epochs（快速验证）
3. 可视化结果，理解热力图输出

**进阶**：
1. 阅读 [model_Mamba/README.md](model_Mamba/README.md:1) 了解门控机制
2. 调整 `gaussian_iou` 参数适配数据集
3. 对比 `use_lidar=True/False` 的性能差异

**专家**：
1. 修改 `ss2d_components.py` 的门控函数（尝试其他融合方式）
2. 在 `polaris_mamba.py` 中添加多尺度输出
3. 实现 Transformer + Mamba 混合架构

---

## 💡 快速实验建议

**实验1：验证门控机制的效果**
```bash
# 不使用 LiDAR
python train_Mamba.py --use_lidar False --experiment_name no_lidar

# 使用 LiDAR 门控
python train_Mamba.py --use_lidar True --experiment_name with_lidar
```

对比 `train_log.csv` 中的 IoU 和 Recall。

---

**实验2：测试模型鲁棒性**
```bash
# 小高斯圆（严格监督）
python train_Mamba.py --gaussian_iou 0.9 --experiment_name strict

# 大高斯圆（宽松监督）
python train_Mamba.py --gaussian_iou 0.5 --experiment_name loose
```

查看对标注噪声的鲁棒性。

---

**实验3：消融研究（Ablation Study）**
```bash
# 基线：mamba_tiny + no lidar
python train_Mamba.py --model mamba_tiny --use_lidar False

# +LiDAR 门控
python train_Mamba.py --model mamba_tiny --use_lidar True

# +更大模型
python train_Mamba.py --model mamba_small --use_lidar True

# +深度图
python train_Mamba.py --model mamba_small --in_channels 2 --use_lidar True
```

---

## 🔗 相关资源

- **原始项目**: PoLaRIS 红外小目标检测
- **参考实现**: [lidar-mamba](https://github.com/...) (Vision Mamba)
- **论文**: CenterNet, Focal Loss, Mamba

---

**祝实验顺利！如有问题，请查阅 [model_Mamba/README.md](model_Mamba/README.md:1) 或提交 Issue。** 🚀
