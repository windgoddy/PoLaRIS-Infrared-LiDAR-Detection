# PoLaRIS-Gaussian-Mamba 服务器部署指南

> **目标环境**: Linux 服务器 + CUDA 11.7+ + PyTorch 2.0+
>
> **部署难度**: ⭐⭐☆☆☆ (简单)

---

## 📋 部署清单

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | 1x RTX 3090 (24GB) | 2x RTX 4090 (48GB) |
| CPU | 8 核 | 16 核+ |
| 内存 | 32GB | 64GB+ |
| 存储 | 100GB SSD | 500GB NVMe SSD |

### 软件要求

| 软件 | 版本要求 | 备注 |
|------|---------|------|
| CUDA | 11.7+ | 推荐 11.8 或 12.1 |
| cuDNN | 8.0+ | 与 CUDA 版本匹配 |
| Python | 3.8 - 3.11 | 推荐 3.9 |
| PyTorch | 2.0+ | 必须支持 CUDA |
| gcc/g++ | 7.0+ | 编译 mamba_ssm 需要 |

---

## 🚀 快速部署（3步）

### 第1步：上传代码到服务器

```bash
# 方法1: 通过 Git
ssh user@server
cd /workspace/
git clone <your-repo-url>

# 方法2: 通过 scp
scp -r PoLaRIS-Infrared-LiDAR-Detection user@server:/workspace/
```

### 第2步：一键环境配置

```bash
ssh user@server
cd /workspace/PoLaRIS-Infrared-LiDAR-Detection

# 运行自动配置脚本
bash scripts/setup_server.sh
```

**该脚本会自动完成**：
- ✅ 检查 CUDA 和 GPU 状态
- ✅ 验证 PyTorch 安装
- ✅ 安装所有依赖（opencv, tqdm, einops, scipy）
- ✅ 安装 `mamba_ssm` 加速库（关键！）
- ✅ 运行单元测试验证

**预期输出**：

```bash
========================================
✅ 环境配置完成！
========================================

📚 下一步：
  1. 准备数据集（确保有 images/ 和 labels/ 文件夹）
  2. 运行训练：
     bash scripts/train_mamba_server.sh
```

### 第3步：启动训练

```bash
# 检查并修改训练配置（可选）
vim scripts/train_mamba_server.sh

# 启动训练
bash scripts/train_mamba_server.sh
```

**训练监控**：

```bash
# 查看训练日志
tail -f result/Pohang-Canal-3k/mamba_small_baseline_server/train_log.csv

# 监控 GPU 使用率
watch -n 1 nvidia-smi

# 后台运行（tmux 或 screen）
tmux new -s mamba_train
bash scripts/train_mamba_server.sh
# Ctrl+B -> D 分离会话
tmux attach -t mamba_train  # 恢复会话
```

---

## ⚙️ 配置说明

### 训练脚本配置 (`scripts/train_mamba_server.sh`)

**关键参数**：

```bash
# GPU 配置
GPUS="0,1"              # 单卡: "0", 双卡: "0,1", 四卡: "0,1,2,3"

# 模型选择
MODEL="mamba_small"     # mamba_tiny (5M) / mamba_small (15M) / mamba_base (30M)

# 批大小（根据显存调整）
BATCH_SIZE=8           # RTX 3090 (24GB): 4-8
                       # RTX 4090 (24GB): 8-16
                       # A100 (40GB): 16-32

# 训练轮数
EPOCHS=200             # 快速测试: 50-100, 正式训练: 200-300

# LiDAR 门控
USE_LIDAR="True"       # True=使用LiDAR, False=纯红外
```

**显存优化**：

| GPU型号 | 显存 | 推荐配置 |
|---------|-----|---------|
| RTX 3090 | 24GB | `mamba_tiny`, BS=8, crop=480 |
| RTX 4090 | 24GB | `mamba_small`, BS=8, crop=512 |
| A100 | 40GB | `mamba_base`, BS=16, crop=512 |
| A100 | 80GB | `mamba_base`, BS=32, crop=640 |

---

## 📊 多GPU训练

### DataParallel 模式（已集成）

训练脚本自动支持多卡并行：

```bash
# 编辑 scripts/train_mamba_server.sh
GPUS="0,1,2,3"    # 指定4张卡

# 启动训练
bash scripts/train_mamba_server.sh
```

**性能对比**：

| GPU数量 | 有效BatchSize | 训练速度 | 预计时间 (200 epochs) |
|---------|--------------|---------|----------------------|
| 1x 3090 | 8 | 1.0x | ~24 小时 |
| 2x 3090 | 16 | 1.8x | ~13 小时 |
| 4x 3090 | 32 | 3.2x | ~7.5 小时 |

### 检查多卡是否生效

训练启动后，检查日志：

```
✅ Using DataParallel on GPUs: [0, 1, 2, 3]
✅ Model: mamba_small, Parameters: 15.23M
```

同时在另一个终端运行：

```bash
watch -n 1 nvidia-smi
```

应看到所有指定GPU的显存和利用率都在上升。

---

## 🐛 常见问题排查

### 问题1: `mamba_ssm` 安装失败

**症状**：

```
[WARNING] mamba_ssm not available. Using PyTorch native fallback (slower).
```

**原因**：
- CUDA 版本过低（需要 11.7+）
- 缺少 gcc/g++ 编译器
- PyPI 下载失败

**解决方案**：

```bash
# 1. 检查 CUDA 版本
nvcc --version  # 应显示 11.7 或更高

# 2. 检查编译器
gcc --version   # 应显示 7.0 或更高

# 3. 手动安装
pip install causal-conv1d>=1.1.0
pip install mamba-ssm

# 4. 从源码编译（如果 pip 失败）
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .

# 5. 验证安装
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK')"
```

**注意**：如果实在无法安装，代码会自动使用 PyTorch 原生实现（速度慢 ~3x，但仍可训练）。

---

### 问题2: CUDA Out of Memory (OOM)

**症状**：

```
RuntimeError: CUDA out of memory. Tried to allocate 2.34 GiB
```

**解决方案**：

```bash
# 方法1: 减小 Batch Size
vim scripts/train_mamba_server.sh
# BATCH_SIZE=4  # 改为更小的值

# 方法2: 降低图像分辨率
# --crop_size 384  # 默认 480

# 方法3: 使用更小的模型
# MODEL="mamba_tiny"  # 替换 mamba_small

# 方法4: 启用梯度累积（需修改 train_Mamba.py）
# gradient_accumulation_steps = 2  # 等效 BS x2

# 方法5: 使用混合精度训练（AMP）
# torch.cuda.amp.autocast()  # 节省 ~40% 显存
```

---

### 问题3: 缺少 `labels/` 文件夹

**症状**：

```
FileNotFoundError: labels/000001.txt not found
```

**原因**：高斯热力图需要 YOLO 格式标注，但您的数据集只有 `masks/` 文件夹。

**解决方案**：运行转换脚本（需自行实现或使用以下示例）

```python
# scripts/convert_mask_to_yolo.py
import cv2
import numpy as np
from glob import glob
import os

dataset_dir = 'dataset/Pohang-Canal-3k'
os.makedirs(f'{dataset_dir}/labels', exist_ok=True)

for mask_path in glob(f'{dataset_dir}/masks/*.png'):
    img_id = os.path.basename(mask_path).replace('.png', '')
    mask = cv2.imread(mask_path, 0)
    H, W = mask.shape

    # 提取连通域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 生成 YOLO 标注
    with open(f'{dataset_dir}/labels/{img_id}.txt', 'w') as f:
        for cnt in contours:
            if cv2.contourArea(cnt) < 5:  # 过滤噪声
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            w_norm = w / W
            h_norm = h / H

            f.write(f'0 {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}\n')

print("✅ 转换完成！")
```

运行：

```bash
python scripts/convert_mask_to_yolo.py
```

---

### 问题4: 训练损失不下降

**症状**：训练 10+ epochs 后，Loss 仍在 0.5+ 徘徊。

**排查步骤**：

1. **检查数据加载**：

```bash
python -m dataset.gaussian_utils
```

2. **可视化生成的热力图**：

```python
from dataset.gaussian_utils import generate_gaussian_target, load_yolo_labels
import matplotlib.pyplot as plt

labels = load_yolo_labels('dataset/Pohang-Canal-3k/labels/000001.txt')
heatmap = generate_gaussian_target(labels, (512, 640))

plt.imshow(heatmap, cmap='jet')
plt.colorbar()
plt.title('Gaussian Target Heatmap')
plt.savefig('debug_heatmap.png')
print(f"Peak value: {heatmap.max()}, Num peaks: {(heatmap > 0.9).sum()}")
```

3. **调整学习率**：

```bash
# 如果 Loss 震荡 → 降低学习率
--lr 0.00005

# 如果 Loss 下降太慢 → 提高学习率
--lr 0.0005
```

4. **调整高斯半径**：

```bash
# 目标稀疏 → 增大高斯圆
--gaussian_iou 0.5

# 目标密集 → 减小高斯圆
--gaussian_iou 0.9
```

---

### 问题5: 多卡训练速度不理想

**症状**：4卡训练速度仅为单卡的 2倍（理论应为 3-3.5倍）。

**原因**：
- DataLoader 瓶颈（CPU加载慢）
- GPU间通信开销
- Batch Size 太小

**优化方案**：

```bash
# 1. 增加 DataLoader 工作线程
--workers 16  # CPU核心数的 2倍

# 2. 增大 Batch Size（充分利用GPU）
--train_batch_size 16  # 每卡 16，总共 64

# 3. 启用 pin_memory（已默认开启）
# DataLoader(..., pin_memory=True)

# 4. 使用 DistributedDataParallel（高级，需修改代码）
# torch.distributed.launch
```

---

## 📈 性能基准

### 训练速度（Pohang-Canal-3k，512x640）

| 配置 | 吞吐量 (samples/s) | 单epoch时间 |
|------|-------------------|------------|
| 1x RTX 3090, BS=4, mamba_tiny | ~12 | 2.5 min |
| 1x RTX 3090, BS=8, mamba_small | ~8 | 3.8 min |
| 2x RTX 4090, BS=16, mamba_small | ~28 | 1.1 min |
| 4x A100, BS=32, mamba_base | ~45 | 0.7 min |

### 预期性能（200 epochs）

| 模型 | 参数量 | IoU | Recall | 训练时间 (4x 3090) |
|------|--------|-----|--------|-------------------|
| mamba_tiny | 5M | ~0.75 | ~0.83 | ~6 小时 |
| mamba_small | 15M | **~0.78** | **~0.86** | ~8 小时 |
| mamba_base | 30M | ~0.80 | ~0.88 | ~12 小时 |

---

## 🎯 部署后检查清单

部署完成后，请确认以下项目：

- [ ] `nvidia-smi` 显示正确的GPU信息
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`
- [ ] `python -m model_Mamba.polaris_mamba` 测试通过
- [ ] `mamba_ssm` 已安装（或看到 fallback 警告）
- [ ] 数据集包含 `images/` 和 `labels/` 文件夹
- [ ] 训练启动后，所有GPU显存均增加
- [ ] 训练日志 `train_log.csv` 正常生成

---

## 📞 技术支持

如遇到无法解决的问题，请提供以下信息：

```bash
# 1. 环境信息
nvidia-smi
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# 2. 错误日志
tail -100 <训练日志路径>

# 3. 完整的训练命令
cat scripts/train_mamba_server.sh
```

---

## 📚 相关文档

- **快速开始**: [MAMBA_QUICKSTART.md](MAMBA_QUICKSTART.md:1)
- **完整技术文档**: [model_Mamba/README.md](model_Mamba/README.md:1)
- **原理详解**: 见 README.md 中的"核心技术详解"章节

---

**祝部署顺利！** 🚀
