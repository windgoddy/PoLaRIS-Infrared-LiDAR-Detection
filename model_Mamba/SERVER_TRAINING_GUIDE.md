# 🚀 服务器端训练快速指南

## 📋 Step 1: 准备 Labels 文件夹

由于您的数据集没有 `labels/` 目录，需要先从原始YOLO标注复制过来。

### 方法1: 使用 Python 脚本（推荐）

```bash
# 进入项目目录
cd ~/code/PoLaRIS-Infrared-LiDAR-Detection

# 运行准备脚本
python model_Mamba/scripts/prepare_labels_simple.py
```

**重要**：如果脚本提示找不到原始标注路径，请编辑脚本修改：

```bash
nano model_Mamba/scripts/prepare_labels_simple.py

# 修改第13行的 ORIGINAL_LABELS_BASE 为正确路径
ORIGINAL_LABELS_BASE = "/home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/PoLaRIS/PoLaRIS"
```

### 方法2: 手动创建链接（快速但不推荐）

如果标注文件已经在正确位置，可以创建符号链接：

```bash
ln -s /path/to/original/labels dataset/Pohang-Canal-3k/labels
```

---

## ⚙️ Step 2: 检查环境（可选）

```bash
# 修复脚本格式（在服务器上运行）
dos2unix model_Mamba/scripts/setup_server.sh

# 或者用 sed
sed -i 's/\r$//' model_Mamba/scripts/setup_server.sh

# 运行环境检查
bash model_Mamba/scripts/setup_server.sh
```

---

## 🏃 Step 3: 开始训练

### 快速启动（使用默认配置）

```bash
# 一键启动训练
bash model_Mamba/scripts/quick_train.sh
```

**默认配置**：
- 数据集: `Pohang-Canal-3k`
- 划分: `50_50_2k_new`
- 模型: `mamba_tiny`
- Batch Size: 8
- GPU: 7（根据您的GPU分析）
- Epochs: 1000

### 自定义训练参数

```bash
# 直接运行 Python 脚本，自定义参数
CUDA_VISIBLE_DEVICES=7 python model_Mamba/train.py \
    --dataset Pohang-Canal-3k \
    --split_method 50_50_2k_new \
    --model mamba_small \
    --batch_size 16 \
    --epochs 1500 \
    --lr 0.0001 \
    --use_lidar True \
    --save_dir result/Pohang-Canal-3k/mamba_small \
    --log_interval 10 \
    --save_interval 50
```

### 可用的模型选项

- `mamba_tiny`: 最小模型，训练最快
- `mamba_small`: 中等模型
- `mamba_base`: 最大模型，精度最高

---

## 📊 Step 4: 监控训练

### 实时查看 GPU 使用情况

```bash
# 使用我们创建的监控脚本
python monitor_gpu.py

# 或者使用 nvidia-smi
watch -n 1 nvidia-smi
```

### 查看训练日志

```bash
# 实时查看日志
tail -f result/Pohang-Canal-3k/mamba_tiny/train.log

# 或者使用 tensorboard（如果有的话）
tensorboard --logdir result/Pohang-Canal-3k/mamba_tiny
```

---

## 🛠️ 常见问题排查

### 问题1: 找不到 labels 目录

**原因**: 没有运行 Step 1

**解决**:
```bash
python model_Mamba/scripts/prepare_labels_simple.py
```

### 问题2: 脚本格式错误 (`\r` 命令未找到)

**原因**: Windows 行尾符问题

**解决**:
```bash
# 方法1: 使用 dos2unix
dos2unix model_Mamba/scripts/*.sh

# 方法2: 使用 sed
find model_Mamba/scripts -name "*.sh" -exec sed -i 's/\r$//' {} \;
```

### 问题3: GPU 内存不足

**原因**: Batch size 太大

**解决**:
```bash
# 减小 batch_size
CUDA_VISIBLE_DEVICES=7 python model_Mamba/train.py \
    --batch_size 4 \
    ... (其他参数)
```

### 问题4: 数据加载失败

**原因**: 图像路径不对

**检查**:
```bash
# 检查图像文件
ls dataset/Pohang-Canal-3k/images/ | head -5

# 应该看到类似: 00_5596.png, 00_5749.png 等
```

---

## 📈 训练进度查看

训练过程中会显示：

```
Epoch 1/1000:
  Batch [10/200] - Loss: 0.4523 - LR: 0.0001
  Batch [20/200] - Loss: 0.4312 - LR: 0.0001
  ...
  
Validation:
  IoU: 0.7234
  Precision: 0.8123
  Recall: 0.7456
  
✅ Best model saved! (IoU: 0.7234)
```

---

## 🎯 完整流程总结

```bash
# 1. SSH 登录服务器
ssh 25-zhangxizhe@59.67.149.90 -p 3001

# 2. 进入项目目录
cd ~/code/PoLaRIS-Infrared-LiDAR-Detection

# 3. 准备 labels
python model_Mamba/scripts/prepare_labels_simple.py

# 4. 检查 labels 已创建
ls dataset/Pohang-Canal-3k/labels/ | wc -l

# 5. 开始训练
bash model_Mamba/scripts/quick_train.sh

# 6. 另开一个终端监控
ssh 25-zhangxizhe@59.67.149.90 -p 3001
cd ~/code/PoLaRIS-Infrared-LiDAR-Detection
python monitor_gpu.py
```

---

## 💡 推荐配置

根据您的 GPU 分析，推荐使用：

**GPU 7** (9436 MiB 空闲) - 最佳选择 ⭐⭐⭐

训练命令：
```bash
CUDA_VISIBLE_DEVICES=7 bash model_Mamba/scripts/quick_train.sh
```

或者如果需要多GPU训练：
```bash
CUDA_VISIBLE_DEVICES=5,7 python model_Mamba/train.py \
    --dataset Pohang-Canal-3k \
    --split_method 50_50_2k_new \
    --model mamba_tiny \
    --batch_size 16 \
    --gpus 0,1  # 这里的0,1对应CUDA_VISIBLE_DEVICES中的5,7
```

---

## 🔄 后台运行训练

如果需要断开 SSH 后继续训练：

```bash
# 使用 nohup
nohup bash model_Mamba/scripts/quick_train.sh > train.log 2>&1 &

# 或使用 screen
screen -S mamba_train
bash model_Mamba/scripts/quick_train.sh
# 按 Ctrl+A, 然后按 D 来断开

# 重新连接
screen -r mamba_train

# 或使用 tmux
tmux new -s mamba_train
bash model_Mamba/scripts/quick_train.sh
# 按 Ctrl+B, 然后按 D 来断开

# 重新连接
tmux attach -t mamba_train
```

---

需要帮助? 请查看详细文档:
- [完整文档](docs/SERVER_DEPLOYMENT.md)
- [快速开始](docs/QUICKSTART.md)
