# PoLaRIS 高级使用指南

> 详细的 DataLoader 和训练技术文档

---

## DataLoader 核心特性

### 1. 16-bit 红外图像处理

**文件位置**: `model/utils_lidar.py`

```python
# 自动检测并处理 16-bit 图像
img, is_16bit = self._load_image(img_path)

# Min-Max 归一化（推荐）
img_normalized = (img - img.min()) / (img.max() - img.min()) * 255.0
```

**关键点**：

- 16-bit 图像范围：0-65535
- 归一化到 [0, 255]：Min-Max 方法保留动态范围
- 最终归一化到 [0.0, 1.0]：`img / 255.0`
- 网络输入：`(B, 1, H, W)` 或 `(B, 2, H, W)`

**为什么使用 Min-Max 归一化**：

```python
# 方法 1: Min-Max（推荐）
normalized = (img - img.min()) / (img.max() - img.min()) * 255
# ✅ 保留暗区细节，动态范围最大化

# 方法 2: 简单缩放
normalized = img / 65535 * 255
# ❌ 可能丢失暗区细节
```

---

### 2. 软标签（Oracle Masks）

**概念**：使用浮点值表示不同置信度的标签

```python
# Oracle Mask 值
oracle_mask = Image.open(path).convert('L')
oracle_mask = oracle_mask / 255.0  # 保留 float 值

# 标签值含义：
# 1.0 - 有 LiDAR 验证的目标（强监督）
# 0.6 - 无 LiDAR 但视觉确认的目标（弱监督）
# 0.0 - 背景
```

**关键点**：

- ⚠️ **禁止二值化**：不要使用 `mask > 0.5`
- 使用软标签兼容的损失函数：`SoftIoULoss`, `SoftBCELoss`, `CombinedSoftLoss`
- 训练时使用 `oracle_masks`，测试时使用 `masks`（GT）

---

### 3. LiDAR 点云处理

**数据格式**：

```python
# .bin 文件格式：float32 二进制
lidar_points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
# 形状：(N, 4) - [x, y, z, intensity]
```

**深度图生成**：

```python
# LiDAR 点云 → 深度图（预处理）
python scripts/generate_depth_maps.py --dataset Pohang-Canal-3k

# 深度图保存为 .npy
depth_map = np.load('dataset/Pohang-Canal-3k/depth_maps/000001.npy')
# 形状：(H, W) - 深度值（米）
```

**双通道输入**：

```python
if in_channels == 2:
    # IR + Depth
    ir = img / 255.0          # 归一化红外
    depth = depth / 80.0      # 归一化深度（假设最大 80m）
    combined = np.stack([ir, depth], axis=0)  # (2, H, W)
```

---

## 训练配置

### 完整训练示例

```python
from model.utils_lidar import PoLaRISTrainLoader, CombinedSoftLoss
from torch.utils.data import DataLoader

# 1. 创建数据集
train_dataset = PoLaRISTrainLoader(
    dataset_dir='dataset/Pohang-Canal-3k',
    img_id=train_img_ids,
    base_size=512,
    crop_size=480,
    transform=None,
    suffix='.png',
    normalize_16bit=True,  # Min-Max 归一化
    in_channels=2          # IR + Depth
)

# 2. 创建 DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

# 3. 训练循环
for batch in train_loader:
    images = batch['image']          # (B, 2, H, W)
    masks = batch['mask']            # (B, 1, H, W) - GT
    oracle_masks = batch['oracle_mask']  # (B, 1, H, W) - 软标签

    # 前向传播
    outputs = model(images)

    # 使用软标签训练
    loss = criterion(outputs, oracle_masks)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

### 损失函数

```python
from model.utils_lidar import CombinedSoftLoss

# 组合损失函数（支持软标签）
criterion = CombinedSoftLoss(
    iou_weight=0.5,
    bce_weight=0.5,
    pos_weight=2.0  # 正样本权重（可调）
)

# 或者单独使用
from model.utils_lidar import SoftIoULoss, SoftBCELoss

iou_loss = SoftIoULoss(pred, oracle_masks)
bce_loss = SoftBCELoss(pred, oracle_masks)
```

---

## 数据增强

### 同步变换

DataLoader 自动同步以下变换：

```python
# _sync_transform() 自动应用：
# 1. 随机缩放 (base_size)
# 2. 随机裁剪 (crop_size)
# 3. 随机水平翻转
# 4. 随机垂直翻转

# 注意：变换同时应用于：
# - IR 图像
# - GT mask
# - Oracle mask
# - Depth map（如果 in_channels=2）
```

**自定义变换**：

```python
import torchvision.transforms as T

custom_transform = T.Compose([
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomRotation(10),
])

train_dataset = PoLaRISTrainLoader(
    dataset_dir='...',
    transform=custom_transform  # 可选
)
```

---

## 常见问题

### Q1: 16-bit 图像精度损失？

**A**: 当前流程会有精度损失：

```
16-bit (0-65535) → Min-Max → uint8 (0-255) → float32 (0-1)
```

灰度级从 65536 降到 256（损失 99.6%），原因是 PIL 数据增强需要 uint8。

**优化方案**（如需要）：

- 使用 `torchvision.transforms` 代替 PIL（支持 float32）
- 跳过 uint8 转换，直接在 float32 上增强

### Q2: 软标签让 Loss 变小？

**A**: 正常现象！

```python
# 硬标签
loss_hard = BCE(pred, [0, 1, 1, 0])  # Loss 较大

# 软标签
loss_soft = BCE(pred, [0, 0.6, 1.0, 0])  # Loss 较小（0.6 降低）
```

**解决方案**（如收敛慢）：

```python
# 方法 1: 增加正样本权重
criterion = CombinedSoftLoss(pos_weight=3.0)

# 方法 2: 增大 loss 系数
total_loss = loss * 2.0
```

### Q3: in_channels=1 和 in_channels=2 的区别？

**A**:

| `in_channels` | 输入通道 | 需要文件 |
| ------------- | -------- | -------- |
| 1 | 仅红外 | `images/` |
| 2 | 红外 + 深度 | `images/` + `depth_maps/` |

```bash
# 仅红外模式
./scripts/train.sh 16bit-ir --dataset Pohang-Canal-3k

# 双通道模式（推荐）
./scripts/train.sh 16bit --dataset Pohang-Canal-3k
```

### Q4: LiDAR 点云如何使用？

**A**: 当前实现中，LiDAR 点云用于**预生成深度图**：

```bash
# 1. 预处理：点云 → 深度图
python scripts/generate_depth_maps.py --dataset Pohang-Canal-3k

# 2. 训练：加载深度图（不是原始点云）
# DataLoader 返回 (B, 2, H, W) - [IR, Depth]
```

如需动态使用点云，需要修改：

1. 在模型中添加点云投影模块
2. 修改 DataLoader 返回点云
3. 修改模型 `forward()` 接受点云输入

---

## 性能优化

### DataLoader 优化

```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,          # 根据 GPU 内存调整
    shuffle=True,
    num_workers=4,         # 多进程加载
    pin_memory=True,       # 加速 GPU 传输
    prefetch_factor=2,     # 预加载 batch 数量
    persistent_workers=True  # 保持 worker 进程
)
```

### 内存优化

```python
# 如果内存不足：
# 1. 减小 batch_size
# 2. 减小 crop_size
# 3. 减少 num_workers
# 4. 使用 in_channels=1（只用红外）

# 示例
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,      # 从 4 减小到 2
    num_workers=2      # 从 4 减小到 2
)
```

---

## 调试技巧

### 测试 DataLoader

```bash
# 快速测试
python scripts/test_dataloader.py

# 预期输出：
# ✅ image shape: (B, 2, H, W)
# ✅ image range: [0.0, 1.0]
# ✅ oracle_mask unique: tensor([0.0000, 0.6000, 1.0000])
```

### 可视化数据

```python
import matplotlib.pyplot as plt

for batch in train_loader:
    img = batch['image'][0, 0].numpy()  # 第一张图，IR 通道
    mask = batch['mask'][0, 0].numpy()
    oracle = batch['oracle_mask'][0, 0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('IR Image')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('GT Mask')
    axes[2].imshow(oracle, cmap='gray')
    axes[2].set_title(f'Oracle Mask\n(values: {torch.unique(oracle)})')
    plt.show()
    break
```

---

## 参数速查

### PoLaRISTrainLoader 参数

```python
PoLaRISTrainLoader(
    dataset_dir,           # str: 数据集根目录
    img_id,                # List[str]: 图像 ID 列表
    base_size=512,         # int: 缩放基准尺寸
    crop_size=480,         # int: 随机裁剪尺寸
    transform=None,        # Optional[Callable]: 自定义变换
    suffix='.png',         # str: 图像后缀
    normalize_16bit=True,  # bool: 16-bit 归一化方式
    in_channels=1          # int: 1 (仅 IR) 或 2 (IR + Depth)
)
```

### PoLaRISTestLoader 参数

```python
PoLaRISTestLoader(
    dataset_dir,           # str: 数据集根目录
    img_id,                # List[str]: 图像 ID 列表
    base_size=512,         # int: 缩放尺寸
    transform=None,        # Optional[Callable]: 自定义变换
    suffix='.png',         # str: 图像后缀
    normalize_16bit=True,  # bool: 16-bit 归一化方式
    in_channels=1          # int: 1 (仅 IR) 或 2 (IR + Depth)
)
```

---

## 返回值格式

### 训练 DataLoader

```python
batch = {
    'image': Tensor,        # (B, C, H, W) - C=1 或 2
    'mask': Tensor,         # (B, 1, H, W) - GT mask
    'oracle_mask': Tensor,  # (B, 1, H, W) - 软标签
    'lidar': List[Tensor],  # List of (N_i, 4) - 点云
    'img_id': List[str],    # 图像 ID
    'is_16bit': List[bool]  # 是否 16-bit
}
```

### 测试 DataLoader

```python
batch = {
    'image': Tensor,        # (B, C, H, W)
    'mask': Tensor,         # (B, 1, H, W)
    'lidar': List[Tensor],  # List of (N_i, 4)
    'img_id': List[str],
    'is_16bit': List[bool]
}
```

---

## 相关脚本

| 脚本 | 用途 |
| ---- | ---- |
| `generate_depth_maps.py` | LiDAR 点云 → 深度图 |
| `generate_oracle_masks.py` | GT + LiDAR → 软标签 |
| `test_dataloader.py` | 测试 DataLoader |
| `verify_dataset.py` | 验证数据集完整性 |
| `diagnose_dataset.py` | 诊断数据集问题 |

---

**更多详情请参考**: [QUICKSTART.md](../QUICKSTART.md)
