# Enhanced DataLoader for PoLaRIS Dataset

## 文件位置
`model/utils_lidar.py`

## 主要特性

### 1. 16-bit 红外图像支持
```python
# 自动检测并处理16-bit图像
img, is_16bit = self._load_image(img_path)

# 两种归一化策略：
# 方法1（推荐）: Min-Max归一化，保留动态范围
img_array = (img_array - img_min) / (img_max - img_min) * 255.0

# 方法2: 简单缩放（可能丢失暗区细节）
img_array = img_array / 65535.0 * 255.0
```

**关键点：**
- 16-bit图像范围：0-65535
- 归一化到 0.0-1.0：`img / 255.0`（在 `__getitem__` 中）
- 网络输入：单通道 Tensor (1, H, W)

### 2. LiDAR 点云加载
```python
# 自动加载 .bin 文件
lidar_points = self._load_lidar(lidar_path)  # (N, 4) [x, y, z, intensity]

# 如果文件不存在，返回空数组
lidar_tensor = torch.from_numpy(lidar_points).float()  # (N, 4)
```

**关键点：**
- 点云格式：`float32` 二进制文件
- 每个点：`[x, y, z, intensity]`
- 点云在世界坐标系，需要在网络中投影到图像平面

### 3. Soft Label 支持（Oracle Masks）
```python
# 加载 Oracle Mask（保留 float 值）
oracle_mask = Image.open(oracle_path).convert('L')

# 归一化到 [0.0, 1.0]，不做二值化！
oracle_mask = oracle_mask / 255.0  # 保留 0.6, 1.0 等软标签值
```

**关键点：**
- **禁止二值化**：不要使用 `mask > 0.5`
- 保留浮点值：
  - `1.0`：有LiDAR点云的目标（高置信度）
  - `0.6`：无LiDAR点云的目标（中置信度，软标签）
  - `0.0`：背景

## 使用方法

### 训练阶段
```python
from model.utils_lidar import PoLaRISTrainLoader, CombinedSoftLoss
from torch.utils.data import DataLoader

# 1. 创建数据集
train_dataset = PoLaRISTrainLoader(
    dataset_dir='dataset/select',
    img_id=train_img_ids,  # ['00_43', '00_44', ...]
    base_size=512,
    crop_size=480,
    transform=None,  # 可选，自定义transform
    suffix='.png',
    normalize_16bit=True  # 推荐使用Min-Max归一化
)

# 2. 创建DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

# 3. 遍历数据
for batch in train_loader:
    images = batch['image']         # (B, 1, H, W) 或 (B, 3, H, W)
    masks = batch['mask']           # (B, 1, H, W) - GT mask
    oracle_masks = batch['oracle_mask']  # (B, 1, H, W) - 软标签
    lidar_points = batch['lidar']   # List of (N_i, 4) tensors
    img_ids = batch['img_id']       # List of strings
    is_16bit = batch['is_16bit']    # List of bools
    
    # 前向传播
    outputs = model(images, lidar_points)
    
    # 计算Loss（使用 oracle_mask 作为 target）
    loss = criterion(outputs, oracle_masks)
```

### 测试阶段
```python
from model.utils_lidar import PoLaRISTestLoader

test_dataset = PoLaRISTestLoader(
    dataset_dir='dataset/select',
    img_id=test_img_ids,
    base_size=512,
    crop_size=480,
    transform=None,
    suffix='.png',
    normalize_16bit=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4
)

for batch in test_loader:
    images = batch['image']
    masks = batch['mask']  # GT mask for evaluation
    lidar_points = batch['lidar']
    
    with torch.no_grad():
        outputs = model(images, lidar_points)
```

## Loss 函数

### 1. SoftIoULoss（推荐）
```python
from model.utils_lidar import SoftIoULoss

criterion = SoftIoULoss(smooth=1e-6)
loss = criterion(pred, oracle_mask)
```

**特点：**
- 支持软标签（float target）
- 计算 Soft IoU：考虑 0.6 等中间值
- 平滑系数避免除零

### 2. SoftBCELoss
```python
from model.utils_lidar import SoftBCELoss

# 可选：设置正样本权重（如果正负样本不平衡）
criterion = SoftBCELoss(pos_weight=torch.tensor([2.0]))
loss = criterion(pred_logits, oracle_mask)
```

**特点：**
- 使用 `BCEWithLogitsLoss`，自动应用 sigmoid
- 支持 `pos_weight` 调整正样本权重
- 预测值应为 logits（未经sigmoid）

### 3. CombinedSoftLoss（最佳）
```python
from model.utils_lidar import CombinedSoftLoss

criterion = CombinedSoftLoss(
    bce_weight=0.5,
    iou_weight=0.5,
    pos_weight=torch.tensor([2.0])  # 可选
)
loss = criterion(pred_logits, oracle_mask)
```

**特点：**
- 结合 BCE 和 IoU 的优点
- BCE：逐像素优化
- IoU：整体形状优化
- 收敛更快更稳定

## 数据增强

### 训练时自动应用
- **水平翻转**：50% 概率
- **随机缩放**：0.5x - 2.0x
- **随机裁剪**：crop_size × crop_size
- **高斯模糊**：50% 概率（仅图像，不模糊mask）

### 注意事项
1. **LiDAR点云不变换**：点云在世界坐标系，网络会处理投影
2. **Oracle Mask使用BILINEAR插值**：保留梯度信息
3. **GT Mask使用NEAREST插值**：保持二值性

## 训练配置建议

### 1. Batch Size
```python
# 16-bit图像占用更多内存
batch_size = 8  # GPU 24GB
batch_size = 4  # GPU 12GB
```

### 2. Learning Rate
```python
# 由于引入Soft Label，Loss值会比以前小
lr = 1e-4  # 初始学习率
# 如果收敛慢，可以适当增加
```

### 3. 正样本权重
```python
# 如果发现模型倾向于预测全背景
pos_weight = torch.tensor([2.0])  # 提高正样本权重

# 或者根据数据集统计
neg_count = ...  # 背景像素数
pos_count = ...  # 前景像素数
pos_weight = torch.tensor([neg_count / pos_count])
```

### 4. 评估指标
```python
# 训练时使用 Oracle Mask
train_loss = criterion(pred, oracle_mask)

# 评估时使用 GT Mask
with torch.no_grad():
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    iou = compute_iou(pred_binary, gt_mask)
    precision = compute_precision(pred_binary, gt_mask)
    recall = compute_recall(pred_binary, gt_mask)
```

## 常见问题

### Q1: 为什么Oracle Mask最大值是153（0.6×255）而不是255？
**A**: 这是Soft Label机制的核心！
- 有LiDAR点云的目标：255 (1.0)
- 无LiDAR点云的目标：153 (0.6)
- 背景：0 (0.0)

归一化后：
- 有LiDAR：1.0（强监督）
- 无LiDAR：0.6（弱监督，避免过拟合纯视觉特征）
- 背景：0.0

### Q2: Loss值比以前小很多，正常吗？
**A**: 正常！Soft Label会导致：
- BCE Loss：target从{0,1}变成[0.0,1.0]，损失值变小
- IoU Loss：部分区域置信度0.6，IoU自然降低

**解决方案**：
1. 提高正样本权重：`pos_weight=2.0`
2. 增大loss系数：`loss = criterion(pred, target) * 2.0`
3. 调整优化器学习率

### Q3: 16-bit图像加载很慢，怎么办？
**A**: 考虑预处理：
```python
# 离线转换16-bit -> 8-bit
from scripts.normalize_infrared_images import normalize_and_save

# 或者在DataLoader中缓存
self.image_cache = {}  # 添加缓存字典
```

### Q4: LiDAR点云怎么传给网络？
**A**: DataLoader返回的是`List[Tensor]`，因为每张图的点数不同：
```python
# 在模型的forward中
def forward(self, images, lidar_points_list):
    batch_size = images.shape[0]
    
    for i in range(batch_size):
        points_i = lidar_points_list[i]  # (N_i, 4)
        # 投影到图像i上
        pixels = project_lidar(points_i, K_cam, T_cam_to_lidar)
        # 构建LiDAR特征图
        lidar_map = create_lidar_feature_map(pixels, images.shape[2:])
```

## 完整训练示例

```python
import torch
from torch.utils.data import DataLoader
from model.utils_lidar import (
    PoLaRISTrainLoader, 
    PoLaRISTestLoader,
    CombinedSoftLoss
)
from model.load_param_data import load_dataset

# 1. 加载数据集划分
train_ids, val_ids, _ = load_dataset(
    root='dataset',
    dataset='select',
    split_method='split_data'
)

# 2. 创建DataLoader
train_dataset = PoLaRISTrainLoader(
    dataset_dir='dataset/select',
    img_id=train_ids,
    base_size=512,
    crop_size=480,
    normalize_16bit=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

# 3. 创建模型和Loss
model = YourModel().cuda()
criterion = CombinedSoftLoss(
    bce_weight=0.5,
    iou_weight=0.5,
    pos_weight=torch.tensor([2.0]).cuda()
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 4. 训练循环
for epoch in range(100):
    model.train()
    for batch in train_loader:
        images = batch['image'].cuda()
        oracle_masks = batch['oracle_mask'].cuda()
        lidar_points = batch['lidar']  # List of tensors
        
        # Forward
        outputs = model(images, lidar_points)
        
        # Loss (使用 oracle_mask，不是 mask！)
        loss = criterion(outputs, oracle_masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## 与旧版DataLoader的区别

| 特性 | 旧版 (utils.py) | 新版 (utils_lidar.py) |
|------|----------------|----------------------|
| 16-bit支持 | ❌ | ✅ Min-Max归一化 |
| LiDAR加载 | ❌ | ✅ .bin文件自动读取 |
| Soft Label | ❌ 二值化 | ✅ 保留float值 |
| 返回格式 | Tuple | Dict（更清晰） |
| Oracle Mask | ❌ | ✅ 作为训练target |
| Loss函数 | 普通BCE/IoU | 支持Soft Label |

## 迁移指南

### 从旧版迁移到新版：

**步骤1：替换导入**
```python
# 旧版
from model.utils import TrainSetLoader, TestSetLoader

# 新版
from model.utils_lidar import PoLaRISTrainLoader, PoLaRISTestLoader
```

**步骤2：修改数据获取**
```python
# 旧版
for img, mask, oracle_mask in train_loader:
    ...

# 新版
for batch in train_loader:
    img = batch['image']
    mask = batch['mask']  # GT mask（评估用）
    oracle_mask = batch['oracle_mask']  # 训练target
    lidar = batch['lidar']
```

**步骤3：修改Loss计算**
```python
# 旧版
criterion = torch.nn.BCEWithLogitsLoss()
loss = criterion(pred, mask)  # 使用GT mask

# 新版
from model.utils_lidar import CombinedSoftLoss
criterion = CombinedSoftLoss()
loss = criterion(pred, oracle_mask)  # 使用Oracle Mask！
```

**步骤4：添加LiDAR处理**
```python
# 在模型中
def forward(self, images, lidar_points_list):
    # IR分支
    ir_features = self.ir_encoder(images)
    
    # LiDAR分支（如果模型支持）
    lidar_features = self.lidar_encoder(lidar_points_list)
    
    # Fusion
    fused = self.fusion(ir_features, lidar_features)
    
    return self.decoder(fused)
```

## 性能优化建议

1. **预加载LiDAR**：如果点云文件很大，考虑预加载到内存
2. **异步加载**：`num_workers=4` 利用多进程
3. **Pin Memory**：`pin_memory=True` 加速GPU传输
4. **混合精度**：使用 `torch.cuda.amp` 减少显存占用

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,  # 加速GPU传输
    drop_last=True
)

# 混合精度训练
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(images, lidar_points)
    loss = criterion(outputs, oracle_masks)
```

## 快速开始：使用预配置的训练脚本

为了方便您在 8-bit 和 16-bit 数据之间切换，我们提供了三个预配置的训练脚本：

### 1. 8-bit 模式（旧版 DataLoader）

```bash
chmod +x scripts/run_Phase3_DualGeo_8bit.sh
./scripts/run_Phase3_DualGeo_8bit.sh
```

**特性**：
- ✅ 8-bit 图像支持
- ✅ 使用旧版 TrainSetLoader
- ✅ 硬标签训练（Binary: 0/1）
- ❌ 不支持 16-bit 图像

### 2. Enhanced 16-bit 模式（新版 DataLoader，推荐）

```bash
chmod +x scripts/run_Phase3_DualGeo_16bit.sh
./scripts/run_Phase3_DualGeo_16bit.sh
```

**特性**：
- ✅ 16-bit 红外图像支持
- ✅ Min-Max 归一化
- ✅ 软标签训练（0.0, 0.6, 1.0）
- ✅ 深度图支持（in_channels=2）
- ✅ LiDAR 点云加载

### 3. IR-Only 16-bit 模式（仅红外，无深度图）

```bash
chmod +x scripts/run_Phase3_DualGeo_16bit_IR_only.sh
./scripts/run_Phase3_DualGeo_16bit_IR_only.sh
```

**特性**：
- ✅ 16-bit 红外图像支持
- ✅ 软标签训练
- ❌ 不使用深度图（in_channels=1）
- 📊 适合对比实验：IR vs IR+Depth

---

## 参数说明

所有脚本都支持以下关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--use_lidar_dataloader` | True/False | 是否使用新的 PoLaRIS DataLoader |
| `--normalize_16bit` | True/False | 16-bit 归一化方式<br>True: Min-Max 归一化（推荐）<br>False: 简单缩放 /65535 |
| `--use_soft_labels` | True/False | 是否使用软标签训练<br>True: 使用 oracle_masks (0.6, 1.0)<br>False: 使用 GT masks (0, 1) |
| `--in_channels` | 1/2 | 输入通道数<br>1: 仅红外图像<br>2: 红外 + 深度图 |

---

## 文件结构检查

在运行新版 DataLoader 之前，请确保您的数据集具有以下结构：

```
dataset/Pohang-Canal/
├── images/              # 红外图像（支持 8-bit .png 或 16-bit .png）
│   ├── 000001.png
│   └── ...
├── masks/               # GT 标签（硬标签，0/255）
│   ├── 000001.png
│   └── ...
├── oracle_masks/        # Oracle 标签（软标签，0/153/255）
│   ├── 000001.png      # 0: 背景, 153 (0.6): 无LiDAR目标, 255 (1.0): 有LiDAR目标
│   └── ...
├── depth_maps/          # 深度图（从 LiDAR 生成，.npy 格式）
│   ├── 000001.npy      # 仅 in_channels=2 时需要
│   └── ...
└── lidar_roi/           # LiDAR 点云（.bin 文件，可选）
    ├── 000001.bin      # (N, 4) float32: [x, y, z, intensity]
    └── ...
```

**检查命令**：
```bash
# 检查 Oracle Masks 是否生成
ls dataset/Pohang-Canal/oracle_masks/ | wc -l

# 检查深度图是否生成（in_channels=2 需要）
ls dataset/Pohang-Canal/depth_maps/*.npy | wc -l

# 检查 16-bit 图像
python -c "from PIL import Image; img = Image.open('dataset/Pohang-Canal/images/000001.png'); print(f'Mode: {img.mode}, Size: {img.size}')"
```

---

## 常见问题 (FAQ)

### Q1: 如何从旧版脚本切换到新版？

**A**: 只需将脚本中的参数修改为：
```bash
--use_lidar_dataloader True \
--normalize_16bit True \
--use_soft_labels True
```

### Q2: 新版 DataLoader 是否兼容旧模型？

**A**: 完全兼容！新版 DataLoader 支持：
- `in_channels=1`: 仅红外图像（与旧版相同）
- `in_channels=2`: 红外 + 深度图（需要 depth_maps/*.npy）

### Q3: 16-bit 图像加载后会损失精度吗？

**A**: 会有轻微损失。当前实现：
```
16-bit (0-65535) → Min-Max 归一化 → uint8 (0-255) → 数据增强 → float32 (0-1)
```

**影响**：65536 级 → 256 级，损失约 99.6% 的灰度级

**优化方案**（如果需要）：
1. 跳过 uint8 转换，直接在 float32 上做数据增强
2. 使用 torchvision transforms（支持 float）

### Q4: Soft Label 的值为什么是 0.6 而不是 0.5？

**A**: 这是设计选择：
- **1.0**：有 LiDAR 点云的目标（强监督，高置信度）
- **0.6**：无 LiDAR 点云的目标（弱监督，避免过拟合纯视觉特征）
- **0.0**：背景

0.6 比 0.5 更高，表明这些目标确实存在，只是缺少几何信息验证。

### Q5: 如果我的数据集没有 oracle_masks，能用新 DataLoader 吗？

**A**: 可以！DataLoader 会自动回退：
- 如果 `oracle_masks/` 文件夹不存在，会使用全零的 oracle_mask
- 训练时设置 `--use_soft_labels False` 使用硬标签

```bash
python train_Phase3.py \
    --use_lidar_dataloader True \
    --use_soft_labels False  # 使用硬标签
```

---

## 下一步

1. **生成 Oracle Masks**（如果还没有）：
   ```bash
   python scripts/generate_oracle_masks.py --dataset Pohang-Canal
   ```

2. **运行训练**：
   ```bash
   # 推荐：使用 16-bit + 软标签
   ./scripts/run_Phase3_DualGeo_16bit.sh

   # 或者：使用旧版（8-bit）
   ./scripts/run_Phase3_DualGeo_8bit.sh
   ```

3. **监控训练**：
   - 检查 Loss 是否下降
   - 使用软标签时，Loss 值会比硬标签小（这是正常的）
   - 如果模型倾向于预测全背景，增加 `pos_weight`
