# Phase 3 模型升级与专项评估 - 实现总结

## 概述
本文档记录了 Phase 3 的三个核心模块的完整实现，旨在提升 MS_CAFNet_DualGeo 模型的性能，特别是在 Label 3（复杂岸边场景）的表现。

---

## 模块 1: 架构升级 - 深度监督 (Deep Supervision)

### 目标
引入深度监督机制以提升小目标 Recall，通过多尺度输出强化模型对不同尺度目标的学习能力。

### 修改文件

#### 1.1 [model/model_Phase3.py](../model/model_Phase3.py)
**修改内容：** 在 `MS_CAFNet_DualGeo` 类中添加深度监督输出头

```python
class MS_CAFNet_DualGeo(nn.Module):
    def __init__(self, num_classes=1, input_channels=2):
        # ... 原有代码 ...

        # 深度监督输出头（新增）
        self.final1 = nn.Conv2d(nb_filter[3], num_classes, kernel_size=1)  # 来自 x3_1 (1/8分辨率)
        self.final2 = nn.Conv2d(nb_filter[2], num_classes, kernel_size=1)  # 来自 x2_2 (1/4分辨率)
        self.final3 = nn.Conv2d(nb_filter[1], num_classes, kernel_size=1)  # 来自 x1_3 (1/2分辨率)
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)   # 来自 x0_4 (原始分辨率)

    def forward(self, input):
        # ... 原有编码-解码代码 ...

        # 深度监督输出（新增）
        out1 = self.final1(x3_1)  # 1/8 分辨率
        out1 = F.interpolate(out1, scale_factor=8, mode='bilinear', align_corners=True)

        out2 = self.final2(x2_2)  # 1/4 分辨率
        out2 = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=True)

        out3 = self.final3(x1_3_fused)  # 1/2 分辨率
        out3 = F.interpolate(out3, scale_factor=2, mode='bilinear', align_corners=True)

        out_final = self.final(x0_4)  # 原始分辨率

        # 返回多尺度预测列表和置信度图
        return [out1, out2, out3, out_final], pred_conf
```

**关键点：**
- 添加了 4 个输出头，分别对应不同解码层
- 所有输出上采样到原始分辨率
- 返回列表格式 `[out1, out2, out3, out_final]` 而非单个输出

#### 1.2 [train_Phase3.py](../train_Phase3.py)
**修改内容：** 更新训练循环以支持深度监督 Loss 计算

```python
# 训练阶段（第195-216行）
if self.args.model == 'MS_CAFNet' or self.args.model == 'MS_CAFNet_DualGeo':
    outputs, pred_conf = self.model(data)

    # 处理列表形式输出（支持深度监督）
    if not isinstance(outputs, list):
        outputs = [outputs]

    # 计算所有输出的平均 Loss
    loss_seg = 0
    for pred in outputs:
        loss_seg += SoftIoULoss(pred, train_target)
    loss_seg /= len(outputs)

    loss_conf = self.conf_loss(pred_conf, oracle_masks)
    loss = loss_seg + 0.5 * loss_conf

    # 使用最终输出计算 batch size
    pred = outputs[-1]

# 测试阶段（第257-273行）
if self.args.model == 'MS_CAFNet' or self.args.model == 'MS_CAFNet_DualGeo':
    outputs, pred_conf = self.model(data)

    if not isinstance(outputs, list):
        outputs = [outputs]

    # 计算所有输出的平均 Loss
    loss = 0
    for pred in outputs:
        loss += SoftIoULoss(pred, labels)
    loss /= len(outputs)

    # 使用最终输出进行评估
    pred = outputs[-1]
```

**关键点：**
- 兼容单输出和列表输出（向后兼容）
- 对所有尺度输出计算 Loss 并取平均
- 仅使用最终输出 `outputs[-1]` 进行指标计算

---

## 模块 2: 参数校准 - 推理阈值调整

### 目标
将推理阈值从 0.5 调整为 0.3，以适配 Soft Label 的最大值为 0.6（而非传统的 1.0）。

### 修改文件

#### 2.1 [model/parse_args_train.py](../model/parse_args_train.py)
**新增参数：**
```python
# Inference threshold (adapted for Soft Labels with max=0.6)
parser.add_argument('--thres', type=float, default=0.3,
                    help='binary threshold for inference (default: 0.3 for Soft Label max=0.6)')
```

#### 2.2 [model/parse_args_test.py](../model/parse_args_test.py)
**新增参数：**
```python
# Inference threshold (adapted for Soft Labels with max=0.6)
parser.add_argument('--thres', type=float, default=0.3,
                    help='binary threshold for inference (default: 0.3 for Soft Label max=0.6)')
```

#### 2.3 [model/metric.py](../model/metric.py)
**修改内容：**

1. **更新 mIoU 类：**
```python
class mIoU():
    def __init__(self, nclass, threshold=0.3):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.threshold = threshold  # 推理阈值（默认0.3适配Soft Labels）
        self.reset()

    def update(self, preds, labels, depth_map=None, use_adaptive_threshold=True):
        correct, labeled = batch_pix_accuracy(preds, labels, depth_map, use_adaptive_threshold, self.threshold)
        inter, union = batch_intersection_union(preds, labels, self.nclass, depth_map, use_adaptive_threshold, self.threshold)
        # ... 后续代码 ...
```

2. **更新 batch_pix_accuracy 函数：**
```python
def batch_pix_accuracy(output, target, depth_map=None, use_adaptive_threshold=True, threshold=0.3):
    # ... 原有代码 ...

    if use_adaptive_threshold and depth_map is not None:
        predict = adaptive_threshold_binarization(output, depth_map,
                                                   threshold_with_lidar=threshold,
                                                   threshold_without_lidar=threshold * 0.7)
    else:
        predict = (output > threshold).float()  # 使用可配置阈值

    # ... 后续代码 ...
```

3. **更新 batch_intersection_union 函数：**
```python
def batch_intersection_union(output, target, nclass, depth_map=None, use_adaptive_threshold=True, threshold=0.3):
    # ... 原有代码 ...

    if use_adaptive_threshold and depth_map is not None:
        predict = adaptive_threshold_binarization(output, depth_map,
                                                   threshold_with_lidar=threshold,
                                                   threshold_without_lidar=threshold * 0.7)
    else:
        predict = (output > threshold).float()  # 使用可配置阈值

    # ... 后续代码 ...
```

#### 2.4 [train_Phase3.py](../train_Phase3.py)
**修改内容：** 传递阈值参数给 mIoU
```python
self.mIoU = mIoU(1, threshold=getattr(args, 'thres', 0.3))  # 使用 args.thres，默认 0.3
```

#### 2.5 [test.py](../test.py)
**修改内容：** 传递阈值参数给 mIoU
```python
self.mIoU = mIoU(1, threshold=getattr(args, 'thres', 0.3))  # 使用 args.thres，默认 0.3
```

**关键点：**
- 阈值可通过命令行参数 `--thres` 配置
- 默认值 0.3 适配 Soft Label max=0.6 的场景
- 动态自适应阈值：有 LiDAR 区域使用 `threshold`，无 LiDAR 区域使用 `threshold * 0.7`

---

## 模块 3: 专项评估 - Label 3 场景虚警率分析

### 目标
实现场景差异化评估，重点分析 Label 3（复杂岸边场景）的性能，证明模型在该场景下的优越性。

### 修改文件

#### 3.1 [test.py](../test.py)
**新增功能：**

1. **导入必要模块：**
```python
import re
from collections import defaultdict
```

2. **新增函数：加载图像类别**
```python
def load_image_categories(dataset_dir):
    """
    从 selection_summary_new.txt 加载图像类别
    返回: dict 映射 image_name (不含.png) 到 category
    """
    selection_file = os.path.join(dataset_dir, 'selection_summary_new.txt')
    image_categories = {}

    if not os.path.exists(selection_file):
        print(f"Warning: {selection_file} not found. Category-specific evaluation disabled.")
        return image_categories

    with open(selection_file, 'r') as f:
        for line in f:
            if '|' in line:
                parts = line.strip().split('|')
                img = parts[0].strip().replace('.png', '')
                cat = int(parts[1].strip())
                if cat in [1, 2, 3]:
                    image_categories[img] = cat

    print(f"✅ Loaded {len(image_categories)} image categories")
    return image_categories
```

3. **新增类：CategoryMetrics**
```python
class CategoryMetrics:
    """为每个类别单独跟踪指标"""
    def __init__(self, nclass, threshold=0.3):
        self.nclass = nclass
        self.threshold = threshold
        self.categories = [1, 2, 3]

        # 每个类别的 mIoU
        self.cat_mIoU = {cat: mIoU(nclass, threshold) for cat in self.categories}
        self.cat_count = {cat: 0 for cat in self.categories}

        # 虚警跟踪（预测 > 阈值 但 GT == 0）
        self.cat_false_alarms = {cat: 0 for cat in self.categories}
        self.cat_total_pixels = {cat: 0 for cat in self.categories}

    def update(self, pred, labels, category):
        """更新特定类别的指标"""
        if category not in self.categories:
            return

        # 更新 mIoU
        self.cat_mIoU[category].update(pred, labels)
        self.cat_count[category] += 1

        # 跟踪虚警：pred > threshold 但 GT == 0 的像素
        pred_binary = (torch.sigmoid(pred) > self.threshold).float()
        gt_binary = (labels > 0).float()

        false_alarm_pixels = ((pred_binary == 1) & (gt_binary == 0)).sum().item()
        total_pixels = labels.numel()

        self.cat_false_alarms[category] += false_alarm_pixels
        self.cat_total_pixels[category] += total_pixels

    def get_results(self):
        """获取各类别的结果"""
        results = {}
        for cat in self.categories:
            if self.cat_count[cat] > 0:
                _, mean_iou = self.cat_mIoU[cat].get()
                fa_rate = self.cat_false_alarms[cat] / (self.cat_total_pixels[cat] + 1e-10)
                results[cat] = {
                    'count': self.cat_count[cat],
                    'mIoU': mean_iou,
                    'false_alarm_rate': fa_rate,
                    'false_alarm_pixels': self.cat_false_alarms[cat],
                    'total_pixels': self.cat_total_pixels[cat]
                }
        return results
```

4. **修改 Trainer 类：初始化类别评估**
```python
# 加载图像类别用于场景差异化评估
self.image_categories = load_image_categories(dataset_dir)
self.category_metrics = CategoryMetrics(1, threshold=getattr(args, 'thres', 0.3))
```

5. **修改测试循环：更新类别指标**
```python
# 场景差异化评估
img_id = val_img_ids[i]
if img_id in self.image_categories:
    category = self.image_categories[img_id]
    self.category_metrics.update(pred, labels, category)
```

6. **测试结束后：打印和保存类别结果**
```python
# 打印类别结果
print("\n" + "="*80)
print("Category-Specific Evaluation Results")
print("="*80)
cat_results = self.category_metrics.get_results()
for cat in sorted(cat_results.keys()):
    result = cat_results[cat]
    print(f"\nCategory {cat} ({result['count']} images):")
    print(f"  mIoU: {result['mIoU']:.4f}")
    print(f"  False Alarm Rate: {result['false_alarm_rate']:.6f}")
    print(f"  False Alarm Pixels: {result['false_alarm_pixels']:,} / {result['total_pixels']:,}")

# 保存类别结果到文件
cat_result_file = os.path.join(dataset_dir, 'value_result', f"{args.st_model}_category_results.txt")
with open(cat_result_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Category-Specific Evaluation Results\n")
    f.write("="*80 + "\n\n")
    for cat in sorted(cat_results.keys()):
        result = cat_results[cat]
        f.write(f"Category {cat} ({result['count']} images):\n")
        f.write(f"  mIoU: {result['mIoU']:.4f}\n")
        f.write(f"  False Alarm Rate: {result['false_alarm_rate']:.6f}\n")
        f.write(f"  False Alarm Pixels: {result['false_alarm_pixels']:,} / {result['total_pixels']:,}\n\n")
print(f"\n✅ Category-specific results saved to: {cat_result_file}")
```

**关键点：**
- 自动从 `dataset/Pohang-Canal-3k/selection_summary_new.txt` 加载类别信息
- 分别跟踪每个类别（1, 2, 3）的 mIoU 和虚警率
- 虚警率定义：预测为目标但 GT 为背景的像素占比
- 结果保存到 `{args.st_model}_category_results.txt`

---

## 使用方法

### 训练（启用深度监督 + 阈值0.3）
```bash
python train_Phase3.py \
    --model MS_CAFNet_DualGeo \
    --dataset Pohang-Canal-3k \
    --use_lidar_dataloader True \
    --use_soft_labels True \
    --thres 0.3 \
    --epochs 200 \
    --lr 0.05
```

### 测试（场景差异化评估）
```bash
python test.py \
    --model MS_CAFNet_DualGeo \
    --dataset Pohang-Canal-3k \
    --model_dir your_model_checkpoint.pth.tar \
    --thres 0.3
```

**输出示例：**
```
================================================================================
Category-Specific Evaluation Results
================================================================================

Category 1 (119 images):
  mIoU: 0.7234
  False Alarm Rate: 0.001234
  False Alarm Pixels: 8,123 / 6,553,600

Category 2 (271 images):
  mIoU: 0.6891
  False Alarm Rate: 0.002156
  False Alarm Pixels: 37,456 / 17,367,040

Category 3 (210 images):
  mIoU: 0.6512
  False Alarm Rate: 0.003421
  False Alarm Pixels: 47,234 / 13,824,000

✅ Category-specific results saved to: dataset/Pohang-Canal-3k/value_result/your_model_category_results.txt
```

---

## 技术亮点

### 1. 深度监督架构
- **多尺度监督**：从 1/8、1/4、1/2、原始分辨率 4 个层级提取特征
- **梯度回传优化**：每个解码层都接收到监督信号，缓解梯度消失
- **小目标增强**：浅层特征保留更多细节，有助于检测小目标

### 2. 自适应阈值
- **Soft Label 适配**：阈值从 0.5 降至 0.3，匹配 Soft Label max=0.6 的特性
- **LiDAR 感知**：有 LiDAR 覆盖区域使用标准阈值，无覆盖区域降低阈值（×0.7）
- **灵活配置**：通过命令行参数 `--thres` 可自由调整

### 3. 场景差异化评估
- **细粒度分析**：分别评估简单海面（Category 1）、一般岸边（Category 2）、复杂岸边（Category 3）
- **虚警率量化**：精确统计每个场景的误检像素，证明模型鲁棒性
- **结果可视化**：自动生成评估报告，便于论文撰写

---

## 预期效果

### 模块 1: 深度监督
- **小目标 Recall ↑**：预计提升 3-5%
- **整体 mIoU ↑**：预计提升 1-2%

### 模块 2: 阈值调整
- **Recall ↑**：阈值降低会提升召回率 5-10%
- **Precision ↓**：可能轻微下降 2-3%（可接受的权衡）
- **F1-score ↑**：总体平衡指标预计提升

### 模块 3: 场景差异化评估
- **Label 3 性能可视化**：证明模型在复杂场景下的优势
- **论文支撑**：提供定量数据支持 "在复杂岸边场景中表现优越" 的论点

---

## 后续建议

1. **超参数调优**：
   - 尝试不同阈值（0.25, 0.3, 0.35）找到最佳平衡
   - 调整深度监督的 Loss 权重（当前均等，可改为 [0.5, 0.7, 0.9, 1.0]）

2. **消融实验**：
   - 对比有/无深度监督的性能差异
   - 对比不同阈值设置的效果

3. **可视化增强**：
   - 绘制 Category 3 的典型成功/失败案例
   - 生成虚警热图，定位高虚警区域

4. **论文撰写**：
   - 使用 Category-specific 结果表格展示各场景性能
   - 强调 Label 3 的 mIoU 和低虚警率作为核心贡献

---

## 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `model/model_Phase3.py` | 修改 | 添加深度监督输出头 |
| `train_Phase3.py` | 修改 | 支持深度监督 Loss 计算 |
| `model/parse_args_train.py` | 新增参数 | `--thres` 推理阈值 |
| `model/parse_args_test.py` | 新增参数 | `--thres` 推理阈值 |
| `model/metric.py` | 修改 | mIoU 类支持可配置阈值 |
| `test.py` | 重大修改 | 添加场景差异化评估 |
| `docs/PHASE3_IMPLEMENTATION.md` | 新增 | 本文档 |

---

## 总结

Phase 3 的三个模块全部实现完毕，涵盖了：
1. **架构层面**：深度监督提升多尺度学习能力
2. **参数层面**：自适应阈值适配 Soft Label 特性
3. **评估层面**：场景差异化分析，定量证明复杂场景优势

所有修改已完成并经过代码审查，可直接用于训练和测试。建议先在小规模数据上验证，确认无误后再进行完整训练。
