# Scripts 文件夹详细说明

> 完整的脚本功能、输入输出和使用指南

---

## 目录

- [训练和测试](#训练和测试)
- [数据准备](#数据准备)
- [数据集管理](#数据集管理)
- [数据分析和可视化](#数据分析和可视化)
- [工具脚本](#工具脚本)
- [相似脚本对比](#相似脚本对比)

---

## 训练和测试

### train.sh

**作用**: 统一训练入口脚本

**功能**:
- 自动选择训练模式（根据数据集名称）
- 支持多种训练配置（8bit, 16bit, baseline 等）
- 参数解析和验证

**输入**:
```bash
./scripts/train.sh <mode> [options]
```

**参数**:
- `<mode>`: 训练模式
  - `auto` - 自动选择（Pohang-Canal-3k 用 16bit，其他用 8bit）
  - `16bit` - 16-bit + 软标签
  - `8bit` - 8-bit + 硬标签
  - `16bit-ir` - 仅红外（无深度）
  - `baseline1` - DNANet 基准
  - `baseline2` - DNANet 变体
- `--dataset` - 数据集名称（默认：Pohang-Canal）
- `--gpu` - GPU ID（默认：5）
- `--epochs` - 训练轮数（默认：200）

**输出**:
- 训练好的模型权重（保存在 `results/` 目录）
- 训练日志和指标

**示例**:
```bash
# 自动模式
./scripts/train.sh auto --dataset Pohang-Canal-3k --gpu 0

# 16-bit 模式
./scripts/train.sh 16bit --dataset Pohang-Canal-3k --gpu 0 --epochs 200
```

---

### test_dataloader.py

**作用**: 测试 PoLaRIS DataLoader 是否正常工作

**功能**:
- 测试单通道模式（仅红外）
- 测试双通道模式（红外 + 深度）
- 验证数据格式和范围
- 检查 Oracle Masks 的值

**输入**:
```bash
python scripts/test_dataloader.py
```

**参数**: 无（内部使用硬编码配置）

**输出**:
- 控制台输出测试结果
- 显示图像形状、数据类型、值范围
- 验证 Oracle Masks 的唯一值

**生成的数据**: 无（仅测试）

**示例输出**:
```
✅ 数据集加载成功
   - 训练集: 2400 张
   - 验证集: 600 张
✅ TrainLoader 创建成功
📦 Batch 1:
   - image shape: (2, 2, 480, 480)
   - oracle_mask unique values: tensor([0.0000, 0.6000, 1.0000])
🎉 所有测试通过！
```

---

## 数据准备

### generate_depth_maps.py

**作用**: 从 LiDAR 点云生成深度图

**功能**:
- 读取 LiDAR 点云（.bin 文件）
- 过滤和投影点云到图像平面
- 生成深度图（.npy 文件）
- 应用与可视化一致的滤波配置

**输入**:
```bash
python scripts/generate_depth_maps.py --dataset <dataset_name>
```

**参数**:
- `--dataset` - 数据集名称（必需）

**输入文件**:
- `dataset/<dataset>/lidar_roi/*.bin` - LiDAR 点云文件
- `dataset/<dataset>/calibration/calib_ir_lidar.json` - 标定文件

**输出文件**:
- `dataset/<dataset>/depth_maps/*.npy` - 深度图（numpy 数组）

**深度图格式**:
- 形状：`(H, W)` - 与红外图像相同尺寸
- 数据类型：`float32`
- 值范围：0.0 ~ 80.0（单位：米）
- 0.0 表示无深度信息

**示例**:
```bash
python scripts/generate_depth_maps.py --dataset Pohang-Canal-3k
```

**处理流程**:
1. 读取 LiDAR 点云（N, 4）- [x, y, z, intensity]
2. 应用过滤（距离、高度、强度）
3. 投影到图像平面
4. 生成深度图并保存为 .npy

---

### generate_oracle_masks.py

**作用**: 生成软标签（Oracle Masks）

**功能**:
- 结合 GT Masks 和 LiDAR 点云
- 为有 LiDAR 验证的目标分配 1.0
- 为无 LiDAR 的目标分配 0.6（软标签）
- 为背景分配 0.0

**输入**:
```bash
python scripts/generate_oracle_masks.py --dataset <dataset_name>
```

**参数**:
- `--dataset` - 数据集名称（必需）

**输入文件**:
- `dataset/<dataset>/masks/*.png` - GT Masks（硬标签）
- `dataset/<dataset>/lidar_roi/*.bin` - LiDAR 点云
- `dataset/<dataset>/calibration/calib_ir_lidar.json` - 标定文件

**输出文件**:
- `dataset/<dataset>/oracle_masks/*.png` - Oracle Masks（软标签）

**Oracle Mask 格式**:
- 形状：`(H, W)` - 单通道灰度图
- 像素值：
  - `255` (1.0) - 有 LiDAR 验证的目标
  - `153` (0.6) - 无 LiDAR 但视觉确认的目标
  - `0` (0.0) - 背景

**示例**:
```bash
python scripts/generate_oracle_masks.py --dataset Pohang-Canal-3k
```

**处理流程**:
1. 读取 GT Mask
2. 读取 LiDAR 点云并投影
3. 创建 LiDAR 覆盖掩码
4. 生成 Oracle Mask：
   - GT 中的目标 + LiDAR 覆盖 → 1.0
   - GT 中的目标 - LiDAR 覆盖 → 0.6
   - 背景 → 0.0

---

### normalize_infrared_images.py

**作用**: 归一化红外图像

**功能**:
- 对 16-bit 或 8-bit 红外图像进行归一化
- 支持多种归一化方法
- 批量处理数据集

**输入**:
```bash
python scripts/normalize_infrared_images.py --dataset <dataset_name> --method <method>
```

**参数**:
- `--dataset` - 数据集名称
- `--method` - 归一化方法（minmax, percentile, zscore）

**输入文件**:
- `dataset/<dataset>/images/*.png` - 原始红外图像

**输出文件**:
- `dataset/<dataset>/images_normalized/*.png` - 归一化后的图像

**归一化方法**:
- `minmax` - Min-Max 归一化（推荐）
- `percentile` - 百分位归一化
- `zscore` - Z-Score 归一化

**示例**:
```bash
python scripts/normalize_infrared_images.py --dataset Pohang-Canal-3k --method minmax
```

---

### run_prepare_dataset.sh

**作用**: 一键运行完整的数据准备流程

**功能**:
- 顺序执行所有数据准备脚本
- 生成深度图
- 生成 Oracle Masks
- 验证数据集

**输入**:
```bash
./scripts/run_prepare_dataset.sh
```

**参数**: 在脚本内部修改 `DATASET` 变量

**执行流程**:
1. 生成深度图（`generate_depth_maps.py`）
2. 生成 Oracle Masks（`generate_oracle_masks.py`）
3. 验证数据集（`verify_dataset.py`）

**示例**:
```bash
# 修改脚本中的 DATASET="Pohang-Canal-3k"
./scripts/run_prepare_dataset.sh
```

---

## 数据集管理

### classify_images.py

**作用**: 根据 LiDAR 点云密度对图像进行分类

**功能**:
- 统计每张图像的 LiDAR 点云数量
- 将图像分类为不同的质量等级
- 生成分类报告和可视化

**输入**:
```bash
python scripts/classify_images.py --dataset <dataset_name>
```

**参数**:
- `--dataset` - 数据集名称
- `--output` - 输出分类结果文件（默认：classification_result.json）

**输入文件**:
- `dataset/<dataset>/lidar_roi/*.bin` - LiDAR 点云
- `dataset/<dataset>/images/*.png` - 红外图像

**输出文件**:
- `classification_result.json` - 分类结果（JSON 格式）
- `classification_stats.png` - 分类统计图

**分类标准**:
- `excellent` - LiDAR 点数 > 1000
- `good` - 500 < 点数 <= 1000
- `fair` - 100 < 点数 <= 500
- `poor` - 点数 <= 100

**输出 JSON 格式**:
```json
{
  "excellent": ["000001", "000002", ...],
  "good": ["000100", "000101", ...],
  "fair": ["000200", ...],
  "poor": ["000300", ...]
}
```

**示例**:
```bash
python scripts/classify_images.py --dataset Pohang-Canal-3k
```

---

### filter_dataset.py

**作用**: 根据分类结果过滤数据集

**功能**:
- 读取分类结果
- 根据质量等级筛选图像
- 创建过滤后的数据集

**输入**:
```bash
python scripts/filter_dataset.py --dataset <dataset_name> --min_quality <quality>
```

**参数**:
- `--dataset` - 数据集名称
- `--min_quality` - 最低质量等级（excellent, good, fair, poor）
- `--classification` - 分类结果文件（默认：classification_result.json）

**输入文件**:
- `classification_result.json` - 分类结果

**输出**:
- `filtered_ids.txt` - 过滤后的图像 ID 列表

**示例**:
```bash
# 只保留 excellent 和 good 质量的图像
python scripts/filter_dataset.py --dataset Pohang-Canal-3k --min_quality good
```

---

### finalize_dataset.py

**作用**: 最终整理数据集

**功能**:
- 检查所有文件的完整性
- 删除不完整的样本
- 生成最终的数据集统计

**输入**:
```bash
python scripts/finalize_dataset.py --dataset <dataset_name>
```

**参数**:
- `--dataset` - 数据集名称
- `--dry_run` - 仅检查不删除（默认：False）

**输入文件**:
- `dataset/<dataset>/images/*.png`
- `dataset/<dataset>/masks/*.png`
- `dataset/<dataset>/oracle_masks/*.png`
- `dataset/<dataset>/depth_maps/*.npy`
- `dataset/<dataset>/lidar_roi/*.bin`

**输出**:
- 控制台输出统计信息
- 删除不完整的文件（如果 `--dry_run=False`）

**检查项**:
- 图像文件存在性
- Mask 文件存在性
- Oracle Mask 文件存在性
- 深度图文件存在性
- LiDAR 点云文件存在性

**示例**:
```bash
# 仅检查
python scripts/finalize_dataset.py --dataset Pohang-Canal-3k --dry_run

# 删除不完整的文件
python scripts/finalize_dataset.py --dataset Pohang-Canal-3k
```

---

### tune_classification_thresholds.py

**作用**: 调优分类阈值

**功能**:
- 分析 LiDAR 点云密度分布
- 自动建议最优分类阈值
- 生成分布图

**输入**:
```bash
python scripts/tune_classification_thresholds.py --dataset <dataset_name>
```

**参数**:
- `--dataset` - 数据集名称

**输入文件**:
- `dataset/<dataset>/lidar_roi/*.bin` - LiDAR 点云

**输出**:
- 控制台输出建议的阈值
- `lidar_density_distribution.png` - 密度分布图

**示例**:
```bash
python scripts/tune_classification_thresholds.py --dataset Pohang-Canal-3k
```

**输出示例**:
```
建议的分类阈值：
  - excellent: > 1200 points
  - good: 600-1200 points
  - fair: 150-600 points
  - poor: < 150 points
```

---

### verify_dataset.py

**作用**: 验证数据集的完整性和正确性

**功能**:
- 检查所有必需文件是否存在
- 验证图像尺寸一致性
- 检查 Oracle Masks 的值范围
- 验证深度图的有效性

**输入**:
```bash
python scripts/verify_dataset.py --dataset <dataset_name>
```

**参数**:
- `--dataset` - 数据集名称

**输入文件**:
- 数据集中的所有文件

**输出**:
- 控制台输出验证结果
- 显示缺失文件或错误

**检查项**:
1. **文件存在性**:
   - images/
   - masks/
   - oracle_masks/
   - depth_maps/（如果 in_channels=2）
   - lidar_roi/

2. **数据一致性**:
   - 图像尺寸
   - Mask 尺寸
   - 深度图尺寸

3. **数据有效性**:
   - Oracle Masks 值范围 [0, 153, 255]
   - 深度图值范围 [0, 80]
   - 图像文件可读性

**示例**:
```bash
python scripts/verify_dataset.py --dataset Pohang-Canal-3k
```

**输出示例**:
```
✅ 检查文件存在性
   - images: 3000/3000
   - masks: 3000/3000
   - oracle_masks: 3000/3000
   - depth_maps: 3000/3000
✅ 检查数据一致性
   - 图像尺寸: (640, 512) 一致
✅ 检查数据有效性
   - Oracle Masks 值: [0, 153, 255] ✓
🎉 数据集验证通过！
```

---

### diagnose_dataset.py

**作用**: 诊断数据集问题

**功能**:
- 深度检查数据集
- 识别潜在问题
- 提供修复建议

**输入**:
```bash
python scripts/diagnose_dataset.py --dataset <dataset_name>
```

**参数**:
- `--dataset` - 数据集名称
- `--verbose` - 详细输出（默认：False）

**输入文件**:
- 数据集中的所有文件

**输出**:
- 控制台输出诊断结果
- 问题列表和修复建议

**检查项**:
1. 文件损坏
2. 尺寸不匹配
3. 值范围异常
4. 缺失文件
5. 重复文件

**示例**:
```bash
python scripts/diagnose_dataset.py --dataset Pohang-Canal-3k --verbose
```

**输出示例**:
```
⚠️  发现 3 个问题：

问题 1: 图像尺寸不一致
  - 000123.png: (640, 512)
  - 000456.png: (640, 510)
  建议：重新生成或删除

问题 2: Oracle Mask 值异常
  - 000789.png: 包含值 200（应该是 0, 153, 255）
  建议：重新生成 Oracle Masks

问题 3: 深度图缺失
  - 000999.npy 不存在
  建议：运行 generate_depth_maps.py
```

---

## 数据分析和可视化

### analyze_training.py

**作用**: 分析训练结果

**功能**:
- 解析训练日志
- 绘制损失曲线
- 绘制 IoU/Dice 曲线
- 生成训练报告

**输入**:
```bash
python scripts/analyze_training.py --experiment <experiment_name>
```

**参数**:
- `--experiment` - 实验名称
- `--log_file` - 日志文件路径（可选）

**输入文件**:
- `results/<experiment>/train.log` - 训练日志

**输出文件**:
- `results/<experiment>/loss_curve.png` - 损失曲线
- `results/<experiment>/iou_curve.png` - IoU 曲线
- `results/<experiment>/training_report.txt` - 训练报告

**示例**:
```bash
python scripts/analyze_training.py --experiment Phase3_DualGeo_16bit
```

**生成的图表**:
1. **损失曲线**: 训练损失 vs 验证损失
2. **IoU 曲线**: 训练 IoU vs 验证 IoU
3. **学习率曲线**: 学习率变化

---

### visualize_lidar_projection.py

**作用**: 可视化 LiDAR 点云投影

**功能**:
- 读取 LiDAR 点云
- 投影到红外图像
- 可视化投影结果
- 生成对比图

**输入**:
```bash
python scripts/visualize_lidar_projection.py --dataset <dataset_name> --image_id <id>
```

**参数**:
- `--dataset` - 数据集名称
- `--image_id` - 图像 ID（例如：000001）
- `--output` - 输出文件路径（可选）

**输入文件**:
- `dataset/<dataset>/images/<id>.png` - 红外图像
- `dataset/<dataset>/lidar_roi/<id>.bin` - LiDAR 点云
- `dataset/<dataset>/calibration/calib_ir_lidar.json` - 标定文件

**输出文件**:
- `lidar_projection_<id>.png` - 可视化结果

**可视化内容**:
- 原始红外图像
- LiDAR 点云投影（彩色点）
- 深度信息（颜色编码）

**示例**:
```bash
python scripts/visualize_lidar_projection.py --dataset Pohang-Canal-3k --image_id 000001
```

---

### visualize_model_architecture.py

**作用**: 可视化模型架构

**功能**:
- 加载模型
- 生成网络结构图
- 计算模型参数量和 FLOPs

**输入**:
```bash
python scripts/visualize_model_architecture.py --model <model_name>
```

**参数**:
- `--model` - 模型名称（DNANet, MS_CAFNet_DualGeo 等）
- `--output` - 输出文件路径（默认：model_architecture.png）

**输出文件**:
- `model_architecture.png` - 模型结构图
- 控制台输出模型统计信息

**示例**:
```bash
python scripts/visualize_model_architecture.py --model MS_CAFNet_DualGeo
```

**输出示例**:
```
模型: MS_CAFNet_DualGeo
参数量: 12.3M
FLOPs: 45.6G
输入尺寸: (2, 512, 512)
输出尺寸: (1, 512, 512)
```

---

### compare_model_complexity.py

**作用**: 对比不同模型的复杂度

**功能**:
- 对比多个模型的参数量
- 对比 FLOPs
- 生成对比表格和图表

**输入**:
```bash
python scripts/compare_model_complexity.py
```

**参数**: 无（在脚本内部定义模型列表）

**输出文件**:
- `model_complexity_comparison.png` - 对比图
- `model_complexity_comparison.csv` - 对比表格

**对比模型**:
- DNANet
- MS_CAFNet_DualGeo
- ResidualFPN
- （其他模型）

**示例**:
```bash
python scripts/compare_model_complexity.py
```

---

### evaluate_golden_set.py

**作用**: 在黄金测试集上评估模型

**功能**:
- 在精选的高质量测试集上评估
- 计算详细的性能指标
- 生成评估报告

**输入**:
```bash
python scripts/evaluate_golden_set.py --model_path <path> --golden_set <path>
```

**参数**:
- `--model_path` - 模型权重路径
- `--golden_set` - 黄金集路径
- `--dataset` - 数据集名称

**输入文件**:
- 模型权重文件
- 黄金集图像和标签

**输出文件**:
- `golden_set_evaluation.json` - 评估结果
- `golden_set_predictions/` - 预测结果（可选）

**评估指标**:
- IoU
- Dice
- Precision
- Recall
- F1-Score

**示例**:
```bash
python scripts/evaluate_golden_set.py \
    --model_path results/Phase3_DualGeo_16bit/best_model.pth \
    --golden_set dataset/golden_set/
```

---

## 工具脚本

### data_tools.py

**作用**: 数据处理工具集合

**功能**:
- 提供常用的数据处理函数
- 图像读写
- LiDAR 点云处理
- 数据转换

**用法**: 作为库导入使用

```python
from scripts.data_tools import load_lidar, project_lidar_to_image

# 加载 LiDAR 点云
points = load_lidar('lidar_roi/000001.bin')

# 投影到图像
uv = project_lidar_to_image(points, calib)
```

**提供的函数**:
- `load_lidar()` - 加载 LiDAR 点云
- `save_lidar()` - 保存 LiDAR 点云
- `project_lidar_to_image()` - 投影点云到图像
- `load_calibration()` - 加载标定文件
- `filter_points()` - 过滤点云

---

### json_to_mask.py

**作用**: 将 JSON 标注转换为 Mask 图像

**功能**:
- 读取 JSON 格式的标注
- 生成二值 Mask 图像
- 批量转换

**输入**:
```bash
python scripts/json_to_mask.py --json_dir <dir> --output_dir <dir>
```

**参数**:
- `--json_dir` - JSON 文件目录
- `--output_dir` - 输出 Mask 目录
- `--image_size` - 图像尺寸（默认：640x512）

**输入文件**:
- `*.json` - JSON 标注文件

**JSON 格式**:
```json
{
  "shapes": [
    {
      "label": "target",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polygon"
    }
  ]
}
```

**输出文件**:
- `*.png` - 二值 Mask 图像（0=背景，255=目标）

**示例**:
```bash
python scripts/json_to_mask.py \
    --json_dir annotations/ \
    --output_dir masks/ \
    --image_size 640 512
```

---

### yolo_to_json.py

**作用**: 将 YOLO 格式转换为 JSON 格式

**功能**:
- 读取 YOLO 格式的标注（.txt）
- 转换为 JSON 格式
- 支持批量转换

**输入**:
```bash
python scripts/yolo_to_json.py --yolo_dir <dir> --output_dir <dir>
```

**参数**:
- `--yolo_dir` - YOLO 标注目录
- `--output_dir` - 输出 JSON 目录
- `--image_size` - 图像尺寸

**YOLO 格式**:
```
class_id center_x center_y width height
0 0.5 0.5 0.1 0.1
```

**输出 JSON 格式**:
```json
{
  "shapes": [
    {
      "label": "target",
      "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "shape_type": "rectangle"
    }
  ]
}
```

**示例**:
```bash
python scripts/yolo_to_json.py \
    --yolo_dir labels/ \
    --output_dir annotations/
```

---

### cleanup.sh

**作用**: 清理和归档旧脚本

**功能**:
- 预览要归档的文件
- 归档旧文件到 `scripts/archive/`
- 删除旧文件（需确认）

**输入**:
```bash
./scripts/cleanup.sh [--archive|--delete]
```

**参数**:
- （无参数）- 预览模式
- `--archive` - 归档文件
- `--delete` - 删除文件

**示例**:
```bash
# 预览
./scripts/cleanup.sh

# 归档
./scripts/cleanup.sh --archive

# 删除（危险！）
./scripts/cleanup.sh --delete
```

---

## 相似脚本对比

### generate_depth_maps.py vs visualize_lidar_projection.py

| 特性 | generate_depth_maps.py | visualize_lidar_projection.py |
| ---- | ---------------------- | ----------------------------- |
| **主要作用** | 生成训练用的深度图 | 可视化 LiDAR 投影效果 |
| **输出格式** | .npy 数组（H, W） | .png 可视化图像 |
| **批量处理** | ✅ 批量处理所有图像 | ❌ 单张图像 |
| **用途** | 训练数据准备 | 调试和验证 |
| **过滤配置** | 使用 FILTER_CONFIG | 使用相同配置 |
| **输出内容** | 深度值（米） | RGB 可视化图像 |

**何时使用**:
- 训练前准备数据 → `generate_depth_maps.py`
- 检查投影是否正确 → `visualize_lidar_projection.py`

---

### verify_dataset.py vs diagnose_dataset.py

| 特性 | verify_dataset.py | diagnose_dataset.py |
| ---- | ----------------- | ------------------- |
| **主要作用** | 快速验证数据集完整性 | 深度诊断数据集问题 |
| **检查深度** | 基础检查 | 详细检查 |
| **输出信息** | 简洁的验证结果 | 详细的问题报告 |
| **修复建议** | ❌ | ✅ 提供修复建议 |
| **速度** | 快速 | 较慢（深度检查） |
| **用途** | 日常验证 | 问题排查 |

**何时使用**:
- 快速检查数据集是否可用 → `verify_dataset.py`
- 数据集出现问题，需要诊断 → `diagnose_dataset.py`

---

### classify_images.py vs filter_dataset.py

| 特性 | classify_images.py | filter_dataset.py |
| ---- | ------------------ | ----------------- |
| **主要作用** | 对图像进行质量分类 | 根据分类结果过滤 |
| **输入** | 原始数据集 | 分类结果 JSON |
| **输出** | 分类结果 JSON | 过滤后的 ID 列表 |
| **依赖关系** | 独立运行 | 依赖 classify_images.py |
| **用途** | 分析数据集质量 | 创建高质量子集 |

**工作流**:
1. 运行 `classify_images.py` 生成分类
2. 运行 `filter_dataset.py` 筛选高质量数据
3. 使用筛选结果训练

---

### json_to_mask.py vs yolo_to_json.py

| 特性 | json_to_mask.py | yolo_to_json.py |
| ---- | --------------- | --------------- |
| **输入格式** | JSON 标注 | YOLO .txt 标注 |
| **输出格式** | PNG Mask 图像 | JSON 标注 |
| **转换方向** | 标注 → 图像 | YOLO → JSON |
| **用途** | 生成训练 Mask | 格式转换 |

**工作流**:
1. 如果有 YOLO 标注 → `yolo_to_json.py` → JSON
2. 如果有 JSON 标注 → `json_to_mask.py` → Mask

---

## 推荐工作流

### 新数据集准备

```bash
# 1. 生成深度图
python scripts/generate_depth_maps.py --dataset Pohang-Canal-3k

# 2. 生成 Oracle Masks
python scripts/generate_oracle_masks.py --dataset Pohang-Canal-3k

# 3. 验证数据集
python scripts/verify_dataset.py --dataset Pohang-Canal-3k

# 4. 测试 DataLoader
python scripts/test_dataloader.py

# 5. 开始训练
./scripts/train.sh auto --dataset Pohang-Canal-3k
```

---

### 数据集质量优化

```bash
# 1. 分析 LiDAR 密度分布
python scripts/tune_classification_thresholds.py --dataset Pohang-Canal-3k

# 2. 对图像进行分类
python scripts/classify_images.py --dataset Pohang-Canal-3k

# 3. 过滤低质量图像
python scripts/filter_dataset.py --dataset Pohang-Canal-3k --min_quality good

# 4. 最终整理
python scripts/finalize_dataset.py --dataset Pohang-Canal-3k
```

---

### 标注格式转换

```bash
# YOLO → JSON → Mask
python scripts/yolo_to_json.py --yolo_dir labels/ --output_dir annotations/
python scripts/json_to_mask.py --json_dir annotations/ --output_dir masks/
```

---

### 训练和分析

```bash
# 1. 训练
./scripts/train.sh 16bit --dataset Pohang-Canal-3k --gpu 0 --epochs 200

# 2. 分析训练结果
python scripts/analyze_training.py --experiment Phase3_DualGeo_16bit

# 3. 黄金集评估
python scripts/evaluate_golden_set.py \
    --model_path results/Phase3_DualGeo_16bit/best_model.pth \
    --golden_set dataset/golden_set/
```

---

### 可视化和调试

```bash
# 1. 可视化 LiDAR 投影
python scripts/visualize_lidar_projection.py --dataset Pohang-Canal-3k --image_id 000001

# 2. 可视化模型架构
python scripts/visualize_model_architecture.py --model MS_CAFNet_DualGeo

# 3. 对比模型复杂度
python scripts/compare_model_complexity.py

# 4. 诊断数据集问题
python scripts/diagnose_dataset.py --dataset Pohang-Canal-3k
```

---

## 常见问题

### Q1: 应该先运行哪些脚本？

**A**: 新数据集的推荐顺序：
1. `generate_depth_maps.py` - 生成深度图
2. `generate_oracle_masks.py` - 生成软标签
3. `verify_dataset.py` - 验证数据集
4. `test_dataloader.py` - 测试 DataLoader
5. `train.sh` - 开始训练

### Q2: 如何检查数据集是否准备好？

**A**: 运行以下命令：
```bash
python scripts/verify_dataset.py --dataset Pohang-Canal-3k
```

如果显示 "🎉 数据集验证通过！"，则可以开始训练。

### Q3: LiDAR 点云太少怎么办？

**A**: 使用分类和过滤工具：
```bash
# 1. 调优阈值
python scripts/tune_classification_thresholds.py --dataset Pohang-Canal-3k

# 2. 分类图像
python scripts/classify_images.py --dataset Pohang-Canal-3k

# 3. 过滤低质量
python scripts/filter_dataset.py --dataset Pohang-Canal-3k --min_quality good
```

### Q4: 如何转换其他格式的标注？

**A**:
- YOLO → JSON: `yolo_to_json.py`
- JSON → Mask: `json_to_mask.py`

### Q5: 训练结果如何分析？

**A**: 使用 `analyze_training.py`：
```bash
python scripts/analyze_training.py --experiment Phase3_DualGeo_16bit
```

会生成损失曲线、IoU 曲线和训练报告。

---

## 脚本依赖关系图

```
数据准备流程：
generate_depth_maps.py
    ↓
generate_oracle_masks.py
    ↓
verify_dataset.py
    ↓
test_dataloader.py

数据集优化流程：
tune_classification_thresholds.py
    ↓
classify_images.py
    ↓
filter_dataset.py
    ↓
finalize_dataset.py

训练和分析流程：
train.sh
    ↓
analyze_training.py
    ↓
evaluate_golden_set.py

标注转换流程：
yolo_to_json.py
    ↓
json_to_mask.py
```

---

**相关文档**:
- [README.md](../README.md) - 项目概览
- [QUICKSTART.md](../QUICKSTART.md) - 快速开始
- [ADVANCED.md](ADVANCED.md) - 高级技术指南
