# PoLaRIS 文档索引

> 快速找到您需要的文档和资源

---

## 📚 文档导航

### 新手入门

1. **[README.md](../README.md)** - 从这里开始 ⭐
   - 项目简介和特点
   - 快速安装和训练
   - 训练模式对比表
   - 常用命令速查

2. **[QUICKSTART.md](../QUICKSTART.md)** - 详细教程
   - 一分钟快速开始
   - 完整工作流程（数据准备 → 训练 → 分析）
   - 常见问题 FAQ
   - 整理成果展示

### 配置和训练

3. **[run_config.sh](../run_config.sh)** - 完整配置文件 🔧
   - 所有可用的训练参数
   - 5 个预定义配置模板
   - 自定义配置示例
   - 详细的参数说明

4. **[scripts/train.sh](../scripts/train.sh)** - 统一训练脚本
   - 自动模式选择
   - 多种训练模式
   - 简洁的命令行接口

### 技术文档

5. **[docs/ADVANCED.md](ADVANCED.md)** - 高级技术指南 📖
   - DataLoader 核心特性
   - 16-bit 图像处理详解
   - 软标签技术细节
   - LiDAR 点云处理
   - 性能优化技巧
   - 调试方法

6. **[docs/SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)** - Scripts 完全手册 📝
   - 所有 21 个脚本的详细说明
   - 输入输出格式规范
   - 相似脚本对比
   - 推荐工作流程
   - 脚本依赖关系图

---

## 🎯 按需求查找

### 我想了解项目

- 项目是什么？→ [README.md](../README.md)
- 有什么特点？→ [README.md - 项目特点](../README.md#项目特点)
- 如何快速开始？→ [README.md - 快速开始](../README.md#快速开始)

### 我想开始训练

- 快速训练命令？→ [QUICKSTART.md - 一分钟快速开始](../QUICKSTART.md#🚀-一分钟快速开始)
- 完整训练流程？→ [QUICKSTART.md - 完整工作流程](../QUICKSTART.md#📚-完整工作流程)
- 自定义配置？→ [run_config.sh](../run_config.sh)
- 训练模式选择？→ [README.md - 训练模式](../README.md#训练模式)

### 我想准备数据

- 数据集结构？→ [README.md - 数据集结构](../README.md#数据集结构)
- 生成深度图？→ [SCRIPTS_GUIDE.md - generate_depth_maps.py](SCRIPTS_GUIDE.md#generate_depth_mapspy)
- 生成软标签？→ [SCRIPTS_GUIDE.md - generate_oracle_masks.py](SCRIPTS_GUIDE.md#generate_oracle_maskspy)
- 验证数据集？→ [SCRIPTS_GUIDE.md - verify_dataset.py](SCRIPTS_GUIDE.md#verify_datasetpy)

### 我想了解技术细节

- 16-bit 如何处理？→ [ADVANCED.md - 16-bit 图像处理](ADVANCED.md#16-bit-图像处理)
- 软标签是什么？→ [ADVANCED.md - 软标签](ADVANCED.md#软标签oracle-masks)
- LiDAR 如何使用？→ [ADVANCED.md - LiDAR 点云处理](ADVANCED.md#lidar-点云处理)
- DataLoader 详解？→ [ADVANCED.md - 训练配置](ADVANCED.md#训练配置)

### 我想找特定脚本

- 所有脚本列表？→ [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md#目录)
- 数据准备脚本？→ [SCRIPTS_GUIDE.md - 数据准备](SCRIPTS_GUIDE.md#数据准备)
- 数据分析脚本？→ [SCRIPTS_GUIDE.md - 数据分析和可视化](SCRIPTS_GUIDE.md#数据分析和可视化)
- 相似脚本对比？→ [SCRIPTS_GUIDE.md - 相似脚本对比](SCRIPTS_GUIDE.md#相似脚本对比)

### 我遇到了问题

- 常见问题？→ [QUICKSTART.md - 常见问题](../QUICKSTART.md#❓-常见问题)
- 技术问题？→ [ADVANCED.md - 常见问题](ADVANCED.md#常见问题)
- 脚本问题？→ [SCRIPTS_GUIDE.md - 常见问题](SCRIPTS_GUIDE.md#常见问题)
- 调试技巧？→ [ADVANCED.md - 调试技巧](ADVANCED.md#调试技巧)

---

## 🗂️ 文档结构总览

```
PoLaRIS-Infrared-LiDAR-Detection/
├── README.md                    # 项目主页（入口）
├── QUICKSTART.md                # 快速开始指南
├── run_config.sh                # 完整配置文件
├── scripts/
│   ├── train.sh                 # 统一训练脚本
│   ├── generate_depth_maps.py   # 生成深度图
│   ├── generate_oracle_masks.py # 生成软标签
│   ├── test_dataloader.py       # 测试 DataLoader
│   └── ... (其他 17 个脚本)
└── docs/
    ├── INDEX.md                 # 本文档（索引）
    ├── ADVANCED.md              # 高级技术指南
    └── SCRIPTS_GUIDE.md         # Scripts 完全手册
```

---

## 📖 推荐阅读路径

### 路径 1: 新手快速上手

```
README.md (5 分钟)
    ↓
QUICKSTART.md (15 分钟)
    ↓
开始训练！
```

### 路径 2: 自定义配置

```
README.md (了解项目)
    ↓
run_config.sh (查看所有参数)
    ↓
修改配置 → 开始训练
```

### 路径 3: 数据准备

```
QUICKSTART.md - 数据准备部分
    ↓
SCRIPTS_GUIDE.md - 数据准备脚本
    ↓
运行脚本 → 验证数据集
```

### 路径 4: 深入学习

```
README.md (概览)
    ↓
ADVANCED.md (技术细节)
    ↓
SCRIPTS_GUIDE.md (脚本详解)
    ↓
自定义开发
```

---

## 🔍 快速查询表

### 命令速查

| 任务 | 命令 | 文档 |
| ---- | ---- | ---- |
| 训练模型 | `./scripts/train.sh auto --dataset Pohang-Canal-3k` | [QUICKSTART.md](../QUICKSTART.md) |
| 生成深度图 | `python scripts/generate_depth_maps.py --dataset <name>` | [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) |
| 生成软标签 | `python scripts/generate_oracle_masks.py --dataset <name>` | [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) |
| 验证数据集 | `python scripts/verify_dataset.py --dataset <name>` | [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) |
| 测试 DataLoader | `python scripts/test_dataloader.py` | [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) |
| 分析训练结果 | `python scripts/analyze_training.py --experiment <name>` | [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) |

### 参数速查

| 参数 | 说明 | 可选值 | 文档 |
| ---- | ---- | ------ | ---- |
| `--dataset` | 数据集名称 | Pohang-Canal, Pohang-Canal-3k, NUDT-SIRST, 等 | [run_config.sh](../run_config.sh) |
| `--model` | 模型名称 | MS_CAFNet_DualGeo, DNANet, 等 | [run_config.sh](../run_config.sh) |
| `--in_channels` | 输入通道数 | 1 (仅 IR), 2 (IR+Depth) | [run_config.sh](../run_config.sh) |
| `--optimizer` | 优化器 | Adam, Adagrad, SGD | [run_config.sh](../run_config.sh) |
| `--use_lidar_dataloader` | 使用新 DataLoader | True, False | [ADVANCED.md](ADVANCED.md) |
| `--use_soft_labels` | 使用软标签 | True, False | [ADVANCED.md](ADVANCED.md) |

### 文件格式速查

| 文件类型 | 格式 | 说明 | 文档 |
| -------- | ---- | ---- | ---- |
| 红外图像 | `.png` | 8-bit 或 16-bit | [ADVANCED.md](ADVANCED.md) |
| GT Mask | `.png` | 单通道，0=背景，255=目标 | [README.md](../README.md) |
| Oracle Mask | `.png` | 单通道，0/153/255 | [ADVANCED.md](ADVANCED.md) |
| 深度图 | `.npy` | numpy 数组，(H, W), float32 | [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) |
| LiDAR 点云 | `.bin` | float32 二进制，(N, 4) | [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) |
| 标定文件 | `.json` | 相机-LiDAR 标定参数 | [ADVANCED.md](ADVANCED.md) |

---

## 🆘 获取帮助

### 文档内查找

使用 `grep` 命令快速查找：

```bash
# 在所有文档中查找关键词
grep -r "深度图" docs/ *.md

# 查找特定脚本的说明
grep -A 20 "generate_depth_maps.py" docs/SCRIPTS_GUIDE.md

# 查找参数说明
grep "in_channels" run_config.sh docs/ADVANCED.md
```

### 按主题查找

| 主题 | 推荐文档 |
| ---- | -------- |
| 入门教程 | [QUICKSTART.md](../QUICKSTART.md) |
| 训练配置 | [run_config.sh](../run_config.sh) |
| DataLoader | [ADVANCED.md](ADVANCED.md) |
| 数据准备 | [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) |
| 问题排查 | [ADVANCED.md - 调试技巧](ADVANCED.md#调试技巧) |
| 性能优化 | [ADVANCED.md - 性能优化](ADVANCED.md#性能优化) |

---

## 📊 文档统计

| 文档 | 行数 | 主要内容 | 适合人群 |
| ---- | ---- | -------- | -------- |
| README.md | ~242 | 项目概览、快速开始 | 所有用户 |
| QUICKSTART.md | ~275 | 详细教程、FAQ | 新用户 |
| run_config.sh | ~600 | 完整配置、参数说明 | 配置人员 |
| docs/ADVANCED.md | ~350 | 技术细节、优化 | 高级用户 |
| docs/SCRIPTS_GUIDE.md | ~1000 | 脚本详解、工作流 | 开发者 |
| **总计** | **~2467** | **完整文档体系** | **全覆盖** |

---

## 🔗 相关资源

### 项目资源

- GitHub 仓库: [PoLaRIS](https://github.com/your-repo)
- 原始 DNANet: [DNANet GitHub](https://github.com/YeRen123455/Infrared-Small-Target-Detection)
- DNANet 论文: [IEEE TIP 2023](https://arxiv.org/pdf/2106.00487.pdf)

### 依赖项目

- ACM: [open-acm](https://github.com/YimianDai/open-acm)
- PSA: [psa](https://github.com/jiwoon-ahn/psa)

---

## ✅ 文档更新记录

| 日期 | 更新内容 |
| ---- | -------- |
| 2026-01 | 创建完整文档体系（5 个文档） |
| 2026-01 | 添加 run_config.sh 配置文件 |
| 2026-01 | 添加 SCRIPTS_GUIDE.md 脚本手册 |
| 2026-01 | 更新 README.md 为 PoLaRIS 项目 |
| 2026-01 | 创建文档索引 INDEX.md |

---

**开始探索吧！** 🚀

建议从 [README.md](../README.md) 开始，然后根据需要查阅其他文档。
