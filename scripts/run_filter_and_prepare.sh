#!/bin/bash
# ./scripts/run_filter_and_prepare.sh
# 数据集清洗和标签准备 - 一键运行脚本

echo "================================================================"
echo "数据集清洗和标签准备 - Pohang-Canal-all"
echo "================================================================"
echo ""
echo "此脚本将执行以下操作："
echo "  1. 过滤没有标签的图像（更新 train.txt 和 test.txt）"
echo "  2. 为有效数据生成二值掩码"
echo ""
echo "优势："
echo "  - 不需要重新运行 prepare_all_data.py（节省时间）"
echo "  - 不删除物理文件，只更新索引"
echo "  - 执行速度快（预计 2-5 分钟）"
echo ""
echo "================================================================"
echo ""

# 步骤 1: 过滤数据集
echo ">>> 步骤 1/2: 过滤数据集（移除无标签图像）"
echo ""
python scripts/filter_dataset.py

# 检查是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 数据集过滤失败，请检查错误信息"
    exit 1
fi

echo ""
echo "================================================================"
echo ""

# 步骤 2: 生成 masks
echo ">>> 步骤 2/2: 生成二值掩码 (YOLO 格式)"
echo ""
python scripts/prepare_labels_yolo.py

# 检查是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 标签准备失败，请检查错误信息"
    exit 1
fi

echo ""
echo "================================================================"
echo "✅ 全部完成！"
echo "================================================================"
echo ""
echo "📁 数据集结构:"
echo "  dataset/Pohang-Canal-all/"
echo "    ├── images/          # 红外图像"
echo "    ├── lidar_roi/       # 点云"
echo "    ├── depth_maps/      # 深度图"
echo "    ├── masks/           # ✨ 二值掩码（新生成）"
echo "    ├── calibration/     # 标定文件"
echo "    └── split_data/      # ✨ 数据集划分（已更新）"
echo "        ├── train.txt    # ✨ 过滤后的训练集"
echo "        └── test.txt     # ✨ 过滤后的测试集"
echo ""
echo "📝 下一步:"
echo "  1. (可选) 生成 oracle masks:"
echo "     python scripts/generate_oracle_masks.py"
echo ""
echo "  2. 开始训练:"
echo "     ./scripts/run_Phase3_improved_v3.sh"
echo ""
echo "================================================================"
