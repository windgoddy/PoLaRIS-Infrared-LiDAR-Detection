#!/bin/bash
# 清理旧的、不再需要的脚本文件

echo "========================================================================"
echo "🧹 清理旧脚本文件"
echo "========================================================================"
echo ""
echo "将要删除以下文件："
echo ""
echo "Python 脚本:"
echo "  - prepare_labels_for_all.py (JSON 格式，不需要)"
echo "  - fix_images_and_masks.py (旧版修复脚本，无归一化)"
echo ""
echo "Shell 脚本:"
echo "  - run_filter_and_prepare.sh"
echo "  - run_fix_dataset.sh"
echo "  - run_prepare_all_data.sh"
echo ""
echo "保留的文件:"
echo "  ✅ prepare_dataset_complete.py - 新的一键式完整准备"
echo "  ✅ run_prepare_dataset.sh - 新的运行脚本"
echo "  ✅ normalize_infrared_images.py - 独立归一化工具"
echo "  ✅ fix_dataset_with_normalization.py - 修复已有数据（应急）"
echo "  ✅ prepare_all_data.py - 旧脚本（备份）"
echo "  ✅ filter_dataset.py - 旧脚本（备份）"
echo "  ✅ prepare_labels_yolo.py - 旧脚本（备份）"
echo ""
echo "========================================================================"
echo ""

read -p "确认删除？(y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ 已取消"
    exit 1
fi

# 删除 Python 脚本
rm -f scripts/prepare_labels_for_all.py
rm -f scripts/fix_images_and_masks.py

# 删除 Shell 脚本
rm -f scripts/run_filter_and_prepare.sh
rm -f scripts/run_fix_dataset.sh
rm -f scripts/run_prepare_all_data.sh

echo ""
echo "✅ 清理完成！"
echo ""
echo "当前脚本结构："
echo "  scripts/"
echo "    ├── prepare_dataset_complete.py  ⭐ 主脚本"
echo "    ├── run_prepare_dataset.sh       ⭐ 运行脚本"
echo "    ├── normalize_infrared_images.py  🔧 工具"
echo "    ├── fix_dataset_with_normalization.py  🔧 修复工具"
echo "    └── [其他工具脚本]"
echo ""
echo "========================================================================"
