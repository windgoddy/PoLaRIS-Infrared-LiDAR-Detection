#!/bin/bash
# ./scripts/run_fix_dataset.sh
# 修复 Pohang-Canal-all 数据集的图像和掩码问题

echo "================================================================"
echo "🔧 数据集修复脚本 - Pohang-Canal-all"
echo "================================================================"
echo ""
echo "此脚本将修复以下问题："
echo "  1. 图像几乎全黑 → 将软链接替换为实际文件复制"
echo "  2. 掩码全黑 → 重新生成 YOLO 格式的掩码"
echo ""
echo "预计耗时: 5-10 分钟（取决于数据量）"
echo ""
echo "================================================================"
echo ""

# 询问用户是否继续
read -p "是否继续？(y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ 已取消"
    exit 1
fi

# 运行修复脚本
echo ">>> 开始修复..."
echo ""
python scripts/fix_images_and_masks.py

# 检查是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 修复失败，请检查错误信息"
    exit 1
fi

echo ""
echo "================================================================"
echo "✅ 修复完成！"
echo "================================================================"
echo ""
echo "📝 下一步:"
echo "  1. 验证结果:"
echo "     python scripts/diagnose_dataset.py"
echo ""
echo "  2. 开始训练:"
echo "     ./scripts/run_Phase3_improved_v3.sh"
echo ""
echo "================================================================"
