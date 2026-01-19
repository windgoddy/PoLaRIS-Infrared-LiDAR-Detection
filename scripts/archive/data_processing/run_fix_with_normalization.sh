#!/bin/bash
# ./scripts/run_fix_with_normalization.sh
# 修复 Pohang-Canal-all 数据集（包含 16-bit → 8-bit 归一化）

echo "================================================================"
echo "🔧 数据集修复脚本 - 包含 16-bit → 8-bit 归一化"
echo "================================================================"
echo ""
echo "此脚本将修复以下问题："
echo "  1. ✨ 16-bit 红外图像 → 归一化到 8-bit (0-255)"
echo "  2. 🎭 重新生成 YOLO 格式的掩码"
echo ""
echo "原理："
echo "  - 红外探测器输出 16-bit 数据 (0-65535)"
echo "  - 实际像素值集中在很窄范围（如 2000-3000）"
echo "  - 使用百分位数归一化拉伸到 0-255"
echo "  - 得到正常对比度的 8-bit 图像"
echo ""
echo "预计耗时: 5-10 分钟"
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
python scripts/fix_dataset_with_normalization.py

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
