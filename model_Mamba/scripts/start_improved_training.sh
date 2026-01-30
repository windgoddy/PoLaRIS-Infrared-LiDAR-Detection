#!/bin/bash
# 改进后的训练启动脚本
# 根据专业分析报告的建议进行了以下关键修复：
# 1. Head bias初始化为-2.19 (sigmoid ≈ 0.1)
# 2. Loss归一化使用max(1.0, num_pos)更稳健
# 3. 添加可视化调试，每个epoch保存预测图

echo "========================================"
echo "PoLaRIS-Mamba 改进训练"
echo "========================================"
echo ""
echo "📋 关键改进："
echo "  ✓ Head bias=-2.19 初始化（防止梯度爆炸）"
echo "  ✓ Loss归一化改用max(1.0, num_pos)（更稳定）"
echo "  ✓ 添加可视化调试（result/*/vis_debug/）"
echo "  ✓ Combined Loss (Focal + Dice) 平衡P/R"
echo "  ✓ Threshold=0.05 减少误检"
echo ""

# 1. 先检查初始化是否正确
echo "步骤 1/3: 验证模型初始化..."
# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_MAMBA_DIR="$(dirname "$SCRIPT_DIR")"
cd "$MODEL_MAMBA_DIR"
python3 scripts/check_initialization.py

if [ $? -ne 0 ]; then
    echo "❌ 初始化检查失败，请检查错误信息"
    exit 1
fi

echo ""
echo "步骤 2/3: 停止旧训练进程..."
pkill -f "model_Mamba/train.py" 2>/dev/null
sleep 2

echo ""
echo "步骤 3/3: 启动改进训练..."
echo ""

# 使用auto_train_gpu.sh自动选择GPU
bash scripts/auto_train_gpu.sh \
    --dataset Pohang-Canal-3k \
    --model mamba_tiny \
    --epochs 1000 \
    --batch_size 4 \
    --loss_type combined \
    --peak_threshold 0.05

echo ""
echo "========================================"
echo "训练已启动！"
echo "========================================"
echo ""
echo "📊 监控指南："
echo "  - 查看日志: tail -f training_mamba_*.log"
echo "  - 查看可视化: ls result/Pohang-Canal-3k/mamba_tiny/vis_debug/"
echo "  - Epoch 0 Loss应该<2.0（之前是4-5）"
echo "  - Epoch 1不应该出现Loss暴涨（之前涨到5.7）"
echo "  - IoU=0正常（因为阈值问题），看可视化图判断"
echo ""
