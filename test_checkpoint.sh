#!/bin/bash
# ============================================================
# 测试已训练模型的 Mask-to-Box IoU
# ============================================================
#
# 🚀 新功能 (2026-02-08):
#   - 自动从 train_log.txt 读取训练配置
#   - 自动检测模型架构 (Progressive, MultiScale, Base)
#   - 自动配置数据加载器 (8-bit/16-bit, IR-only/IR+LiDAR)
#   - 只需提供权重文件路径即可运行测试！
#
# 用法:
#   bash test_checkpoint.sh <checkpoint_path> [gpu_id] [OPTIONS]
#
# 示例:
#   # 最简单的用法（推荐）- 自动配置一切
#   bash test_checkpoint.sh model_Mamba/result/Pohang-Canal-3k_mamba_tiny_progressive_20260207_213457/latest_best_model.pth
#
#   # 使用不同 GPU
#   bash test_checkpoint.sh result/DNANet_baseline_8bit_xxx/latest_best_model.pth.tar 1
#
#   # 自定义评估策略
#   bash test_checkpoint.sh result/xxx/best_model.pth.tar 0 --eval_strategy fixed --threshold 0.3
# ============================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 默认参数
CHECKPOINT=""
GPU=0
THRESHOLD=0.5
BATCH_SIZE=4
EVAL_STRATEGY="auto"

# 解析参数
if [ $# -lt 1 ]; then
    echo "❌ 错误: 缺少 checkpoint 路径"
    echo ""
    echo "用法: bash test_checkpoint.sh <checkpoint_path> [gpu_id] [OPTIONS]"
    echo ""
    echo "🚀 新功能: 自动从 train_log.txt 读取配置"
    echo "   模型架构、数据格式、加载器等参数会自动配置，无需手动指定！"
    echo ""
    echo "选项:"
    echo "  --threshold VALUE        固定阈值 (default: 0.5, 仅用于fixed策略)"
    echo "  --batch_size VALUE       批次大小 (default: 4)"
    echo "  --eval_strategy STRATEGY 评估策略 (auto|fixed|dynamic, default: auto)"
    echo "                          auto: 所有模型用动态阈值扫描（推荐）"
    echo "                          fixed: 使用固定阈值"
    echo "                          dynamic: 动态阈值扫描 [0.1-0.9]"
    echo ""
    echo "示例:"
    echo "  # 最简单用法（推荐）- 完全自动配置"
    echo "  bash test_checkpoint.sh model_Mamba/result/xxx/latest_best_model.pth"
    echo ""
    echo "  # 使用不同GPU"
    echo "  bash test_checkpoint.sh result/DNANet_xxx/latest_best_model.pth.tar 1"
    echo ""
    echo "  # 使用固定阈值策略"
    echo "  bash test_checkpoint.sh result/xxx.pth 0 --eval_strategy fixed --threshold 0.3"
    echo ""
    echo "  # 所有模型都用动态阈值（与训练一致）"
    echo "  bash test_checkpoint.sh result/xxx.pth 0 --eval_strategy dynamic"
    exit 1
fi

CHECKPOINT="$1"
shift  # 移除第一个参数

# 可选参数: GPU ID (位置参数)
if [[ $# -ge 1 && ! "$1" =~ ^-- ]]; then
    GPU="$1"
    shift
fi

# 解析命名参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --eval_strategy)
            EVAL_STRATEGY="$2"
            shift 2
            ;;
        *)
            echo "⚠️  未知参数: $1（忽略）"
            shift
            ;;
    esac
done

# 检查 checkpoint 是否存在
if [ ! -f "$SCRIPT_DIR/$CHECKPOINT" ]; then
    echo "❌ 错误: Checkpoint 文件不存在: $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "🧪 测试 Mask-to-Box IoU"
echo "=========================================="
echo "  Checkpoint: $CHECKPOINT"
echo "  GPU: $GPU"
echo "  Threshold: $THRESHOLD (如果使用固定阈值)"
echo "  Eval Strategy: $EVAL_STRATEGY"
echo ""
echo "📋 注意: 配置将从 train_log.txt 自动加载"
echo "=========================================="
echo ""

# 运行测试
# 大部分参数现在由 test_box_iou.py 从 train_log.txt 自动读取
cd "$SCRIPT_DIR"

python test_box_iou.py \
    --checkpoint "$CHECKPOINT" \
    --gpu "$GPU" \
    --threshold "$THRESHOLD" \
    --batch_size "$BATCH_SIZE" \
    --eval_strategy "$EVAL_STRATEGY"

echo ""
echo "=========================================="
echo "✅ 测试完成!"
echo "=========================================="
echo ""
echo "📊 结果已保存到:"
echo "  $(dirname "$CHECKPOINT")/box_iou_test_results.txt"
echo ""
echo "💡 提示: 所有配置已从 train_log.txt 自动加载"
echo ""
