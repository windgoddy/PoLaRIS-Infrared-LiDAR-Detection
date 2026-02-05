#!/bin/bash
# =============================================================================
# 多尺度Mamba模型完整训练脚本 (Full Training)
# =============================================================================
#
# ⚠️ 重要：仅在pilot training成功后使用此脚本
#
# 前置条件：
#   - pilot training (50 epochs) 的 Box IoU > 0.58
#   - 极端阈值占比有改善迹象
#
# 训练策略：
#   - 500 epochs（与Plan C对齐）
#   - CosineAnnealingWarmRestarts (T_0=50, 10次restart)
#   - Hybrid Loss (dice=2.5, proj=2.0)
#   - Deep Supervision enabled
#
# 预期结果：
#   - Box IoU: 0.60-0.64 (vs 0.574 baseline)
#   - 极端阈值: <65% (vs 66% baseline)
#
# 用法：
#   bash train_multiscale_full.sh [GPU_ID]
#
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ============================================================
# 参数配置
# ============================================================

GPU=${1:-0}
EPOCHS=500  # 完整训练

# 模型配置
MODEL_TYPE="mamba_tiny_multiscale"
USE_LIDAR=True
USE_DEEP_SUPERVISION=True

# 数据配置
DATASET="Pohang-Canal-3k"
SPLIT_METHOD="50_50_2k_new"
BATCH_SIZE=4

# 训练配置
LR=0.0001
OPTIMIZER="AdamW"
SCHEDULER="CosineAnnealingWarmRestarts"

# Loss配置
LOSS_TYPE="hybrid"
DICE_WEIGHT=2.5
PROJECTION_WEIGHT=2.0

# 实验名称
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="MultiScale_Full_d2.5p2_${TIMESTAMP}"

# ============================================================
# 警告和确认
# ============================================================

echo "=========================================="
echo "⚠️  多尺度Mamba模型完整训练"
echo "=========================================="
echo ""
echo "⚠️  此训练将消耗大量时间和计算资源"
echo ""
echo "配置信息:"
echo "  GPU:              $GPU"
echo "  模型:             $MODEL_TYPE (with Deep Supervision)"
echo "  训练轮数:         $EPOCHS epochs (~24-36小时)"
echo "  实验名称:         $EXPERIMENT_NAME"
echo ""
echo "📋 训练前检查清单:"
echo "  ✓ Pilot training的Box IoU > 0.58?"
echo "  ✓ GPU显存充足 (建议16GB+)?"
echo "  ✓ 磁盘空间充足 (需要~10GB)?"
echo "  ✓ 训练期间不会断电?"
echo ""
read -p "确认所有条件满足后，按回车键开始训练，或 Ctrl+C 取消..."

# ============================================================
# 开始训练
# ============================================================

echo ""
echo "🚀 开始训练..."
echo ""

# 创建实验目录（Python 期望目录已存在）
mkdir -p "result/$EXPERIMENT_NAME"
echo "✓ 已创建实验目录: result/$EXPERIMENT_NAME"
echo ""

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --root "../dataset" \
    --model "$MODEL_TYPE" \
    --dataset "$DATASET" \
    --split_method "$SPLIT_METHOD" \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --optimizer "$OPTIMIZER" \
    --scheduler "$SCHEDULER" \
    --loss_type "$LOSS_TYPE" \
    --dice_weight $DICE_WEIGHT \
    --projection_weight $PROJECTION_WEIGHT \
    --use_lidar $USE_LIDAR \
    --experiment_name "$EXPERIMENT_NAME" \
    --save_dir "result/$EXPERIMENT_NAME" \
    --save_interval 50

# ============================================================
# 训练完成
# ============================================================

echo ""
echo "=========================================="
echo "✅ 完整训练完成！"
echo "=========================================="
echo ""
echo "📁 结果保存在: result/$EXPERIMENT_NAME"
echo ""

SAVE_DIR="result/$EXPERIMENT_NAME"

if [ -f "$SAVE_DIR/latest_best_model.pth" ]; then
    echo "🔍 评估最佳checkpoint..."
    echo ""

    # Box IoU测试
    bash ../test_checkpoint.sh "$SAVE_DIR/latest_best_model.pth" $GPU

    echo ""
    echo "=========================================="
    echo "📊 最终结果分析"
    echo "=========================================="
    echo ""
    echo "📈 对比分析建议:"
    echo ""
    echo "1. Box IoU对比:"
    echo "   - Baseline (Plan C):      0.574"
    echo "   - MultiScale (期望):      0.60-0.64"
    echo "   - 实际结果:               查看上方输出"
    echo ""
    echo "2. 置信度校准对比:"
    echo "   - Baseline (Plan C):      66.2% 极端阈值"
    echo "   - MultiScale (期望):      <65%"
    echo "   - 实际结果:               查看上方'Best threshold distribution'"
    echo ""
    echo "3. 固定阈值鲁棒性测试:"
    echo "   bash ../compare_threshold_robustness.sh \\"
    echo "     --mamba result/$EXPERIMENT_NAME/latest_best_model.pth \\"
    echo "     --gpu $GPU"
    echo ""
    echo "4. 论文撰写:"
    echo "   - 参考 SCIENTIFIC_STORY_PLAN.md"
    echo "   - 重点突出多尺度融合和Deep Supervision的贡献"
    echo "   - Category-based分析（Cat1/2/3性能差异）"
    echo ""

else
    echo "⚠️  未找到最佳checkpoint"
fi
