#!/bin/bash
# =============================================================================
# 多尺度Mamba模型试验训练脚本 (Pilot Training)
# =============================================================================
#
# 目的：验证多尺度特征融合 + Deep Supervision是否有效
#
# 策略：
#   1. 短期训练（50 epochs）快速验证方向
#   2. 如果Box IoU显示提升（>0.58），再启动完整训练
#
# 预期结果：
#   - 当前baseline（Plan C）: Box IoU = 0.574
#   - 乐观情况：Box IoU = 0.59-0.61 (+2-6%)
#   - 如果<0.58：说明方向有问题，需要调整
#
# 用法：
#   bash train_multiscale_pilot.sh [GPU_ID]
#
# 示例：
#   bash train_multiscale_pilot.sh 0  # 使用GPU 0
#
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ============================================================
# 参数配置
# ============================================================

GPU=${1:-0}  # 默认使用GPU 0
EPOCHS=50     # 短期验证：50 epochs

# 模型配置
MODEL_TYPE="mamba_tiny_multiscale"  # 新的多尺度模型
USE_LIDAR=True
USE_DEEP_SUPERVISION=True

# 数据配置
DATASET="Pohang-Canal-3k"
SPLIT_METHOD="50_50_2k_new"
BATCH_SIZE=4

# 训练配置
LR=0.0001
OPTIMIZER="AdamW"
SCHEDULER="CosineAnnealingWarmRestarts"  # 使用Plan C的调度器

# Loss配置（使用Plan C的配置）
LOSS_TYPE="hybrid"
DICE_WEIGHT=2.5
PROJECTION_WEIGHT=2.0

# 实验名称
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="MultiScale_PilotTest_${TIMESTAMP}"

# ============================================================
# 开始训练
# ============================================================

echo "=========================================="
echo "🚀 多尺度Mamba模型试验训练"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  GPU:              $GPU"
echo "  模型:             $MODEL_TYPE (with Deep Supervision)"
echo "  训练轮数:         $EPOCHS epochs (快速验证)"
echo "  学习率调度:       $SCHEDULER"
echo "  Loss类型:         $LOSS_TYPE (dice=$DICE_WEIGHT, proj=$PROJECTION_WEIGHT)"
echo "  实验名称:         $EXPERIMENT_NAME"
echo ""
echo "📊 预期结果:"
echo "  当前baseline:     Box IoU = 0.574"
echo "  目标:             Box IoU > 0.58 (提升>1%)"
echo "  如果达成:         启动完整训练 (500 epochs)"
echo ""
read -p "按回车键开始训练，或 Ctrl+C 取消..."

# 创建实验目录（Python 期望目录已存在）
mkdir -p "result/$EXPERIMENT_NAME"
echo "✓ 已创建实验目录: result/$EXPERIMENT_NAME"
echo ""

# 运行训练
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
    --save_interval 10

# ============================================================
# 训练完成后自动评估
# ============================================================

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo ""
echo "📁 结果保存在: result/$EXPERIMENT_NAME"
echo ""

# 查找最佳checkpoint
SAVE_DIR="result/$EXPERIMENT_NAME"
if [ -f "$SAVE_DIR/latest_best_model.pth" ]; then
    echo "🔍 正在评估最佳checkpoint..."
    echo ""

    # 运行Box IoU测试
    bash ../test_checkpoint.sh "$SAVE_DIR/latest_best_model.pth" $GPU

    echo ""
    echo "=========================================="
    echo "📊 评估完成"
    echo "=========================================="
    echo ""
    echo "📈 下一步行动建议:"
    echo ""
    echo "1. 查看 Box IoU 结果"
    echo "   - 如果 Box IoU > 0.58:  ✅ 方向正确，启动完整训练"
    echo "   - 如果 Box IoU < 0.58:  ⚠️  需要调整架构或超参数"
    echo ""
    echo "2. 启动完整训练（如果验证成功）:"
    echo "   bash scripts/train_multiscale_full.sh $GPU"
    echo ""
    echo "3. 对比置信度校准:"
    echo "   检查极端阈值占比是否改善（目标: <60%）"
    echo ""

else
    echo "⚠️  未找到最佳checkpoint，请检查训练日志"
fi
