#!/bin/bash
# 多尺度Mamba模型快速训练脚本
# 使用方法: bash scripts/train_multiscale_simple.sh [GPU_ID]
# Ctrl+C: 优雅停止训练（保存 checkpoint）

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_MAMBA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$MODEL_MAMBA_DIR")"

echo "📂 Directories:"
echo "  Script:       $SCRIPT_DIR"
echo "  model_Mamba:  $MODEL_MAMBA_DIR"
echo "  Project root: $PROJECT_ROOT"
echo ""

# 优雅退出处理
TRAIN_PID=""
TAIL_PID=""

cleanup() {
    echo ""
    echo "⚠️  收到中断信号，正在优雅停止训练..."

    # 停止 tail
    if [ -n "$TAIL_PID" ] && ps -p $TAIL_PID > /dev/null 2>&1; then
        kill $TAIL_PID 2>/dev/null || true
        echo "  ✓ 已停止日志显示"
    fi

    # 向训练进程发送 SIGINT（让它有机会保存 checkpoint）
    if [ -n "$TRAIN_PID" ] && ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo "  ⏳ 正在等待训练进程保存 checkpoint（PID: $TRAIN_PID）..."
        kill -INT $TRAIN_PID 2>/dev/null || true

        # 等待最多 30 秒让它保存
        for i in {1..30}; do
            if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
                echo "  ✓ 训练进程已优雅退出"
                exit 0
            fi
            sleep 1
        done

        # 如果 30 秒后还没退出，强制终止
        echo "  ⚠️  超时，强制终止..."
        kill -9 $TRAIN_PID 2>/dev/null || true
    fi

    echo "  ✓ 清理完成"
    exit 0
}

# 捕获 Ctrl+C (SIGINT) 和 SIGTERM
trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "PoLaRIS-Mamba 多尺度模型训练"
echo "=========================================="
echo ""
echo "💡 提示: 按 Ctrl+C 可优雅停止训练（会保存 checkpoint）"
echo ""

# ============================================================
# 参数配置
# ============================================================

GPU=${1:-0}
EPOCHS=50
BATCH_SIZE=4

# 模型配置
MODEL="mamba_tiny_multiscale"
DATASET="Pohang-Canal-3k"
SPLIT_METHOD="50_50_2k_new"

# 训练配置
LR=0.0001
OPTIMIZER="AdamW"
SCHEDULER="CosineAnnealingWarmRestarts"

# Loss配置（使用Plan C的配置）
LOSS_TYPE="hybrid"
DICE_WEIGHT=2.5
PROJECTION_WEIGHT=2.0

# 实验名称
DT_STRING=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="MultiScale_Pilot_${DT_STRING}"

echo "📋 配置信息:"
echo "  GPU:              $GPU"
echo "  模型:             $MODEL (with Deep Supervision)"
echo "  训练轮数:         $EPOCHS epochs (~2-3小时)"
echo "  Batch Size:       $BATCH_SIZE"
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
echo ""

# ============================================================
# 创建实验目录
# ============================================================

DIR_NAME="${EXPERIMENT_NAME}_${DATASET}_${MODEL}_${DT_STRING}"
SAVE_DIR="$PROJECT_ROOT/model_Mamba/result/$DIR_NAME"
mkdir -p "$SAVE_DIR"

echo "📁 实验目录: $SAVE_DIR"
LOG_FILE="$SAVE_DIR/training_gpu${GPU}_${DT_STRING}.log"
echo "📝 日志文件: $LOG_FILE"
echo ""

# ============================================================
# 启动训练
# ============================================================

echo "🚀 启动训练..."
echo ""

CUDA_VISIBLE_DEVICES=$GPU python "$MODEL_MAMBA_DIR/train.py" \
    --root "$PROJECT_ROOT/dataset" \
    --dataset "$DATASET" \
    --split_method "$SPLIT_METHOD" \
    --model "$MODEL" \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --optimizer "$OPTIMIZER" \
    --scheduler "$SCHEDULER" \
    --loss_type "$LOSS_TYPE" \
    --dice_weight $DICE_WEIGHT \
    --projection_weight $PROJECTION_WEIGHT \
    --experiment_name "$EXPERIMENT_NAME" \
    --save_dir "$SAVE_DIR" \
    --save_interval 10 \
    --gpus 0 \
    > $LOG_FILE 2>&1 &

TRAIN_PID=$!
echo "进程 PID: $TRAIN_PID"
echo ""

# ============================================================
# 监控训练启动
# ============================================================

echo "⏳ 监控训练启动中（最多等待 300 秒以确认无 OOM）..."
echo ""

SUCCESS=false
EPOCH_STARTED=false
TAIL_PID=""

for i in {1..300}; do
    sleep 1

    # 检查进程是否还在运行
    if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
        # 停止 tail 进程
        if [ -n "$TAIL_PID" ]; then
            kill $TAIL_PID 2>/dev/null || true
        fi

        # 进程已退出，检查是否是 OOM
        if tail -100 $LOG_FILE | grep -q "OutOfMemoryError"; then
            echo ""
            echo "❌ GPU $GPU (Batch Size $BATCH_SIZE) OOM"
            echo ""
            echo "建议："
            echo "  1. 降低 batch size 到 2 或 1"
            echo "  2. 重新运行: bash scripts/train_multiscale_simple.sh $GPU"
            echo ""
            tail -20 $LOG_FILE
            exit 1
        else
            # 其他错误
            echo ""
            echo "❌ 训练失败，错误信息："
            tail -20 $LOG_FILE
            exit 1
        fi
    fi

    # 检查是否出现 Epoch 0，如果是则立即开始显示日志
    if [ "$EPOCH_STARTED" = false ] && tail -20 $LOG_FILE 2>/dev/null | grep -q "Epoch 0:.*%"; then
        EPOCH_STARTED=true
        echo ""
        echo "✓ Epoch 0 已开始，开始实时显示训练日志..."
        echo "  (继续监控 1 分钟以确认稳定性)"
        echo "=========================================="

        # 在后台启动 tail -f，输出到终端
        tail -f $LOG_FILE &
        TAIL_PID=$!
    fi

    # 如果已经开始训练且运行超过 1 分钟（60秒），认为稳定
    if [ "$EPOCH_STARTED" = true ] && [ $i -ge 60 ]; then
        SUCCESS=true
        echo ""
        echo "=========================================="
        echo "✅ 训练稳定运行 1 分钟，确认成功！"
        echo "=========================================="
        echo ""
        echo "按 Ctrl+C 退出日志查看（训练继续后台运行）"
        echo ""

        # 等待 tail 进程（让它继续在前台显示）
        wait $TAIL_PID 2>/dev/null || true

        # 如果用户 Ctrl+C 退出，显示后续监控命令
        echo ""
        echo "=========================================="
        echo "训练仍在后台运行中"
        echo "=========================================="
        echo ""
        echo "📊 重新查看训练进度:"
        echo "  tail -f \"$LOG_FILE\""
        echo ""
        echo "🛑 停止训练:"
        echo "  kill $TRAIN_PID"
        echo ""
        echo "📈 查看 GPU 使用:"
        echo "  watch -n 1 nvidia-smi"
        echo ""
        echo "📁 实验目录:"
        echo "  $SAVE_DIR"
        echo ""

        exit 0
    fi
done

# 如果 300 秒后还没开始训练
if [ "$EPOCH_STARTED" = false ]; then
    echo ""
    echo "❌ 训练在 5 分钟内未启动，可能遇到问题"
    echo ""
    echo "最新日志:"
    tail -50 $LOG_FILE
    echo ""

    # 清理进程
    if ps -p $TRAIN_PID > /dev/null 2>&1; then
        kill $TRAIN_PID 2>/dev/null || true
    fi

    exit 1
fi

exit 0
