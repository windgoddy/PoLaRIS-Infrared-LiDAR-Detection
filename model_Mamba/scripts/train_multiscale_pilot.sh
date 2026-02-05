#!/bin/bash
# =============================================================================
# 多尺度Mamba模型试验训练脚本 (Pilot Training)
# 自动选择 GPU 和 batch size，OOM 时自动切换
# =============================================================================
#
# 用法：
#   bash train_multiscale_pilot.sh              # 自动选择GPU（按显存排序）
#   bash train_multiscale_pilot.sh 4            # 优先尝试GPU 4
#
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ============================================================
# 参数配置
# ============================================================

# 可选：手动指定起始GPU（如果提供命令行参数）
MANUAL_GPU="${1:-}"

EPOCHS=50

# 模型配置
MODEL_TYPE="mamba_tiny_multiscale"
USE_LIDAR=True

# 数据配置
DATASET="Pohang-Canal-3k"
SPLIT_METHOD="50_50_2k_new"

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
EXPERIMENT_NAME="MultiScale_Pilot_${TIMESTAMP}"

# 依次尝试不同的 batch_size（从大到小）
BATCH_SIZES=(4 2 1)

# ============================================================
# 优雅退出处理
# ============================================================

# 优雅退出处理
TRAIN_PID=""
TAIL_PID=""

cleanup() {
    echo ""
    echo "⚠️  收到中断信号，正在优雅停止训练..."

    if [ -n "$TAIL_PID" ] && ps -p $TAIL_PID > /dev/null 2>&1; then
        kill $TAIL_PID 2>/dev/null || true
        echo "  ✓ 已停止日志显示"
    fi

    if [ -n "$TRAIN_PID" ] && ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo "  ⏳ 正在等待训练进程保存 checkpoint（PID: $TRAIN_PID）..."
        kill -INT $TRAIN_PID 2>/dev/null || true

        for i in {1..30}; do
            if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
                echo "  ✓ 训练进程已优雅退出"
                exit 0
            fi
            sleep 1
        done

        echo "  ⚠️  超时，强制终止..."
        kill -9 $TRAIN_PID 2>/dev/null || true
    fi

    echo "  ✓ 清理完成"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================================
# 开始执行
# ============================================================

echo "=========================================="
echo "🚀 多尺度Mamba模型试验训练（自动模式）"
echo "=========================================="
echo ""
echo "💡 提示: 按 Ctrl+C 可优雅停止训练（会保存 checkpoint）"
echo ""

# 显示使用说明
if [ -n "$MANUAL_GPU" ]; then
    echo "⚙️  手动模式: 从 GPU $MANUAL_GPU 开始尝试"
    echo ""
fi

echo "📋 配置信息:"
echo "  模型:             $MODEL_TYPE (with Deep Supervision)"
echo "  训练轮数:         $EPOCHS epochs (快速验证)"
echo "  学习率调度:       $SCHEDULER"
echo "  Loss类型:         $LOSS_TYPE (dice=$DICE_WEIGHT, proj=$PROJECTION_WEIGHT)"
echo "  实验名称:         $EXPERIMENT_NAME"
echo ""
echo "📊 预期结果:"
echo "  当前baseline:     Box IoU = 0.574"
echo "  目标:             Box IoU > 0.58 (提升>1%)"
echo ""

# ============================================================
# 获取可用 GPU 列表
# ============================================================

echo "📌 检测可用 GPU..."

if [ -n "$MANUAL_GPU" ]; then
    echo "  ⚙️  用户指定从 GPU $MANUAL_GPU 开始尝试"
    # 先尝试用户指定的GPU，然后尝试其他GPU
    OTHER_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | awk '{print $1}' | tr -d ',' | grep -v "^${MANUAL_GPU}$" || true)
    GPU_LIST="$MANUAL_GPU $OTHER_GPUS"
else
    # 自动按显存从大到小排序
    GPU_LIST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | awk '{print $1}' | tr -d ',')
fi

echo "GPU 显存状态（从大到小）:"
nvidia-smi --query-gpu=index,name,memory.free,memory.used --format=csv,noheader
echo ""

# ============================================================
# 依次尝试 GPU 和 batch size
# ============================================================

for GPU_ID in $GPU_LIST; do
    echo "=========================================="
    echo "🔄 尝试 GPU $GPU_ID"
    echo "=========================================="

    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        echo ""
        echo "📝 配置: GPU=$GPU_ID, Batch Size=$BATCH_SIZE"

        # 创建实验目录
        SAVE_DIR="result/$EXPERIMENT_NAME"
        mkdir -p "$SAVE_DIR"

        # 生成日志文件名
        LOG_FILE="$SAVE_DIR/training_gpu${GPU_ID}_bs${BATCH_SIZE}_${TIMESTAMP}.log"

        echo "📁 实验目录: $SAVE_DIR"
        echo "📝 日志文件: $LOG_FILE"
        echo ""
        echo "🚀 启动训练..."

        # 启动训练
        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
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
            --save_dir "$SAVE_DIR" \
            --save_interval 10 \
            --gpus 0 \
            > $LOG_FILE 2>&1 &

        TRAIN_PID=$!
        echo "进程 PID: $TRAIN_PID"
        echo ""

        # 监控训练启动（最多等待 300 秒）
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
                if tail -100 $LOG_FILE 2>/dev/null | grep -q "OutOfMemoryError"; then
                    echo ""
                    echo "❌ GPU $GPU_ID (Batch Size $BATCH_SIZE) OOM，尝试下一个配置..."
                    break
                else
                    # 其他错误
                    echo ""
                    echo "❌ 训练失败，错误信息："
                    tail -20 $LOG_FILE
                    break
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
                echo "📋 配置信息:"
                echo "  GPU:              $GPU_ID"
                echo "  Batch Size:       $BATCH_SIZE"
                echo "  实验目录:         $SAVE_DIR"
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
                echo "📊 训练完成后，运行以下命令评估结果:"
                echo "  bash ../test_checkpoint.sh \"$SAVE_DIR/latest_best_model.pth\" $GPU_ID"
                echo ""

                exit 0
            fi
        done

        # 如果成功启动，退出脚本
        if [ "$SUCCESS" = true ]; then
            exit 0
        fi

        # 清理失败的进程
        if ps -p $TRAIN_PID > /dev/null 2>&1; then
            echo "清理失败的训练进程 $TRAIN_PID..."
            kill $TRAIN_PID 2>/dev/null || true
            sleep 2
        fi
    done
done

# 所有 GPU 和 batch size 都失败了
echo ""
echo "=========================================="
echo "❌ 所有 GPU 配置都失败了"
echo "=========================================="
echo ""
echo "建议："
echo "  1. 检查数据集路径是否正确"
echo "  2. 检查最新的日志文件:"
echo "     ls -lht result/$EXPERIMENT_NAME/*.log | head -5"
echo "  3. 考虑减小模型或图像分辨率"
echo ""
exit 1
