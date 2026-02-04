#!/bin/bash
# 自动选择 GPU 并启动训练，OOM 时自动切换到下一个 GPU
# 使用方法: bash scripts/auto_train_gpu.sh
# Ctrl+C: 优雅停止训练（保存 checkpoint）

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 推导出 model_Mamba 目录的绝对路径
MODEL_MAMBA_DIR="$(dirname "$SCRIPT_DIR")"
# 推导出项目根目录的绝对路径
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

echo "========================================"
echo "PoLaRIS-Mamba 自动 GPU 训练启动器"
echo "========================================"
echo ""
echo "💡 提示: 按 Ctrl+C 可优雅停止训练（会保存 checkpoint）"
echo ""

# ============================================================
# 🔧 实验配置选择 (Updated 2026-02-04)
# ============================================================
# 选择预设配置：
#   - "PLAN_B": WarmRestarts + 原始loss权重 (d4.0_p1.0, 800ep)
#   - "PLAN_C": WarmRestarts + 调整loss权重 (d2.5_p2.0, 500ep)
#   - "CUSTOM": 自定义配置（手动设置下方所有参数）
# ============================================================
EXPERIMENT_CONFIG="PLAN_B"  # 修改此处选择配置: PLAN_B | PLAN_C | CUSTOM

# 根据实验配置设置参数
if [ "$EXPERIMENT_CONFIG" = "PLAN_B" ]; then
    echo "📋 使用预设: Plan B - WarmRestarts 长期训练"
    echo "   - Scheduler: CosineAnnealingWarmRestarts"
    echo "   - Epochs: 800"
    echo "   - Loss: Hybrid (dice=4.0, projection=1.0)"
    echo ""

    EPOCHS=800
    LOSS_TYPE="hybrid"
    DICE_WEIGHT=4.0
    PROJECTION_WEIGHT=1.0
    SCHEDULER="CosineAnnealingWarmRestarts"
    EXPERIMENT_NAME="Hybrid_warmrestarts_d4p1"

elif [ "$EXPERIMENT_CONFIG" = "PLAN_C" ]; then
    echo "📋 使用预设: Plan C - WarmRestarts + 调整loss权重"
    echo "   - Scheduler: CosineAnnealingWarmRestarts"
    echo "   - Epochs: 500"
    echo "   - Loss: Hybrid (dice=2.5, projection=2.0)"
    echo ""

    EPOCHS=500
    LOSS_TYPE="hybrid"
    DICE_WEIGHT=2.5
    PROJECTION_WEIGHT=2.0
    SCHEDULER="CosineAnnealingWarmRestarts"
    EXPERIMENT_NAME="Hybrid_warmrestarts_d2.5p2"

else
    echo "📋 使用自定义配置"
    echo ""

    # 自定义参数配置（仅在 CUSTOM 模式下生效）
    EPOCHS=200
    LOSS_TYPE="hybrid"
    DICE_WEIGHT=4.0
    PROJECTION_WEIGHT=1.0
    SCHEDULER="CosineAnnealingLR"  # CosineAnnealingLR | CosineAnnealingWarmRestarts | StepLR

    # 动态生成实验名称
    if [ "$SCHEDULER" = "CosineAnnealingWarmRestarts" ]; then
        SCHED_SUFFIX="warmrestarts"
    else
        SCHED_SUFFIX="cosine"
    fi

    if [ "$LOSS_TYPE" = "hybrid" ]; then
        EXPERIMENT_NAME="Hybrid_${SCHED_SUFFIX}_d${DICE_WEIGHT}_p${PROJECTION_WEIGHT}"
    elif [ "$LOSS_TYPE" = "projection" ]; then
        EXPERIMENT_NAME="Projection_${SCHED_SUFFIX}"
    else
        EXPERIMENT_NAME="BCEDice_${SCHED_SUFFIX}_d${DICE_WEIGHT}"
    fi
fi

# 通用训练参数（所有配置共用）
DATASET="Pohang-Canal-3k"
SPLIT_METHOD="50_50_2k_new"
MODEL="mamba_tiny"
LR=0.0002  # 2e-4, optimal learning rate
PEAK_THRESHOLD=0.35
SAVE_INTERVAL=50

# Loss 参数配置（可被上方预设覆盖）
FOCAL_ALPHA=0.25
FOCAL_GAMMA=2.5
PROJECTION_MODE="max"
OHEM_RATIO=0.0

# 获取可用 GPU 列表（按空闲显存从大到小排序）
echo ""
echo "📌 检测可用 GPU..."
GPU_LIST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | awk '{print $1}' | tr -d ',')

echo "GPU 显存状态（从大到小）:"
nvidia-smi --query-gpu=index,name,memory.free,memory.used --format=csv,noheader

# 尝试不同的 batch_size（从大到小）
BATCH_SIZES=(4 2 1)

# 依次尝试 GPU
for GPU_ID in $GPU_LIST; do
    echo ""
    echo "========================================"
    echo "🔄 尝试 GPU $GPU_ID"
    echo "========================================"
    
    # 依次尝试不同的 batch_size
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        echo ""
        echo "📝 配置: Config=$EXPERIMENT_CONFIG, GPU=$GPU_ID, BS=$BATCH_SIZE, LR=$LR"
        echo "   实验名: $EXPERIMENT_NAME"

        # 生成实验目录名（与 train.py 的 create_experiment_dir 保持一致）
        DT_STRING=$(date +%Y%m%d_%H%M%S)
        if [ -n "$EXPERIMENT_NAME" ]; then
            DIR_NAME="${EXPERIMENT_NAME}_${DATASET}_${MODEL}_${DT_STRING}"
        else
            DIR_NAME="${DATASET}_${MODEL}_${DT_STRING}"
        fi
        SAVE_DIR="$PROJECT_ROOT/model_Mamba/result/$DIR_NAME"

        # 提前创建实验目录
        mkdir -p "$SAVE_DIR"
        echo "📁 实验目录: $SAVE_DIR"

        # 生成日志文件名（保存在实验目录下）
        LOG_FILE="$SAVE_DIR/training_mamba_gpu${GPU_ID}_bs${BATCH_SIZE}_${DT_STRING}.log"

        # 启动训练 (Updated 2026-02-04: Added scheduler parameter)
        echo "🚀 启动训练..."
        echo "  Training script: $MODEL_MAMBA_DIR/train.py"
        echo "  Log file: $LOG_FILE"
        echo "  Loss type: $LOSS_TYPE"
        echo "  Scheduler: $SCHEDULER"
        echo "  Epochs: $EPOCHS"
        CUDA_VISIBLE_DEVICES=$GPU_ID python "$MODEL_MAMBA_DIR/train.py" \
            --root "$PROJECT_ROOT/dataset" \
            --dataset $DATASET \
            --split_method $SPLIT_METHOD \
            --model $MODEL \
            --train_batch_size $BATCH_SIZE \
            --test_batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --lr $LR \
            --scheduler $SCHEDULER \
            --peak_threshold $PEAK_THRESHOLD \
            --save_interval $SAVE_INTERVAL \
            --experiment_name $EXPERIMENT_NAME \
            --loss_type $LOSS_TYPE \
            --focal_alpha $FOCAL_ALPHA \
            --focal_gamma $FOCAL_GAMMA \
            --dice_weight $DICE_WEIGHT \
            --projection_weight $PROJECTION_WEIGHT \
            --projection_mode $PROJECTION_MODE \
            --ohem_ratio $OHEM_RATIO \
            --save_dir "$SAVE_DIR" \
            --gpus 0 \
            > $LOG_FILE 2>&1 &
        
        TRAIN_PID=$!
        echo "进程 PID: $TRAIN_PID"
        
        # 等待训练启动并监控前 5 分钟（确保通过 OOM 高风险期）
        echo "⏳ 监控训练启动中（最多等待 300 秒以确认无 OOM）..."
        
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
            if [ "$EPOCH_STARTED" = false ] && tail -20 $LOG_FILE | grep -q "Epoch 0:.*%"; then
                EPOCH_STARTED=true
                echo ""
                echo "✓ Epoch 0 已开始，开始实时显示训练日志..."
                echo "  (继续监控 1 分钟以确认稳定性)"
                echo "========================================"
                
                # 在后台启动 tail -f，输出到终端
                tail -f $LOG_FILE &
                TAIL_PID=$!
            fi
            
            # 如果已经开始训练且运行超过 1 分钟（60秒），认为稳定
            if [ "$EPOCH_STARTED" = true ] && [ $i -ge 60 ]; then
                SUCCESS=true
                echo ""
                echo "========================================"
                echo "✅ 训练稳定运行 1 分钟，确认成功！"
                echo "========================================"
                echo ""
                echo "按 Ctrl+C 退出日志查看（训练继续后台运行）"
                echo ""
                
                # 等待 tail 进程（让它继续在前台显示）
                wait $TAIL_PID 2>/dev/null || true
                
                # 如果用户 Ctrl+C 退出，显示后续监控命令
                echo ""
                echo "========================================"
                echo "训练仍在后台运行中"
                echo "========================================"
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
                echo "📁 日志文件位置:"
                echo "  $LOG_FILE"
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

# 所有 GPU 都失败了
echo ""
echo "========================================"
echo "❌ 所有 GPU 配置都失败了"
echo "========================================"
echo ""
echo "建议："
echo "  1. 检查数据集路径是否正确"
echo "  2. 检查最新的日志文件:"
echo "     ls -lht $SCRIPT_DIR/training_mamba_*.log | head -5"
echo "  3. 考虑减小模型或图像分辨率"
echo ""
exit 1
