#!/bin/bash
# 自动选择 GPU 并启动训练，OOM 时自动切换到下一个 GPU
# 使用方法: bash scripts/auto_train_gpu.sh

set -e

echo "========================================"
echo "PoLaRIS-Mamba 自动 GPU 训练启动器"
echo "========================================"

# 训练参数配置
DATASET="Pohang-Canal-3k"
SPLIT_METHOD="50_50_2k_new"
MODEL="mamba_tiny"
EPOCHS=1000
LR=0.001

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
        echo "📝 配置: GPU=$GPU_ID, Batch Size=$BATCH_SIZE"
        
        # 生成日志文件名
        LOG_FILE="training_mamba_gpu${GPU_ID}_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S).log"
        
        # 启动训练
        echo "🚀 启动训练..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python model_Mamba/train.py \
            --dataset $DATASET \
            --split_method $SPLIT_METHOD \
            --model $MODEL \
            --train_batch_size $BATCH_SIZE \
            --test_batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --lr $LR \
            --gpus 0 \
            > $LOG_FILE 2>&1 &
        
        TRAIN_PID=$!
        echo "进程 PID: $TRAIN_PID"
        
        # 等待训练启动并监控前 5 分钟（确保通过 OOM 高风险期）
        echo "⏳ 监控训练启动中（最多等待 300 秒以确认无 OOM）..."
        
        SUCCESS=false
        EPOCH_STARTED=false
        
        for i in {1..300}; do
            sleep 1
            
            # 检查进程是否还在运行
            if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
                # 进程已退出，检查是否是 OOM
                if tail -100 $LOG_FILE | grep -q "OutOfMemoryError"; then
                    echo "❌ GPU $GPU_ID (Batch Size $BATCH_SIZE) OOM，尝试下一个配置..."
                    break
                else
                    # 其他错误
                    echo "❌ 训练失败，错误信息："
                    tail -20 $LOG_FILE
                    break
                fi
            fi
            
            # 检查是否出现 Epoch 0
            if [ "$EPOCH_STARTED" = false ] && tail -20 $LOG_FILE | grep -q "Epoch 0:.*%"; then
                EPOCH_STARTED=true
                echo "  ✓ Epoch 0 已开始，继续监控稳定性..."
            fi
            
            # 如果已经开始训练且运行超过 3 分钟（180秒），认为稳定
            if [ "$EPOCH_STARTED" = true ] && [ $i -ge 180 ]; then
                SUCCESS=true
                echo ""
                echo "✅ 训练稳定运行 3 分钟，确认成功！"
                echo "========================================"
                echo "配置信息:"
                echo "  GPU: $GPU_ID"
                echo "  Batch Size: $BATCH_SIZE"
                echo "  Dataset: $DATASET"
                echo "  Model: $MODEL"
                echo "  Epochs: $EPOCHS"
                echo "  Log File: $LOG_FILE"
                echo "  PID: $TRAIN_PID"
                echo "========================================"
                echo ""
                echo "📊 开始实时显示训练日志（按 Ctrl+C 退出查看，训练继续后台运行）"
                echo ""
                sleep 2
                
                # 实时显示训练日志
                tail -f $LOG_FILE
                
                # 如果用户 Ctrl+C 退出，显示后续监控命令
                echo ""
                echo "========================================"
                echo "训练仍在后台运行中"
                echo "========================================"
                echo ""
                echo "📊 重新查看训练进度:"
                echo "  tail -f $LOG_FILE"
                echo ""
                echo "🛑 停止训练:"
                echo "  kill $TRAIN_PID"
                echo ""
                echo "📈 查看 GPU 使用:"
                echo "  watch -n 1 nvidia-smi"
                echo ""
                
                exit 0
            fi
            
            # 每 15 秒显示一次进度
            if [ $((i % 15)) -eq 0 ]; then
                echo "  等待中... ${i}s (Epoch started: $EPOCH_STARTED)"
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
echo "     ls -lht training_mamba_*.log | head -5"
echo "  3. 考虑减小模型或图像分辨率"
echo ""
exit 1
