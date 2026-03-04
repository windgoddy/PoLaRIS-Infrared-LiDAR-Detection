#!/bin/bash

# ============================================================
# 统一训练脚本 - PoLaRIS Infrared-LiDAR Detection
# ============================================================
# 用法:
#   ./scripts/train.sh <mode> [options]
#
# 模式:
#   baseline1    - DNANet + 8-bit images（对比基准）
#   cat2         - DNANet + Cat2数据集（小目标场景专用）
#   dnanet_lidar - DNANet + LiDAR（多模态，公平对比）
#   16bit-ir     - MS_CAFNet + 16-bit + 仅红外（改进模型）
#   16bit        - MS_CAFNet + 16-bit + 深度图（改进模型，完整）
#   mamba_unetpp - Mamba-UNet++ 混合架构（新：Mamba + UNet++密集连接）
#
# 选项:
#   --gpu <N>              指定GPU编号（默认：auto自动选择）
#   --dataset <name>       指定数据集（默认：Pohang-Canal-3k）
#   --split-method <name>  数据集划分方法（默认：50_50_2k，cat2模式默认：50_50_cat2）
#   --epochs <N>           训练轮数（默认：2000）
#   --oracle-masks <name>  Oracle masks文件夹名称（默认：oracle_masks）
#   --thres <float>        推理阈值（默认：自动根据模式选择）
#   --batch-size <N>       训练batch size（默认：8）
#
# 示例:
#   ./scripts/train.sh baseline1                                        # DNANet baseline，自动选择GPU
#   ./scripts/train.sh cat2 --gpu 0                                     # Cat2数据集，指定GPU 0
#   ./scripts/train.sh cat2 --batch-size 4                              # Cat2数据集，batch=4
#   ./scripts/train.sh dnanet_lidar --split-method 50_50_cat2           # DNANet多模态+Cat2
#   ./scripts/train.sh baseline1 --split-method 50_50_cat2              # baseline1用Cat2数据
#   ./scripts/train.sh 16bit-ir --gpu 1                                 # MS_CAFNet 16-bit 无深度图
# ============================================================

# 自动GPU选择函数
select_gpu() {
    local manual_gpu=$1

    if [ "$manual_gpu" != "auto" ]; then
        echo "$manual_gpu"
        return 0
    fi

    if ! command -v nvidia-smi &> /dev/null; then
        echo "0"
        return 0
    fi

    # 获取GPU信息（ID, 显存总量MB, 已用显存MB）
    gpu_info=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
               awk -F', ' '{free=$2-$3; print $1, $2, free}')

    if [ -z "$gpu_info" ]; then
        echo "0"
        return 0
    fi

    # 选择空闲显存最大的GPU
    best_gpu=$(echo "$gpu_info" | sort -k3 -rn | head -1 | awk '{print $1}')
    echo "$best_gpu"
}

# 默认参数
MODE="16bit"
MANUAL_GPU="auto"
DATASET="Pohang-Canal-3k"
SPLIT_METHOD=""  # 空字符串表示根据模式自动选择
EPOCHS=2000
ORACLE_MASKS="oracle_masks"
THRESHOLD=""  # 空字符串表示自动选择
BATCH_SIZE=8
RESUME=""     # checkpoint 路径，空字符串表示从头开始
SAVE_DIR=""   # 指定实验目录（resume时应与原目录一致）

# 解析第一个参数作为模式（如果提供）
if [[ $# -gt 0 && $1 =~ ^(baseline1|baseline2|cat2|dnanet_lidar|16bit-ir|16bit|mamba_unetpp)$ ]]; then
    MODE="$1"
    shift
fi

# 解析其他命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            MANUAL_GPU="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --split-method)
            SPLIT_METHOD="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --oracle-masks)
            ORACLE_MASKS="$2"
            shift 2
            ;;
        --thres)
            THRESHOLD="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        *)
            echo "警告: 未知参数 '$1'"
            shift
            ;;
    esac
done

# 自动选择数据集划分方法（如果未指定）
if [[ -z "$SPLIT_METHOD" ]]; then
    case $MODE in
        cat2)
            SPLIT_METHOD="50_50_cat2"
            ;;
        *)
            SPLIT_METHOD="50_50_2k"
            ;;
    esac
fi

# 自动选择GPU
GPU=$(select_gpu "$MANUAL_GPU")

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU

# 自动设置阈值（如果用户未指定）
if [[ -z "$THRESHOLD" ]]; then
    case $MODE in
        baseline1|baseline2)
            THRESHOLD="0.5"  # Hard Labels 使用 0.5
            ;;
        cat2)
            THRESHOLD="0.3"  # Cat2小目标使用较低阈值
            ;;
        16bit|16bit-ir)
            THRESHOLD="0.5"  # Soft Labels 使用 0.5
            ;;
        *)
            THRESHOLD="0.3"  # 默认 0.3
            ;;
    esac
fi

echo "============================================================"
echo "训练模式: $MODE"
if [ "$MANUAL_GPU" == "auto" ]; then
    echo "GPU: $GPU (自动选择)"
else
    echo "GPU: $GPU"
fi
echo "数据集划分: $SPLIT_METHOD"
echo "Batch Size: $BATCH_SIZE"
echo "推理阈值: $THRESHOLD"
echo "============================================================"

# 根据模式选择配置
case $MODE in
    baseline1)
        echo "🔹 Baseline: DNANet + 8-bit images"
        python train.py \
            --experiment_name DNANet_baseline_8bit \
            --model DNANet \
            --dataset $DATASET \
            --image_folder images-8bit \
            --train_batch_size $BATCH_SIZE \
            --test_batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --optimizer Adagrad \
            --lr 0.05 \
            --deep_supervision True \
            --backbone resnet_18 \
            --channel_size three \
            --seed 42 \
            --thres $THRESHOLD \
            --suffix .png \
            --split_method $SPLIT_METHOD \
            --workers 4
        ;;

    cat2)
        echo "🎯 Cat2 Focused: DNANet + 8-bit + 小目标场景专用"
        python train.py \
            --experiment_name Cat2_Focused_DNANet \
            --model DNANet \
            --dataset $DATASET \
            --mode TXT \
            --split_method $SPLIT_METHOD \
            --in_channels 3 \
            --image_folder images-8bit \
            --base_size 256 \
            --crop_size 256 \
            --train_batch_size $BATCH_SIZE \
            --test_batch_size $BATCH_SIZE \
            --optimizer Adagrad \
            --lr 0.05 \
            --scheduler CosineAnnealingLR \
            --deep_supervision True \
            --epochs $EPOCHS \
            --thres $THRESHOLD \
            --seed 42 \
            --workers 4
        ;;

    dnanet_lidar)
        echo "🔷 DNANet + LiDAR: 纯DNANet多模态（公平对比baseline）"
        python train_Phase3.py \
            --experiment_name DNANet_LiDAR_baseline \
            --model DNANet \
            --dataset $DATASET \
            --in_channels 2 \
            --epochs $EPOCHS \
            --optimizer Adagrad \
            --lr 0.05 \
            --scheduler CosineAnnealingLR \
            --deep_supervision True \
            --backbone resnet_18 \
            --channel_size three \
            --seed 42 \
            --use_lidar_dataloader True \
            --normalize_16bit True \
            --use_soft_labels True \
            --thres $THRESHOLD \
            --train_batch_size $BATCH_SIZE \
            --test_batch_size $BATCH_SIZE \
            --suffix .png \
            --split_method $SPLIT_METHOD \
            --workers 4
        ;;

    baseline2)
        echo "🔹 Baseline: MS_CAFNet_DualGeo + 8-bit images"
        python train.py \
            --experiment_name MS_CAFNet_baseline_8bit \
            --model MS_CAFNet_DualGeo \
            --dataset Pohang-Canal-3k \
            --image_folder images-8bit \
            --train_batch_size 8 \
            --epochs $EPOCHS \
            --optimizer Adagrad \
            --lr 0.05 \
            --deep_supervision True \
            --backbone resnet_18 \
            --channel_size three \
            --seed 42 \
            --thres $THRESHOLD \
            --suffix .png \
            --split_method 50_50_2k \
            --workers 4
        ;;

    16bit)
        echo "🔸 PoLaRIS: 16-bit + 深度图（完整模型）"
        python train_Phase3.py \
            --experiment_name PoLaRIS_16bit_full \
            --model MS_CAFNet_DualGeo \
            --dataset Pohang-Canal-3k \
            --epochs $EPOCHS \
            --optimizer Adam \
            --lr 0.0001 \
            --weight_decay 5e-4 \
            --scheduler CosineAnnealingLR \
            --min_lr 1e-6 \
            --in_channels 2 \
            --deep_supervision False \
            --seed 42 \
            --use_lidar_dataloader True \
            --normalize_16bit True \
            --use_soft_labels True \
            --thres $THRESHOLD \
            --train_batch_size 16 \
            --test_batch_size 4 \
            --backbone resnet_18 \
            --suffix .png \
            --split_method 50_50_2k_new \
            --workers 4 \
            --oracle_masks_folder oracle_masks2
        ;;

    16bit-ir)
        echo "🔹 PoLaRIS: 16-bit + 仅红外（无深度图）"
        python train_Phase3.py \
            --experiment_name PoLaRIS_16bit_IR_only \
            --model MS_CAFNet_DualGeo \
            --dataset Pohang-Canal-3k \
            --epochs $EPOCHS \
            --optimizer Adam \
            --lr 0.0001 \
            --weight_decay 5e-4 \
            --scheduler CosineAnnealingLR \
            --min_lr 1e-6 \
            --in_channels 1 \
            --deep_supervision False \
            --seed 42 \
            --use_lidar_dataloader True \
            --normalize_16bit True \
            --use_soft_labels True \
            --thres $THRESHOLD \
            --oracle_masks_folder $ORACLE_MASKS \
            --backbone resnet_34 \
            --train_batch_size 16 \
            --test_batch_size 16 \
            --suffix .png \
            --split_method 50_50_2k \
            --workers 4
        ;;

    mamba_unetpp)
        echo "🎯 Mamba-UNet++ V2: 混合架构（优化版）"
        echo "  ✓ Spatial Restore Layer (修复语义代沟)"
        echo "  ✓ Deep Supervision (UNet++ 核心)"
        echo "  ✓ 残差连接 (稳定训练)"
        echo "  ✓ Dice Weight = 4.0 (提高像素精度)"

        # 在 cd 之前将相对路径转为绝对路径，避免进入子目录后路径失效
        ROOT_ABS="$(pwd)"
        if [[ -n "$RESUME" && "$RESUME" != /* ]]; then
            RESUME="$ROOT_ABS/$RESUME"
        fi
        if [[ -n "$SAVE_DIR" && "$SAVE_DIR" != /* ]]; then
            SAVE_DIR="$ROOT_ABS/$SAVE_DIR"
        fi

        cd model_Mamba

        # OOM 自动重试逻辑：从用户指定的 batch size 开始，逐步降低
        BATCH_SIZES=($BATCH_SIZE)
        for fallback in 4 2 1; do
            if [ $fallback -lt $BATCH_SIZE ]; then
                BATCH_SIZES+=($fallback)
            fi
        done
        SUCCESS=false
        LAST_LOG_FILE=""

        for TRY_BATCH in "${BATCH_SIZES[@]}"; do
            echo ""
            echo "🔄 尝试 Batch Size = $TRY_BATCH"
            # 每次重试使用独立日志，避免旧 OOM 信息干扰新一轮的错误判断
            LOG_FILE="/tmp/mamba_unetpp_train_$$_bs${TRY_BATCH}.log"
            LAST_LOG_FILE="$LOG_FILE"

            # [FIX 2026-03-03] 导师诊断：BatchNorm + 小Batch的致命组合
            # 核心修复：
            # 1. 模型代码已改用 GroupNorm（不依赖 Batch Size）
            # 2. 降低学习率 (0.0005 → 0.0004) ← GroupNorm 更稳定，可用稍高 LR
            # 3. 缩短训练周期 (500 → 300) ← 配合 GroupNorm 快速收敛
            # 4. 保持数据增强、梯度累积、深度监督
            python train.py \
                --experiment_name MambaUNetPP_V5_Cat2_HybridNoDice \
                --model mamba_unetpp \
                --root ../dataset/ \
                --dataset $DATASET \
                --split_method $SPLIT_METHOD \
                --in_channels 3 \
                --image_folder images-8bit \
                --suffix .png \
                --base_size 256 \
                --crop_size 256 \
                --train_batch_size $TRY_BATCH \
                --test_batch_size $TRY_BATCH \
                --gradient_accumulation_steps 4 \
                --epochs 500 \
                --optimizer AdamW \
                --lr 0.0004 \
                --weight_decay 0.05 \
                --scheduler CosineAnnealingLR \
                --min_lr 1e-6 \
                --seed 42 \
                --use_polaris_loader False \
                --normalize_16bit False \
                --loss_type hybrid \
                --dice_weight 0.1 \
                --projection_weight 2.0 \
                --focal_gamma 4.0 \
                --peak_threshold $THRESHOLD \
                --workers 4 \
                --use_lidar False \
                --use_deep_supervision True \
                --use_augmentation True \
                ${RESUME:+--resume "$RESUME"} \
                ${SAVE_DIR:+--save_dir "$SAVE_DIR"} > "$LOG_FILE" 2>&1 &

            TRAIN_PID=$!
            echo "📝 训练进程 PID: $TRAIN_PID"
            echo "📁 日志文件: $LOG_FILE"

            # 监控训练启动（等待 60 秒）
            EPOCH_STARTED=false
            for i in {1..60}; do
                sleep 1

                # 检查进程是否还在运行
                if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
                    # 检查是否 OOM
                    if grep -qE "OutOfMemoryError|CUDA out of memory" "$LOG_FILE"; then
                        echo "❌ OOM"
                        break
                    else
                        echo "❌ 训练失败"
                        tail -10 "$LOG_FILE" | grep -E "Error|Exception" || echo "查看日志: $LOG_FILE"
                        break
                    fi
                fi

                # 检查是否开始训练
                if [ "$EPOCH_STARTED" = false ] && grep -qE "Epoch.*0|epoch.*0|Epoch 0:" "$LOG_FILE"; then
                    EPOCH_STARTED=true
                    echo "✓ 训练已启动，Batch Size = $TRY_BATCH 可行"
                    echo "✓ 继续训练中... (日志: $LOG_FILE)"
                    SUCCESS=true
                    break
                fi
            done

            if [ "$SUCCESS" = true ]; then
                # 显示实时训练日志
                echo ""
                echo "=========================================="
                echo "开始显示训练日志 (Ctrl+C 退出查看但不停止训练)"
                echo "=========================================="
                tail -f "$LOG_FILE" &
                TAIL_PID=$!

                # 等待训练完成
                wait $TRAIN_PID
                TRAIN_EXIT_CODE=$?

                # 停止 tail
                kill $TAIL_PID 2>/dev/null || true

                echo ""
                echo "=========================================="
                if [ $TRAIN_EXIT_CODE -eq 0 ]; then
                    echo "✅ 训练成功完成 (Batch Size = $TRY_BATCH)"
                    echo "=========================================="
                    break
                elif grep -qE "OutOfMemoryError|CUDA out of memory" "$LOG_FILE"; then
                    echo "❌ 训练中途 OOM (Batch Size = $TRY_BATCH)"
                    echo "=========================================="
                    SUCCESS=false
                    if [ "$TRY_BATCH" = "${BATCH_SIZES[-1]}" ]; then
                        echo "❌ 即使 Batch Size = $TRY_BATCH 仍然 OOM，放弃"
                        break
                    fi
                    echo "⚠️  尝试更小的 batch size..."
                    sleep 2
                else
                    echo "❌ 训练异常退出 (退出码: $TRAIN_EXIT_CODE)"
                    echo "查看完整日志: $LOG_FILE"
                    echo "=========================================="
                    break
                fi
            else
                # 杀死进程（如果还在运行）
                kill $TRAIN_PID 2>/dev/null || true
                wait $TRAIN_PID 2>/dev/null || true

                if [ "$TRY_BATCH" != "${BATCH_SIZES[-1]}" ]; then
                    echo "⚠️  Batch Size $TRY_BATCH 不可行，尝试更小的 batch..."
                    sleep 2
                    continue
                else
                    echo "❌ 即使 Batch Size = 1 仍然失败"
                    break
                fi
            fi
        done

        cd ..

        if [ "$SUCCESS" = false ]; then
            echo ""
            echo "============================================================"
            echo "❌ 训练失败，请查看日志: $LOG_FILE"
            echo "============================================================"
            exit 1
        fi
        ;;

    *)
        echo "❌ 未知模式: $MODE"
        echo "支持的模式: baseline1, cat2, dnanet_lidar, 16bit-ir, 16bit, mamba_unetpp"
        echo ""
        echo "实验设计："
        echo "  baseline1    - DNANet + 8-bit（单模态baseline）"
        echo "  cat2         - DNANet + Cat2数据集（小目标场景专用）"
        echo "  dnanet_lidar - DNANet + LiDAR（多模态，公平对比）"
        echo "  16bit-ir     - MS_CAFNet + 16-bit IR-only（改进模型）"
        echo "  16bit        - MS_CAFNet + 16-bit + LiDAR（改进模型完整版）"
        echo "  mamba_unetpp - Mamba-UNet++混合架构（新：结合Mamba与UNet++）"
        exit 1
        ;;
esac

echo "============================================================"
echo "✅ 训练完成!"
echo "============================================================"
