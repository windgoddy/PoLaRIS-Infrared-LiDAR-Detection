#!/bin/bash
# =============================================================================
# Mamba多尺度模型训练脚本 (支持LiDAR和IR-only模式 + 自动GPU选择)
# =============================================================================
#
# 功能：
#   1. 支持3种训练模式：LiDAR / IR-only / IR-only-RGB
#   2. 自动选择可用GPU（按显存从大到小）
#   3. OOM自动重试（降低batch size或换GPU）
#   4. Ctrl+C优雅退出（保存checkpoint）
#
# 用法：
#   # 自动选择GPU（推荐）
#   bash model_Mamba/scripts/train_multiscale_full.sh [mode] [gpu] [model_type] [bit_depth] [loader]
#
#   # 指定GPU
#   bash model_Mamba/scripts/train_multiscale_full.sh [mode] [GPU_ID] [model_type] [bit_depth] [loader]
#
#   mode选项：
#     lidar       - LiDAR模式（默认）: IR + Depth + LiDAR点云
#     ir_only     - IR-only模式: 单通道IR（公平对比）
#     ir_only_rgb - IR-only RGB模式: 3通道RGB（完全匹配DNANet）
#
#   gpu选项：
#     auto        - 自动选择GPU（默认）
#     0,1,2...    - 手动指定GPU ID
#
#   model_type选项（可选）：
#     mamba_tiny_multiscale   - 小模型+多尺度融合（默认，embed_dim=64，patch_size=4）
#     mamba_tiny_progressive  - 小模型+渐进式解码器（embed_dim=64，patch_size=2，U-Net风格）
#     mamba_tiny              - 小模型（embed_dim=64）
#     mamba_small             - 中等模型（embed_dim=96）
#     mamba_small_progressive - 中等模型+渐进式解码器（embed_dim=96，patch_size=2）
#     mamba_base              - 大模型（embed_dim=128）
#
#   bit_depth选项（可选）：
#     16          - 使用16位图像（默认，images文件夹）
#     8           - 使用8位图像（images-8bit文件夹）
#
#   loader选项（可选，仅对lidar模式有效）：
#     polaris     - 使用PoLaRISTrainLoader（默认，支持16-bit + LiDAR）
#     traditional - 使用TrainSetLoader（8-bit only，DNANet兼容）
#
#   示例：
#     bash model_Mamba/scripts/train_multiscale_full.sh                                      # LiDAR，自动GPU，默认模型，16bit，polaris loader
#     bash model_Mamba/scripts/train_multiscale_full.sh lidar auto auto 8                    # LiDAR，自动GPU，默认模型，8bit，polaris loader
#     bash model_Mamba/scripts/train_multiscale_full.sh lidar auto auto 8 traditional        # LiDAR，8bit，traditional loader
#     bash model_Mamba/scripts/train_multiscale_full.sh ir_only                              # IR-only，自动GPU，默认模型，16bit
#     bash model_Mamba/scripts/train_multiscale_full.sh ir_only auto mamba_tiny_progressive  # IR-only，渐进式解码器（推荐）
#     bash model_Mamba/scripts/train_multiscale_full.sh lidar 0                              # LiDAR，GPU 0，默认模型，16bit
#     bash model_Mamba/scripts/train_multiscale_full.sh ir_only auto mamba_small             # IR-only，自动GPU，中等模型，16bit
#     bash model_Mamba/scripts/train_multiscale_full.sh lidar 1 mamba_base 8                 # LiDAR，GPU 1，大模型，8bit
#
#   示例：
#       python model_Mamba/train.py \
#        --normalize_mode minmax （percentile \ clahe \ global）
#        --mode lidar （ir_only \ ir_only_rgb）
#        --gpu auto （0 \ 1 \ 2 ...）
#        --model_type mamba_tiny_multiscale （mamba_tiny \ mamba_small \ mamba_base）
#        --bit_depth 16 （8）
#        --loader polaris （traditional）
#        --epochs 2000
#        --use_deep_supervision True （False）
#        --dataset Pohang-Canal-3k
#        --split_method 50_50_2k_new
#        --base_size 256
#        --crop_size 256
#        --lr 0.0001
#        --optimizer AdamW （SGD \ Adam）
#        --scheduler CosineAnnealingWarmRestarts （StepLR \ MultiStepLR）
#        --loss_type hybrid （bce \ dice \ focal）
#        --dice_weight 2.5
#        --projection_weight 2.0
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ============================================================
# 优雅退出处理
# ============================================================
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

        # 等待最多 30 秒
        for i in {1..30}; do
            if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
                echo "  ✓ 训练进程已优雅退出"
                exit 0
            fi
            sleep 1
        done

        # 超时强制终止
        echo "  ⚠️  超时，强制终止..."
        kill -9 $TRAIN_PID 2>/dev/null || true
    fi

    echo "  ✓ 清理完成"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================================
# 参数配置
# ============================================================

MODE=${1:-lidar}  # 训练模式
MANUAL_GPU=${2:-}  # 手动指定的GPU（空或"auto"则自动选择）
MODEL_TYPE=${3:-mamba_tiny_multiscale}  # 模型类型（可选）
BIT_DEPTH=${4:-16}  # 图像位深度：8 或 16（可选，默认16bit）
LOADER_TYPE=${5:-polaris}  # DataLoader类型：polaris 或 traditional（可选，默认polaris）
NORMALIZE_MODE=${6:-minmax}  # 归一化模式：minmax/global/percentile/clahe（可选，默认minmax）

# 处理"auto"作为GPU参数的情况
if [ "$MANUAL_GPU" == "auto" ]; then
    MANUAL_GPU=""
fi

# 处理"auto"作为MODEL_TYPE参数的情况
if [ "$MODEL_TYPE" == "auto" ] || [ -z "$MODEL_TYPE" ]; then
    MODEL_TYPE="mamba_tiny_multiscale"
fi

# 处理"auto"作为LOADER_TYPE参数的情况
if [ "$LOADER_TYPE" == "auto" ] || [ -z "$LOADER_TYPE" ]; then
    LOADER_TYPE="polaris"
fi

# 验证模型类型
case $MODEL_TYPE in
    "mamba_tiny"|"mamba_tiny_multiscale"|"mamba_tiny_progressive"|"mamba_small"|"mamba_small_progressive"|"mamba_base")
        # 有效的模型类型
        ;;
    *)
        echo "❌ 错误: 未知模型类型 '$MODEL_TYPE'"
        echo "支持的模型: mamba_tiny | mamba_tiny_multiscale | mamba_tiny_progressive | mamba_small | mamba_small_progressive | mamba_base"
        exit 1
        ;;
esac

EPOCHS=2000

# 根据模式设置配置
# 注意：所有模式都使用 train.py（已支持IR-only）
case $MODE in
    "lidar")
        USE_LIDAR=True
        IN_CHANNELS=2
        USE_LIDAR_LOADER=True
        TRAIN_SCRIPT="train.py"
        MODE_DESC="LiDAR模式 (IR + Depth + LiDAR)"
        ;;
    "ir_only")
        USE_LIDAR=False
        IN_CHANNELS=1
        USE_LIDAR_LOADER=False
        TRAIN_SCRIPT="train.py"  # ← 改为使用 train.py
        MODE_DESC="IR-only模式 (公平对比，单通道)"
        ;;
    "ir_only_rgb")
        USE_LIDAR=False
        IN_CHANNELS=3
        USE_LIDAR_LOADER=False
        TRAIN_SCRIPT="train.py"  # ← 改为使用 train.py
        MODE_DESC="IR-only RGB模式 (完全匹配DNANet)"
        ;;
    *)
        echo "❌ 错误: 未知模式 '$MODE'"
        echo "支持的模式: lidar | ir_only | ir_only_rgb"
        exit 1
        ;;
esac

USE_DEEP_SUPERVISION=True

# 数据配置
DATASET="Pohang-Canal-3k"
SPLIT_METHOD="50_50_2k_new"
BASE_SIZE=256
CROP_SIZE=256

# 根据位深度设置图像文件夹和归一化选项
if [ "$BIT_DEPTH" == "8" ]; then
    IMAGE_FOLDER="images-8bit"
    NORMALIZE_16BIT="False"
    BIT_DESC="8bit"
elif [ "$BIT_DEPTH" == "16" ]; then
    IMAGE_FOLDER="images"
    NORMALIZE_16BIT="True"
    BIT_DESC="16bit"
else
    echo "❌ 错误: 未知位深度 '$BIT_DEPTH'"
    echo "支持的位深度: 8 | 16"
    exit 1
fi

# 根据loader类型设置参数（仅对lidar模式有效）
if [ "$MODE" == "lidar" ]; then
    if [ "$LOADER_TYPE" == "polaris" ]; then
        USE_POLARIS_LOADER="True"
        LOADER_DESC="PoLaRIS (16-bit + LiDAR)"
    elif [ "$LOADER_TYPE" == "traditional" ]; then
        USE_POLARIS_LOADER="False"
        LOADER_DESC="Traditional (8-bit, DNANet-compatible)"
    else
        echo "❌ 错误: 未知Loader类型 '$LOADER_TYPE'"
        echo "支持的Loader: polaris | traditional"
        exit 1
    fi
else
    # IR-only模式总是使用TrainSetLoader
    USE_POLARIS_LOADER="False"
    LOADER_DESC="TrainSetLoader (IR-only)"
fi

# 训练配置
LR=0.0001
OPTIMIZER="AdamW"
SCHEDULER="CosineAnnealingWarmRestarts"

# Loss配置
LOSS_TYPE="hybrid"
DICE_WEIGHT=2.5
PROJECTION_WEIGHT=2.0

# 模型大小信息（用于显示和命名）
case $MODEL_TYPE in
    "mamba_tiny"|"mamba_tiny_multiscale"|"mamba_tiny_progressive")
        MODEL_SIZE="tiny"
        EMBED_DIM=64
        ;;
    "mamba_small"|"mamba_small_progressive")
        MODEL_SIZE="small"
        EMBED_DIM=96
        ;;
    "mamba_base")
        MODEL_SIZE="base"
        EMBED_DIM=128
        ;;
esac

# 实验名称（包含模型大小、位深度和loader类型）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
case $MODE in
    "lidar")
        if [ "$USE_POLARIS_LOADER" == "True" ]; then
            EXPERIMENT_NAME="${MODEL_SIZE}_LiDAR_${BIT_DESC}_PoLaRIS_d${DICE_WEIGHT}p${PROJECTION_WEIGHT}_${TIMESTAMP}"
        else
            EXPERIMENT_NAME="${MODEL_SIZE}_LiDAR_${BIT_DESC}_Traditional_d${DICE_WEIGHT}p${PROJECTION_WEIGHT}_${TIMESTAMP}"
        fi
        ;;
    "ir_only")
        EXPERIMENT_NAME="${MODEL_SIZE}_IR_Only_${BIT_DESC}_d${DICE_WEIGHT}p${PROJECTION_WEIGHT}_${TIMESTAMP}"
        ;;
    "ir_only_rgb")
        EXPERIMENT_NAME="${MODEL_SIZE}_IR_Only_RGB_${BIT_DESC}_d${DICE_WEIGHT}p${PROJECTION_WEIGHT}_${TIMESTAMP}"
        ;;
esac

# ============================================================
# 训练信息显示
# ============================================================

echo "=========================================="
echo "🚀 Mamba多尺度模型训练 (自动GPU)"
echo "=========================================="
echo ""
echo "训练模式:         $MODE_DESC"
echo "图像位深度:       $BIT_DESC ($IMAGE_FOLDER)"
echo "DataLoader:       $LOADER_DESC"
echo ""
echo "配置信息:"
echo "  模型:             $MODEL_TYPE (embed_dim=$EMBED_DIM)"
echo "  输入:             ${IN_CHANNELS}-channel"
echo "  LiDAR:            $USE_LIDAR"
echo "  Deep Supervision: $USE_DEEP_SUPERVISION"
echo "  训练轮数:         $EPOCHS epochs"
echo "  Batch Sizes:      尝试 16/8/4/2 (自动降级)"
echo "  损失权重:         Dice=$DICE_WEIGHT, Projection=$PROJECTION_WEIGHT"
echo "  实验名称:         $EXPERIMENT_NAME"
echo ""
if [ "$MODE" == "lidar" ]; then
    echo "📋 预期结果 (LiDAR模式):"
    echo "  - Box IoU: 0.60-0.64"
    echo "  - Seg IoU: 0.55-0.60"
else
    echo "📋 预期结果 (IR-only模式 - 公平对比):"
    echo "  - 目标: 接近或超过DNANet的Seg IoU: 0.82+"
    echo "  - 如果低于DNANet: 说明架构需改进"
    echo "  - 如果接近DNANet: 说明LiDAR引入了噪声"
fi
echo ""
echo "💡 提示: 按 Ctrl+C 可优雅停止训练（会保存 checkpoint）"
echo ""

# ============================================================
# GPU 选择
# ============================================================

if [ -n "$MANUAL_GPU" ]; then
    # 手动指定GPU
    echo "🔧 使用手动指定的GPU: $MANUAL_GPU"
    GPU_LIST="$MANUAL_GPU"
else
    # 自动选择GPU
    echo "📌 自动检测可用 GPU..."
    GPU_LIST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | awk '{print $1}' | tr -d ',')

    echo "GPU 显存状态（从大到小）:"
    nvidia-smi --query-gpu=index,name,memory.free,memory.used --format=csv,noheader
fi

# Batch size候选值（从大到小）
BATCH_SIZES=(16 8 4 2)

# ============================================================
# 开始训练 (自动重试逻辑)
# ============================================================

echo ""
echo "🚀 开始训练..."
echo ""

# 依次尝试GPU
for GPU_ID in $GPU_LIST; do
    echo ""
    echo "========================================"
    echo "🔄 尝试 GPU $GPU_ID"
    echo "========================================"

    # 依次尝试不同batch size
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        echo ""
        echo "📝 配置: Mode=$MODE, GPU=$GPU_ID, Batch Size=$BATCH_SIZE"

        # 创建实验目录
        mkdir -p "result/$EXPERIMENT_NAME"
        SAVE_DIR="result/$EXPERIMENT_NAME"

        # 日志文件
        LOG_FILE="$SAVE_DIR/training_gpu${GPU_ID}_bs${BATCH_SIZE}_${TIMESTAMP}.log"
        echo "📁 日志: $LOG_FILE"

        # 所有模式统一使用 train.py（已支持IR-only）
        echo "🚀 启动训练..."

        # 注意：CUDA_VISIBLE_DEVICES 会重新编号GPU为0，所以--gpus参数固定为"0"
        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
            --root "../dataset" \
            --model "$MODEL_TYPE" \
            --dataset "$DATASET" \
            --split_method "$SPLIT_METHOD" \
            --base_size $BASE_SIZE \
            --crop_size $CROP_SIZE \
            --train_batch_size $BATCH_SIZE \
            --test_batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --lr $LR \
            --optimizer "$OPTIMIZER" \
            --scheduler "$SCHEDULER" \
            --use_lidar "$USE_LIDAR" \
            --in_channels $IN_CHANNELS \
            --normalize_16bit "$NORMALIZE_16BIT" \
            --normalize_mode "$NORMALIZE_MODE" \
            --use_polaris_loader "$USE_POLARIS_LOADER" \
            --image_folder "$IMAGE_FOLDER" \
            --loss_type "$LOSS_TYPE" \
            --dice_weight $DICE_WEIGHT \
            --projection_weight $PROJECTION_WEIGHT \
            --gpus "0" \
            --workers 4 \
            --seed 42 \
            --suffix .png \
            > $LOG_FILE 2>&1 &

        TRAIN_PID=$!
        echo "进程 PID: $TRAIN_PID"

        # 监控训练启动（等待5分钟确认无OOM）
        echo "⏳ 监控训练启动中（最多等待 300 秒）..."

        SUCCESS=false
        EPOCH_STARTED=false
        TAIL_PID=""

        for i in {1..300}; do
            sleep 1

            # 检查进程是否还在运行
            if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
                # 停止 tail
                if [ -n "$TAIL_PID" ]; then
                    kill $TAIL_PID 2>/dev/null || true
                fi

                # 检查是否OOM
                if tail -100 $LOG_FILE | grep -qE "OutOfMemoryError|CUDA out of memory"; then
                    echo ""
                    echo "❌ GPU $GPU_ID (Batch Size $BATCH_SIZE) OOM，尝试下一个配置..."
                    break
                else
                    echo ""
                    echo "❌ 训练失败，错误信息："
                    tail -20 $LOG_FILE
                    break
                fi
            fi

            # 检查是否开始训练（出现Epoch 0）
            if [ "$EPOCH_STARTED" = false ] && tail -20 $LOG_FILE | grep -qE "Epoch.*0|epoch.*0|Epoch 0:"; then
                EPOCH_STARTED=true
                echo ""
                echo "✓ Epoch 0 已开始，显示实时日志..."
                echo "  (继续监控 60 秒确认稳定性)"
                echo "========================================"

                # 后台启动 tail -f
                tail -f $LOG_FILE &
                TAIL_PID=$!
            fi

            # 运行60秒后认为稳定
            if [ "$EPOCH_STARTED" = true ] && [ $i -ge 60 ]; then
                SUCCESS=true
                echo ""
                echo "========================================"
                echo "✅ 训练稳定运行 60 秒，确认成功！"
                echo "========================================"
                echo ""
                echo "📊 训练信息:"
                echo "  GPU:        $GPU_ID"
                echo "  Batch Size: $BATCH_SIZE"
                echo "  模式:       $MODE"
                echo "  实验名:     $EXPERIMENT_NAME"
                echo "  日志:       $LOG_FILE"
                echo ""
                echo "按 Ctrl+C 退出日志查看（训练继续后台运行）"
                echo ""

                # 等待 tail（前台显示）
                wait $TAIL_PID 2>/dev/null || true

                # Ctrl+C后显示监控命令
                echo ""
                echo "========================================"
                echo "训练仍在后台运行中"
                echo "========================================"
                echo ""
                echo "📊 查看训练进度:"
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
        done

        # 成功则退出
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

# ============================================================
# 所有配置都失败
# ============================================================

echo ""
echo "========================================"
echo "❌ 所有 GPU 配置都失败了"
echo "========================================"
echo ""
echo "建议："
echo "  1. 检查数据集路径是否正确"
echo "  2. 检查最新的日志文件:"
echo "     ls -lht result/$EXPERIMENT_NAME/training_*.log | head -5"
echo "  3. 考虑减小模型或图像分辨率"
echo "  4. 尝试手动指定GPU: bash $0 $MODE 0"
echo ""
exit 1
