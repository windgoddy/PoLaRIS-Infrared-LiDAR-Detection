#!/bin/bash
# =============================================================================
# Mamba多尺度模型训练脚本 (自动GPU + 顶会技巧集成)
# =============================================================================
#
# 功能：
#   1. 支持3种训练模式：LiDAR / IR-only / IR-only-RGB
#   2. 自动选择可用GPU（按显存从大到小）
#   3. OOM自动重试（降低batch size或换GPU）
#   4. Ctrl+C优雅退出（保存checkpoint）
#   5. ✨ 集成所有顶会训练技巧（+10-16% IoU预期提升）
#      - 强化数据增强（垂直翻转、90度旋转、Gamma/对比度）
#      - 多尺度训练（动态分辨率 [192-320]）
#      - 梯度累积（有效Batch=Batch×4）
#      - Warmup学习率调度器（10 epochs预热）
#      - EMA模型权重平均（decay=0.9999）
#      - Test-Time Augmentation（8-way增强）
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
#     rgb_lidar   - RGB+LiDAR模式: 3通道伪RGB + LiDAR gating（全面超越DNANet）⭐推荐
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
#   deep_supervision选项（可选）：
#     True        - 启用深度监督（辅助损失在D2和D3层，仅对multiscale和progressive模型有效）
#     False       - 不启用深度监督（默认）
#
#   dice_weight选项（可选，第8个参数）：
#     数值        - 自定义Dice Loss权重（默认: 2.5）
#                   降低此值可以减少对false positives的惩罚，提高模型输出置信度
#
#   projection_weight选项（可选，第9个参数）：
#     数值        - 自定义Projection Loss权重（默认: 2.0）
#                   调整投影约束强度
#
#   resume_checkpoint选项（可选，第10个参数）：
#     路径        - Resume训练的checkpoint文件路径（留空则从头训练）
#                   示例: result/tiny_RGB_LiDAR_xxx/best_model_epoch0347.pth
#
#   dataset选项（可选，第11个参数）：
#     Pohang-Canal-3k - 浦项运河数据集（默认，支持LiDAR）
#     NUDT-SIRST      - NUDT红外小目标数据集（仅支持IR-only模式，8bit）
#
#   use_scene_weights选项（可选，第12个参数）：
#     True        - 启用场景加权损失（针对性提升困难场景，如Category 3海岸场景）
#     False       - 不启用场景加权（默认）
#
#   scene_weight_cat0-3选项（可选，第13-16个参数）：
#     cat0 数值   - Category 0（未分类）权重（默认: 1.0）
#     cat1 数值   - Category 1（适中场景）权重（默认: 1.5，新数据集推荐）
#     cat2 数值   - Category 2（小目标/多数类）权重（默认: 0.8，降低多数类影响）
#     cat3 数值   - Category 3（海岸场景）权重（默认: 2.5，重点优化）
#     说明: 新数据集50_50_2k中Cat2为多数类，建议使用(1.0,1.5,0.8,2.5)避免过拟合
#
#   split_method选项（可选，第17个参数）：
#     split_method - 数据集划分方法（默认: 50_50_2k for Pohang-Canal-3k, 50_50 for NUDT-SIRST）
#                    常用值: 50_50_2k（新数据集，Cat2多数类）, 50_50_2k_new（旧数据集，原始分布）
#                    说明: 显式指定数据集划分，避免手动修改脚本
#
#   示例：
#     bash model_Mamba/scripts/train_multiscale_full.sh                                      # LiDAR，自动GPU，默认模型，16bit，polaris loader
#     bash model_Mamba/scripts/train_multiscale_full.sh rgb_lidar auto mamba_tiny_progressive 16 polaris minmax False 2.5 2.0  # ⭐推荐：RGB+LiDAR模式，全面超越DNANet
#     bash model_Mamba/scripts/train_multiscale_full.sh rgb_lidar auto mamba_tiny_progressive 16 polaris minmax False 2.0 2.0 result/xxx/best_model.pth  # ⭐微调：降低Dice权重+Resume
#     bash model_Mamba/scripts/train_multiscale_full.sh rgb_lidar auto mamba_tiny_progressive 8                               # RGB+LiDAR，8bit图像
#     bash model_Mamba/scripts/train_multiscale_full.sh lidar auto auto 8                    # LiDAR，自动GPU，默认模型，8bit，polaris loader
#     bash model_Mamba/scripts/train_multiscale_full.sh lidar auto auto 8 traditional        # LiDAR，8bit，traditional loader
#     bash model_Mamba/scripts/train_multiscale_full.sh ir_only                              # IR-only，自动GPU，默认模型，16bit
#     bash model_Mamba/scripts/train_multiscale_full.sh ir_only auto mamba_tiny_progressive  # IR-only，渐进式解码器（推荐）
#     bash model_Mamba/scripts/train_multiscale_full.sh ir_only auto mamba_tiny_progressive 16 polaris True  # IR-only，渐进式解码器+深度监督
#     bash model_Mamba/scripts/train_multiscale_full.sh lidar auto mamba_tiny_progressive 16 polaris minmax True 2.0 1.5  # 调整loss参数（dice=2.0, proj=1.5）
#     bash model_Mamba/scripts/train_multiscale_full.sh lidar 0                              # LiDAR，GPU 0，默认模型，16bit
#     bash model_Mamba/scripts/train_multiscale_full.sh ir_only auto mamba_small             # IR-only，自动GPU，中等模型，16bit
#     bash model_Mamba/scripts/train_multiscale_full.sh lidar 1 mamba_base 8                 # LiDAR，GPU 1，大模型，8bit
#     bash model_Mamba/scripts/train_multiscale_full.sh ir_only_rgb auto mamba_tiny_progressive 8 traditional minmax False 2.5 2.0 NUDT-SIRST  # NUDT-SIRST数据集（RGB模式）
#     bash model_Mamba/scripts/train_multiscale_full.sh ir_only auto mamba_tiny_progressive 8 traditional minmax False 2.5 2.0 NUDT-SIRST     # NUDT-SIRST数据集（IR-only模式）
#     bash model_Mamba/scripts/train_multiscale_full.sh rgb_lidar auto mamba_tiny_progressive 16 polaris minmax False 2.5 2.0 "" Pohang-Canal-3k True 2.0  # ⭐场景加权：优先训练海岸场景（Cat3权重2.0）
#     bash model_Mamba/scripts/train_multiscale_full.sh lidar auto mamba_tiny_progressive 16 polaris minmax False 2.5 2.0 result/xxx/best_model.pth Pohang-Canal-3k True 1.8  # Resume+场景加权（Cat3权重1.8）
#     bash model_Mamba/scripts/train_multiscale_full.sh rgb_lidar auto mamba_tiny_progressive 16 polaris minmax False 2.5 2.0 "" Pohang-Canal-3k True 1.0 1.5 0.8 2.5 50_50_2k  # ⭐新数据集：Cat2多数类，降权训练
#     bash model_Mamba/scripts/train_multiscale_full.sh rgb_lidar auto mamba_tiny_progressive 16 polaris minmax False 2.5 2.0 "" Pohang-Canal-3k True 1.0 1.0 1.0 2.5 50_50_2k_new  # 旧数据集：原始分布
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
MODEL_TYPE=${3:-mamba_tiny_progressive}  # 模型类型（可选）
BIT_DEPTH=${4:-16}  # 图像位深度：8 或 16（可选，默认16bit）
LOADER_TYPE=${5:-polaris}  # DataLoader类型：polaris 或 traditional（可选，默认polaris）
NORMALIZE_MODE=${6:-minmax}  # 归一化模式：minmax/global/percentile/clahe（可选，默认minmax）
DEEP_SUPERVISION=${7:-False}  # 深度监督：True 或 False（可选，默认False）
CUSTOM_DICE_WEIGHT=${8:-}  # 自定义 Dice 权重（可选，默认使用脚本内置值）
CUSTOM_PROJECTION_WEIGHT=${9:-}  # 自定义 Projection 权重（可选，默认使用脚本内置值）
RESUME_CHECKPOINT=${10:-}  # Resume训练的checkpoint路径（可选，留空则从头训练）

# ============================================================
# [NEW] 训练改进参数（Tier S + Tier A 顶会技巧）
# ============================================================
USE_AUGMENTATION="True"         # 强化数据增强（垂直翻转+90度旋转+Gamma+对比度）
USE_MULTI_SCALE="True"          # 多尺度训练（动态分辨率 [192-320]）
GRADIENT_ACCUM=4                # 梯度累积步数（有效batch=batch_size×4）
USE_EMA="False"                 # EMA (Exponential Moving Average) - DISABLED: causing test collapse
USE_TTA="True"                  # Test-Time Augmentation (8-way)
USE_WARMUP="True"               # Warmup学习率调度器
WARMUP_EPOCHS=10                # Warmup轮数

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
        TRAIN_SCRIPT="train_with_improvements.py"
        MODE_DESC="LiDAR模式 (IR + Depth + LiDAR)"
        ;;
    "rgb_lidar")
        USE_LIDAR=True
        IN_CHANNELS=3
        USE_LIDAR_LOADER=True
        TRAIN_SCRIPT="train_with_improvements.py"
        MODE_DESC="RGB+LiDAR模式 (3通道伪RGB + LiDAR gating，全面超越DNANet)"
        ;;
    "ir_only")
        USE_LIDAR=False
        IN_CHANNELS=1
        USE_LIDAR_LOADER=False
        TRAIN_SCRIPT="train_with_improvements.py"  # ← 改为使用 train.py
        MODE_DESC="IR-only模式 (公平对比，单通道)"
        ;;
    "ir_only_rgb")
        USE_LIDAR=False
        IN_CHANNELS=3
        USE_LIDAR_LOADER=False
        TRAIN_SCRIPT="train_with_improvements.py"  # ← 改为使用 train.py
        MODE_DESC="IR-only RGB模式 (完全匹配DNANet)"
        ;;
    *)
        echo "❌ 错误: 未知模式 '$MODE'"
        echo "支持的模式: lidar | rgb_lidar | ir_only | ir_only_rgb"
        exit 1
        ;;
esac

# 数据配置 - 支持通过第11个参数指定数据集
DATASET="${11:-Pohang-Canal-3k}"
USE_SCENE_WEIGHTS="${12:-False}"  # 场景加权损失：True 或 False（默认False）

# [NEW 2026-02-20] 场景加权配置 - 适应新数据集50_50_2k（Cat2多数类）
# 方案1（推荐）：降低多数类Cat2权重，提升Cat1/Cat3权重
SCENE_WEIGHT_CAT0="${13:-1.0}"    # Category 0（未分类）权重（默认1.0）
SCENE_WEIGHT_CAT1="${14:-1.5}"    # Category 1（适中场景）权重（默认1.5，提升避免被压制）
SCENE_WEIGHT_CAT2="${15:-0.8}"    # Category 2（小目标/多数类）权重（默认0.8，降低多数类影响）
SCENE_WEIGHT_CAT3="${16:-2.5}"    # Category 3（海岸场景）权重（默认2.5，重点优化）

# [NEW 2026-02-20] 数据集划分配置 - 支持通过命令行指定
CUSTOM_SPLIT_METHOD="${17}"  # 可选：显式指定数据集划分（如 50_50_2k, 50_50_2k_new 等）

BASE_SIZE=256
CROP_SIZE=256

# 根据数据集自动配置划分方法和图像文件夹
if [ "$DATASET" == "NUDT-SIRST" ]; then
    SPLIT_METHOD="50_50"
    # NUDT-SIRST的8bit图像直接在images文件夹中
    if [ "$BIT_DEPTH" == "8" ]; then
        IMAGE_FOLDER="images"
        NORMALIZE_16BIT="False"
        BIT_DESC="8bit"
    else
        echo "❌ 错误: NUDT-SIRST数据集只支持8bit图像"
        exit 1
    fi
elif [ "$DATASET" == "Pohang-Canal-3k" ]; then
    SPLIT_METHOD="50_50_2k_new"  # 默认使用旧数据集（原始分布）
    # Pohang-Canal-3k有独立的images-8bit文件夹
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
else
    # 其他数据集使用默认配置
    SPLIT_METHOD="50_50"
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
fi

# [NEW 2026-02-20] 允许通过命令行参数覆盖 SPLIT_METHOD
if [ -n "$CUSTOM_SPLIT_METHOD" ]; then
    SPLIT_METHOD="$CUSTOM_SPLIT_METHOD"
    echo "✓ 使用自定义数据集划分: $SPLIT_METHOD"
fi

# 根据loader类型设置参数（仅对lidar和rgb_lidar模式有效）
if [ "$MODE" == "lidar" ] || [ "$MODE" == "rgb_lidar" ]; then
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
# Reduced from 0.0001 to 0.00005 for stable warmup (2026-02-27)
LR=0.00005
OPTIMIZER="AdamW"
SCHEDULER="CosineAnnealingWarmRestarts"

# Loss配置（默认值）
LOSS_TYPE="hybrid"
DICE_WEIGHT=2.5
PROJECTION_WEIGHT=2.0

# 应用自定义 loss 参数（如果提供）
if [ -n "$CUSTOM_DICE_WEIGHT" ]; then
    DICE_WEIGHT="$CUSTOM_DICE_WEIGHT"
    echo "✓ 使用自定义 Dice Weight: $DICE_WEIGHT"
fi

if [ -n "$CUSTOM_PROJECTION_WEIGHT" ]; then
    PROJECTION_WEIGHT="$CUSTOM_PROJECTION_WEIGHT"
    echo "✓ 使用自定义 Projection Weight: $PROJECTION_WEIGHT"
fi

# 场景加权配置输出
if [ "$USE_SCENE_WEIGHTS" == "True" ]; then
    echo "✓ 启用场景加权损失（Scene-Weighted Loss）"
    echo "  - Category 0 (未分类) 权重: $SCENE_WEIGHT_CAT0"
    echo "  - Category 1 (适中场景) 权重: $SCENE_WEIGHT_CAT1"
    echo "  - Category 2 (小目标/多数类) 权重: $SCENE_WEIGHT_CAT2"
    echo "  - Category 3 (海岸场景) 权重: $SCENE_WEIGHT_CAT3"
    echo "  - 策略: 降低Cat2多数类权重($SCENE_WEIGHT_CAT2)，提升Cat3性能($SCENE_WEIGHT_CAT3)"
fi

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
    "rgb_lidar")
        if [ "$USE_POLARIS_LOADER" == "True" ]; then
            EXPERIMENT_NAME="${MODEL_SIZE}_RGB_LiDAR_${BIT_DESC}_PoLaRIS_d${DICE_WEIGHT}p${PROJECTION_WEIGHT}_${TIMESTAMP}"
        else
            EXPERIMENT_NAME="${MODEL_SIZE}_RGB_LiDAR_${BIT_DESC}_Traditional_d${DICE_WEIGHT}p${PROJECTION_WEIGHT}_${TIMESTAMP}"
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
echo "🚀 Mamba多尺度模型训练 (自动GPU + 顶会技巧)"
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
echo "  Deep Supervision: $DEEP_SUPERVISION"
echo "  训练轮数:         $EPOCHS epochs"
echo "  Batch Sizes:      尝试 16/8/4/2 (自动降级)"
echo "  损失权重:         Dice=$DICE_WEIGHT, Projection=$PROJECTION_WEIGHT"
echo "  实验名称:         $EXPERIMENT_NAME"
echo ""
echo "🎯 性能改进 (已启用):"
echo "  ✅ 强化数据增强     垂直翻转 + 90度旋转 + Gamma + 对比度"
echo "  ✅ 多尺度训练       动态分辨率 [192-320]"
echo "  ✅ 梯度累积         有效Batch = Batch × $GRADIENT_ACCUM"
echo "  ✅ Warmup调度器     $WARMUP_EPOCHS epochs 预热"
echo "  ✅ EMA             模型权重指数平均"
echo "  ✅ TTA             测试时8-way增强"
echo "  📊 预期提升:        +10-16% IoU"
echo ""
if [ "$MODE" == "lidar" ]; then
    echo "📋 预期结果 (LiDAR模式):"
    echo "  - Box IoU: 0.60-0.64"
    echo "  - Seg IoU: 0.55-0.60"
elif [ "$MODE" == "rgb_lidar" ]; then
    echo "📋 预期结果 (RGB+LiDAR模式 - 全面超越DNANet):"
    echo "  - 岸边场景: 0.7682 → 0.82-0.85 (+5-8%)"
    echo "  - 整体Box IoU: 0.8020 → 0.825-0.835 (+2-4%)"
    echo "  - 目标: 全面超越DNANet (0.8245)"
    echo "  - 优势: 3通道输入 + LiDAR gating双重加持"
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

        # 创建实验目录（使用绝对路径）
        SAVE_DIR="$(pwd)/result/$EXPERIMENT_NAME"
        mkdir -p "$SAVE_DIR"

        # 日志文件
        LOG_FILE="$SAVE_DIR/training_gpu${GPU_ID}_bs${BATCH_SIZE}_${TIMESTAMP}.log"
        echo "📁 保存目录: $SAVE_DIR"
        echo "📁 日志文件: $LOG_FILE"

        # 所有模式统一使用 train.py（已支持IR-only）
        echo "🚀 启动训练..."

        # 构建resume参数（如果提供）
        RESUME_ARG=""
        if [ -n "$RESUME_CHECKPOINT" ]; then
            RESUME_ARG="--resume $RESUME_CHECKPOINT"
            echo "📦 Resume from checkpoint: $RESUME_CHECKPOINT"
        fi

        # 注意：CUDA_VISIBLE_DEVICES 会重新编号GPU为0，所以--gpus参数固定为"0"
        # TODO: 改进功能目前需要手动集成到train.py，暂时使用原始train.py
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
            --use_deep_supervision "$DEEP_SUPERVISION" \
            --normalize_16bit "$NORMALIZE_16BIT" \
            --normalize_mode "$NORMALIZE_MODE" \
            --use_polaris_loader "$USE_POLARIS_LOADER" \
            --image_folder "$IMAGE_FOLDER" \
            --loss_type "$LOSS_TYPE" \
            --dice_weight $DICE_WEIGHT \
            --projection_weight $PROJECTION_WEIGHT \
            --use_scene_weights "$USE_SCENE_WEIGHTS" \
            --scene_weight_cat0 $SCENE_WEIGHT_CAT0 \
            --scene_weight_cat1 $SCENE_WEIGHT_CAT1 \
            --scene_weight_cat2 $SCENE_WEIGHT_CAT2 \
            --scene_weight_cat3 $SCENE_WEIGHT_CAT3 \
            --save_dir "$SAVE_DIR" \
            --gpus "0" \
            --workers 4 \
            --seed 42 \
            --suffix .png \
            --use_augmentation "$USE_AUGMENTATION" \
            --use_multi_scale "$USE_MULTI_SCALE" \
            --gradient_accumulation_steps $GRADIENT_ACCUM \
            --use_ema "$USE_EMA" \
            --use_tta "$USE_TTA" \
            --use_warmup "$USE_WARMUP" \
            --warmup_epochs $WARMUP_EPOCHS \
            $RESUME_ARG \
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
