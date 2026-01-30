#!/bin/bash
# PoLaRIS-Gaussian-Mamba 服务器训练脚本
# 用途：在服务器上一键启动训练
# 作者：PoLaRIS Team
# 日期：2026-01-30

# ==================== 配置区域 ====================

# 数据集配置
DATASET="Pohang-Canal-3k"
DATASET_ROOT="dataset/"
IMAGE_FOLDER="images"
SPLIT_METHOD="50_50"

# 模型配置
MODEL="mamba_small"              # 可选: mamba_tiny, mamba_small, mamba_base
USE_LIDAR="True"                # 是否使用 LiDAR 门控
IN_CHANNELS=1                   # 1=仅红外, 2=红外+深度图

# 训练参数
EPOCHS=200
BATCH_SIZE=8                    # 根据显存调整（单卡 16GB 建议 4-8）
LR=0.0001
OPTIMIZER="AdamW"
SCHEDULER="CosineAnnealingLR"

# GPU 配置
GPUS="0,1"                      # 使用的 GPU ID，多卡用逗号分隔（如 "0,1,2,3"）

# 高斯热力图配置
GAUSSIAN_IOU=0.7                # 高斯半径 IoU 阈值（0.5-0.9）
HEATMAP_DOWNSCALE=1             # 热力图下采样（1=全分辨率, 4=1/4 分辨率）

# 实验名称（用于区分不同实验）
EXPERIMENT_NAME="baseline_server"

# 其他配置
WORKERS=8                       # DataLoader 工作线程数
SAVE_INTERVAL=20                # 每 N 个 epoch 保存一次检查点

# ==================== 训练启动 ====================

echo "========================================"
echo "PoLaRIS-Gaussian-Mamba 训练启动"
echo "========================================"
echo ""
echo "📊 配置信息："
echo "  数据集: $DATASET"
echo "  模型: $MODEL"
echo "  LiDAR 门控: $USE_LIDAR"
echo "  GPU: $GPUS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  学习率: $LR"
echo "  实验名称: $EXPERIMENT_NAME"
echo ""

# 检查数据集是否存在
if [ ! -d "$DATASET_ROOT/$DATASET" ]; then
    echo "❌ 错误: 数据集目录不存在: $DATASET_ROOT/$DATASET"
    exit 1
fi

# 检查必需文件夹
if [ ! -d "$DATASET_ROOT/$DATASET/images" ]; then
    echo "❌ 错误: images/ 文件夹不存在"
    exit 1
fi

if [ ! -d "$DATASET_ROOT/$DATASET/labels" ]; then
    echo "❌ 错误: labels/ 文件夹不存在（需要 YOLO 格式标注）"
    echo "请先运行标注转换脚本或手动创建 labels/ 文件夹"
    exit 1
fi

# 检查 GPU 可用性
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU 状态:"
    nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
    echo ""
else
    echo "⚠️  警告: nvidia-smi 未找到"
fi

# 显示训练命令
echo "========================================"
echo "🚀 开始训练..."
echo "========================================"
echo ""

# 执行训练（从项目根目录运行）
python model_Mamba/train.py \
    --model $MODEL \
    --dataset $DATASET \
    --root $DATASET_ROOT \
    --image_folder $IMAGE_FOLDER \
    --split_method $SPLIT_METHOD \
    --in_channels $IN_CHANNELS \
    --use_lidar $USE_LIDAR \
    --epochs $EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $BATCH_SIZE \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --gaussian_iou $GAUSSIAN_IOU \
    --heatmap_downscale $HEATMAP_DOWNSCALE \
    --workers $WORKERS \
    --gpus $GPUS \
    --experiment_name $EXPERIMENT_NAME \
    --save_interval $SAVE_INTERVAL \
    --normalize_16bit True \
    --base_size 512 \
    --crop_size 480 \
    --seed 42

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ 训练完成！"
    echo "========================================"
    echo ""
    echo "📁 结果保存在: result/$DATASET/${MODEL}_${EXPERIMENT_NAME}/"
    echo ""
    echo "📊 可视化预测结果："
    echo "  bash scripts/visualize_mamba.sh"
    echo ""
else
    echo ""
    echo "❌ 训练失败，请检查错误日志"
    exit 1
fi
