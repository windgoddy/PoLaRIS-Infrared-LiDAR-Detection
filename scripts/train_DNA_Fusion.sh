#!/bin/bash
#
# DNA-FusionNet Early Fusion 训练启动脚本
#

# 设置参数
GPU_ID=${1:-5}
EPOCHS=${2:-50}

DATASET="Pohang-Canal-3k"
SPLIT_METHOD="50_50_2k_new"
BATCH_SIZE=4
LR=0.0001

echo "===================================="
echo "DNA-FusionNet Early Fusion 训练"
echo "===================================="
echo "GPU: $GPU_ID"
echo "Epochs: $EPOCHS"
echo "Dataset: $DATASET"
echo "Split: $SPLIT_METHOD"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "===================================="
echo ""

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 启动训练
python model_DNAFusionNet/train_DNAFusion_EarlyFusion.py \
    --dataset $DATASET \
    --split_method $SPLIT_METHOD \
    --gpu 0 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --threshold 0.3

echo ""
echo "===================================="
echo "训练完成！"
echo "===================================="
