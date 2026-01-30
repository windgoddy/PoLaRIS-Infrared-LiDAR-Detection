#!/bin/bash
# PoLaRIS-Gaussian-Mamba 可视化脚本
# 用途：生成预测结果的可视化图像
# 作者：PoLaRIS Team
# 日期：2026-01-30

# ==================== 配置区域 ====================

# 模型配置（需要与训练时一致）
MODEL="mamba_small"
USE_LIDAR="True"
EXPERIMENT_NAME="baseline_server"

# 数据集配置
DATASET="Pohang-Canal-3k"
DATASET_ROOT="dataset/"

# 检查点路径（自动查找 best_model.pth）
CHECKPOINT="result/$DATASET/${MODEL}_${EXPERIMENT_NAME}/best_model.pth"

# 可视化参数
NUM_SAMPLES=20                  # 可视化样本数量
PEAK_THRESHOLD=0.5              # 峰值检测阈值
NMS_RADIUS=10                   # 非极大值抑制半径
OUTPUT_DIR="visualizations/${MODEL}_${EXPERIMENT_NAME}"

# GPU 配置
GPUS="0"

# ==================== 可视化启动 ====================

echo "========================================"
echo "PoLaRIS-Gaussian-Mamba 可视化"
echo "========================================"
echo ""

# 检查模型文件是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: 模型文件不存在: $CHECKPOINT"
    echo ""
    echo "可用的模型文件："
    find result/$DATASET/ -name "*.pth" 2>/dev/null || echo "  未找到任何模型文件"
    echo ""
    echo "请修改脚本中的 CHECKPOINT 路径，或先运行训练脚本"
    exit 1
fi

echo "✅ 找到模型文件: $CHECKPOINT"
echo ""

# 检查模型信息
if python -c "import torch; ckpt=torch.load('$CHECKPOINT', map_location='cpu'); print(f'模型信息: Epoch={ckpt[\"epoch\"]}, IoU={ckpt.get(\"best_iou\", \"N/A\")}')" 2>/dev/null; then
    echo ""
else
    echo "⚠️  无法读取模型文件信息"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "🎨 开始生成可视化..."
echo "========================================"
echo ""
echo "配置："
echo "  样本数量: $NUM_SAMPLES"
echo "  峰值阈值: $PEAK_THRESHOLD"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 执行可视化（从项目根目录运行）
python model_Mamba/visualize.py \
    --checkpoint $CHECKPOINT \
    --model $MODEL \
    --use_lidar $USE_LIDAR \
    --dataset $DATASET \
    --root $DATASET_ROOT \
    --num_samples $NUM_SAMPLES \
    --output_dir $OUTPUT_DIR \
    --peak_threshold $PEAK_THRESHOLD \
    --nms_radius $NMS_RADIUS \
    --gpus $GPUS

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ 可视化完成！"
    echo "========================================"
    echo ""
    echo "📁 可视化结果保存在: $OUTPUT_DIR/"
    echo ""
    echo "生成的文件数量:"
    ls -1 "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l || echo "0"
    echo ""
    echo "📊 查看结果:"
    echo "  cd $OUTPUT_DIR && ls -lh"
    echo ""
    echo "💡 提示: 可以使用 scp 下载到本地查看:"
    echo "  scp -r user@server:$(pwd)/$OUTPUT_DIR ."
    echo ""
else
    echo ""
    echo "❌ 可视化失败，请检查错误日志"
    exit 1
fi
