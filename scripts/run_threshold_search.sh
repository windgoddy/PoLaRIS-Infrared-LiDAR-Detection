#!/bin/bash
# 最优阈值搜索 - 快速启动脚本
# 用法: bash scripts/run_threshold_search.sh

# ============================================================================
# 配置参数（请根据实际情况修改）
# ============================================================================

# 模型路径
MODEL_PATH="result/PoLaRIS_16bit_full_Pohang-Canal-3k_MS_CAFNet_DualGeo_19_01_2026_17_16_02_wDS/best_model_epoch0287_mIoU0.6674.pth.tar"

# 数据集路径
# 注意：脚本会在 dataset_dir/split_data/test.txt 或 dataset_dir/50_50/test.txt 中查找
DATASET_DIR="dataset/Pohang-Canal"

# 数据划分（train/test/val）
# 如果使用 50_50 分组，脚本会自动在 dataset/Pohang-Canal/50_50/test.txt 中查找
SPLIT="test"

# 输出目录
OUTPUT_DIR="results/threshold_search_$(date +%Y%m%d_%H%M%S)"

# 阈值范围
TH_MIN=0.1
TH_MAX=0.7
TH_STEP=0.05

# 优化目标（iou/f1/precision/recall）
METRIC="iou"

# 批处理大小
BATCH_SIZE=8

# 是否进行分组分析
GROUP_ANALYSIS="--group_analysis"

# 是否保存预测结果
SAVE_PREDICTIONS=""

# ============================================================================
# 执行脚本
# ============================================================================

echo "========================================================================"
echo "🚀 最优阈值搜索"
echo "========================================================================"
echo "模型路径: $MODEL_PATH"
echo "数据集: $DATASET_DIR"
echo "数据划分: $SPLIT"
echo "阈值范围: [$TH_MIN, $TH_MAX], 步长: $TH_STEP"
echo "优化目标: $METRIC"
echo "输出目录: $OUTPUT_DIR"
echo "========================================================================"
echo ""

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_PATH"
    echo "请修改 MODEL_PATH 变量为正确的模型路径"
    exit 1
fi

# 检查数据集目录是否存在
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ 错误: 数据集目录不存在: $DATASET_DIR"
    echo "请修改 DATASET_DIR 变量为正确的数据集路径"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行脚本
python scripts/find_optimal_threshold.py \
    --model_path "$MODEL_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --th_min $TH_MIN \
    --th_max $TH_MAX \
    --th_step $TH_STEP \
    --metric "$METRIC" \
    --batch_size $BATCH_SIZE \
    $GROUP_ANALYSIS \
    $SAVE_PREDICTIONS

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ 阈值搜索完成！"
    echo "========================================================================"
    echo "结果保存在: $OUTPUT_DIR"
    echo ""
    echo "生成的文件:"
    echo "  - threshold_curve.png          # 阈值-性能曲线图"
    echo "  - group_comparison.png         # 分组对比图（如果启用）"
    echo "  - threshold_search_summary.json # 结果摘要（JSON格式）"
    echo ""
    echo "查看结果摘要:"
    echo "  cat $OUTPUT_DIR/threshold_search_summary.json | python -m json.tool"
    echo ""
else
    echo ""
    echo "❌ 执行失败，请检查错误信息"
    exit 1
fi
