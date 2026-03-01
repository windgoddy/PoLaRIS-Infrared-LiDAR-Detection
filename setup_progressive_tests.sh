#!/bin/bash
# 一键设置渐进式测试环境
# 在服务器上运行此脚本创建所有必要的测试集

echo "======================================================================"
echo "设置渐进式Cat3测试环境"
echo "======================================================================"

# Step 1: 检查必要文件
echo ""
echo "[1/3] 检查必要文件..."
if [ ! -f "cat3_difficulty_analysis.json" ]; then
    echo "❌ 错误: cat3_difficulty_analysis.json 不存在"
    echo "请先运行: python analyze_cat3_difficulty.py --checkpoint <你的checkpoint>"
    exit 1
fi

if [ ! -f "create_progressive_test_splits.py" ]; then
    echo "❌ 错误: create_progressive_test_splits.py 不存在"
    exit 1
fi

echo "✓ 所有必要文件存在"

# Step 2: 创建渐进式测试集
echo ""
echo "[2/3] 创建渐进式测试集..."
python create_progressive_test_splits.py \
  --difficulty_json cat3_difficulty_analysis.json \
  --num_easy 20 \
  --num_medium 50

if [ $? -ne 0 ]; then
    echo "❌ 创建测试集失败"
    exit 1
fi

# Step 3: 验证创建结果
echo ""
echo "[3/3] 验证测试集..."
SPLITS=(
    "dataset/Pohang-Canal-3k/progressive_test_1_baseline"
    "dataset/Pohang-Canal-3k/progressive_test_2_add_easy"
    "dataset/Pohang-Canal-3k/progressive_test_3_add_medium"
    "dataset/Pohang-Canal-3k/progressive_test_4_all_cat3"
)

ALL_OK=true
for split in "${SPLITS[@]}"; do
    if [ -f "$split/train.txt" ] && [ -f "$split/test.txt" ]; then
        echo "  ✓ $split"
    else
        echo "  ❌ $split (缺少 train.txt 或 test.txt)"
        ALL_OK=false
    fi
done

echo ""
if [ "$ALL_OK" = true ]; then
    echo "======================================================================"
    echo "✅ 环境设置完成！现在可以运行渐进式测试："
    echo "======================================================================"
    echo ""
    echo "# 测试DNANet:"
    echo "./progressive_test_commands.sh \\"
    echo "  result/DNANet_baseline_8bit_Pohang-Canal-3k_DNANet_28_01_2026_17_37_58_wDS/best_model_epoch1326_mIoU0.8437.pth.tar \\"
    echo "  --model DNANet --normalize 8bit"
    echo ""
    echo "# 测试Mamba:"
    echo "./progressive_test_commands.sh \\"
    echo "  model_Mamba/result/tiny_LiDAR_16bit_PoLaRIS_d2.5p2.0_20260212_221812/best_model_epoch1148_IoU0.7071_BoxIoU0.7928.pth"
    echo ""
    echo "======================================================================"
else
    echo "❌ 部分测试集创建失败，请检查错误信息"
    exit 1
fi
