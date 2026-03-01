#!/bin/bash
# Progressive Cat3 Testing Commands
# 渐进式Cat3测试脚本 - 支持任意模型
#
# 使用方法:
#   ./progressive_test_commands.sh <checkpoint_path> [额外的test_box_iou.py参数...]
#
# 示例:
#   # 测试特定checkpoint
#   ./progressive_test_commands.sh model_Mamba/result/xxx/best_model.pth
#
#   # 测试并指定GPU
#   ./progressive_test_commands.sh model_Mamba/result/xxx/best_model.pth --gpu 1
#
#   # 测试DNANet等其他模型
#   ./progressive_test_commands.sh dnanet_model/best_model.pth --model DNANet
#
# 功能说明:
#   按照Easy→Medium→Hard顺序逐步加入Cat3样本，观察性能衰减点

# 检查参数
if [ $# -lt 1 ]; then
    echo "❌ 错误: 缺少checkpoint参数"
    echo ""
    echo "使用方法:"
    echo "  $0 <checkpoint_path> [额外参数...]"
    echo ""
    echo "示例:"
    echo "  $0 model_Mamba/result/tiny_LiDAR_16bit_PoLaRIS_d2.5p2.0_20260212_221812/best_model_epoch1148_IoU0.7071_BoxIoU0.7928.pth"
    echo "  $0 model_Mamba/result/xxx/best_model.pth --gpu 1"
    echo "  $0 result/DNANet_baseline_8bit_Pohang-Canal-3k_DNANet_28_01_2026_17_37_58_wDS/best_model_epoch1326_mIoU0.8437.pth.tar --model DNANet --normalize 8bit"
    echo ""
    exit 1
fi

CHECKPOINT="$1"
shift  # 移除第一个参数，保留其余参数

# 检查checkpoint文件是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: checkpoint文件不存在: $CHECKPOINT"
    exit 1
fi

echo "======================================================================"
echo "渐进式Cat3测试"
echo "======================================================================"
echo "Checkpoint: $CHECKPOINT"
echo "额外参数: $@"
echo "======================================================================"
echo ""

# Test 1: Baseline (Cat1+Cat2 only, no Cat3)
echo "【测试 1/4】Baseline (仅Cat1+Cat2, 无Cat3)"
echo "----------------------------------------------------------------------"
python test_box_iou.py \
  --checkpoint "$CHECKPOINT" \
  --split_method progressive_test_1_baseline \
  "$@"
echo ""

# Test 2: Baseline + 20 Easy Cat3
echo "【测试 2/4】Baseline + 20个简单Cat3 (IoU≥0.8)"
echo "----------------------------------------------------------------------"
python test_box_iou.py \
  --checkpoint "$CHECKPOINT" \
  --split_method progressive_test_2_add_easy \
  "$@"
echo ""

# Test 3: Baseline + Easy + 50 Medium Cat3
echo "【测试 3/4】Baseline + Easy + 50个中等Cat3 (IoU 0.5-0.8)"
echo "----------------------------------------------------------------------"
python test_box_iou.py \
  --checkpoint "$CHECKPOINT" \
  --split_method progressive_test_3_add_medium \
  "$@"
echo ""

# Test 4: Baseline + All Cat3 (1015 samples)
echo "【测试 4/4】Baseline + 全部Cat3 (1015个样本)"
echo "----------------------------------------------------------------------"
python test_box_iou.py \
  --checkpoint "$CHECKPOINT" \
  --split_method progressive_test_4_all_cat3 \
  "$@"
echo ""

echo "======================================================================"
echo "✅ 渐进式测试完成！"
echo "======================================================================"
