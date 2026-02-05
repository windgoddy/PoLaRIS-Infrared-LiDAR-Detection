#!/bin/bash
# ============================================================
# 验证阈值一致性：对比不同阈值下的IoU差异
# ============================================================
# 用法:
#   bash scripts/verify_threshold_consistency.sh <checkpoint_path>
#
# 示例:
#   bash scripts/verify_threshold_consistency.sh result/DNANet_baseline_8bit_Pohang-Canal-3k_DNANet_28_01_2026_17_37_58_wDS/latest_best_model.pth.tar
# ============================================================

set -e

CHECKPOINT="$1"

if [ -z "$CHECKPOINT" ]; then
    echo "❌ 错误: 请提供 checkpoint 路径"
    echo ""
    echo "用法: bash scripts/verify_threshold_consistency.sh <checkpoint_path>"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: Checkpoint 文件不存在: $CHECKPOINT"
    exit 1
fi

echo "========================================================================"
echo "阈值一致性验证"
echo "========================================================================"
echo "Checkpoint: $CHECKPOINT"
echo ""
echo "将使用以下阈值进行测试："
echo "  1. 0.3 (DNANet训练时默认)"
echo "  2. 0.5 (测试时默认)"
echo "  3. 动态扫描 [0.1-0.9] (Mamba风格)"
echo ""
echo "目标：验证 Segmentation IoU 和 Box IoU 的一致性"
echo "========================================================================"
echo ""

# 创建结果目录
RESULT_DIR=$(dirname "$CHECKPOINT")
VERIFY_DIR="$RESULT_DIR/threshold_verification_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$VERIFY_DIR"

echo "📁 结果将保存到: $VERIFY_DIR"
echo ""

# ============================================================
# Test 1: 阈值 0.3 (训练时默认)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Test 1: 固定阈值 0.3 (训练时默认)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python test_box_iou.py \
    --checkpoint "$CHECKPOINT" \
    --threshold 0.3 \
    --eval_strategy fixed \
    --gpu 0 \
    --batch_size 4 \
    | tee "$VERIFY_DIR/test_threshold_0.3.log"

echo ""
echo "✅ Test 1 完成"
echo ""
sleep 2

# ============================================================
# Test 2: 阈值 0.5 (测试时默认)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Test 2: 固定阈值 0.5 (测试时默认)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python test_box_iou.py \
    --checkpoint "$CHECKPOINT" \
    --threshold 0.5 \
    --eval_strategy fixed \
    --gpu 0 \
    --batch_size 4 \
    | tee "$VERIFY_DIR/test_threshold_0.5.log"

echo ""
echo "✅ Test 2 完成"
echo ""
sleep 2

# ============================================================
# Test 3: 动态扫描 (Mamba风格)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Test 3: 动态阈值扫描 [0.1-0.9] (Mamba风格)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python test_box_iou.py \
    --checkpoint "$CHECKPOINT" \
    --eval_strategy dynamic \
    --gpu 0 \
    --batch_size 4 \
    | tee "$VERIFY_DIR/test_dynamic_sweep.log"

echo ""
echo "✅ Test 3 完成"
echo ""

# ============================================================
# 汇总结果
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 测试结果汇总"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 提取关键指标
echo "从日志中提取结果..."
echo ""

SUMMARY_FILE="$VERIFY_DIR/summary.txt"

cat > "$SUMMARY_FILE" << EOF
======================================================================
阈值一致性验证结果汇总
======================================================================
Checkpoint: $CHECKPOINT
测试时间: $(date '+%Y-%m-%d %H:%M:%S')

======================================================================
测试配置
======================================================================

Test 1: 固定阈值 0.3 (训练时默认)
  - 评估策略: fixed
  - 阈值: 0.3

Test 2: 固定阈值 0.5 (测试时默认)
  - 评估策略: fixed
  - 阈值: 0.5

Test 3: 动态扫描 [0.1-0.9] (Mamba风格)
  - 评估策略: dynamic
  - 阈值范围: 0.1 - 0.9

======================================================================
关键指标对比
======================================================================

EOF

# 提取 Segmentation IoU 和 Box IoU
echo "正在解析测试结果..."

for TEST_NAME in "threshold_0.3" "threshold_0.5" "dynamic_sweep"; do
    LOG_FILE="$VERIFY_DIR/test_${TEST_NAME}.log"
    
    if [ -f "$LOG_FILE" ]; then
        # 提取 Segmentation IoU
        SEG_IOU=$(grep -E "Segmentation IoU|mean_iou" "$LOG_FILE" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        
        # 提取 Box IoU
        BOX_IOU=$(grep -E "Mask-to-Box IoU|mean_box_iou" "$LOG_FILE" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        
        # 计算差异
        if [ -n "$SEG_IOU" ] && [ -n "$BOX_IOU" ]; then
            DIFF=$(python3 -c "print(abs($SEG_IOU - $BOX_IOU))")
            DIFF_PCT=$(python3 -c "print(abs($SEG_IOU - $BOX_IOU) / $SEG_IOU * 100)")
            
            echo "[$TEST_NAME]" >> "$SUMMARY_FILE"
            echo "  Segmentation IoU: $SEG_IOU" >> "$SUMMARY_FILE"
            echo "  Box IoU:          $BOX_IOU" >> "$SUMMARY_FILE"
            echo "  差异 (绝对值):     $DIFF" >> "$SUMMARY_FILE"
            echo "  差异 (百分比):     ${DIFF_PCT}%" >> "$SUMMARY_FILE"
            echo "" >> "$SUMMARY_FILE"
        else
            echo "[$TEST_NAME]" >> "$SUMMARY_FILE"
            echo "  ⚠️  无法提取指标（请检查日志）" >> "$SUMMARY_FILE"
            echo "" >> "$SUMMARY_FILE"
        fi
    fi
done

cat >> "$SUMMARY_FILE" << EOF
======================================================================
分析建议
======================================================================

1. 如果 threshold_0.3 和 threshold_0.5 的差异 < 1%：
   ✅ Segmentation IoU 和 Box IoU 计算一致
   
2. 如果 threshold_0.3 的 mIoU 更接近训练时记录的值：
   ✅ 应该使用阈值 0.3 进行测试（匹配训练配置）
   
3. 如果 dynamic_sweep 的 mIoU 显著高于固定阈值：
   💡 考虑在训练时也使用动态扫描（但会变慢）
   
4. 如果不同测试的 Box IoU 差异很大：
   ⚠️  可能存在其他问题（数据加载、预处理等）

======================================================================
详细日志
======================================================================

- Test 1: $VERIFY_DIR/test_threshold_0.3.log
- Test 2: $VERIFY_DIR/test_threshold_0.5.log
- Test 3: $VERIFY_DIR/test_dynamic_sweep.log

======================================================================
EOF

# 显示汇总
cat "$SUMMARY_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 验证完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 所有结果已保存到: $VERIFY_DIR"
echo "📄 汇总报告: $VERIFY_DIR/summary.txt"
echo ""
