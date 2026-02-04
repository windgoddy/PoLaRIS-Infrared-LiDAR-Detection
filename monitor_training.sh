#!/bin/bash
# ============================================================
# 监控训练进度，定期测试checkpoint
# ============================================================
# 用法:
#   bash monitor_training.sh <experiment_dir> [OPTIONS]
#
# 示例:
#   bash monitor_training.sh model_Mamba/result/Hybrid_warmrestarts_d4p1_*
#   bash monitor_training.sh model_Mamba/result/Hybrid_warmrestarts_d2.5p2_* --interval 100
#   bash monitor_training.sh model_Mamba/result/Hybrid_* --eval_strategy dynamic --batch_size 1
# ============================================================

set -e

# 默认参数
INTERVAL=50           # 测试间隔（每N个epoch测试一次）
EVAL_STRATEGY="dynamic"  # 评估策略：dynamic|fixed|auto
BATCH_SIZE=1          # 批次大小（dynamic扫描建议用1避免OOM）
THRESHOLD=0.5         # 固定阈值（仅在fixed策略下使用）
BACKGROUND=false      # 是否后台运行

# 解析参数
if [ $# -lt 1 ]; then
    echo "❌ 错误: 缺少实验目录路径"
    echo ""
    echo "用法: bash monitor_training.sh <experiment_dir> [OPTIONS]"
    echo ""
    echo "必需参数:"
    echo "  experiment_dir          实验目录路径（支持通配符）"
    echo ""
    echo "可选参数:"
    echo "  --interval N            测试间隔epoch数 (default: 50)"
    echo "  --eval_strategy STR     评估策略 (dynamic|fixed|auto, default: dynamic)"
    echo "  --batch_size N          批次大小 (default: 1)"
    echo "  --threshold VALUE       固定阈值 (default: 0.5, 仅在fixed策略下使用)"
    echo "  --background            后台运行（将输出重定向到日志文件）"
    echo ""
    echo "示例:"
    echo "  bash monitor_training.sh model_Mamba/result/Hybrid_warmrestarts_d4p1_*"
    echo "  bash monitor_training.sh model_Mamba/result/Hybrid_* --interval 100"
    echo "  bash monitor_training.sh model_Mamba/result/Hybrid_* --background"
    exit 1
fi

EXP_DIR="$1"
shift

# 解析可选参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --eval_strategy)
            EVAL_STRATEGY="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        *)
            echo "⚠️  未知参数: $1（忽略）"
            shift
            ;;
    esac
done

# 展开通配符（如果使用了通配符）
EXP_DIR=$(echo $EXP_DIR)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "==========================================="
echo "📊 训练监控系统"
echo "==========================================="
echo "  实验目录: $EXP_DIR"
echo "  测试间隔: 每 $INTERVAL epoch"
echo "  评估策略: $EVAL_STRATEGY"
echo "  批次大小: $BATCH_SIZE"
if [ "$EVAL_STRATEGY" = "fixed" ]; then
    echo "  固定阈值: $THRESHOLD"
fi
echo "==========================================="
echo ""

# 检查实验目录是否存在
if [ ! -d "$EXP_DIR" ]; then
    echo "❌ 错误: 实验目录不存在: $EXP_DIR"
    exit 1
fi

# 如果后台运行，重定向输出
if [ "$BACKGROUND" = true ]; then
    MONITOR_LOG="$EXP_DIR/monitor_training.log"
    echo "📝 后台运行模式，日志输出到: $MONITOR_LOG"
    exec > "$MONITOR_LOG" 2>&1
    echo "==========================================="
    echo "📊 训练监控系统 (后台运行)"
    echo "==========================================="
    echo "  开始时间: $(date)"
    echo "  实验目录: $EXP_DIR"
    echo "  测试间隔: 每 $INTERVAL epoch"
    echo "  评估策略: $EVAL_STRATEGY"
    echo "==========================================="
    echo ""
fi

# 查找所有checkpoint
echo "🔍 搜索 checkpoint 文件..."
CHECKPOINTS=$(find "$EXP_DIR" -name "checkpoint_epoch_*.pth" -o -name "checkpoint_epoch*.pth" | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ 未找到checkpoint文件"
    echo ""
    echo "💡 提示："
    echo "  1. 确认实验目录路径正确"
    echo "  2. 确认训练已开始保存checkpoint"
    echo "  3. checkpoint文件命名格式应为: checkpoint_epoch_*.pth 或 checkpoint_epoch*.pth"
    exit 1
fi

CHECKPOINT_COUNT=$(echo "$CHECKPOINTS" | wc -l | tr -d ' ')
echo "✓ 找到 $CHECKPOINT_COUNT 个checkpoints"
echo ""

# 创建结果汇总文件
SUMMARY_FILE="$EXP_DIR/checkpoint_comparison_${EVAL_STRATEGY}.txt"
SUMMARY_CSV="$EXP_DIR/checkpoint_comparison_${EVAL_STRATEGY}.csv"

# CSV 表头
echo "Epoch,Seg_IoU,Box_IoU,Extreme_Pct,Threshold_01_Pct,Threshold_09_Pct,Test_Time" > "$SUMMARY_CSV"

# 创建人类可读的汇总文件表头
{
    echo "==========================================="
    echo "Checkpoint 性能对比汇总"
    echo "==========================================="
    echo "实验目录: $EXP_DIR"
    echo "评估策略: $EVAL_STRATEGY"
    echo "测试时间: $(date)"
    echo "==========================================="
    echo ""
} > "$SUMMARY_FILE"

# 已测试的checkpoint记录（避免重复测试）
TESTED_EPOCHS=""
if [ -f "$SUMMARY_CSV" ]; then
    TESTED_EPOCHS=$(tail -n +2 "$SUMMARY_CSV" | cut -d',' -f1)
fi

# 测试每个checkpoint
TESTED_COUNT=0
SKIPPED_COUNT=0

for ckpt in $CHECKPOINTS; do
    # 提取epoch数（支持多种命名格式）
    if [[ "$ckpt" =~ checkpoint_epoch_([0-9]+)\.pth ]] || [[ "$ckpt" =~ checkpoint_epoch([0-9]+)\.pth ]]; then
        epoch="${BASH_REMATCH[1]}"
    else
        echo "⚠️  无法解析epoch数: $ckpt (跳过)"
        continue
    fi

    # 检查是否已测试过
    if echo "$TESTED_EPOCHS" | grep -q "^${epoch}$"; then
        echo "⏭️  Epoch $epoch 已测试，跳过"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi

    # 检查是否符合测试间隔
    if [ $((epoch % INTERVAL)) -ne 0 ]; then
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi

    echo "==========================================="
    echo "🧪 测试 Epoch $epoch ($((TESTED_COUNT + 1))/$CHECKPOINT_COUNT)"
    echo "==========================================="
    echo "  Checkpoint: $(basename "$ckpt")"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # 记录测试开始时间
    TEST_START=$(date +%s)

    # 运行测试
    if [ "$EVAL_STRATEGY" = "fixed" ]; then
        bash test_checkpoint.sh "$ckpt" 0 \
            --eval_strategy fixed \
            --threshold "$THRESHOLD" \
            --batch_size "$BATCH_SIZE"
    else
        bash test_checkpoint.sh "$ckpt" 0 \
            --eval_strategy "$EVAL_STRATEGY" \
            --batch_size "$BATCH_SIZE"
    fi

    # 记录测试结束时间
    TEST_END=$(date +%s)
    TEST_DURATION=$((TEST_END - TEST_START))

    # 提取结果
    result_file="$(dirname "$ckpt")/box_iou_test_results.txt"
    if [ -f "$result_file" ]; then
        # 提取 IoU 指标
        seg_iou=$(grep "Segmentation IoU" "$result_file" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
        box_iou=$(grep "Mask-to-Box IoU" "$result_file" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")

        # 提取阈值分布
        thresh_01=$(grep "0.1:" "$result_file" | grep -oE "[0-9]+\.[0-9]+%" | head -1 | sed 's/%//' || echo "0")
        thresh_09=$(grep "0.9:" "$result_file" | grep -oE "[0-9]+\.[0-9]+%" | head -1 | sed 's/%//' || echo "0")

        # 计算极端阈值占比
        if [[ "$thresh_01" != "0" && "$thresh_09" != "0" ]]; then
            extreme=$(awk "BEGIN {printf \"%.1f\", ${thresh_01} + ${thresh_09}}")
        else
            extreme="N/A"
        fi

        # 保存到 CSV
        echo "$epoch,$seg_iou,$box_iou,$extreme,$thresh_01,$thresh_09,$TEST_DURATION" >> "$SUMMARY_CSV"

        # 显示结果摘要
        echo ""
        echo "📈 结果:"
        echo "  Seg IoU:        $seg_iou"
        echo "  Box IoU:        $box_iou"
        echo "  极端阈值:       $extreme%"
        echo "  测试耗时:       ${TEST_DURATION}s"

        TESTED_COUNT=$((TESTED_COUNT + 1))
    else
        echo ""
        echo "⚠️  未找到结果文件: $result_file"
    fi

    echo ""
done

echo "==========================================="
echo "✅ 监控完成"
echo "==========================================="
echo "  总checkpoint数: $CHECKPOINT_COUNT"
echo "  已测试:        $TESTED_COUNT"
echo "  跳过:          $SKIPPED_COUNT"
echo "==========================================="
echo ""

# 生成人类可读的汇总报告
{
    echo "测试完成统计:"
    echo "  - 总checkpoint数: $CHECKPOINT_COUNT"
    echo "  - 已测试: $TESTED_COUNT"
    echo "  - 跳过: $SKIPPED_COUNT"
    echo ""
    echo "==========================================="
    echo "性能汇总表格"
    echo "==========================================="
    echo ""
} >> "$SUMMARY_FILE"

# 将CSV转换为对齐的表格
if [ -f "$SUMMARY_CSV" ] && [ $(wc -l < "$SUMMARY_CSV") -gt 1 ]; then
    cat "$SUMMARY_CSV" | column -t -s, >> "$SUMMARY_FILE"
else
    echo "⚠️  没有测试数据" >> "$SUMMARY_FILE"
fi

echo "" >> "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"
echo "🏆 最佳性能" >> "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 找到最佳checkpoint（只处理有效数据）
if [ -f "$SUMMARY_CSV" ] && [ $(wc -l < "$SUMMARY_CSV") -gt 1 ]; then
    # 最佳 Box IoU
    best_box_line=$(tail -n +2 "$SUMMARY_CSV" | grep -v "N/A" | sort -t, -k3 -rn | head -1)
    if [ -n "$best_box_line" ]; then
        best_box_epoch=$(echo "$best_box_line" | cut -d',' -f1)
        best_box_iou=$(echo "$best_box_line" | cut -d',' -f3)
        {
            echo "🥇 最佳 Box IoU: $best_box_iou (Epoch $best_box_epoch)"
            echo "   完整数据: $best_box_line"
            echo ""
        } >> "$SUMMARY_FILE"
    fi

    # 最佳 Seg IoU
    best_seg_line=$(tail -n +2 "$SUMMARY_CSV" | grep -v "N/A" | sort -t, -k2 -rn | head -1)
    if [ -n "$best_seg_line" ]; then
        best_seg_epoch=$(echo "$best_seg_line" | cut -d',' -f1)
        best_seg_iou=$(echo "$best_seg_line" | cut -d',' -f2)
        {
            echo "🥇 最佳 Seg IoU: $best_seg_iou (Epoch $best_seg_epoch)"
            echo "   完整数据: $best_seg_line"
            echo ""
        } >> "$SUMMARY_FILE"
    fi

    # 最佳置信度校准（极端阈值最低）
    best_calib_line=$(tail -n +2 "$SUMMARY_CSV" | grep -v "N/A" | sort -t, -k4 -n | head -1)
    if [ -n "$best_calib_line" ]; then
        best_calib_epoch=$(echo "$best_calib_line" | cut -d',' -f1)
        best_calib_extreme=$(echo "$best_calib_line" | cut -d',' -f4)
        {
            echo "🎯 最佳置信度校准: $best_calib_extreme% 极端阈值 (Epoch $best_calib_epoch)"
            echo "   完整数据: $best_calib_line"
            echo ""
        } >> "$SUMMARY_FILE"
    fi
fi

{
    echo "==========================================="
    echo "📁 文件位置"
    echo "==========================================="
    echo ""
    echo "CSV数据: $SUMMARY_CSV"
    echo "文本报告: $SUMMARY_FILE"
    echo ""
} >> "$SUMMARY_FILE"

# 显示汇总报告
echo "📊 性能汇总:"
echo ""
cat "$SUMMARY_FILE"
echo ""

echo "==========================================="
echo "📁 结果文件"
echo "==========================================="
echo "  CSV数据:   $SUMMARY_CSV"
echo "  文本报告:  $SUMMARY_FILE"
echo "==========================================="
echo ""

# 如果是后台运行，发送通知（如果有 say 命令）
if [ "$BACKGROUND" = true ] && command -v say &> /dev/null; then
    say "Training monitoring completed. Tested $TESTED_COUNT checkpoints."
fi
