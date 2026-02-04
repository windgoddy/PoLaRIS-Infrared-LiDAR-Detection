#!/bin/bash
# ============================================================
# 对比不同模型在多个固定阈值下的性能（阈值鲁棒性测试）
# ============================================================
# 用法:
#   bash compare_threshold_robustness.sh
#
# 这个脚本会：
# 1. 测试Mamba和DNANet在阈值0.3, 0.4, 0.5, 0.6, 0.7下的性能
# 2. 生成性能对比表格和CSV文件
# 3. 计算性能方差和可用阈值范围
# ============================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🎯 阈值鲁棒性对比实验"
echo "=========================================="
echo ""

# 模型路径配置
MAMBA_CHECKPOINT="model_Mamba/result/Hybrid_a0.25_g2.5_d4.0_p1.0_max_Pohang-Canal-3k_mamba_tiny_20260203_230723/latest_best_model.pth"
DNANET_CHECKPOINT="result/DNANet_baseline_8bit_Pohang-Canal-3k_DNANet_03_02_2026_19_32_11_wDS/latest_best_model.pth.tar"

# 检查模型文件是否存在
if [ ! -f "$MAMBA_CHECKPOINT" ]; then
    echo "❌ Mamba checkpoint 不存在: $MAMBA_CHECKPOINT"
    exit 1
fi

if [ ! -f "$DNANET_CHECKPOINT" ]; then
    echo "❌ DNANet checkpoint 不存在: $DNANET_CHECKPOINT"
    exit 1
fi

echo "✓ Mamba:  $(basename "$(dirname "$MAMBA_CHECKPOINT")")"
echo "✓ DNANet: $(basename "$(dirname "$DNANET_CHECKPOINT")")"
echo ""

# 测试阈值列表
THRESHOLDS=(0.3 0.4 0.5 0.6 0.7)

# 结果目录
RESULT_DIR="threshold_robustness_comparison"
mkdir -p "$RESULT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_CSV="$RESULT_DIR/threshold_comparison_${TIMESTAMP}.csv"
SUMMARY_TXT="$RESULT_DIR/threshold_comparison_${TIMESTAMP}.txt"

# 初始化CSV文件
echo "Model,Threshold,Seg_IoU,Box_IoU,Extreme_Pct,Threshold_01_Pct,Threshold_09_Pct" > "$SUMMARY_CSV"

# 初始化文本报告
{
    echo "=========================================="
    echo "阈值鲁棒性对比实验报告"
    echo "=========================================="
    echo "生成时间: $(date)"
    echo ""
    echo "测试模型:"
    echo "  - Mamba:  $(basename "$(dirname "$MAMBA_CHECKPOINT")")"
    echo "  - DNANet: $(basename "$(dirname "$DNANET_CHECKPOINT")")"
    echo ""
    echo "测试阈值: ${THRESHOLDS[*]}"
    echo ""
    echo "=========================================="
    echo ""
} > "$SUMMARY_TXT"

# 函数：提取结果
extract_results() {
    local result_file=$1
    local model=$2
    local threshold=$3

    if [ ! -f "$result_file" ]; then
        echo "⚠️  未找到结果文件: $result_file"
        return
    fi

    # 提取指标
    seg_iou=$(grep "Segmentation IoU" "$result_file" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    box_iou=$(grep "Mask-to-Box IoU" "$result_file" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")

    # 提取阈值分布（动态扫描才有，固定阈值策略返回N/A）
    thresh_01=$(grep "0.1:" "$result_file" | grep -oE "[0-9]+\.[0-9]+%" | head -1 | sed 's/%//' || echo "")
    thresh_09=$(grep "0.9:" "$result_file" | grep -oE "[0-9]+\.[0-9]+%" | head -1 | sed 's/%//' || echo "")

    # 计算极端阈值占比
    if [[ -n "$thresh_01" && -n "$thresh_09" ]]; then
        # 使用 bc 进行浮点运算（更安全）
        extreme=$(echo "$thresh_01 + $thresh_09" | bc)
    elif [[ -n "$thresh_01" ]]; then
        extreme="$thresh_01"
    elif [[ -n "$thresh_09" ]]; then
        extreme="$thresh_09"
    else
        extreme="N/A"
    fi
    
    # 如果阈值为空，用0填充（避免CSV格式错误）
    thresh_01="${thresh_01:-0}"
    thresh_09="${thresh_09:-0}"

    # 保存到CSV
    echo "$model,$threshold,$seg_iou,$box_iou,$extreme,$thresh_01,$thresh_09" >> "$SUMMARY_CSV"

    # 显示结果
    echo "  Seg IoU: $seg_iou"
    echo "  Box IoU: $box_iou"
    echo "  极端阈值: $extreme%"
}

# 测试Mamba模型
echo "=========================================="
echo "🔬 测试 Mamba 模型"
echo "=========================================="
echo ""

for threshold in "${THRESHOLDS[@]}"; do
    echo "----------------------------------------"
    echo "阈值: $threshold"
    echo "----------------------------------------"

    # 运行测试
    bash test_checkpoint.sh "$MAMBA_CHECKPOINT" 0 \
        --eval_strategy fixed \
        --threshold "$threshold" \
        --batch_size 1

    # 提取结果
    result_file="$(dirname "$MAMBA_CHECKPOINT")/box_iou_test_results.txt"
    extract_results "$result_file" "Mamba" "$threshold"

    echo ""
done

# 测试DNANet模型
echo "=========================================="
echo "🔬 测试 DNANet 模型"
echo "=========================================="
echo ""

for threshold in "${THRESHOLDS[@]}"; do
    echo "----------------------------------------"
    echo "阈值: $threshold"
    echo "----------------------------------------"

    # 运行测试
    bash test_checkpoint.sh "$DNANET_CHECKPOINT" 0 \
        --eval_strategy fixed \
        --threshold "$threshold" \
        --batch_size 4

    # 提取结果
    result_file="$(dirname "$DNANET_CHECKPOINT")/box_iou_test_results.txt"
    extract_results "$result_file" "DNANet" "$threshold"

    echo ""
done

echo "=========================================="
echo "📊 生成对比报告"
echo "=========================================="
echo ""

# 生成表格（使用column命令对齐）
{
    echo "=========================================="
    echo "详细结果表格"
    echo "=========================================="
    echo ""
} >> "$SUMMARY_TXT"

cat "$SUMMARY_CSV" | column -t -s, >> "$SUMMARY_TXT"

# 计算统计信息
{
    echo ""
    echo "=========================================="
    echo "统计分析"
    echo "=========================================="
    echo ""
} >> "$SUMMARY_TXT"

# 使用Python进行统计分析
python3 - "$SUMMARY_CSV" << 'PYTHON_SCRIPT' >> "$SUMMARY_TXT"
import pandas as pd
import numpy as np
import sys

# 读取CSV（从命令行参数获取）
csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# 确保数值列是float类型
for col in ['Seg_IoU', 'Box_IoU', 'Extreme_Pct']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 分别统计Mamba和DNANet
mamba_df = df[df['Model'] == 'Mamba'].copy()
dnanet_df = df[df['Model'] == 'DNANet'].copy()

print("【Mamba 性能统计】")
print(f"  最佳 Box IoU: {mamba_df['Box_IoU'].max():.4f} (阈值 {mamba_df.loc[mamba_df['Box_IoU'].idxmax(), 'Threshold']:.1f})")
print(f"  最差 Box IoU: {mamba_df['Box_IoU'].min():.4f} (阈值 {mamba_df.loc[mamba_df['Box_IoU'].idxmin(), 'Threshold']:.1f})")
print(f"  平均 Box IoU: {mamba_df['Box_IoU'].mean():.4f}")
print(f"  标准差:       {mamba_df['Box_IoU'].std():.4f}")
print(f"  变异系数:     {(mamba_df['Box_IoU'].std() / mamba_df['Box_IoU'].mean() * 100):.2f}%")
print()

print("【DNANet 性能统计】")
print(f"  最佳 Box IoU: {dnanet_df['Box_IoU'].max():.4f} (阈值 {dnanet_df.loc[dnanet_df['Box_IoU'].idxmax(), 'Threshold']:.1f})")
print(f"  最差 Box IoU: {dnanet_df['Box_IoU'].min():.4f} (阈值 {dnanet_df.loc[dnanet_df['Box_IoU'].idxmin(), 'Threshold']:.1f})")
print(f"  平均 Box IoU: {dnanet_df['Box_IoU'].mean():.4f}")
print(f"  标准差:       {dnanet_df['Box_IoU'].std():.4f}")
print(f"  变异系数:     {(dnanet_df['Box_IoU'].std() / dnanet_df['Box_IoU'].mean() * 100):.2f}%")
print()

print("【鲁棒性对比】")
mamba_std = mamba_df['Box_IoU'].std()
dnanet_std = dnanet_df['Box_IoU'].std()
print(f"  Mamba 方差:   {mamba_std**2:.6f}")
print(f"  DNANet 方差:  {dnanet_std**2:.6f}")
print(f"  稳定性提升:   {(dnanet_std**2 / mamba_std**2):.1f}x (Mamba更稳定)" if mamba_std < dnanet_std else f"  稳定性下降:   {(mamba_std**2 / dnanet_std**2):.1f}x (DNANet更稳定)")
print()

# 计算可用阈值范围（性能下降<5%的范围）
mamba_max_iou = mamba_df['Box_IoU'].max()
dnanet_max_iou = dnanet_df['Box_IoU'].max()

mamba_usable = mamba_df[mamba_df['Box_IoU'] >= mamba_max_iou * 0.95]['Threshold']
dnanet_usable = dnanet_df[dnanet_df['Box_IoU'] >= dnanet_max_iou * 0.95]['Threshold']

print("【可用阈值范围】(性能下降<5%)")
if len(mamba_usable) > 0:
    print(f"  Mamba:  {mamba_usable.min():.1f} - {mamba_usable.max():.1f} (宽度: {mamba_usable.max() - mamba_usable.min():.1f})")
else:
    print(f"  Mamba:  无")

if len(dnanet_usable) > 0:
    print(f"  DNANet: {dnanet_usable.min():.1f} - {dnanet_usable.max():.1f} (宽度: {dnanet_usable.max() - dnanet_usable.min():.1f})")
else:
    print(f"  DNANet: 无")
print()

print("【置信度校准】")
print(f"  Mamba 平均极端阈值:  {mamba_df['Extreme_Pct'].mean():.1f}%")
print(f"  DNANet 平均极端阈值: {dnanet_df['Extreme_Pct'].mean():.1f}%")
print(f"  Mamba 优势:          {(dnanet_df['Extreme_Pct'].mean() - mamba_df['Extreme_Pct'].mean()):.1f}% (更少极端阈值)")
print()

PYTHON_SCRIPT

# 添加结论
{
    echo ""
    echo "=========================================="
    echo "🎯 关键发现"
    echo "=========================================="
    echo ""
    echo "1. 性能稳定性："
    echo "   - 查看上方标准差对比"
    echo "   - 标准差越小，模型越稳定"
    echo ""
    echo "2. 阈值鲁棒性："
    echo "   - 查看可用阈值范围"
    echo "   - 范围越宽，部署越容易"
    echo ""
    echo "3. 置信度校准："
    echo "   - 查看极端阈值占比"
    echo "   - 占比越低，校准越好"
    echo ""
    echo "=========================================="
    echo "📁 文件位置"
    echo "=========================================="
    echo ""
    echo "CSV数据:   $SUMMARY_CSV"
    echo "文本报告:  $SUMMARY_TXT"
    echo ""
} >> "$SUMMARY_TXT"

# 显示报告
echo "📊 完整报告："
echo ""
cat "$SUMMARY_TXT"

echo ""
echo "=========================================="
echo "✅ 实验完成"
echo "=========================================="
echo ""
echo "结果文件："
echo "  - CSV数据:   $SUMMARY_CSV"
echo "  - 文本报告:  $SUMMARY_TXT"
echo ""
echo "💡 提示："
echo "  可以使用这些数据绘制阈值-性能曲线图（论文关键图表）"
echo ""
