#!/bin/bash

# ============================================================
# Scripts 文件夹清理脚本
# ============================================================
# 用法:
#   ./scripts/cleanup.sh          # 查看要归档的文件
#   ./scripts/cleanup.sh --archive # 归档旧文件
#   ./scripts/cleanup.sh --delete  # 删除旧文件（危险！）
# ============================================================

MODE=${1:---dry-run}

echo "============================================================"
echo "Scripts 文件夹清理工具"
echo "============================================================"

# 创建归档文件夹
if [[ "$MODE" == "--archive" || "$MODE" == "--delete" ]]; then
    mkdir -p scripts/archive/{training,data_processing,visualization,misc}
fi

# 定义要归档的文件
TRAINING_SCRIPTS=(
    "run_Phase3.sh"
    "run_Phase3_DualGeo.sh"
    "run_Phase3_DualGeo_8bit.sh"
    "run_Phase3_DualGeo_16bit.sh"
    "run_Phase3_DualGeo_16bit_IR_only.sh"
    "run_Phase3_DualGeo_all_dataset.sh"
    "run_Phase3_ResidualFPN.sh"
    "run_Phase3_improved_v3.sh"
    "run_baseline1.sh"
    "run_baseline2.sh"
    "run_evaluate_golden_set.sh"
)

DATA_PROCESSING=(
    "fix_dataset_with_normalization.py"
    "run_fix_with_normalization.sh"
    "fix_line_endings.py"
    "prepare_all_data.py"
    "prepare_dataset_complete.py"
    "generate_single_oracle.py"
)

VISUALIZATION=(
    "visualize_raw_01_fixed.py"
    "run_visualize_lidar.sh"
)

MISC=(
    "prepare_labels_yolo.py"
    "generate_lr_images.py"
    "project_lidar_to_ir_minimal.py"
)

# 统计函数
count_files() {
    local count=0
    for file in "$@"; do
        if [[ -f "scripts/$file" ]]; then
            ((count++))
        fi
    done
    echo $count
}

# 显示要处理的文件
show_files() {
    local category=$1
    shift
    local files=("$@")

    echo ""
    echo "📦 $category:"
    for file in "${files[@]}"; do
        if [[ -f "scripts/$file" ]]; then
            echo "   - $file"
        fi
    done
}

# 处理文件
process_files() {
    local action=$1
    local dest=$2
    shift 2
    local files=("$@")

    for file in "${files[@]}"; do
        if [[ -f "scripts/$file" ]]; then
            if [[ "$action" == "archive" ]]; then
                mv "scripts/$file" "$dest/"
                echo "   ✓ 已归档: $file"
            elif [[ "$action" == "delete" ]]; then
                rm "scripts/$file"
                echo "   ✗ 已删除: $file"
            fi
        fi
    done
}

# 执行清理
case $MODE in
    --archive)
        echo ""
        echo "📦 开始归档旧文件..."
        echo "============================================================"

        echo ""
        echo "📦 训练脚本 → scripts/archive/training/"
        process_files "archive" "scripts/archive/training" "${TRAINING_SCRIPTS[@]}"

        echo ""
        echo "📦 数据处理脚本 → scripts/archive/data_processing/"
        process_files "archive" "scripts/archive/data_processing" "${DATA_PROCESSING[@]}"

        echo ""
        echo "📦 可视化脚本 → scripts/archive/visualization/"
        process_files "archive" "scripts/archive/visualization" "${VISUALIZATION[@]}"

        echo ""
        echo "📦 其他脚本 → scripts/archive/misc/"
        process_files "archive" "scripts/archive/misc" "${MISC[@]}"

        echo ""
        echo "============================================================"
        echo "✅ 归档完成！旧文件已移动到 scripts/archive/"
        echo ""
        echo "恢复文件: cp scripts/archive/training/run_Phase3.sh scripts/"
        ;;

    --delete)
        echo ""
        read -p "⚠️  确定要删除这些文件吗？(输入 YES 确认): " confirm

        if [[ "$confirm" == "YES" ]]; then
            echo ""
            echo "🗑️  开始删除旧文件..."
            echo "============================================================"

            echo ""
            echo "🗑️  训练脚本"
            process_files "delete" "" "${TRAINING_SCRIPTS[@]}"

            echo ""
            echo "🗑️  数据处理脚本"
            process_files "delete" "" "${DATA_PROCESSING[@]}"

            echo ""
            echo "🗑️  可视化脚本"
            process_files "delete" "" "${VISUALIZATION[@]}"

            echo ""
            echo "🗑️  其他脚本"
            process_files "delete" "" "${MISC[@]}"

            echo ""
            echo "============================================================"
            echo "✅ 删除完成！"
        else
            echo ""
            echo "❌ 已取消删除操作"
        fi
        ;;

    --dry-run|*)
        echo ""
        echo "🔍 预览模式 - 以下文件将被处理："
        echo "============================================================"

        show_files "训练脚本（已被 train.sh 替代）" "${TRAINING_SCRIPTS[@]}"
        show_files "数据处理脚本（一次性使用）" "${DATA_PROCESSING[@]}"
        show_files "可视化脚本（特定用途）" "${VISUALIZATION[@]}"
        show_files "其他脚本" "${MISC[@]}"

        echo ""
        echo "============================================================"
        TOTAL=$(($(count_files "${TRAINING_SCRIPTS[@]}") + \
                 $(count_files "${DATA_PROCESSING[@]}") + \
                 $(count_files "${VISUALIZATION[@]}") + \
                 $(count_files "${MISC[@]}")))
        echo "📊 总计: $TOTAL 个文件将被处理"
        echo ""
        echo "执行归档: ./scripts/cleanup.sh --archive"
        echo "执行删除: ./scripts/cleanup.sh --delete  (⚠️  危险！)"
        ;;
esac

echo "============================================================"
