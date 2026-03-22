#!/usr/bin/env bash
# ============================================================
# HALO Hyperparameter Sweep — tmux 并行启动脚本
# ============================================================
# 用法:
#   bash paper/run_sweep_tmux.sh
#
# 修改下面的 GPU 分配和参数，然后运行。
# 脚本会：
#   1. 生成所有伪标签（串行，很快）
#   2. 在新的 tmux 窗口里并行启动各训练任务
# ============================================================

set -e
cd "$(dirname "$0")/.."   # 切换到项目根目录

# ──────────────────────────────────────────────────────────────
# 配置区：根据你的实际情况修改
# ──────────────────────────────────────────────────────────────
CONDA_ENV="LIDAR_POLARIS_env"       # conda 环境名
SESSION="halo_sweep"                # tmux session 名
EPOCHS=1500
SEED=42

# sigma sweep: 每个 (dataset, sigma) 分配一个 GPU
# 格式: "DATASET sigma GPU"
JOBS=(
    "NUAA-SIRST  1.5  3"
    "NUAA-SIRST  2.0  4"
    "NUDT-SIRST  1.5  5"
    "NUDT-SIRST  2.0  6"
    "IRSTD-1k    1.5  7"
    "IRSTD-1k    2.0  8"
)

# ──────────────────────────────────────────────────────────────
# Step 1: 生成所有伪标签
# ──────────────────────────────────────────────────────────────
echo "======================================================"
echo " Step 1: 生成伪标签..."
echo "======================================================"
conda run -n "$CONDA_ENV" python paper/hyperparam_sweep_e2e.py \
    --sweep sigma --sigma_values 1.5,2.0 --gen_only
echo ""

# ──────────────────────────────────────────────────────────────
# Step 2: 创建 tmux session 并在各窗口启动训练
# ──────────────────────────────────────────────────────────────
echo "======================================================"
echo " Step 2: 在 tmux session '$SESSION' 中启动训练..."
echo "======================================================"

# 如果 session 已存在则删除
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "main"

EXPAND_FIXED=1.5
WIN_IDX=0

for JOB in "${JOBS[@]}"; do
    # 解析 job 配置
    DS=$(echo "$JOB" | awk '{print $1}')
    SIGMA=$(echo "$JOB" | awk '{print $2}')
    GPU=$(echo "$JOB" | awk '{print $3}')

    # 根据数据集确定 image_folder 和 split_method
    case "$DS" in
        "IRSTD-1k")
            IMG_FOLDER="IRSTD1k_Img"
            SPLIT="50_50"
            ;;
        *)
            IMG_FOLDER="images"
            SPLIT="50_50"
            ;;
    esac

    # 构造参数
    TAG="sigma${SIGMA}_expand${EXPAND_FIXED}"
    # 将 sigma 格式化为 3 位小数（如 1.5 → 1.500）
    SIGMA_FMT=$(printf "%.3f" "$SIGMA")
    MASK_FOLDER="masks_sweep_sigma${SIGMA_FMT}_expand${EXPAND_FIXED}"
    EXP_TAG="sweep_${DS}_sigma${SIGMA_FMT}_expand${EXPAND_FIXED}"
    WIN_NAME="${DS:0:4}_s${SIGMA}"   # 短名如 NUAA_s1.5

    # 创建新窗口（第一个 job 用已有的 main 窗口）
    if [ "$WIN_IDX" -eq 0 ]; then
        tmux rename-window -t "$SESSION:0" "$WIN_NAME"
    else
        tmux new-window -t "$SESSION" -n "$WIN_NAME"
    fi

    # 构造训练命令
    TRAIN_CMD="conda activate $CONDA_ENV && \
CUDA_VISIBLE_DEVICES=$GPU python train.py \
--model DNANet \
--dataset $DS \
--split_method $SPLIT \
--root dataset \
--image_folder $IMG_FOLDER \
--mask_folder $MASK_FOLDER \
--epochs $EPOCHS \
--experiment_name $EXP_TAG \
--seed $SEED \
--in_channels 1 \
--base_size 256 \
--crop_size 256 \
--deep_supervision True \
--train_batch_size 4 \
--test_batch_size 4 \
2>&1 | tee result/sweep_logs/${EXP_TAG}.log; echo '=== DONE: $EXP_TAG ==='"

    tmux send-keys -t "$SESSION:$WIN_IDX" "$TRAIN_CMD" Enter

    echo "  Window '$WIN_NAME': $DS sigma=$SIGMA → GPU $GPU"
    WIN_IDX=$((WIN_IDX + 1))
done

echo ""
echo "======================================================"
echo " 全部启动完成！"
echo " 查看进度: tmux attach -t $SESSION"
echo " 切换窗口: Ctrl+B + 数字键 / Ctrl+B + n"
echo " 所有 log:  ls result/sweep_logs/"
echo "======================================================"
echo ""
echo " 训练完成后，运行以下命令汇总结果："
echo "   python paper/hyperparam_sweep_e2e.py --collect \\"
echo "     --sweep sigma --sigma_values 1.5,2.0"
echo "======================================================"
