#!/bin/bash
# P1 鲁棒训练脚本
# ================================================
# 功能：
#   - OOM 或崩溃时自动切换 GPU (默认顺序: 5,0,1,2,3,4,6,7)
#   - 用 .done 文件标记已完成实验，支持断点续跑
#   - 每次尝试的完整日志保存到 logs/p1/
# ================================================
# 用法：
#   nohup ./run_p1_all.sh > logs/p1_main.log 2>&1 &
#   tail -f logs/p1_main.log        # 查看主进度
#   ls logs/p1/done/                # 查看哪些实验已完成
# ================================================

set -uo pipefail

# ── 训练超参数 ──────────────────────────────────
EPOCHS=400
LR=0.05
BS=4

# ── GPU 优先级（OOM 后按此顺序尝试）─────────────
GPUS=(5 0 1 2 3 4 6 7)

# ── 日志 / 断点目录 ─────────────────────────────
LOG_DIR="logs/p1"
DONE_DIR="logs/p1/done"
mkdir -p "$LOG_DIR" "$DONE_DIR"

# ================================================
# run_experiment <dataset> <mask_folder> <image_folder> <exp_name>
# ================================================
run_experiment() {
    local DATASET=$1
    local MASK_FOLDER=$2
    local IMAGE_FOLDER=$3
    local EXP_NAME=$4

    local DONE_FILE="$DONE_DIR/${EXP_NAME}.done"

    # 已完成则跳过
    if [ -f "$DONE_FILE" ]; then
        echo "[SKIP] ${EXP_NAME} — 已完成"
        return 0
    fi

    echo ""
    echo "════════════════════════════════════════════════"
    echo "▶  START  ${EXP_NAME}"
    echo "   dataset=${DATASET}  mask=${MASK_FOLDER}"
    echo "   $(date)"
    echo "════════════════════════════════════════════════"

    local attempt=0
    while true; do
        for GPU in "${GPUS[@]}"; do
            attempt=$((attempt + 1))
            local LOG_FILE="${LOG_DIR}/${EXP_NAME}_gpu${GPU}_try${attempt}.log"

            echo "  → [try ${attempt}] GPU=${GPU}  log=${LOG_FILE}"

            # 运行训练，同时写日志 + 终端可见
            # pipefail 确保 python 退出码能被 $? 捕获
            set +e
            CUDA_VISIBLE_DEVICES=${GPU} python train.py \
                --model DNANet \
                --deep_supervision True \
                --dataset "${DATASET}" \
                --mask_folder "${MASK_FOLDER}" \
                --image_folder "${IMAGE_FOLDER}" \
                --epochs ${EPOCHS} \
                --lr ${LR} \
                --train_batch_size ${BS} \
                --gpus "${GPU}" \
                --experiment_name "${EXP_NAME}" \
                2>&1 | tee "${LOG_FILE}"
            local EXIT_CODE=${PIPESTATUS[0]}
            set -e

            if [ ${EXIT_CODE} -eq 0 ]; then
                echo "  ✓  DONE  ${EXP_NAME}  (GPU=${GPU}, try=${attempt})"
                touch "${DONE_FILE}"
                return 0
            fi

            # 判断失败原因
            if grep -qE "CUDA out of memory|OutOfMemoryError|CUDA error" "${LOG_FILE}"; then
                echo "  ✗  GPU${GPU} OOM — 切换下一 GPU (等 15s)..."
            else
                echo "  ✗  GPU${GPU} 失败 (exit=${EXIT_CODE}) — 切换下一 GPU (等 15s)..."
            fi
            sleep 15
        done

        # 所有 GPU 都失败了
        echo "  !! 全部 GPU 均失败，等待 120s 后重新轮询..."
        sleep 120
    done
}

# ================================================
# 12 组实验
# ================================================
echo "P1 训练启动 — $(date)"
echo "断点文件: ${DONE_DIR}/"
echo "日志目录: ${LOG_DIR}/"

# ── NUDT-SIRST ──────────────────────────────────
run_experiment NUDT-SIRST masks            images  P1_A_gt_nudt
run_experiment NUDT-SIRST masks_box_fill   images  P1_B_box_nudt
run_experiment NUDT-SIRST masks_bsnr       images  P1_C_bsnr_nudt
run_experiment NUDT-SIRST masks_bsnr_gauss images  P1_D_pag_nudt

# ── NUAA-SIRST ──────────────────────────────────
run_experiment NUAA-SIRST masks            images  P1_A_gt_nuaa
run_experiment NUAA-SIRST masks_box_fill   images  P1_B_box_nuaa
run_experiment NUAA-SIRST masks_bsnr       images  P1_C_bsnr_nuaa
run_experiment NUAA-SIRST masks_bsnr_gauss images  P1_D_pag_nuaa

# ── IRSTD-1k ────────────────────────────────────
run_experiment IRSTD-1k masks            IRSTD1k_Img  P1_A_gt_irstd
run_experiment IRSTD-1k masks_box_fill   IRSTD1k_Img  P1_B_box_irstd
run_experiment IRSTD-1k masks_bsnr       IRSTD1k_Img  P1_C_bsnr_irstd
run_experiment IRSTD-1k masks_bsnr_gauss IRSTD1k_Img  P1_D_pag_irstd

echo ""
echo "✓  全部 12 组训练完成 — $(date)"
