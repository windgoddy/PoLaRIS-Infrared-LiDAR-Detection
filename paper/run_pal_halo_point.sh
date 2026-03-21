#!/bin/bash
# ============================================================
# 用 HALO p* 点标注跑 PAL + DNANet（NUDT-SIRST）
#
# 实验目的：验证"框监督→物理热点p*→点监督迭代训练"是否可行
#   - p* 由 HALO 的 B-SNR argmax 自动生成（无需人工点击）
#   - 喂给 PAL 的 masks_centroid 格式（与人工质心点标注格式相同）
#   - 对比 PAL 原始结果（人工点）：NUDT 73.29%
#
# 在服务器项目根目录下运行：
#   bash paper/run_pal_halo_point.sh
# ============================================================
set -e

DATASET="NUDT-SIRST"
DATASET_ROOT="dataset/"
DATASET_DIR="${DATASET_ROOT}${DATASET}"
SPLIT_FILE="${DATASET_DIR}/50_50/train.txt"
VAL_SPLIT_FILE="${DATASET_DIR}/50_50/test.txt"
PAL_DIR="PAL"
PAL_DATASET_NAME="NUDT_SIRST_HALO_point"
PSTAR_DIR="${DATASET_DIR}/masks_pstar_point"

echo "============================================================"
echo "Step 1: 生成 HALO p* 点标注"
echo "============================================================"
python paper/gen_pstar_point_labels.py \
    --dataset_dir "${DATASET_DIR}" \
    --split_file "${SPLIT_FILE}" \
    --output_dir "${PSTAR_DIR}" \
    --label_folder "labels_box" \
    --image_folder "images" \
    --expand_ratio 1.5 \
    --temperature 3.0

echo ""
echo "============================================================"
echo "Step 2: 构建 PAL 数据集目录结构"
echo "============================================================"

PAL_DATA="${PAL_DIR}/dataset/${PAL_DATASET_NAME}"
mkdir -p "${PAL_DATA}/origin/img"
mkdir -p "${PAL_DATA}/origin/masks_centroid"
mkdir -p "${PAL_DATA}/val/img"
mkdir -p "${PAL_DATA}/val/mask"

# 训练图像：从 train.txt 复制（或建软链接）
echo "  复制训练图像..."
while IFS= read -r img_id; do
    [ -z "$img_id" ] && continue
    src_img="${DATASET_DIR}/images/${img_id}.png"
    src_pt="${PSTAR_DIR}/${img_id}.png"
    [ -f "$src_img" ] && cp "$src_img" "${PAL_DATA}/origin/img/${img_id}.png"
    [ -f "$src_pt"  ] && cp "$src_pt"  "${PAL_DATA}/origin/masks_centroid/${img_id}.png"
done < "${SPLIT_FILE}"

# 验证集：图像 + GT 掩码
echo "  复制验证集..."
while IFS= read -r img_id; do
    [ -z "$img_id" ] && continue
    cp "${DATASET_DIR}/images/${img_id}.png" "${PAL_DATA}/val/img/${img_id}.png" 2>/dev/null || true
    cp "${DATASET_DIR}/masks/${img_id}.png"  "${PAL_DATA}/val/mask/${img_id}.png" 2>/dev/null || true
done < "${VAL_SPLIT_FILE}"

echo "  训练图像: $(ls ${PAL_DATA}/origin/img/ | wc -l)"
echo "  点标注:   $(ls ${PAL_DATA}/origin/masks_centroid/ | wc -l)"
echo "  验证图像: $(ls ${PAL_DATA}/val/img/ | wc -l)"
echo "  验证掩码: $(ls ${PAL_DATA}/val/mask/ | wc -l)"

echo ""
echo "============================================================"
echo "Step 3: 运行 PAL + DNANet 训练"
echo "  choose_model     = DNA"
echo "  choose_dataset   = ${PAL_DATASET_NAME}"
echo "  choose_dataset_type = masks_centroid"
echo "============================================================"

cd "${PAL_DIR}"
CUDA_VISIBLE_DEVICES=0 python train_model.py 2>&1 | tee "../logs/pal_halo_point_nudt.log"

echo ""
echo "============================================================"
echo "Step 4: 用我们的评估协议重新评估 best checkpoint"
echo "============================================================"
cd ..  # 回到项目根目录

# 找 PAL 保存的最佳 checkpoint（mIoU 最优）
BEST_CKPT=$(ls PAL/result/DNA__${PAL_DATASET_NAME}__masks_centroid__*/best_mIoU_checkpoint_*.pth.tar 2>/dev/null | head -1)

if [ -z "$BEST_CKPT" ]; then
    echo "  [WARN] 未找到 best checkpoint，请手动指定路径运行："
    echo "  python paper/eval_pal_checkpoint.py --ckpt <路径> --dataset NUDT-SIRST"
else
    echo "  找到 checkpoint: ${BEST_CKPT}"
    python paper/eval_pal_checkpoint.py \
        --ckpt    "${BEST_CKPT}" \
        --dataset "NUDT-SIRST" \
        --root    "dataset/" \
        --split   "50_50/test.txt" \
        2>&1 | tee -a logs/pal_halo_point_nudt.log
fi

echo ""
echo "============================================================"
echo "全部完成！对比参考："
echo "  PAL 论文（人工质心点）：NUDT = 73.29%"
echo "  HALO P1_D（框监督）  ：NUDT = 62.97%"
echo "  本实验（HALO p* 点）  ：见上方输出"
echo "============================================================"
