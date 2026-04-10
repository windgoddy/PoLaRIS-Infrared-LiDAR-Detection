#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOXINSTSEG_ROOT="${BOXINSTSEG_ROOT:-$PROJECT_ROOT/external/BoxInstSeg}"
CONFIG_PATH="$PROJECT_ROOT/baselines/boxinstseg/configs/box_levelset_nuaa_r50_fpn_3x.py"
COCO_PREP="$PROJECT_ROOT/baselines/boxinstseg/prepare_irstd_coco.py"
EXPORT_SCRIPT="$PROJECT_ROOT/baselines/boxinstseg/export_boxinstseg_results.py"
EVAL_SCRIPT="$PROJECT_ROOT/baselines/boxinstseg/eval_file_predictions.py"
WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/baselines/boxinstseg/work_dirs/box_levelset_nuaa_r50_fpn_3x}"
RESULTS_PKL="${RESULTS_PKL:-$WORK_DIR/boxlevelset_nuaa_val.pkl}"
PRED_DIR="${PRED_DIR:-$WORK_DIR/exported_probs}"

usage() {
  cat <<EOF
Usage:
  bash baselines/boxinstseg/run_boxlevelset_nuaa.sh prepare
  bash baselines/boxinstseg/run_boxlevelset_nuaa.sh train
  bash baselines/boxinstseg/run_boxlevelset_nuaa.sh test <checkpoint.pth>
  bash baselines/boxinstseg/run_boxlevelset_nuaa.sh export <results.pkl>
  bash baselines/boxinstseg/run_boxlevelset_nuaa.sh eval [pred_dir]

Environment overrides:
  BOXINSTSEG_ROOT   default: $BOXINSTSEG_ROOT
  WORK_DIR          default: $WORK_DIR
  RESULTS_PKL       default: $RESULTS_PKL
  PRED_DIR          default: $PRED_DIR
EOF
}

require_boxinstseg() {
  if [[ ! -d "$BOXINSTSEG_ROOT" ]]; then
    echo "BoxInstSeg root not found: $BOXINSTSEG_ROOT" >&2
    echo "Set BOXINSTSEG_ROOT or clone the repo first." >&2
    exit 1
  fi
}

cmd="${1:-}"
case "$cmd" in
  prepare)
    python "$COCO_PREP" --dataset NUAA-SIRST --root "$PROJECT_ROOT/dataset"
    ;;

  train)
    require_boxinstseg
    mkdir -p "$WORK_DIR"
    (
      cd "$BOXINSTSEG_ROOT"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        python tools/train.py "$CONFIG_PATH" --work-dir "$WORK_DIR"
    )
    ;;

  test)
    require_boxinstseg
    ckpt="${2:-}"
    if [[ -z "$ckpt" ]]; then
      echo "Checkpoint path is required for test." >&2
      usage
      exit 1
    fi
    mkdir -p "$WORK_DIR"
    (
      cd "$BOXINSTSEG_ROOT"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        python tools/test.py "$CONFIG_PATH" "$ckpt" --out "$RESULTS_PKL"
    )
    echo "Saved MMDet results to $RESULTS_PKL"
    ;;

  export)
    results_path="${2:-$RESULTS_PKL}"
    python "$EXPORT_SCRIPT" \
      --dataset NUAA-SIRST \
      --root "$PROJECT_ROOT/dataset" \
      --results "$results_path" \
      --out-dir "$PRED_DIR"
    ;;

  eval)
    pred_dir="${2:-$PRED_DIR}"
    python "$EVAL_SCRIPT" \
      --dataset NUAA-SIRST \
      --root "$PROJECT_ROOT/dataset" \
      --pred-dir "$pred_dir"
    ;;

  *)
    usage
    exit 1
    ;;
esac
