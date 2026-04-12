#!/usr/bin/env bash
# Usage:
#   bash baselines/boxinstseg/run_boxlevelset.sh <dataset> <cmd> [extra args]
#
# dataset:
#   NUAA-SIRST | NUDT-SIRST | IRSTD-1k
#   or short aliases: nuaa | nudt | irstd | irstd1k
#
# cmd:
#   prepare
#   train
#   test <ckpt>
#   export [results.pkl]
#   eval [pred_dir]
#
# Examples:
#   bash baselines/boxinstseg/run_boxlevelset.sh nuaa train
#   bash baselines/boxinstseg/run_boxlevelset.sh nudt train
#   bash baselines/boxinstseg/run_boxlevelset.sh irstd test work_dirs/.../latest.pth
#   bash baselines/boxinstseg/run_boxlevelset.sh NUAA-SIRST eval

set -euo pipefail

DATASET="${1:-}"
CMD="${2:-}"

if [[ -z "$DATASET" || -z "$CMD" ]]; then
  echo "Usage: bash $0 <NUAA-SIRST|NUDT-SIRST|IRSTD-1k|nuaa|nudt|irstd|irstd1k> <prepare|train|test|export|eval> [args]"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOXINSTSEG_ROOT="${BOXINSTSEG_ROOT:-$PROJECT_ROOT/external/BoxInstSeg}"
COCO_PREP="$PROJECT_ROOT/baselines/boxinstseg/prepare_irstd_coco.py"
EXPORT_SCRIPT="$PROJECT_ROOT/baselines/boxinstseg/export_boxinstseg_results.py"
EVAL_SCRIPT="$PROJECT_ROOT/baselines/boxinstseg/eval_file_predictions.py"

export POLARIS_PROJECT_ROOT="${POLARIS_PROJECT_ROOT:-$PROJECT_ROOT}"

ensure_runtime_env() {
  local conda_lib=""
  local torch_lib=""
  local cuda_lib="/usr/local/cuda-11.8/lib64"

  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    conda_lib="${CONDA_PREFIX}/lib"
  fi

  torch_lib="$(python - <<'PY'
import os
try:
    import torch
    print(os.path.join(os.path.dirname(torch.__file__), "lib"))
except Exception:
    print("")
PY
)"

  local new_ld=""
  for path in "$conda_lib" "$torch_lib" "$cuda_lib"; do
    if [[ -n "$path" && -d "$path" ]]; then
      if [[ -z "$new_ld" ]]; then
        new_ld="$path"
      else
        new_ld="${new_ld}:$path"
      fi
    fi
  done

  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    if [[ -n "$new_ld" ]]; then
      export LD_LIBRARY_PATH="${new_ld}:${LD_LIBRARY_PATH}"
    fi
  elif [[ -n "$new_ld" ]]; then
    export LD_LIBRARY_PATH="$new_ld"
  fi
}

# 根据数据集自动选 config 和 work_dir
case "${DATASET,,}" in
  nuaa-sirst|nuaa)
    CONFIG_PATH="$PROJECT_ROOT/baselines/boxinstseg/configs/box_levelset_nuaa_r50_fpn_3x.py"
    WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/baselines/boxinstseg/work_dirs/box_levelset_nuaa_r50_fpn_3x}"
    DATASET="NUAA-SIRST"
    ;;
  nudt-sirst|nudt)
    CONFIG_PATH="$PROJECT_ROOT/baselines/boxinstseg/configs/box_levelset_nudt_r50_fpn_3x.py"
    WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/baselines/boxinstseg/work_dirs/box_levelset_nudt_r50_fpn_3x}"
    DATASET="NUDT-SIRST"
    ;;
  irstd-1k|irstd1k|irstd)
    CONFIG_PATH="$PROJECT_ROOT/baselines/boxinstseg/configs/box_levelset_irstd1k_r50_fpn_3x.py"
    WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/baselines/boxinstseg/work_dirs/box_levelset_irstd1k_r50_fpn_3x}"
    DATASET="IRSTD-1k"
    ;;
  *)
    echo "Unknown dataset: $DATASET. Choose from NUAA-SIRST, NUDT-SIRST, IRSTD-1k, nuaa, nudt, irstd, irstd1k" >&2
    exit 1
    ;;
esac

RESULTS_PKL="${RESULTS_PKL:-$WORK_DIR/results_val.pkl}"
PRED_DIR="${PRED_DIR:-$WORK_DIR/exported_probs}"

require_boxinstseg() {
  if [[ ! -d "$BOXINSTSEG_ROOT" ]]; then
    echo "BoxInstSeg root not found: $BOXINSTSEG_ROOT" >&2
    exit 1
  fi
}

ensure_runtime_env

case "$CMD" in
  prepare)
    python "$COCO_PREP" --dataset "$DATASET" --root "$PROJECT_ROOT/dataset"
    ;;

  train)
    require_boxinstseg
    mkdir -p "$WORK_DIR"
    (
      cd "$BOXINSTSEG_ROOT"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        python -W ignore tools/train.py "$CONFIG_PATH" --work-dir "$WORK_DIR"
    )
    ;;

  test)
    require_boxinstseg
    ckpt="${3:-}"
    if [[ -z "$ckpt" ]]; then
      echo "Checkpoint path required: bash $0 $DATASET test <ckpt.pth>" >&2
      exit 1
    fi
    mkdir -p "$WORK_DIR"
    (
      cd "$BOXINSTSEG_ROOT"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        python -W ignore tools/test.py "$CONFIG_PATH" "$ckpt" --out "$RESULTS_PKL"
    )
    echo "Saved results to $RESULTS_PKL"
    ;;

  export)
    results_path="${3:-$RESULTS_PKL}"
    python "$EXPORT_SCRIPT" \
      --dataset "$DATASET" \
      --root "$PROJECT_ROOT/dataset" \
      --results "$results_path" \
      --out-dir "$PRED_DIR"
    ;;

  eval)
    pred_dir="${3:-$PRED_DIR}"
    python "$EVAL_SCRIPT" \
      --dataset "$DATASET" \
      --root "$PROJECT_ROOT/dataset" \
      --pred-dir "$pred_dir"
    ;;

  *)
    echo "Unknown command: $CMD. Choose from prepare|train|test|export|eval" >&2
    exit 1
    ;;
esac
