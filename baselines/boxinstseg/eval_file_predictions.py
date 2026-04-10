#!/usr/bin/env python3
"""
Evaluate file-based prediction maps using the same IRSTD-style protocol used in this repo:
1. Threshold sweep for best mIoU.
2. Fixed-threshold Pd / Fa using connected-component matching (distance < 3px).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "NUAA-SIRST": {
        "image_folder": "images",
        "mask_folder": "masks",
        "split_folder": "50_50",
    },
    "NUDT-SIRST": {
        "image_folder": "images",
        "mask_folder": "masks",
        "split_folder": "50_50",
    },
    "IRSTD-1k": {
        "image_folder": "IRSTD1k_Img",
        "mask_folder": "masks",
        "split_folder": "50_50",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exported BoxInstSeg predictions.")
    parser.add_argument("--dataset", default="NUAA-SIRST", choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--split-method", default=None)
    parser.add_argument("--pred-dir", required=True, help="Directory containing probability PNGs.")
    parser.add_argument("--mask-folder", default=None, help="Override GT mask folder.")
    parser.add_argument("--fixed-threshold", type=float, default=0.5, help="Threshold for Pd/Fa report.")
    parser.add_argument("--sweep-start", type=float, default=0.05)
    parser.add_argument("--sweep-stop", type=float, default=0.95)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    return parser.parse_args()


def read_split_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compute_dataset_iou(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray], threshold: float) -> float:
    total_inter = 0
    total_union = 0
    for pred, gt in zip(preds, gts):
        pred_bin = (pred >= threshold).astype(np.uint8)
        inter = int((pred_bin & gt).sum())
        union = int(pred_bin.sum() + gt.sum() - inter)
        if union == 0:
            total_inter += 1
            total_union += 1
        else:
            total_inter += inter
            total_union += union
    return float(total_inter / total_union) if total_union > 0 else 0.0


def connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)


def compute_pd_fa(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray], threshold: float) -> Tuple[float, float]:
    false_alarm_area = 0.0
    total_image_area = 0.0
    matched_targets = 0.0
    total_targets = 0.0

    for pred, gt in zip(preds, gts):
        pred_bin = (pred >= threshold).astype(np.uint8)
        total_image_area += float(pred_bin.shape[0] * pred_bin.shape[1])

        pred_n, _, pred_stats, pred_centroids = connected_components(pred_bin)
        gt_n, _, _, gt_centroids = connected_components(gt.astype(np.uint8))

        pred_used: set[int] = set()
        total_targets += float(gt_n - 1)

        for gt_idx in range(1, gt_n):
            gt_cx, gt_cy = gt_centroids[gt_idx]
            for pred_idx in range(1, pred_n):
                if pred_idx in pred_used:
                    continue
                pred_cx, pred_cy = pred_centroids[pred_idx]
                if math.hypot(float(pred_cx - gt_cx), float(pred_cy - gt_cy)) < 3.0:
                    matched_targets += 1.0
                    pred_used.add(pred_idx)
                    break

        for pred_idx in range(1, pred_n):
            if pred_idx not in pred_used:
                false_alarm_area += float(pred_stats[pred_idx, cv2.CC_STAT_AREA])

    fa = false_alarm_area / total_image_area if total_image_area > 0 else 0.0
    pd = matched_targets / total_targets if total_targets > 0 else 0.0
    return pd, fa


def main() -> None:
    args = parse_args()
    ds_cfg = DATASET_CONFIGS[args.dataset]
    dataset_dir = Path(args.root) / args.dataset
    split_folder = args.split_method or ds_cfg["split_folder"]
    mask_folder = args.mask_folder or ds_cfg["mask_folder"]

    pred_dir = Path(args.pred_dir)
    split_ids = read_split_ids(dataset_dir / split_folder / "test.txt")
    gt_dir = dataset_dir / mask_folder

    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    missing: List[str] = []
    resized_gt = 0

    for sample_id in split_ids:
        pred_path = pred_dir / f"{sample_id}.png"
        gt_path = gt_dir / f"{sample_id}.png"
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            missing.append(sample_id)
            continue
        if gt.shape != pred.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            resized_gt += 1
        preds.append(pred.astype(np.float32) / 255.0)
        gts.append((gt > 127).astype(np.uint8))

    if not preds:
        raise RuntimeError(f"No valid predictions found in {pred_dir}")

    thresholds = np.arange(args.sweep_start, args.sweep_stop + 1e-9, args.sweep_step)
    best_thresh = float(thresholds[0])
    best_miou = -1.0
    sweep_results: List[Tuple[float, float]] = []
    for thr in thresholds:
        miou = compute_dataset_iou(preds, gts, float(thr))
        sweep_results.append((float(thr), float(miou)))
        if miou > best_miou:
            best_miou = float(miou)
            best_thresh = float(thr)

    pd_fixed, fa_fixed = compute_pd_fa(preds, gts, threshold=args.fixed_threshold)
    pd_best, fa_best = compute_pd_fa(preds, gts, threshold=best_thresh)

    print(f"Dataset        : {args.dataset}")
    print(f"Prediction dir : {pred_dir}")
    print(f"Samples used   : {len(preds)} / {len(split_ids)}")
    if missing:
        print(f"Missing preds  : {len(missing)}")
    if resized_gt:
        print(f"Resized GT     : {resized_gt}")
    print("")
    print("mIoU @ thresholds")
    for thr, miou in sweep_results:
        marker = "  <-- best" if abs(thr - best_thresh) < 1e-9 else ""
        print(f"  thresh={thr:.2f}  mIoU={miou * 100:.2f}%{marker}")
    print("")
    print(f"Best mIoU      : {best_miou * 100:.2f}% @ threshold={best_thresh:.2f}")
    print(f"Pd / Fa @ 0.50 : {pd_fixed * 100:.2f}% / {fa_fixed:.4e}")
    print(f"Pd / Fa @ best : {pd_best * 100:.2f}% / {fa_best:.4e}")


if __name__ == "__main__":
    main()
