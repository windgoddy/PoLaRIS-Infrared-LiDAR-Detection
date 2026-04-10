#!/usr/bin/env python3
"""
Convert BoxInstSeg / MMDetection pickle results into per-image probability maps.

Expected result item for BoxLevelset:
    (bbox_results, (mask_results, score_results))

where:
    bbox_results[class_id] -> Nx5 [x1, y1, x2, y2, score]
    mask_results[class_id] -> list[Tensor|ndarray|RLE]
    score_results[class_id] -> list[Tensor|float]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover
    mask_utils = None


DATASET_IMAGE_FOLDERS = {
    "NUAA-SIRST": "images",
    "NUDT-SIRST": "images",
    "IRSTD-1k": "IRSTD1k_Img",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MMDet results to IRSTD probability PNGs.")
    parser.add_argument("--dataset", default="NUAA-SIRST", choices=sorted(DATASET_IMAGE_FOLDERS.keys()))
    parser.add_argument("--root", default="dataset")
    parser.add_argument(
        "--coco-root",
        default=None,
        help="Prepared COCO root. Defaults to dataset/<DATASET>/boxinstseg_coco.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Which COCO annotation file order to use when mapping result index -> image id.",
    )
    parser.add_argument("--results", required=True, help="Path to MMDetection pickle result file.")
    parser.add_argument("--out-dir", required=True, help="Directory to save probability PNGs.")
    parser.add_argument(
        "--binary-dir",
        default=None,
        help="Optional directory to also save binary PNGs thresholded at --binary-threshold.",
    )
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_image_order(ann_path: Path) -> List[Tuple[str, int, int]]:
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [(img["file_name"], int(img["width"]), int(img["height"])) for img in data["images"]]


def normalize_score(score: Any) -> float:
    if torch is not None and isinstance(score, torch.Tensor):
        return float(score.detach().cpu().item())
    if isinstance(score, np.ndarray):
        return float(score.item())
    return float(score)


def decode_mask(mask_obj: Any) -> np.ndarray:
    if torch is not None and isinstance(mask_obj, torch.Tensor):
        return mask_obj.detach().cpu().numpy().astype(bool)
    if isinstance(mask_obj, np.ndarray):
        return mask_obj.astype(bool)
    if isinstance(mask_obj, dict):
        if mask_utils is None:
            raise RuntimeError("pycocotools is required to decode RLE masks.")
        decoded = mask_utils.decode(mask_obj)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        return decoded.astype(bool)
    raise TypeError(f"Unsupported mask type: {type(mask_obj)!r}")


def unpack_result(result: Any) -> Tuple[Sequence[Any], Sequence[Sequence[Any]], Sequence[Sequence[Any]]]:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError("Expected MMDet result item to be a tuple: (bbox_results, segm_results).")

    bbox_results, segm_result = result
    if isinstance(segm_result, tuple) and len(segm_result) == 2:
        mask_results, score_results = segm_result
    else:
        mask_results = segm_result
        score_results = []
        for class_boxes in bbox_results:
            if len(class_boxes) == 0:
                score_results.append([])
            else:
                score_results.append([float(row[-1]) for row in class_boxes])
    return bbox_results, mask_results, score_results


def build_probability_map(
    result: Any,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    height, width = image_shape
    prob_map = np.zeros((height, width), dtype=np.float32)
    _, mask_results, score_results = unpack_result(result)

    for class_masks, class_scores in zip(mask_results, score_results):
        for mask_obj, score in zip(class_masks, class_scores):
            score_f = float(np.clip(normalize_score(score), 0.0, 1.0))
            if score_f <= 0:
                continue
            mask = decode_mask(mask_obj)
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            prob_map[mask] = np.maximum(prob_map[mask], score_f)
    return prob_map


def main() -> None:
    args = parse_args()
    coco_root = Path(args.coco_root) if args.coco_root else Path(args.root) / args.dataset / "boxinstseg_coco"
    ann_name = "instances_train2017.json" if args.split == "train" else "instances_val2017.json"
    ann_path = coco_root / "annotations" / ann_name
    image_infos = load_image_order(ann_path)
    results = load_pickle(Path(args.results))

    if len(results) != len(image_infos):
        raise ValueError(
            f"Result length mismatch: {len(results)} results vs {len(image_infos)} images from {ann_path}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_dir = Path(args.binary_dir) if args.binary_dir else None
    if binary_dir is not None:
        binary_dir.mkdir(parents=True, exist_ok=True)

    for result, (file_name, width, height) in zip(results, image_infos):
        prob_map = build_probability_map(result=result, image_shape=(height, width))
        image_id = Path(file_name).stem
        prob_u8 = np.clip(np.round(prob_map * 255.0), 0, 255).astype(np.uint8)
        ok = cv2.imwrite(str(out_dir / f"{image_id}.png"), prob_u8)
        if not ok:
            raise IOError(f"Failed to write probability map for {image_id}")

        if binary_dir is not None:
            pred_bin = (prob_map >= args.binary_threshold).astype(np.uint8) * 255
            ok = cv2.imwrite(str(binary_dir / f"{image_id}.png"), pred_bin)
            if not ok:
                raise IOError(f"Failed to write binary mask for {image_id}")

    print(f"Saved {len(image_infos)} probability maps to {out_dir}")
    if binary_dir is not None:
        print(f"Saved binary masks to {binary_dir} @ threshold={args.binary_threshold:.2f}")


if __name__ == "__main__":
    main()
