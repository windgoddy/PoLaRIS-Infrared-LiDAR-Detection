#!/usr/bin/env python3
"""
Prepare IRSTD datasets for BoxInstSeg / BoxLevelset.

This script converts the existing PoLaRIS weak-box dataset layout:

    dataset/<DATASET>/{images|IRSTD1k_Img,masks,labels_box,50_50}

into a COCO-style layout expected by BoxInstSeg:

    dataset/<DATASET>/boxinstseg_coco/
      annotations/
        instances_train2017.json
        instances_val2017.json
      train2017/
      val2017/

Design choices:
1. Weak boxes come from `labels_box` and are used as COCO `bbox`.
2. GT masks are used only to build per-instance `segmentation` for evaluation.
3. Grayscale IR images are converted to repeated 3-channel RGB PNGs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "NUAA-SIRST": {
        "image_folder": "images",
        "mask_folder": "masks",
        "label_folder": "labels_box",
        "split_folder": "50_50",
    },
    "NUDT-SIRST": {
        "image_folder": "images",
        "mask_folder": "masks",
        "label_folder": "labels_box",
        "split_folder": "50_50",
    },
    "IRSTD-1k": {
        "image_folder": "IRSTD1k_Img",
        "mask_folder": "masks",
        "label_folder": "labels_box",
        "split_folder": "50_50",
    },
}


@dataclass
class BoxAnn:
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_xywh: Tuple[float, float, float, float]


@dataclass
class ComponentAnn:
    comp_id: int
    mask: np.ndarray
    area: int
    centroid_xy: Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert IRSTD layout to COCO for BoxInstSeg.")
    parser.add_argument(
        "--dataset",
        default="NUAA-SIRST",
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset to convert.",
    )
    parser.add_argument(
        "--root",
        default="dataset",
        help="Dataset root containing NUAA-SIRST / NUDT-SIRST / IRSTD-1k.",
    )
    parser.add_argument(
        "--split-method",
        default=None,
        help="Override split folder. Defaults to dataset config value (usually 50_50).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output COCO directory. Defaults to dataset/<DATASET>/boxinstseg_coco.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated RGB images if they already exist.",
    )
    return parser.parse_args()


def read_split_ids(split_path: Path) -> List[str]:
    with split_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img


def write_rgb_repeat(src_path: Path, dst_path: Path, overwrite: bool) -> Tuple[int, int]:
    if dst_path.exists() and not overwrite:
        existing = cv2.imread(str(dst_path), cv2.IMREAD_COLOR)
        if existing is None:
            raise FileNotFoundError(f"Existing RGB image is unreadable: {dst_path}")
        return existing.shape[1], existing.shape[0]

    gray = load_grayscale(src_path)
    if gray.ndim != 2:
        raise ValueError(f"Expected single-channel grayscale image: {src_path}")
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst_path), rgb)
    if not ok:
        raise IOError(f"Failed to write RGB image: {dst_path}")
    return rgb.shape[1], rgb.shape[0]


def resolve_existing_path(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No existing file found in candidates:\n" + "\n".join(str(p) for p in candidates))


def load_yolo_boxes(label_path: Path, width: int, height: int) -> List[BoxAnn]:
    boxes: List[BoxAnn] = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = parts[:5]
            cx = float(cx) * width
            cy = float(cy) * height
            bw = float(bw) * width
            bh = float(bh) * height

            x1 = max(0.0, cx - bw / 2.0)
            y1 = max(0.0, cy - bh / 2.0)
            x2 = min(float(width), cx + bw / 2.0)
            y2 = min(float(height), cy + bh / 2.0)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(
                BoxAnn(
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                )
            )
    return boxes


def load_components(mask_path: Path) -> List[ComponentAnn]:
    gt = load_grayscale(mask_path)
    gt_bin = (gt > 127).astype(np.uint8)
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(gt_bin, connectivity=8)

    components: List[ComponentAnn] = []
    for comp_id in range(1, num_labels):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp_mask = (label_map == comp_id).astype(np.uint8)
        centroid_x, centroid_y = centroids[comp_id]
        components.append(
            ComponentAnn(
                comp_id=comp_id,
                mask=comp_mask,
                area=area,
                centroid_xy=(float(centroid_x), float(centroid_y)),
            )
        )
    return components


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def component_bbox(mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return (x1, y1, x2, y2)


def overlap_pixels(box_xyxy: Sequence[float], mask: np.ndarray) -> int:
    h, w = mask.shape
    x1, y1, x2, y2 = box_xyxy
    ix1 = max(0, int(math.floor(x1)))
    iy1 = max(0, int(math.floor(y1)))
    ix2 = min(w, int(math.ceil(x2)))
    iy2 = min(h, int(math.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return int(mask[iy1:iy2, ix1:ix2].sum())


def assign_components_to_boxes(
    boxes: Sequence[BoxAnn], components: Sequence[ComponentAnn]
) -> Tuple[List[Optional[ComponentAnn]], int]:
    assigned: List[Optional[ComponentAnn]] = [None] * len(boxes)
    used_components: set[int] = set()
    duplicate_matches = 0

    for box_idx, box in enumerate(boxes):
        best_key = None
        best_comp: Optional[ComponentAnn] = None
        box_xyxy = box.bbox_xyxy
        box_cx = 0.5 * (box_xyxy[0] + box_xyxy[2])
        box_cy = 0.5 * (box_xyxy[1] + box_xyxy[3])

        def rank(comp: ComponentAnn) -> Tuple[int, int, float, float, float]:
            cx, cy = comp.centroid_xy
            center_inside = int(box_xyxy[0] <= cx <= box_xyxy[2] and box_xyxy[1] <= cy <= box_xyxy[3])
            overlap = overlap_pixels(box_xyxy, comp.mask)
            iou = bbox_iou(box_xyxy, component_bbox(comp.mask))
            dist = math.hypot(cx - box_cx, cy - box_cy)
            return (int(overlap > 0), center_inside, float(overlap), iou, -dist)

        for comp in components:
            if comp.comp_id in used_components:
                continue
            cur_key = rank(comp)
            if best_key is None or cur_key > best_key:
                best_key = cur_key
                best_comp = comp

        if best_comp is None and components:
            for comp in components:
                cur_key = rank(comp)
                if best_key is None or cur_key > best_key:
                    best_key = cur_key
                    best_comp = comp
            if best_comp is not None:
                duplicate_matches += 1

        assigned[box_idx] = best_comp
        if best_comp is not None:
            used_components.add(best_comp.comp_id)

    return assigned, duplicate_matches


def encode_binary_mask_to_rle(mask: np.ndarray) -> Dict[str, object]:
    """
    Produce uncompressed COCO RLE.
    """
    flat = np.asarray(mask, order="F", dtype=np.uint8).reshape(-1, order="F")
    counts: List[int] = []
    last_val = 0
    run_len = 0
    for val in flat:
        if int(val) == last_val:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            last_val = int(val)
    counts.append(run_len)
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def build_annotation(
    ann_id: int,
    image_id: int,
    category_id: int,
    box: BoxAnn,
    component: Optional[ComponentAnn],
    image_hw: Tuple[int, int],
) -> Dict[str, object]:
    h, w = image_hw
    if component is None or component.area <= 0:
        fallback_mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = box.bbox_xyxy
        ix1 = max(0, int(math.floor(x1)))
        iy1 = max(0, int(math.floor(y1)))
        ix2 = min(w, int(math.ceil(x2)))
        iy2 = min(h, int(math.ceil(y2)))
        fallback_mask[iy1:iy2, ix1:ix2] = 1
        segmentation = encode_binary_mask_to_rle(fallback_mask)
        area = int(fallback_mask.sum())
    else:
        segmentation = encode_binary_mask_to_rle(component.mask)
        area = int(component.area)

    x, y, bw, bh = box.bbox_xywh
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": 0,
        "area": area,
        "bbox": [round(x, 3), round(y, 3), round(bw, 3), round(bh, 3)],
        "segmentation": segmentation,
    }


def build_coco_json(
    ids: Sequence[str],
    dataset_name: str,
    dataset_dir: Path,
    source_image_dir: Path,
    source_mask_dir: Path,
    source_label_dir: Path,
    target_image_dir: Path,
    overwrite: bool,
) -> Tuple[Dict[str, object], Dict[str, int]]:
    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []
    ann_id = 1

    stats = {
        "num_images": 0,
        "num_boxes": 0,
        "num_components": 0,
        "duplicate_matches": 0,
        "fallback_segmentation_boxes": 0,
    }

    for image_id, sample_id in enumerate(ids, start=1):
        img_candidates = [source_image_dir / f"{sample_id}.png"]
        mask_candidates = [source_mask_dir / f"{sample_id}.png"]
        label_candidates = [source_label_dir / f"{sample_id}.txt"]

        if dataset_name == "NUAA-SIRST":
            img_candidates.append(dataset_dir.parent / "sirst-master" / "images" / f"{sample_id}.png")
            mask_candidates.append(dataset_dir.parent / "sirst-master" / "masks" / f"{sample_id}_pixels0.png")

        src_img = resolve_existing_path(img_candidates)
        src_mask = resolve_existing_path(mask_candidates)
        src_label = resolve_existing_path(label_candidates)
        dst_img = target_image_dir / f"{sample_id}.png"

        width, height = write_rgb_repeat(src_img, dst_img, overwrite=overwrite)
        boxes = load_yolo_boxes(src_label, width=width, height=height)
        components = load_components(src_mask)
        matched_components, duplicate_matches = assign_components_to_boxes(boxes, components)

        images.append(
            {
                "id": image_id,
                "file_name": dst_img.name,
                "width": width,
                "height": height,
            }
        )

        stats["num_images"] += 1
        stats["num_boxes"] += len(boxes)
        stats["num_components"] += len(components)
        stats["duplicate_matches"] += duplicate_matches

        for box, comp in zip(boxes, matched_components):
            if comp is None:
                stats["fallback_segmentation_boxes"] += 1
            annotations.append(
                build_annotation(
                    ann_id=ann_id,
                    image_id=image_id,
                    category_id=1,
                    box=box,
                    component=comp,
                    image_hw=(height, width),
                )
            )
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "target"}],
    }
    return coco, stats


def main() -> None:
    args = parse_args()
    ds_cfg = DATASET_CONFIGS[args.dataset]

    dataset_dir = Path(args.root) / args.dataset
    split_folder = args.split_method or ds_cfg["split_folder"]
    out_dir = Path(args.out_dir) if args.out_dir else dataset_dir / "boxinstseg_coco"

    source_image_dir = dataset_dir / ds_cfg["image_folder"]
    source_mask_dir = dataset_dir / ds_cfg["mask_folder"]
    source_label_dir = dataset_dir / ds_cfg["label_folder"]
    split_dir = dataset_dir / split_folder

    train_ids = read_split_ids(split_dir / "train.txt")
    val_ids = read_split_ids(split_dir / "test.txt")

    train_img_dir = out_dir / "train2017"
    val_img_dir = out_dir / "val2017"
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    train_coco, train_stats = build_coco_json(
        ids=train_ids,
        dataset_name=args.dataset,
        dataset_dir=dataset_dir,
        source_image_dir=source_image_dir,
        source_mask_dir=source_mask_dir,
        source_label_dir=source_label_dir,
        target_image_dir=train_img_dir,
        overwrite=args.overwrite,
    )
    val_coco, val_stats = build_coco_json(
        ids=val_ids,
        dataset_name=args.dataset,
        dataset_dir=dataset_dir,
        source_image_dir=source_image_dir,
        source_mask_dir=source_mask_dir,
        source_label_dir=source_label_dir,
        target_image_dir=val_img_dir,
        overwrite=args.overwrite,
    )

    with (ann_dir / "instances_train2017.json").open("w", encoding="utf-8") as f:
        json.dump(train_coco, f)
    with (ann_dir / "instances_val2017.json").open("w", encoding="utf-8") as f:
        json.dump(val_coco, f)

    summary = {
        "dataset": args.dataset,
        "source_root": str(dataset_dir),
        "output_root": str(out_dir),
        "train": train_stats,
        "val": val_stats,
    }
    with (ann_dir / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
