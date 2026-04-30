#!/usr/bin/env python3
"""
SCR-binned diagnostic for standalone DNANet+HALO checkpoints.

This follows the logic of HALO+PAL's scr_diagnostic_stepA.py:
  - SCR = (mu_target - mu_bg) / sigma_bg
  - pure-background protocol: exclude all GT target pixels from context stats
  - analysis-time context expand_ratio = 3.0
  - image SCR = minimum SCR across target components
  - report model performance by SCR bins

Default checkpoints are the DNANet+HALO (D group) weights listed in
paper copy/ADVISOR_REPORT.md.

Example:
  python analysis/scr_diagnostic_halo_dnanet.py --device cuda:0 --cache_json
  python analysis/scr_diagnostic_halo_dnanet.py --threshold 0.3 --output_dir analysis/results/scr_halo_dnanet
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.load_param_data import load_param  # noqa: E402
from model.model_DNANet import DNANet, Res_CBAM_block  # noqa: E402


SCR_EXPAND_RATIO = 3.0
SCR_EPSILON = 1e-4
SCR_BINS = [0.0, 3.0, 6.0, 9.0, 12.0, float("inf")]
SCR_BIN_LABELS = ["[0,3)", "[3,6)", "[6,9)", "[9,12)", "[12+)"]


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    dataset: str
    split_method: Optional[str]
    image_folder: str
    checkpoint: str
    color: str
    scr_hint: str


DEFAULT_DATASETS: Dict[str, DatasetConfig] = {
    "NUAA": DatasetConfig(
        key="NUAA",
        label="NUAA-SIRST",
        dataset="NUAA-SIRST",
        split_method="50_50",
        image_folder="images",
        checkpoint="/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/result/"
        "P1_D_pag_nuaa_NUAA-SIRST_DNANet_13_03_2026_21_12_40_wDS_new/"
        "best_model_epoch0417_mIoU0.7353_BoxIoU0.7089.pth.tar",
        color="#1565C0",
        scr_hint="high SCR",
    ),
    "NUDT": DatasetConfig(
        key="NUDT",
        label="NUDT-SIRST",
        dataset="NUDT-SIRST",
        split_method="50_50",
        image_folder="images",
        checkpoint="/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/result/"
        "P1_D_pag_nudt_NUDT-SIRST_DNANet_15_03_2026_01_30_01_wDS_new/"
        "best_model_epoch0440_mIoU0.6749_BoxIoU0.8307.pth.tar",
        color="#2E7D32",
        scr_hint="low SCR",
    ),
    "IRSTD": DatasetConfig(
        key="IRSTD",
        label="IRSTD-1K",
        dataset="IRSTD-1k",
        split_method=None,
        image_folder="IRSTD1k_Img",
        checkpoint="/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/result/"
        "P1_D_pag_irstd_IRSTD-1k_DNANet_14_03_2026_17_23_59_wDS_new/"
        "best_model_epoch0273_mIoU0.6433_BoxIoU0.6243.pth.tar",
        color="#E65100",
        scr_hint="high SCR",
    ),
}


def resolve_split_file(dataset_dir: Path, split_method: Optional[str], name: str) -> Path:
    if split_method:
        return dataset_dir / split_method / name
    return dataset_dir / name


def load_test_ids(dataset_dir: Path, split_method: Optional[str]) -> List[str]:
    test_txt = resolve_split_file(dataset_dir, split_method, "test.txt")
    if not test_txt.exists():
        raise FileNotFoundError(f"Test split not found: {test_txt}")
    return [line.strip() for line in test_txt.read_text().splitlines() if line.strip()]


def find_existing_image(path_without_ext: Path, suffixes: Sequence[str]) -> Path:
    for suffix in suffixes:
        path = path_without_ext.with_suffix(suffix)
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find file for stem: {path_without_ext}")


def compute_scr_values(
    image_gray: np.ndarray,
    mask: np.ndarray,
    expand_ratio: float = SCR_EXPAND_RATIO,
    epsilon: float = SCR_EPSILON,
) -> List[float]:
    """Compute target-level SCR with pure-background context statistics."""
    height, width = image_gray.shape
    image = image_gray.astype(np.float32)
    binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    scr_values: List[float] = []
    for label_idx in range(1, num_labels):
        x1 = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y1 = int(stats[label_idx, cv2.CC_STAT_TOP])
        box_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        x2 = x1 + box_w
        y2 = y1 + box_h
        if box_w <= 0 or box_h <= 0:
            continue

        component_mask = labels[y1:y2, x1:x2] == label_idx
        target_pixels = image[y1:y2, x1:x2][component_mask]
        if target_pixels.size == 0:
            continue

        pad_x = max(int((expand_ratio - 1.0) / 2.0 * box_w), 3)
        pad_y = max(int((expand_ratio - 1.0) / 2.0 * box_h), 3)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(width, x2 + pad_x)
        cy2 = min(height, y2 + pad_y)

        context = image[cy1:cy2, cx1:cx2]
        context_target = binary[cy1:cy2, cx1:cx2].astype(bool)
        background_pixels = context[~context_target]
        if background_pixels.size < 2:
            continue

        scr = (float(target_pixels.mean()) - float(background_pixels.mean())) / (
            float(background_pixels.std()) + epsilon
        )
        scr_values.append(scr)

    return scr_values


def compute_image_scr(image_gray: np.ndarray, mask: np.ndarray) -> float:
    values = compute_scr_values(image_gray, mask)
    return float(np.min(values)) if values else float("nan")


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def bin_label(scr: float) -> Optional[str]:
    if math.isnan(scr):
        return None
    for idx in range(len(SCR_BINS) - 1):
        if SCR_BINS[idx] <= scr < SCR_BINS[idx + 1]:
            return SCR_BIN_LABELS[idx]
    return None


class ScrDiagnosticDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        dataset_cfg: DatasetConfig,
        ids: Sequence[str],
        base_size: int,
        in_channels: int,
        suffixes: Sequence[str],
    ) -> None:
        self.dataset_root = dataset_root
        self.cfg = dataset_cfg
        self.ids = list(ids)
        self.base_size = base_size
        self.in_channels = in_channels
        self.suffixes = tuple(suffixes)
        self.image_dir = dataset_root / dataset_cfg.dataset / dataset_cfg.image_folder
        self.mask_dir = dataset_root / dataset_cfg.dataset / "masks"
        if in_channels == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.ids[idx]
        image_path = find_existing_image(self.image_dir / image_id, self.suffixes)
        mask_path = find_existing_image(self.mask_dir / image_id, self.suffixes)

        image_gray = np.array(Image.open(image_path).convert("L"))
        mask_gray = np.array(Image.open(mask_path).convert("L"))
        has_target = bool(mask_gray.max() > 127)
        scr = compute_image_scr(image_gray, mask_gray) if has_target else float("nan")

        image_mode = "L" if self.in_channels == 1 else "RGB"
        image = Image.open(image_path).convert(image_mode)
        mask = Image.open(mask_path).convert("L")
        image = image.resize((self.base_size, self.base_size), Image.BILINEAR)
        mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)

        image_tensor = self.transform(image)
        mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": image_id,
            "scr": torch.tensor(scr, dtype=torch.float32),
            "has_target": torch.tensor(has_target, dtype=torch.bool),
        }


def strip_module_prefix(state_dict: dict) -> dict:
    return {str(k).replace("module.", "", 1): v for k, v in state_dict.items()}


def detect_dnanet_params(state_dict: dict) -> Tuple[int, bool]:
    in_channels = 3
    deep_supervision = any(k.startswith("final1.") or ".final1." in k for k in state_dict)
    for key, value in state_dict.items():
        if "conv0_0.0.conv1.weight" in key or "conv0_0.0.weight" in key:
            if hasattr(value, "shape") and len(value.shape) == 4:
                in_channels = int(value.shape[1])
                break
    return in_channels, deep_supervision


def load_dnanet_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, dict, int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = strip_module_prefix(state_dict)
    in_channels, deep_supervision = detect_dnanet_params(state_dict)

    nb_filter, num_blocks = load_param("three", "resnet_18")
    model = DNANet(
        num_classes=1,
        input_channels=in_channels,
        block=Res_CBAM_block,
        num_blocks=num_blocks,
        nb_filter=nb_filter,
        deep_supervision=deep_supervision,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    info = {
        "epoch": checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None,
        "mIoU": checkpoint.get("mean_IOU", checkpoint.get("mIoU", checkpoint.get("iou", None)))
        if isinstance(checkpoint, dict)
        else None,
        "box_IOU": checkpoint.get("box_IOU", checkpoint.get("box_iou", None))
        if isinstance(checkpoint, dict)
        else None,
        "in_channels": in_channels,
        "deep_supervision": deep_supervision,
    }
    return model, info, in_channels


def evaluate_dataset(
    cfg: DatasetConfig,
    dataset_root: Path,
    checkpoint_path: Path,
    device: torch.device,
    base_size: int,
    batch_size: int,
    workers: int,
    threshold: float,
    suffixes: Sequence[str],
) -> Tuple[List[dict], dict]:
    model, checkpoint_info, in_channels = load_dnanet_checkpoint(checkpoint_path, device)
    ids = load_test_ids(dataset_root / cfg.dataset, cfg.split_method)
    dataset = ScrDiagnosticDataset(dataset_root, cfg, ids, base_size, in_channels, suffixes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    records: List[dict] = []
    total_inter = 0
    total_union = 0
    total_images = 0
    total_with_valid_scr = 0

    desc = f"{cfg.label}"
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, ncols=100):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            output = model(images)
            if isinstance(output, (list, tuple)):
                output = output[-1]
            probs = torch.sigmoid(output)
            pred_bin = (probs > threshold).cpu().numpy().astype(bool)
            gt_bin = (masks > 0.5).cpu().numpy().astype(bool)

            ids_batch = batch["id"]
            scr_batch = batch["scr"].cpu().numpy()
            has_target_batch = batch["has_target"].cpu().numpy().astype(bool)

            for idx, image_id in enumerate(ids_batch):
                pred = pred_bin[idx, 0]
                gt = gt_bin[idx, 0]
                inter = int(np.logical_and(pred, gt).sum())
                union = int(np.logical_or(pred, gt).sum())
                iou = 1.0 if union == 0 else float(inter) / float(union)
                scr = float(scr_batch[idx])
                if np.isnan(scr):
                    scr_value = None
                    label = None
                else:
                    scr_value = scr
                    label = bin_label(scr)
                    total_with_valid_scr += 1

                records.append(
                    {
                        "dataset": cfg.key,
                        "dataset_label": cfg.label,
                        "id": image_id,
                        "scr": scr_value,
                        "scr_bin": label,
                        "iou": iou,
                        "intersection": inter,
                        "union": union,
                        "has_target": bool(has_target_batch[idx]),
                    }
                )
                total_inter += inter
                total_union += union
                total_images += 1

    overall = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_info": checkpoint_info,
        "n_images": total_images,
        "n_valid_scr": total_with_valid_scr,
        "global_iou": float(total_inter / total_union) if total_union > 0 else float("nan"),
        "mean_per_image_iou": float(np.mean([r["iou"] for r in records])) if records else float("nan"),
    }
    return records, overall


def summarize_records(records: Sequence[dict]) -> List[dict]:
    rows: List[dict] = []
    for label in SCR_BIN_LABELS:
        subset = [r for r in records if r["scr_bin"] == label]
        if subset:
            inter = sum(int(r["intersection"]) for r in subset)
            union = sum(int(r["union"]) for r in subset)
            scr_values = [float(r["scr"]) for r in subset if r["scr"] is not None]
            ious = [float(r["iou"]) for r in subset]
            rows.append(
                {
                    "scr_bin": label,
                    "n": len(subset),
                    "scr_mean": float(np.mean(scr_values)),
                    "scr_median": float(np.median(scr_values)),
                    "global_iou": float(inter / union) if union > 0 else float("nan"),
                    "mean_per_image_iou": float(np.mean(ious)),
                    "std_per_image_iou": float(np.std(ious)),
                }
            )
        else:
            rows.append(
                {
                    "scr_bin": label,
                    "n": 0,
                    "scr_mean": float("nan"),
                    "scr_median": float("nan"),
                    "global_iou": float("nan"),
                    "mean_per_image_iou": float("nan"),
                    "std_per_image_iou": float("nan"),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown_summary(all_records: Dict[str, List[dict]], all_overall: Dict[str, dict]) -> None:
    print("\n" + "=" * 92)
    print("SCR-binned Diagnostic - DNANet+HALO standalone")
    print(f"SCR: pure_background, expand_ratio={SCR_EXPAND_RATIO}; image SCR = min target SCR")
    print("=" * 92)
    header = "| SCR Bin |"
    sep = "| --- |"
    for key in all_records:
        header += f" {key} mean IoU | N | {key} global IoU |"
        sep += " ---: | ---: | ---: |"
    print(header)
    print(sep)

    summaries = {key: {row["scr_bin"]: row for row in summarize_records(recs)} for key, recs in all_records.items()}
    for label in SCR_BIN_LABELS:
        line = f"| {label} |"
        for key in all_records:
            row = summaries[key][label]
            if row["n"] == 0:
                line += " - | 0 | - |"
            else:
                line += f" {row['mean_per_image_iou']:.4f} | {row['n']} | {row['global_iou']:.4f} |"
        print(line)

    print("\nOverall:")
    for key, overall in all_overall.items():
        print(
            f"  {key}: n={overall['n_images']}, valid SCR={overall['n_valid_scr']}, "
            f"mean per-image IoU={overall['mean_per_image_iou']:.4f}, "
            f"global IoU={overall['global_iou']:.4f}"
        )


def plot_results(all_records: Dict[str, List[dict]], output_dir: Path, selected_cfgs: Dict[str, DatasetConfig]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not available; skip plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(all_records), figsize=(5 * len(all_records), 4.2), sharey=True)
    if len(all_records) == 1:
        axes = [axes]
    for ax, (key, records) in zip(axes, all_records.items()):
        cfg = selected_cfgs[key]
        valid = [r for r in records if r["scr"] is not None]
        scrs = np.array([r["scr"] for r in valid], dtype=np.float32)
        ious = np.array([r["iou"] for r in valid], dtype=np.float32)
        if len(valid) > 0:
            ax.scatter(scrs, ious, s=18, alpha=0.45, color=cfg.color, linewidths=0)
        ax.axvline(3.0, color="#666666", linestyle=":", linewidth=1)
        ax.axvline(6.0, color="#999999", linestyle=":", linewidth=1)
        ax.set_title(f"{cfg.label} ({cfg.scr_hint})")
        ax.set_xlabel("SCR (min per image)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Per-image IoU")
    fig.suptitle("DNANet+HALO performance vs SCR")
    fig.tight_layout()
    scatter_path = output_dir / "fig1_scatter_scr_vs_iou.png"
    fig.savefig(scatter_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {scatter_path}")

    x = np.arange(len(SCR_BIN_LABELS))
    width = 0.8 / max(1, len(all_records))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for idx, (key, records) in enumerate(all_records.items()):
        cfg = selected_cfgs[key]
        summary = summarize_records(records)
        means = [row["mean_per_image_iou"] if row["n"] > 0 else np.nan for row in summary]
        ns = [row["n"] for row in summary]
        offset = (idx - (len(all_records) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, label=cfg.label, color=cfg.color, alpha=0.84)
        for bar, n in zip(bars, ns):
            if n > 0 and not np.isnan(bar.get_height()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={n}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(SCR_BIN_LABELS)
    ax.set_ylabel("Mean per-image IoU")
    ax.set_xlabel("SCR bin")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.18)
    ax.set_title("DNANet+HALO IoU by SCR bin")
    fig.tight_layout()
    bar_path = output_dir / "fig2_binned_iou_per_scr.png"
    fig.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {bar_path}")


def cache_ready(cache_path: Path, dataset_keys: Iterable[str]) -> bool:
    if not cache_path.exists():
        return False
    try:
        cached = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return False
    return all(key in cached.get("records", {}) for key in dataset_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="SCR-binned diagnostic for DNANet+HALO checkpoints")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu")
    parser.add_argument("--dataset_root", default=str(ROOT / "dataset"), help="Dataset root directory")
    parser.add_argument("--output_dir", default=str(ROOT / "analysis" / "results" / "scr_halo_dnanet"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS.keys()), choices=list(DEFAULT_DATASETS.keys()))
    parser.add_argument("--threshold", type=float, default=0.3, help="Binary threshold after sigmoid")
    parser.add_argument("--base_size", type=int, default=256, help="Evaluation resize size, matching test.py default")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--suffixes", nargs="+", default=[".png", ".bmp", ".jpg", ".jpeg"])
    parser.add_argument("--cache_json", action="store_true", help="Reuse/save per-image JSON cache")
    parser.add_argument("--no_plots", action="store_true", help="Skip matplotlib figures")
    parser.add_argument("--nuaa_checkpoint", default=DEFAULT_DATASETS["NUAA"].checkpoint)
    parser.add_argument("--nudt_checkpoint", default=DEFAULT_DATASETS["NUDT"].checkpoint)
    parser.add_argument("--irstd_checkpoint", default=DEFAULT_DATASETS["IRSTD"].checkpoint)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    ckpt_override = {
        "NUAA": args.nuaa_checkpoint,
        "NUDT": args.nudt_checkpoint,
        "IRSTD": args.irstd_checkpoint,
    }
    selected_cfgs = {key: DEFAULT_DATASETS[key] for key in args.datasets}

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if str(device) != args.device:
        print(f"[WARN] Requested {args.device}, but CUDA is unavailable. Falling back to CPU.")

    cache_path = output_dir / "per_image_data.json"
    all_records: Dict[str, List[dict]] = {}
    all_overall: Dict[str, dict] = {}

    if args.cache_json and cache_ready(cache_path, args.datasets):
        cached = json.loads(cache_path.read_text())
        for key in args.datasets:
            all_records[key] = cached["records"][key]
            all_overall[key] = cached["overall"][key]
        print(f"Loaded cache: {cache_path}")
    else:
        for key, cfg in selected_cfgs.items():
            checkpoint_path = Path(ckpt_override[key])
            print(f"\n[{cfg.label}] checkpoint: {checkpoint_path}")
            records, overall = evaluate_dataset(
                cfg=cfg,
                dataset_root=dataset_root,
                checkpoint_path=checkpoint_path,
                device=device,
                base_size=args.base_size,
                batch_size=args.batch_size,
                workers=args.workers,
                threshold=args.threshold,
                suffixes=args.suffixes,
            )
            all_records[key] = records
            all_overall[key] = overall

        if args.cache_json:
            cache_path.write_text(json.dumps({"records": all_records, "overall": all_overall}, indent=2))
            print(f"Saved cache: {cache_path}")

    print_markdown_summary(all_records, all_overall)

    per_image_rows = [row for records in all_records.values() for row in records]
    write_csv(
        output_dir / "per_image_scr_iou.csv",
        per_image_rows,
        ["dataset", "dataset_label", "id", "scr", "scr_bin", "iou", "intersection", "union", "has_target"],
    )

    summary_rows: List[dict] = []
    for key, records in all_records.items():
        for row in summarize_records(records):
            summary_rows.append({"dataset": key, "dataset_label": selected_cfgs[key].label, **row})
    write_csv(
        output_dir / "scr_bin_summary.csv",
        summary_rows,
        [
            "dataset",
            "dataset_label",
            "scr_bin",
            "n",
            "scr_mean",
            "scr_median",
            "global_iou",
            "mean_per_image_iou",
            "std_per_image_iou",
        ],
    )
    (output_dir / "overall.json").write_text(json.dumps(all_overall, indent=2))

    if not args.no_plots:
        plot_results(all_records, output_dir, selected_cfgs)

    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
