#!/usr/bin/env python3
"""
End-to-End Hyperparameter Sweep for HALO (Training mIoU, not Pseudo-Label IoU)
================================================================================
修正之前 hyperparameter_sensitivity_ablation.py 只测伪标签 IoU 的问题。
本脚本对每个参数设置完整执行：生成伪标签 → 训练 DNANet → 记录最佳训练 mIoU。

扫描参数（默认聚焦最关键对比，可按需扩展）：
  1. gauss_sigma_ratio: [1.0, 1.274, 1.5, 2.0]
     核心问题：sigma=2.0 的伪标签 IoU 更高，训练 mIoU 是否也更高？
  2. expand_ratio: [1.0, 1.5, 2.0]
     核心问题：1.0/1.5 在伪标签层面等价（3px下界），训练层面是否仍等价？

用法：
    # 最关键的对比：只跑 sigma sweep（约 4 × 3 = 12 次训练，约 6-12h）
    python paper/hyperparam_sweep_e2e.py --sweep sigma --dataset NUAA-SIRST --gpu 0

    # 完整 expand sweep（3 × 3 = 9 次训练）
    python paper/hyperparam_sweep_e2e.py --sweep expand --dataset NUDT-SIRST --gpu 1

    # 只跑最关键的 1.5 vs 2.0 对比（2 × 3 = 6 次训练，约 3-6h）
    python paper/hyperparam_sweep_e2e.py --sweep sigma --sigma_values 1.5,2.0 --gpu 0

    # 快速验证（减少 epoch，用于趋势判断，不用于报告）
    python paper/hyperparam_sweep_e2e.py --sweep sigma --epochs 500 --gpu 0

输出：
    result/hyperparam_sweep_e2e_TIMESTAMP/
    ├── sigma_sweep_NUAA-SIRST.csv     # sigma sweep 各设置训练 mIoU
    ├── expand_sweep_NUAA-SIRST.csv    # expand sweep 各设置训练 mIoU
    └── summary.txt                    # 汇总表格（对应 ADVISOR_REPORT.md 超参数消融节）
"""

import os
import sys
import argparse
import subprocess
import shutil
import csv
import json
import time
import glob
import importlib.util
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 路径设置
# ──────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 加载 bsnr_mask_utils（避免直接 import model_Mamba 触发 torch 依赖检查失败）
_spec = importlib.util.spec_from_file_location(
    "bsnr_mask_utils",
    os.path.join(ROOT, "model_Mamba", "dataset", "bsnr_mask_utils.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
generate_bsnr_mask = _mod.generate_bsnr_mask
_load_yolo_labels = _mod._load_yolo_labels

import cv2


# ──────────────────────────────────────────────────────────────────────────────
# 数据集配置（与主实验完全一致）
# ──────────────────────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    'NUAA-SIRST': {
        'img_dir':    'dataset/NUAA-SIRST/images',
        'mask_dir':   'dataset/NUAA-SIRST/masks',
        'label_dir':  'dataset/NUAA-SIRST/labels_box',
        'split_file': 'dataset/NUAA-SIRST/50_50/train.txt',
        'split_method': '50_50',
        'image_folder': 'images',
        'base_mask_dir': 'dataset/NUAA-SIRST',
    },
    'NUDT-SIRST': {
        'img_dir':    'dataset/NUDT-SIRST/images',
        'mask_dir':   'dataset/NUDT-SIRST/masks',
        'label_dir':  'dataset/NUDT-SIRST/labels_box',
        'split_file': 'dataset/NUDT-SIRST/50_50/train.txt',
        'split_method': '50_50',
        'image_folder': 'images',
        'base_mask_dir': 'dataset/NUDT-SIRST',
    },
    'IRSTD-1k': {
        'img_dir':    'dataset/IRSTD-1k/IRSTD1k_Img',
        'mask_dir':   'dataset/IRSTD-1k/masks',
        'label_dir':  'dataset/IRSTD-1k/labels_box',
        'split_file': 'dataset/IRSTD-1k/50_50/train.txt',
        'split_method': '50_50',
        'image_folder': 'IRSTD1k_Img',
        'base_mask_dir': 'dataset/IRSTD-1k',
    },
}

# 扫描参数默认值
SIGMA_VALUES  = [1.0, 1.274, 1.5, 2.0]
EXPAND_VALUES = [1.0, 1.5, 2.0]


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: 生成伪标签并保存到临时目录
# ──────────────────────────────────────────────────────────────────────────────
def generate_pseudolabels(dataset_name, cfg, output_mask_dir,
                          expand_ratio=1.5, sigma_ratio=1.5,
                          temperature=3.0):
    """
    对整个数据集（train split）生成 PAG 伪标签，保存为 PNG。
    output_mask_dir: 目标目录（会被 train.py 的 --mask_folder 参数引用）
    """
    os.makedirs(output_mask_dir, exist_ok=True)

    # 读取训练集 ID（只需生成训练集的伪标签，测试集始终用 GT masks/）
    split_file = os.path.join(ROOT, cfg['split_file'])
    if not os.path.exists(split_file):
        # fallback: 用 labels_box 目录下所有 txt
        label_dir = os.path.join(ROOT, cfg['label_dir'])
        img_ids = [f[:-4] for f in sorted(os.listdir(label_dir)) if f.endswith('.txt')]
    else:
        with open(split_file) as f:
            img_ids = [line.strip() for line in f if line.strip()]

    img_dir   = os.path.join(ROOT, cfg['img_dir'])
    label_dir = os.path.join(ROOT, cfg['label_dir'])
    generated = 0

    for img_id in img_ids:
        # 找图像文件（支持 .png / .bmp / .jpg）
        img_path = None
        for ext in ['.png', '.bmp', '.jpg']:
            candidate = os.path.join(img_dir, img_id + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        H, W = image.shape[:2]

        # 读取 YOLO 标签
        label_path = os.path.join(label_dir, img_id + '.txt')
        labels = _load_yolo_labels(label_path)

        # 生成 PAG 伪标签（spatial_gaussian=True 对应 D 组）
        mask = generate_bsnr_mask(
            labels, (H, W), image,
            expand_ratio=expand_ratio,
            temperature=temperature,
            fg_threshold=0.0,   # 保留连续值，不做硬截断
            soft_label=True,
            max_value=1.0,
            spatial_gaussian=True,      # D 组：PAG
            gauss_sigma_ratio=sigma_ratio,
        )

        # 保存为 uint8 PNG（与主实验 masks_bsnr_gauss/ 格式完全一致）
        out_path = os.path.join(output_mask_dir, img_id + '.png')
        mask_u8 = (mask * 255).clip(0, 255).astype('uint8')
        cv2.imwrite(out_path, mask_u8)
        generated += 1

    print(f"  [GenLabel] {dataset_name} sigma={sigma_ratio} expand={expand_ratio}: "
          f"generated {generated}/{len(img_ids)} masks → {output_mask_dir}")
    return generated


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: 调用 train.py 训练，捕获最佳 mIoU
# ──────────────────────────────────────────────────────────────────────────────
def run_training(dataset_name, cfg, mask_folder_name, experiment_tag,
                 epochs=1500, gpu='0', seed=42):
    """
    调用 train.py，mask_folder 指向已生成的伪标签目录。
    使用 --experiment_name 为每次训练打标签，之后在 result/ 目录查找 checkpoint。
    """
    # train.py 通过 make_dir() 自动创建:
    #   result/{experiment_name}_{dataset}_{model}_{timestamp}_wDS/
    # 所以用 experiment_tag 作为前缀来定位输出目录
    cmd = [
        sys.executable, os.path.join(ROOT, 'train.py'),
        '--model',           'DNANet',
        '--dataset',         dataset_name,
        '--split_method',    cfg['split_method'],
        '--root',            os.path.join(ROOT, 'dataset'),
        '--mask_folder',     mask_folder_name,
        '--image_folder',    cfg['image_folder'],
        '--epochs',          str(epochs),
        '--experiment_name', experiment_tag,
        '--gpus',            gpu,
        '--seed',            str(seed),
        '--train_batch_size', '4',
        '--test_batch_size',  '4',
        '--deep_supervision', 'True',
        '--in_channels',     '1',
        '--base_size',       '256',
        '--crop_size',       '256',
    ]

    print(f"\n  [Train] {dataset_name} mask={mask_folder_name} epochs={epochs}")
    print(f"  experiment_tag={experiment_tag}  CUDA_VISIBLE_DEVICES={gpu}")

    # 记录开始时间，用于在 result/ 目录中定位新建的 checkpoint 目录
    t_start = time.time()

    # 将 stdout/stderr 同时写到 result/sweep_logs/ 以便调试
    log_dir = os.path.join(ROOT, 'result', 'sweep_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{experiment_tag}.log')

    # train.py 使用 model.cuda()（无 device 参数），必须通过 CUDA_VISIBLE_DEVICES 指定 GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu

    with open(log_file, 'w') as flog:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=flog, stderr=subprocess.STDOUT, env=env)

    if proc.returncode != 0:
        print(f"  [WARN] Training returned non-zero exit code {proc.returncode}")

    # 在 result/ 中找 experiment_tag 前缀且创建时间晚于 t_start 的目录
    best_miou = _parse_best_miou_by_tag(experiment_tag, t_start, log_file)
    print(f"  [Result] best_mIoU = {best_miou:.4f}")
    return best_miou


def _parse_best_miou_by_tag(experiment_tag, t_start, fallback_log=None):
    """
    在 result/ 目录中定位 experiment_tag 开头的最新训练结果目录，
    然后从 checkpoint 文件名中解析最佳 mIoU。
    """
    result_root = os.path.join(ROOT, 'result')
    # 匹配所有以 experiment_tag 开头的子目录
    candidates = []
    if os.path.isdir(result_root):
        for name in os.listdir(result_root):
            full = os.path.join(result_root, name)
            if name.startswith(experiment_tag) and os.path.isdir(full):
                mtime = os.path.getmtime(full)
                if mtime >= t_start:
                    candidates.append(full)

    if not candidates:
        # 回退：尝试从 log 文件解析
        if fallback_log and os.path.exists(fallback_log):
            return _parse_miou_from_log(fallback_log)
        return float('nan')

    # 取最新的目录
    result_dir = max(candidates, key=os.path.getmtime)
    return _parse_best_miou_from_dir(result_dir, fallback_log)


def _parse_best_miou_from_dir(result_dir, fallback_log=None):
    """
    从 checkpoint 文件名中解析最佳 mIoU。
    checkpoint 文件名格式（来自 save_model in model/utils.py）：
        best_model_epoch{epoch:04d}_mIoU{mean_IOU:.4f}.pth.tar
        best_model_epoch{epoch:04d}_mIoU{mean_IOU:.4f}_BoxIoU{box_iou:.4f}.pth.tar
    """
    import re
    pattern = os.path.join(result_dir, 'best_model_epoch*_mIoU*.pth.tar')
    ckpts = glob.glob(pattern)
    if not ckpts:
        # 回退：读 log 文件
        log_file = os.path.join(result_dir, '*_best_IoU_IoU.log')
        log_files = glob.glob(log_file)
        if log_files:
            return _parse_miou_from_log(log_files[0])
        if fallback_log and os.path.exists(fallback_log):
            return _parse_miou_from_log(fallback_log)
        return float('nan')

    best = 0.0
    miou_re = re.compile(r'_mIoU([0-9]+\.[0-9]+)')
    for ckpt in ckpts:
        name = os.path.basename(ckpt)
        m = miou_re.search(name)
        if m:
            try:
                val = float(m.group(1))
                if 0.0 < val <= 1.0:
                    best = max(best, val)
            except ValueError:
                pass
    return best if best > 0 else float('nan')


def _parse_miou_from_log(log_file):
    """
    从训练 log 中提取最后出现的最佳 mIoU。
    支持两种 log 格式：
      1. train.py 最终 summary: "Best Segmentation IoU : 0.7069"
      2. save_model log (_best_IoU_IoU.log): "... mIoU 0.7069"
    """
    import re
    best = 0.0
    miou_pattern = re.compile(
        r'(?:mIoU|Best Segmentation IoU|best_iou)\s*[:\s]+([0-9]+\.[0-9]+)',
        re.IGNORECASE
    )
    try:
        with open(log_file) as f:
            for line in f:
                m = miou_pattern.search(line)
                if m:
                    try:
                        v = float(m.group(1))
                        if 0.0 < v <= 1.0:
                            best = max(best, v)
                    except ValueError:
                        pass
    except Exception:
        pass
    return best if best > 0 else float('nan')


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: 主扫描逻辑
# ──────────────────────────────────────────────────────────────────────────────
def run_sigma_sweep(datasets, sigma_values, out_root, epochs, gpu, seed):
    """gauss_sigma_ratio sweep（固定 expand_ratio=1.5）"""
    results = {}  # {dataset: {sigma: miou}}
    EXPAND_FIXED = 1.5

    for ds_name in datasets:
        cfg = DATASET_CONFIGS[ds_name]
        results[ds_name] = {}
        ds_base = os.path.join(ROOT, cfg['base_mask_dir'])

        for sigma in sigma_values:
            tag = f"sigma{sigma:.3f}_expand{EXPAND_FIXED:.1f}"
            mask_folder_name = f"masks_sweep_{tag}"
            mask_dir = os.path.join(ds_base, mask_folder_name)
            # experiment_tag 作为 train.py --experiment_name 的前缀
            experiment_tag = f"sweep_{ds_name}_{tag}"

            # 生成伪标签
            generate_pseudolabels(
                ds_name, cfg, mask_dir,
                expand_ratio=EXPAND_FIXED,
                sigma_ratio=sigma,
            )

            # 训练
            miou = run_training(
                ds_name, cfg, mask_folder_name, experiment_tag,
                epochs=epochs, gpu=gpu, seed=seed,
            )
            results[ds_name][sigma] = miou

            # 训练完成后删除临时伪标签目录（节省磁盘）
            if os.path.isdir(mask_dir):
                shutil.rmtree(mask_dir)
                print(f"  [Clean] Removed temp mask dir: {mask_dir}")

    return results


def run_expand_sweep(datasets, expand_values, out_root, epochs, gpu, seed):
    """expand_ratio sweep（固定 gauss_sigma_ratio=1.5）"""
    results = {}
    SIGMA_FIXED = 1.5

    for ds_name in datasets:
        cfg = DATASET_CONFIGS[ds_name]
        results[ds_name] = {}
        ds_base = os.path.join(ROOT, cfg['base_mask_dir'])

        for expand in expand_values:
            tag = f"sigma{SIGMA_FIXED:.1f}_expand{expand:.1f}"
            mask_folder_name = f"masks_sweep_{tag}"
            mask_dir = os.path.join(ds_base, mask_folder_name)
            experiment_tag = f"sweep_{ds_name}_{tag}"

            generate_pseudolabels(
                ds_name, cfg, mask_dir,
                expand_ratio=expand,
                sigma_ratio=SIGMA_FIXED,
            )

            miou = run_training(
                ds_name, cfg, mask_folder_name, experiment_tag,
                epochs=epochs, gpu=gpu, seed=seed,
            )
            results[ds_name][expand] = miou

            if os.path.isdir(mask_dir):
                shutil.rmtree(mask_dir)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: 输出汇总表格（可直接贴入 ADVISOR_REPORT.md）
# ──────────────────────────────────────────────────────────────────────────────
def print_and_save_results(results, param_name, param_values, out_root,
                           default_val, sweep_type):
    """打印 markdown 格式结果表，并保存 CSV。"""
    datasets = list(results.keys())

    # 打印 markdown 表格
    header = f"| {param_name} |" + "".join(f" {ds} |" for ds in datasets) + " 均值 | vs default |"
    sep    = "|---" * (len(datasets) + 3) + "|"
    print(f"\n\n{'='*70}")
    print(f"  端到端训练 mIoU — {sweep_type} sweep（正确版本）")
    print(f"{'='*70}")
    print(header)
    print(sep)

    rows = []
    default_mean = None
    for val in param_values:
        miou_per_ds = [results[ds].get(val, float('nan')) for ds in datasets]
        mean_val = np.nanmean(miou_per_ds)
        is_default = abs(val - default_val) < 1e-6
        marker = " ← default" if is_default else ""
        if is_default:
            default_mean = mean_val

        vals_str = "".join(f" {v*100:.2f}% |" for v in miou_per_ds)
        print(f"| {val}{marker} |{vals_str} {mean_val*100:.2f}% | — |")
        rows.append({'param': val, **{ds: results[ds].get(val, float('nan')) for ds in datasets},
                     'mean': mean_val})

    # 补充 vs default 列
    print("\n（补充 vs default）")
    print(header)
    print(sep)
    for val in param_values:
        miou_per_ds = [results[ds].get(val, float('nan')) for ds in datasets]
        mean_val = np.nanmean(miou_per_ds)
        is_default = abs(val - default_val) < 1e-6
        marker = " ← default" if is_default else ""
        vs = f"{(mean_val - default_mean)*100:+.2f}%" if default_mean is not None else "—"
        vals_str = "".join(f" {v*100:.2f}% |" for v in miou_per_ds)
        print(f"| {val}{marker} |{vals_str} {mean_val*100:.2f}% | {vs} |")

    # 保存 CSV
    csv_path = os.path.join(out_root, f"{sweep_type}_sweep.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['param'] + datasets + ['mean'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV saved → {csv_path}")

    # 保存 JSON（便于后续绘图）
    json_path = os.path.join(out_root, f"{sweep_type}_sweep.json")
    with open(json_path, 'w') as f:
        json.dump({'param_name': param_name, 'default': default_val,
                   'results': {ds: {str(k): v for k, v in results[ds].items()}
                                for ds in datasets}}, f, indent=2)
    print(f"  JSON saved → {json_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='End-to-End Hyperparameter Sweep for HALO (Training mIoU)')
    parser.add_argument('--sweep', choices=['sigma', 'expand', 'both'],
                        default='sigma',
                        help='Which parameter to sweep (default: sigma)')
    parser.add_argument('--dataset', default='all',
                        choices=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k', 'all'],
                        help='Dataset(s) to run (default: all)')
    parser.add_argument('--sigma_values', type=str, default=None,
                        help='Comma-separated sigma_ratio values, e.g. "1.5,2.0" '
                             '(default: 1.0,1.274,1.5,2.0)')
    parser.add_argument('--expand_values', type=str, default=None,
                        help='Comma-separated expand_ratio values, e.g. "1.0,1.5,2.0"')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Training epochs per run (default: 1500; use 500 for quick check)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output root dir (default: result/hyperparam_sweep_e2e_TIMESTAMP)')
    args = parser.parse_args()

    # 参数解析
    datasets = (list(DATASET_CONFIGS.keys()) if args.dataset == 'all'
                else [args.dataset])
    sigma_values  = ([float(x) for x in args.sigma_values.split(',')]
                     if args.sigma_values else SIGMA_VALUES)
    expand_values = ([float(x) for x in args.expand_values.split(',')]
                     if args.expand_values else EXPAND_VALUES)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_root = args.out_dir or os.path.join(ROOT, 'result', f'hyperparam_sweep_e2e_{timestamp}')
    os.makedirs(out_root, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  HALO End-to-End Hyperparameter Sweep")
    print(f"  sweep={args.sweep}  datasets={datasets}")
    print(f"  epochs={args.epochs}  gpu={args.gpu}  seed={args.seed}")
    print(f"  output → {out_root}")
    print(f"{'='*70}")

    # 估算总训练时间
    n_sigma  = len(sigma_values)  if args.sweep in ('sigma',  'both') else 0
    n_expand = len(expand_values) if args.sweep in ('expand', 'both') else 0
    n_runs   = (n_sigma + n_expand) * len(datasets)
    print(f"  预计训练次数: {n_runs}  （每次约 30-60min for 1500 epoch on 1 GPU）\n")

    # 执行 sweep
    if args.sweep in ('sigma', 'both'):
        print(f"\n>>> sigma_ratio sweep: {sigma_values}  (expand_ratio=1.5 fixed)")
        sigma_results = run_sigma_sweep(
            datasets, sigma_values, out_root, args.epochs, args.gpu, args.seed)
        print_and_save_results(
            sigma_results, 'sigma_ratio', sigma_values, out_root,
            default_val=1.5, sweep_type='sigma')

    if args.sweep in ('expand', 'both'):
        print(f"\n>>> expand_ratio sweep: {expand_values}  (sigma_ratio=1.5 fixed)")
        expand_results = run_expand_sweep(
            datasets, expand_values, out_root, args.epochs, args.gpu, args.seed)
        print_and_save_results(
            expand_results, 'expand_ratio', expand_values, out_root,
            default_val=1.5, sweep_type='expand')

    print(f"\n{'='*70}")
    print(f"  All done. Results in: {out_root}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
