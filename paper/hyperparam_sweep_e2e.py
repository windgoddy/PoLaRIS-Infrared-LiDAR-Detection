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
def _make_train_cmd(dataset_name, cfg, mask_folder_name, experiment_tag,
                    epochs=1500, gpu='0', seed=42):
    """构造 train.py 调用命令（不执行）。"""
    return [
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


def run_training(dataset_name, cfg, mask_folder_name, experiment_tag,
                 epochs=1500, gpu='0', seed=42):
    """
    单进程串行训练（sequential 模式）。
    train.py 用 model.cuda()，必须通过 CUDA_VISIBLE_DEVICES 指定 GPU。
    """
    cmd = _make_train_cmd(dataset_name, cfg, mask_folder_name, experiment_tag,
                          epochs, gpu, seed)
    print(f"\n  [Train] {dataset_name} mask={mask_folder_name} epochs={epochs}")
    print(f"  experiment_tag={experiment_tag}  GPU={gpu}")

    t_start = time.time()
    log_dir  = os.path.join(ROOT, 'result', 'sweep_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{experiment_tag}.log')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu

    with open(log_file, 'w') as flog:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=flog, stderr=subprocess.STDOUT, env=env)

    if proc.returncode != 0:
        print(f"  [WARN] Training returned non-zero exit code {proc.returncode}")

    best_miou = _parse_best_miou_by_tag(experiment_tag, t_start, log_file)
    print(f"  [Result] best_mIoU = {best_miou:.4f}")
    return best_miou


def launch_training_async(dataset_name, cfg, mask_folder_name, experiment_tag,
                          epochs=1500, gpu='0', seed=42):
    """
    异步启动训练进程（parallel 模式）。
    返回 (Popen对象, log_file路径, t_start, experiment_tag)。
    """
    cmd = _make_train_cmd(dataset_name, cfg, mask_folder_name, experiment_tag,
                          epochs, gpu, seed)
    print(f"  [Launch] {dataset_name} sigma/expand={experiment_tag.split('_')[-1]}  GPU={gpu}")
    print(f"           log → result/sweep_logs/{experiment_tag}.log")

    log_dir  = os.path.join(ROOT, 'result', 'sweep_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{experiment_tag}.log')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    t_start = time.time()

    flog = open(log_file, 'w')
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=flog, stderr=subprocess.STDOUT, env=env)
    return proc, flog, log_file, t_start, experiment_tag


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


def run_parallel_sweep(jobs, gpu_list):
    """
    并行模式：
      jobs = [(ds_name, cfg, param_val, param_type, mask_folder, experiment_tag, mask_dir), ...]
      gpu_list = ['3', '4', '5', '6', ...]  按 round-robin 分配给各 job

    流程：
      1. 依次生成所有伪标签（串行，很快）
      2. 同时启动所有训练进程（并行，每个进程独占一个 GPU）
      3. 等待全部完成，收集 mIoU
      4. 清理临时伪标签目录
    """
    # Step 1: 批量生成伪标签
    print(f"\n[Parallel] Step 1: 生成所有伪标签...")
    for job in jobs:
        ds_name, cfg, param_val, param_type, mask_folder, exp_tag, mask_dir = job
        expand = 1.5 if param_type == 'sigma' else param_val
        sigma  = param_val if param_type == 'sigma' else 1.5
        generate_pseudolabels(ds_name, cfg, mask_dir,
                              expand_ratio=expand, sigma_ratio=sigma)

    # Step 2: 并行启动所有训练
    print(f"\n[Parallel] Step 2: 并行启动 {len(jobs)} 个训练进程...")
    procs = []
    for i, job in enumerate(jobs):
        ds_name, cfg, param_val, param_type, mask_folder, exp_tag, mask_dir = job
        gpu = gpu_list[i % len(gpu_list)]
        proc, flog, log_file, t_start, exp_tag = launch_training_async(
            ds_name, cfg, mask_folder, exp_tag,
            epochs=jobs[0][0] if False else _EPOCHS_GLOBAL,
            gpu=gpu, seed=_SEED_GLOBAL,
        )
        procs.append((proc, flog, log_file, t_start, exp_tag, job))

    # Step 3: 等待全部完成
    print(f"\n[Parallel] Step 3: 等待所有训练完成...")
    results_map = {}  # exp_tag -> miou
    for proc, flog, log_file, t_start, exp_tag, job in procs:
        proc.wait()
        flog.close()
        rc = proc.returncode
        if rc != 0:
            print(f"  [WARN] {exp_tag}: exit code {rc}")
        miou = _parse_best_miou_by_tag(exp_tag, t_start, log_file)
        print(f"  [Done] {exp_tag}: best_mIoU = {miou:.4f}")
        results_map[exp_tag] = miou

    # Step 4: 清理临时伪标签
    for job in jobs:
        _, _, _, _, _, _, mask_dir = job
        if os.path.isdir(mask_dir):
            shutil.rmtree(mask_dir)
            print(f"  [Clean] {mask_dir}")

    return results_map


# 全局变量，parallel 模式下通过闭包传递 epochs/seed
_EPOCHS_GLOBAL = 1500
_SEED_GLOBAL   = 42


def build_sigma_jobs(datasets, sigma_values):
    """构造 sigma sweep 的 job 列表。"""
    EXPAND_FIXED = 1.5
    jobs = []
    for ds_name in datasets:
        cfg = DATASET_CONFIGS[ds_name]
        ds_base = os.path.join(ROOT, cfg['base_mask_dir'])
        for sigma in sigma_values:
            tag = f"sigma{sigma:.3f}_expand{EXPAND_FIXED:.1f}"
            mask_folder = f"masks_sweep_{tag}"
            mask_dir    = os.path.join(ds_base, mask_folder)
            exp_tag     = f"sweep_{ds_name}_{tag}"
            jobs.append((ds_name, cfg, sigma, 'sigma', mask_folder, exp_tag, mask_dir))
    return jobs


def build_expand_jobs(datasets, expand_values):
    """构造 expand sweep 的 job 列表。"""
    SIGMA_FIXED = 1.5
    jobs = []
    for ds_name in datasets:
        cfg = DATASET_CONFIGS[ds_name]
        ds_base = os.path.join(ROOT, cfg['base_mask_dir'])
        for expand in expand_values:
            tag = f"sigma{SIGMA_FIXED:.1f}_expand{expand:.1f}"
            mask_folder = f"masks_sweep_{tag}"
            mask_dir    = os.path.join(ds_base, mask_folder)
            exp_tag     = f"sweep_{ds_name}_{tag}"
            jobs.append((ds_name, cfg, expand, 'expand', mask_folder, exp_tag, mask_dir))
    return jobs


def jobs_to_results(jobs, results_map):
    """将 results_map 重组为 {dataset: {param_val: miou}} 格式。"""
    results = {}
    for job in jobs:
        ds_name, _, param_val, _, _, exp_tag, _ = job
        if ds_name not in results:
            results[ds_name] = {}
        results[ds_name][param_val] = results_map.get(exp_tag, float('nan'))
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
        description='End-to-End Hyperparameter Sweep for HALO (Training mIoU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 串行（单 GPU）
  python paper/hyperparam_sweep_e2e.py --sweep sigma --sigma_values 1.5,2.0 --gpu 3

  # 并行（多 GPU，每个 job 独占一个 GPU）
  python paper/hyperparam_sweep_e2e.py --sweep sigma --sigma_values 1.5,2.0 --gpus 3,4,5,6,7,8

  # 并行 + 指定数据集
  python paper/hyperparam_sweep_e2e.py --sweep both --gpus 3,4,5,6,7,8
""")
    parser.add_argument('--sweep', choices=['sigma', 'expand', 'both'],
                        default='sigma')
    parser.add_argument('--dataset', default='all',
                        choices=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k', 'all'])
    parser.add_argument('--sigma_values', type=str, default=None,
                        help='逗号分隔，如 "1.5,2.0"（默认: 1.0,1.274,1.5,2.0）')
    parser.add_argument('--expand_values', type=str, default=None,
                        help='逗号分隔，如 "1.0,1.5,2.0"')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--gpu', type=str, default=None,
                        help='串行模式：单个 GPU ID（如 "3"）')
    parser.add_argument('--gpus', type=str, default=None,
                        help='并行模式：逗号分隔的 GPU ID 列表（如 "3,4,5,6"）'
                             '，job 数量超过 GPU 数时按 round-robin 分配')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--gen_only', action='store_true',
                        help='只生成伪标签，不训练（配合 tmux 手动启动训练）')
    parser.add_argument('--collect', action='store_true',
                        help='只汇总已完成训练的结果（从 result/ 目录读取 checkpoint）')
    args = parser.parse_args()

    # 确定串行/并行模式
    if args.gpus:
        gpu_list  = [g.strip() for g in args.gpus.split(',')]
        parallel  = True
    else:
        gpu_list  = [args.gpu or '0']
        parallel  = False

    # 参数解析
    datasets = (list(DATASET_CONFIGS.keys()) if args.dataset == 'all'
                else [args.dataset])
    sigma_values  = ([float(x) for x in args.sigma_values.split(',')]
                     if args.sigma_values else SIGMA_VALUES)
    expand_values = ([float(x) for x in args.expand_values.split(',')]
                     if args.expand_values else EXPAND_VALUES)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_root  = args.out_dir or os.path.join(ROOT, 'result', f'hyperparam_sweep_e2e_{timestamp}')
    os.makedirs(out_root, exist_ok=True)

    # 把 epochs/seed 写入全局，供 run_parallel_sweep 内部使用
    global _EPOCHS_GLOBAL, _SEED_GLOBAL
    _EPOCHS_GLOBAL = args.epochs
    _SEED_GLOBAL   = args.seed

    n_sigma  = len(sigma_values)  if args.sweep in ('sigma',  'both') else 0
    n_expand = len(expand_values) if args.sweep in ('expand', 'both') else 0
    n_runs   = (n_sigma + n_expand) * len(datasets)

    print(f"\n{'='*70}")
    print(f"  HALO End-to-End Hyperparameter Sweep")
    print(f"  mode={'gen_only' if args.gen_only else ('parallel' if parallel else 'sequential')}")
    print(f"  sweep={args.sweep}  datasets={datasets}")
    print(f"  epochs={args.epochs}  gpus={gpu_list}  seed={args.seed}")
    print(f"  训练次数: {n_runs}  output → {out_root}")
    print(f"{'='*70}\n")

    # ── gen_only 模式：只生成伪标签，打印训练命令供手动执行 ──
    if args.gen_only:
        all_sigma_jobs  = build_sigma_jobs(datasets, sigma_values)  if args.sweep in ('sigma', 'both') else []
        all_expand_jobs = build_expand_jobs(datasets, expand_values) if args.sweep in ('expand', 'both') else []
        all_jobs = all_sigma_jobs + all_expand_jobs

        print("[gen_only] 生成所有伪标签...")
        for job in all_jobs:
            ds_name, cfg, param_val, param_type, mask_folder, exp_tag, mask_dir = job
            expand = 1.5 if param_type == 'sigma' else param_val
            sigma  = param_val if param_type == 'sigma' else 1.5
            generate_pseudolabels(ds_name, cfg, mask_dir,
                                  expand_ratio=expand, sigma_ratio=sigma)

        print(f"\n[gen_only] 完成！共生成 {len(all_jobs)} 套伪标签。")
        print("\n以下是对应的训练命令（在不同 GPU/tmux 窗口中执行）：\n")
        for i, job in enumerate(all_jobs):
            ds_name, cfg, param_val, param_type, mask_folder, exp_tag, mask_dir = job
            img_folder = cfg['image_folder']
            split      = cfg['split_method']
            print(f"# Job {i+1}: {exp_tag}")
            print(f"CUDA_VISIBLE_DEVICES=? python train.py \\")
            print(f"    --model DNANet --dataset {ds_name} --split_method {split} \\")
            print(f"    --root dataset --image_folder {img_folder} --mask_folder {mask_folder} \\")
            print(f"    --epochs {args.epochs} --experiment_name {exp_tag} \\")
            print(f"    --seed {args.seed} --in_channels 1 --base_size 256 --crop_size 256 \\")
            print(f"    --deep_supervision True --train_batch_size 4 --test_batch_size 4")
            print()
        return

    # ── collect 模式：扫描已完成的 checkpoint，直接输出结果表格 ──
    if args.collect:
        all_sigma_jobs  = build_sigma_jobs(datasets, sigma_values)  if args.sweep in ('sigma', 'both') else []
        all_expand_jobs = build_expand_jobs(datasets, expand_values) if args.sweep in ('expand', 'both') else []

        if all_sigma_jobs:
            print("[collect] 扫描 sigma sweep 结果...")
            results_map = {}
            for job in all_sigma_jobs:
                _, _, _, _, _, exp_tag, _ = job
                miou = _parse_best_miou_by_tag(exp_tag, t_start=0)
                print(f"  {exp_tag}: {miou:.4f}" if not (miou != miou) else f"  {exp_tag}: not found")
                results_map[exp_tag] = miou
            sigma_results = jobs_to_results(all_sigma_jobs, results_map)
            print_and_save_results(sigma_results, 'sigma_ratio', sigma_values, out_root,
                                   default_val=1.5, sweep_type='sigma')

        if all_expand_jobs:
            print("[collect] 扫描 expand sweep 结果...")
            results_map = {}
            for job in all_expand_jobs:
                _, _, _, _, _, exp_tag, _ = job
                miou = _parse_best_miou_by_tag(exp_tag, t_start=0)
                print(f"  {exp_tag}: {miou:.4f}" if not (miou != miou) else f"  {exp_tag}: not found")
                results_map[exp_tag] = miou
            expand_results = jobs_to_results(all_expand_jobs, results_map)
            print_and_save_results(expand_results, 'expand_ratio', expand_values, out_root,
                                   default_val=1.5, sweep_type='expand')
        return

    if parallel:
        # ── 并行模式：一次性收集所有 job，同时启动 ──
        all_sigma_jobs  = build_sigma_jobs(datasets, sigma_values)  if args.sweep in ('sigma', 'both') else []
        all_expand_jobs = build_expand_jobs(datasets, expand_values) if args.sweep in ('expand', 'both') else []
        all_jobs = all_sigma_jobs + all_expand_jobs

        print(f"[Parallel] {len(all_jobs)} jobs → {len(gpu_list)} GPUs ({gpu_list})")
        results_map = run_parallel_sweep(all_jobs, gpu_list)

        if all_sigma_jobs:
            sigma_results = jobs_to_results(all_sigma_jobs, results_map)
            print_and_save_results(sigma_results, 'sigma_ratio', sigma_values, out_root,
                                   default_val=1.5, sweep_type='sigma')
        if all_expand_jobs:
            expand_results = jobs_to_results(all_expand_jobs, results_map)
            print_and_save_results(expand_results, 'expand_ratio', expand_values, out_root,
                                   default_val=1.5, sweep_type='expand')
    else:
        # ── 串行模式（原逻辑）──
        gpu = gpu_list[0]
        if args.sweep in ('sigma', 'both'):
            print(f">>> sigma_ratio sweep: {sigma_values}  (expand_ratio=1.5 fixed)")
            sigma_results = run_sigma_sweep(datasets, sigma_values, out_root,
                                            args.epochs, gpu, args.seed)
            print_and_save_results(sigma_results, 'sigma_ratio', sigma_values, out_root,
                                   default_val=1.5, sweep_type='sigma')
        if args.sweep in ('expand', 'both'):
            print(f">>> expand_ratio sweep: {expand_values}  (sigma_ratio=1.5 fixed)")
            expand_results = run_expand_sweep(datasets, expand_values, out_root,
                                              args.epochs, gpu, args.seed)
            print_and_save_results(expand_results, 'expand_ratio', expand_values, out_root,
                                   default_val=1.5, sweep_type='expand')

    print(f"\n{'='*70}")
    print(f"  All done. Results in: {out_root}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
