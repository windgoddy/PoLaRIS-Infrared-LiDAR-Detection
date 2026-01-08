#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析训练日志，对比所有实验结果
"""

import os
import re
import glob
from pathlib import Path
from datetime import datetime

def analyze_iou_log(log_path):
    """解析 mIoU 日志文件 - 完整训练历史分析"""
    if not os.path.exists(log_path):
        return None

    import numpy as np
    from datetime import datetime as dt

    # 存储完整训练历史
    history = {
        'epochs': [],
        'train_losses': [],
        'test_losses': [],
        'ious': [],
        'timestamps': []
    }

    best_epoch = 0
    best_iou = 0.0
    train_loss = 0.0
    test_loss = 0.0

    with open(log_path, 'r') as f:
        lines = f.readlines()

        # 解析所有行，构建完整训练历史
        for line in lines:
            # 格式：24/12/2025 21:36:46 - 0045:	 - train_loss: 0.1390:	 - test_loss: 0.3624:	 mIoU 0.6442
            # 提取时间戳
            timestamp_match = re.search(r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})', line)
            timestamp = None
            if timestamp_match:
                try:
                    timestamp = dt.strptime(timestamp_match.group(1), '%d/%m/%Y %H:%M:%S')
                except:
                    pass

            match = re.search(r'(\d+):\s*-\s*train_loss:\s*([\d.]+):\s*-\s*test_loss:\s*([\d.]+):\s*mIoU\s*([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                t_loss = float(match.group(2))
                v_loss = float(match.group(3))
                iou = float(match.group(4))

                history['epochs'].append(epoch)
                history['train_losses'].append(t_loss)
                history['test_losses'].append(v_loss)
                history['ious'].append(iou)
                history['timestamps'].append(timestamp)

        # 从所有轮次中找到真正的最佳结果
        if len(history['ious']) > 0:
            ious_array = np.array(history['ious'])
            # 找到最大 IoU 的索引
            best_idx = np.argmax(ious_array)
            # 提取对应的最佳指标
            best_iou = ious_array[best_idx]
            best_epoch = history['epochs'][best_idx]
            train_loss = history['train_losses'][best_idx]
            test_loss = history['test_losses'][best_idx]
        else:
            # 如果没有找到任何记录，保持默认值
            ious_array = np.array([])

    # 计算稳定性指标
    stability_metrics = {}
    if len(ious_array) > 0:

        # 1. IoU 标准差（越小越稳定）
        stability_metrics['iou_std'] = np.std(ious_array)

        # 2. IoU 最大波动
        stability_metrics['iou_range'] = np.max(ious_array) - np.min(ious_array)

        # 3. 最后 10 个 epoch 的平均 IoU（检验最终稳定性）
        if len(ious_array) >= 10:
            stability_metrics['last10_mean'] = np.mean(ious_array[-10:])
            stability_metrics['last10_std'] = np.std(ious_array[-10:])
        else:
            stability_metrics['last10_mean'] = np.mean(ious_array)
            stability_metrics['last10_std'] = np.std(ious_array)

        # 4. 有效训练比例（IoU > 阈值的 epoch 占比）
        threshold = 0.6  # 可调整
        stability_metrics['effective_ratio'] = np.sum(ious_array > threshold) / len(ious_array)

        # 5. 收敛速度（达到最佳 IoU 90% 时的 epoch）
        target_iou = best_iou * 0.9
        converged_epochs = [e for e, iou in zip(history['epochs'], history['ious']) if iou >= target_iou]
        stability_metrics['convergence_epoch'] = converged_epochs[0] if converged_epochs else -1

        # 6. 过拟合检测（test_loss / train_loss 比值）
        if len(history['test_losses']) > 0 and len(history['train_losses']) > 0:
            test_train_ratio = np.array(history['test_losses']) / (np.array(history['train_losses']) + 1e-8)
            stability_metrics['avg_overfit_ratio'] = np.mean(test_train_ratio[-10:])  # 最后10个epoch平均

        # 7. 训练时间分析（如果有时间戳）
        valid_timestamps = [ts for ts in history['timestamps'] if ts is not None]
        if len(valid_timestamps) >= 2:
            # 计算总训练时间
            total_time = (valid_timestamps[-1] - valid_timestamps[0]).total_seconds()
            stability_metrics['total_time_seconds'] = total_time
            stability_metrics['total_time_hours'] = total_time / 3600

            # 计算平均每个 epoch 的时间
            if len(valid_timestamps) > 1:
                time_diffs = [(valid_timestamps[i] - valid_timestamps[i-1]).total_seconds()
                             for i in range(1, len(valid_timestamps))]
                if time_diffs:
                    stability_metrics['avg_epoch_time_seconds'] = np.mean(time_diffs)
                    stability_metrics['avg_epoch_time_minutes'] = np.mean(time_diffs) / 60

        # 8. 训练效率指标（IoU 提升速度）
        if len(history['ious']) > 1:
            # 前期提升速度（前 10 个 epoch 的平均提升）
            early_epochs = min(10, len(history['ious']))
            if early_epochs > 1:
                early_improvement = (history['ious'][early_epochs-1] - history['ious'][0]) / early_epochs
                stability_metrics['early_improvement_rate'] = early_improvement

    return {
        'epoch': best_epoch,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'mean_IoU': best_iou,
        'history': history,
        'stability': stability_metrics
    }

def analyze_other_metric_log(log_path):
    """解析其他指标日志文件"""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None

    # 最后几行包含 Recall 和 Precision
    recall_line = None
    precision_line = None

    for i in range(len(lines) - 1, -1, -1):
        if 'Recall' in lines[i]:
            recall_line = lines[i]
        if 'Precision' in lines[i]:
            precision_line = lines[i]
        if recall_line and precision_line:
            break

    recall = []
    precision = []

    if recall_line:
        recall = [float(x) for x in re.findall(r'[\d.]+', recall_line.split('Recall')[1])]

    if precision_line:
        precision = [float(x) for x in re.findall(r'[\d.]+', precision_line.split('Precision')[1])]

    return {
        'recall': recall,
        'precision': precision
    }

def extract_experiment_name(dir_name):
    """从文件夹名称中提取实验名称"""
    # 格式: {experiment_name}_{dataset}_{model}_{timestamp}_wDS
    # 或者: {dataset}_{model}_{timestamp}_wDS (无 experiment_name)

    parts = dir_name.split('_')

    # 检查是否包含已知的数据集名称
    known_datasets = ['Pohang-Canal', 'NUDT-SIRST', 'NUAA-SIRST']

    for i, part in enumerate(parts):
        if any(ds in '_'.join(parts[i:i+2]) for ds in known_datasets):
            # 找到数据集位置，之前的就是 experiment_name
            if i > 0:
                return '_'.join(parts[:i])
            else:
                # 没有 experiment_name，使用时间戳作为唯一标识
                # 提取时间戳部分 (dd_mm_yyyy_hh_mm_ss)
                if len(parts) >= 8:
                    timestamp = '_'.join(parts[-7:-1])  # 去掉最后的 wDS/woDS
                    model = ''
                    if 'MS_CAFNet_DualGeo' in dir_name:
                        model = 'MS_CAFNet_DualGeo'
                    elif 'MS_CAFNet' in dir_name or 'MSCAFNet' in dir_name:
                        model = 'MS_CAFNet'
                    elif 'DNANet' in dir_name:
                        model = 'DNANet'

                    return f"{model}_{timestamp}" if model else timestamp
                else:
                    # 降级方案：使用模型名称
                    if 'MS_CAFNet_DualGeo' in dir_name:
                        return 'MS_CAFNet_DualGeo (old format)'
                    elif 'MS_CAFNet' in dir_name or 'MSCAFNet' in dir_name:
                        return 'MS_CAFNet (old format)'
                    elif 'DNANet' in dir_name:
                        return 'DNANet (old format)'
                    else:
                        return 'Unknown'

    return dir_name

def get_experiment_info(result_dir):
    """从训练日志中提取实验配置信息"""
    train_log = os.path.join(result_dir, 'train_log.txt')

    # 从文件夹名称中提取基本信息（作为降级方案）
    dir_name = os.path.basename(result_dir)

    # 识别模型类型（优先匹配更具体的模型名）
    if 'MS_CAFNet_DualGeo' in dir_name:
        default_model = 'MS_CAFNet_DualGeo'
    elif 'MS_CAFNet' in dir_name or 'MSCAFNet' in dir_name:
        default_model = 'MS_CAFNet'
    elif 'DNANet' in dir_name:
        default_model = 'DNANet'
    else:
        default_model = 'N/A'

    info = {
        'model': default_model,
        'in_channels': 'N/A',
        'optimizer': 'N/A',
        'lr': 'N/A',
        'epochs': 'N/A'
    }

    if os.path.exists(train_log):
        with open(train_log, 'r') as f:
            content = f.read()

            # 提取关键信息 - 尝试多种格式
            # 格式1: 'model': 'DNANet'
            model_match = re.search(r"['\"]model['\"]:\s*['\"](\w+)['\"]", content)
            if model_match:
                info['model'] = model_match.group(1)

            # 格式2: in_channels: 2 或 'in_channels': 2
            channels_match = re.search(r"['\"]?in_channels['\"]?:\s*(\d+)", content)
            if channels_match:
                info['in_channels'] = channels_match.group(1)

            # 优化器
            optimizer_match = re.search(r"['\"]?optimizer['\"]?:\s*['\"]?(\w+)['\"]?", content)
            if optimizer_match:
                info['optimizer'] = optimizer_match.group(1)

            # 学习率 - 支持科学计数法
            lr_match = re.search(r"['\"]?lr['\"]?:\s*([\d.eE+-]+)", content)
            if lr_match:
                info['lr'] = lr_match.group(1)

            # epochs
            epochs_match = re.search(r"['\"]?epochs['\"]?:\s*(\d+)", content)
            if epochs_match:
                info['epochs'] = epochs_match.group(1)

    return info

def find_all_result_dirs(base_dir='result'):
    """找到所有结果文件夹"""
    if not os.path.exists(base_dir):
        return []

    # 获取所有子文件夹
    result_dirs = [d for d in glob.glob(os.path.join(base_dir, '*'))
                   if os.path.isdir(d)]

    # 按修改时间排序（最新的在前）
    result_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    return result_dirs

def analyze_single_experiment(result_dir):
    """分析单个实验的结果"""
    dir_name = os.path.basename(result_dir)
    experiment_name = extract_experiment_name(dir_name)

    # 查找日志文件
    iou_logs = glob.glob(os.path.join(result_dir, '*_best_IoU_IoU.log'))
    other_logs = glob.glob(os.path.join(result_dir, '*_best_IoU_other_metric.log'))

    result = {
        'dir_name': dir_name,
        'experiment_name': experiment_name,
        'has_logs': len(iou_logs) > 0,
        'iou_data': None,
        'metric_data': None,
        'config': get_experiment_info(result_dir)
    }

    # 分析 IoU 日志
    if iou_logs:
        result['iou_data'] = analyze_iou_log(iou_logs[0])

    # 分析其他指标
    if other_logs:
        result['metric_data'] = analyze_other_metric_log(other_logs[0])

    return result

def print_comparison_table(experiments):
    """打印实验对比表格 - 基础性能"""
    print("\n" + "=" * 180)
    print("📊 实验结果对比 - 基础性能指标")
    print("=" * 180)

    # 表头
    header = f"{'#':<4} {'实验名称':<40} {'模型':<12} {'输入':<6} {'最佳Epoch':<10} {'最佳mIoU':<10} {'Train Loss':<12} {'Test Loss':<12} {'状态':<10}"
    print(header)
    print("-" * 180)

    # 找到最佳 IoU
    best_iou = 0.0
    for exp in experiments:
        if exp['iou_data'] and exp['iou_data']['mean_IoU'] > best_iou:
            best_iou = exp['iou_data']['mean_IoU']

    # 打印每个实验
    for idx, exp in enumerate(experiments, 1):
        name = exp['experiment_name'][:38]  # 截断过长的名称
        model = exp['config']['model']
        channels = exp['config']['in_channels']

        if exp['iou_data']:
            epoch = f"{exp['iou_data']['epoch']}"
            iou = f"{exp['iou_data']['mean_IoU']:.4f}"
            train_loss = f"{exp['iou_data']['train_loss']:.4f}"
            test_loss = f"{exp['iou_data']['test_loss']:.4f}"

            # 标记最佳结果
            if exp['iou_data']['mean_IoU'] == best_iou:
                status = "🏆 BEST"
                iou = f"*{iou}*"
            else:
                status = "✅"
        else:
            epoch = "N/A"
            iou = "N/A"
            train_loss = "N/A"
            test_loss = "N/A"
            status = "⚠️  No logs"

        row = f"{idx:<4} {name:<40} {model:<12} {channels:<6} {epoch:<10} {iou:<10} {train_loss:<12} {test_loss:<12} {status:<10}"
        print(row)

    print("=" * 180)

def print_efficiency_table(experiments):
    """打印训练效率对比表格 - 突出 Phase3 优势"""
    # 过滤出有稳定性数据的实验
    valid_experiments = [exp for exp in experiments
                        if exp['iou_data'] and 'stability' in exp['iou_data']
                        and exp['iou_data']['stability']]

    if not valid_experiments:
        return

    print("\n" + "=" * 180)
    print("⚡ 训练效率与时间成本分析")
    print("=" * 180)

    header = f"{'#':<4} {'实验名称':<40} {'总耗时':<12} {'每轮耗时':<12} {'收敛轮次':<12} {'IoU提升率':<14} {'效率评级':<12}"
    print(header)
    print("-" * 180)

    for idx, exp in enumerate(valid_experiments, 1):
        name = exp['experiment_name'][:38]
        stability = exp['iou_data']['stability']

        # 提取指标
        total_hours = stability.get('total_time_hours', -1)
        avg_epoch_minutes = stability.get('avg_epoch_time_minutes', -1)
        convergence = stability.get('convergence_epoch', -1)
        improvement_rate = stability.get('early_improvement_rate', 0)

        # 格式化输出
        if total_hours > 0:
            if total_hours < 1:
                time_str = f"{total_hours*60:.1f}分钟"
            else:
                time_str = f"{total_hours:.2f}小时"
        else:
            time_str = "N/A"

        if avg_epoch_minutes > 0:
            if avg_epoch_minutes < 1:
                epoch_time_str = f"{avg_epoch_minutes*60:.1f}秒"
            else:
                epoch_time_str = f"{avg_epoch_minutes:.2f}分钟"
        else:
            epoch_time_str = "N/A"

        conv_str = f"{convergence}轮" if convergence > 0 else "未收敛"
        improvement_str = f"+{improvement_rate:.4f}/轮"

        # 效率评级（综合收敛速度和提升率）
        efficiency_score = 0
        if convergence > 0 and convergence < 20:
            efficiency_score += 3
        elif convergence > 0 and convergence < 40:
            efficiency_score += 2
        elif convergence > 0 and convergence < 60:
            efficiency_score += 1

        if improvement_rate > 0.05:  # 前期快速提升
            efficiency_score += 2
        elif improvement_rate > 0.03:
            efficiency_score += 1

        if efficiency_score >= 4:
            efficiency_rating = "⚡⚡⚡ 极快"
        elif efficiency_score >= 3:
            efficiency_rating = "⚡⚡ 快速"
        elif efficiency_score >= 2:
            efficiency_rating = "⚡ 中等"
        else:
            efficiency_rating = "🐌 缓慢"

        row = f"{idx:<4} {name:<40} {time_str:<12} {epoch_time_str:<12} {conv_str:<12} {improvement_str:<14} {efficiency_rating:<12}"
        print(row)

    print("=" * 180)

    # 指标说明
    print("\n💡 效率指标说明:")
    print("  - 总耗时: 从第一个epoch到最后一个epoch的总时间")
    print("  - 每轮耗时: 平均每个epoch的训练时间")
    print("  - 收敛轮次: 达到最佳IoU 90%的轮次（越少越好）")
    print("  - IoU提升率: 前10轮平均每轮IoU提升（越大越好，表示快速学习）")
    print("  - 效率评级: 综合收敛速度和提升率的总体评价")

def print_stability_table(experiments):
    """打印稳定性分析表格"""
    # 过滤出有稳定性数据的实验
    valid_experiments = [exp for exp in experiments
                        if exp['iou_data'] and 'stability' in exp['iou_data']
                        and exp['iou_data']['stability']]

    if not valid_experiments:
        return

    print("\n" + "=" * 180)
    print("📈 训练稳定性与收敛性分析")
    print("=" * 180)

    header = f"{'#':<4} {'实验名称':<40} {'IoU波动':<12} {'最后10轮':<12} {'有效比例':<12} {'收敛Epoch':<12} {'过拟合比':<12} {'评级':<10}"
    print(header)
    print("-" * 180)

    for idx, exp in enumerate(valid_experiments, 1):
        name = exp['experiment_name'][:38]
        stability = exp['iou_data']['stability']

        # 提取指标
        iou_std = stability.get('iou_std', 0)
        last10_mean = stability.get('last10_mean', 0)
        effective_ratio = stability.get('effective_ratio', 0)
        convergence = stability.get('convergence_epoch', -1)
        overfit_ratio = stability.get('avg_overfit_ratio', 0)

        # 格式化输出
        std_str = f"{iou_std:.4f}"
        last10_str = f"{last10_mean:.4f}"
        eff_str = f"{effective_ratio*100:.1f}%"
        conv_str = f"{convergence}" if convergence > 0 else "未收敛"
        overfit_str = f"{overfit_ratio:.2f}x"

        # 综合评级
        score = 0
        if iou_std < 0.05:  # 稳定性好
            score += 2
        elif iou_std < 0.1:
            score += 1

        if effective_ratio > 0.8:  # 有效训练比例高
            score += 2
        elif effective_ratio > 0.6:
            score += 1

        if convergence > 0 and convergence < 30:  # 快速收敛
            score += 2
        elif convergence > 0 and convergence < 50:
            score += 1

        if overfit_ratio < 2.5:  # 无明显过拟合
            score += 2
        elif overfit_ratio < 3.5:
            score += 1

        # 评级映射
        if score >= 7:
            rating = "⭐⭐⭐ 优秀"
        elif score >= 5:
            rating = "⭐⭐ 良好"
        elif score >= 3:
            rating = "⭐ 一般"
        else:
            rating = "❌ 不稳定"

        row = f"{idx:<4} {name:<40} {std_str:<12} {last10_str:<12} {eff_str:<12} {conv_str:<12} {overfit_str:<12} {rating:<10}"
        print(row)

    print("=" * 180)

    # 指标说明
    print("\n💡 指标说明:")
    print("  - IoU波动: 标准差，越小越稳定（优秀 < 0.05, 良好 < 0.1）")
    print("  - 最后10轮: 训练末期平均IoU，反映最终性能")
    print("  - 有效比例: IoU > 0.6 的epoch占比，反映训练效率（优秀 > 80%, 良好 > 60%）")
    print("  - 收敛Epoch: 达到最佳IoU 90%时的轮次（越小越快，优秀 < 30, 良好 < 50）")
    print("  - 过拟合比: Test Loss / Train Loss（越小越好，优秀 < 2.5x, 良好 < 3.5x）")
    print("  - 评级: 综合上述5项指标的总体评价")

def print_detailed_analysis(experiments):
    """打印详细分析"""
    # 过滤出有有效数据的实验
    valid_experiments = [exp for exp in experiments if exp['iou_data']]

    if not valid_experiments:
        print("\n⚠️  没有找到有效的实验结果")
        return

    # 找到最佳实验
    best_exp = max(valid_experiments, key=lambda x: x['iou_data']['mean_IoU'])

    print(f"\n🏆 最佳实验: {best_exp['experiment_name']}")
    print("-" * 60)
    print(f"  📁 文件夹: {best_exp['dir_name']}")
    print(f"  📈 mean IoU: {best_exp['iou_data']['mean_IoU']:.4f}")
    print(f"  🎯 最佳 Epoch: {best_exp['iou_data']['epoch']}")
    print(f"  📉 Train Loss: {best_exp['iou_data']['train_loss']:.4f}")
    print(f"  📉 Test Loss: {best_exp['iou_data']['test_loss']:.4f}")

    # 稳定性信息
    if 'stability' in best_exp['iou_data'] and best_exp['iou_data']['stability']:
        stability = best_exp['iou_data']['stability']

        print(f"\n  ⚡ 训练效率:")
        # 时间信息
        total_hours = stability.get('total_time_hours', -1)
        if total_hours > 0:
            if total_hours < 1:
                print(f"    • 总训练时间: {total_hours*60:.1f} 分钟")
            else:
                print(f"    • 总训练时间: {total_hours:.2f} 小时")

        avg_epoch_minutes = stability.get('avg_epoch_time_minutes', -1)
        if avg_epoch_minutes > 0:
            if avg_epoch_minutes < 1:
                print(f"    • 每轮平均时间: {avg_epoch_minutes*60:.1f} 秒")
            else:
                print(f"    • 每轮平均时间: {avg_epoch_minutes:.2f} 分钟")

        improvement_rate = stability.get('early_improvement_rate', 0)
        if improvement_rate > 0:
            print(f"    • 前期学习速度: +{improvement_rate:.4f} IoU/轮")

        print(f"\n  📊 训练稳定性:")
        print(f"    • IoU 标准差: {stability.get('iou_std', 0):.4f}")
        print(f"    • 最后10轮平均: {stability.get('last10_mean', 0):.4f}")
        print(f"    • 有效训练比例: {stability.get('effective_ratio', 0)*100:.1f}%")

        conv_epoch = stability.get('convergence_epoch', -1)
        if conv_epoch > 0:
            print(f"    • 收敛速度: 第 {conv_epoch} 轮达到90%最佳性能")
        else:
            print(f"    • 收敛速度: 未达到稳定收敛")

        overfit = stability.get('avg_overfit_ratio', 0)
        if overfit < 2.5:
            overfit_status = "✅ 无过拟合"
        elif overfit < 3.5:
            overfit_status = "⚠️  轻微过拟合"
        else:
            overfit_status = "❌ 明显过拟合"
        print(f"    • 过拟合检测: {overfit:.2f}x ({overfit_status})")

    # 检测性能
    if best_exp['metric_data'] and best_exp['metric_data']['recall']:
        idx = 5 if len(best_exp['metric_data']['recall']) > 5 else len(best_exp['metric_data']['recall']) // 2
        print(f"\n  🎯 检测性能 (@ IoU 0.5):")
        print(f"    • Recall: {best_exp['metric_data']['recall'][idx]:.4f}")
        print(f"    • Precision: {best_exp['metric_data']['precision'][idx]:.4f}")

    # 性能对比
    print("\n📊 性能提升分析:")

    # 查找不同模型类型的实验
    baselines = [exp for exp in valid_experiments if 'baseline' in exp['experiment_name'].lower()]
    phase3_exps = [exp for exp in valid_experiments if 'phase3' in exp['experiment_name'].lower() and 'dualgeo' not in exp['experiment_name'].lower()]
    dualgeo_exps = [exp for exp in valid_experiments if 'dualgeo' in exp['experiment_name'].lower() or exp['config']['model'] == 'MS_CAFNet_DualGeo']

    # 对比 Baseline vs Phase3
    if baselines and phase3_exps:
        # 找到最佳实验
        baseline_best = max(baselines, key=lambda x: x['iou_data']['mean_IoU'])
        phase3_best = max(phase3_exps, key=lambda x: x['iou_data']['mean_IoU'])

        baseline_iou = baseline_best['iou_data']['mean_IoU']
        phase3_iou = phase3_best['iou_data']['mean_IoU']

        improvement = ((phase3_iou - baseline_iou) / baseline_iou) * 100

        print(f"  📊 Baseline 最佳 IoU: {baseline_iou:.4f}")
        print(f"  📊 Phase3 (原模型) 最佳 IoU: {phase3_iou:.4f}")
        print(f"  {'📈' if improvement > 0 else '📉'} 相比 Baseline: {improvement:+.2f}%")

        # 添加 Recall 和 Precision 对比
        if (baseline_best['metric_data'] and baseline_best['metric_data']['recall'] and
            phase3_best['metric_data'] and phase3_best['metric_data']['recall']):
            idx = 5 if len(baseline_best['metric_data']['recall']) > 5 else len(baseline_best['metric_data']['recall']) // 2

            baseline_recall = baseline_best['metric_data']['recall'][idx]
            phase3_recall = phase3_best['metric_data']['recall'][idx]
            baseline_precision = baseline_best['metric_data']['precision'][idx]
            phase3_precision = phase3_best['metric_data']['precision'][idx]

            recall_improvement = ((phase3_recall - baseline_recall) / baseline_recall) * 100
            precision_improvement = ((phase3_precision - baseline_precision) / baseline_precision) * 100

            print(f"\n  🎯 检测性能对比 (@ IoU 0.5):")
            print(f"    Recall:    {baseline_recall:.4f} → {phase3_recall:.4f} ({recall_improvement:+.2f}%)")
            print(f"    Precision: {baseline_precision:.4f} → {phase3_precision:.4f} ({precision_improvement:+.2f}%)")

        if improvement > 5:
            print("\n  ✅ Phase3 显著优于 Baseline")
        elif improvement > 0:
            print("\n  ⚡ Phase3 略优于 Baseline")
        else:
            print("\n  ⚠️  Phase3 未带来提升，需要调整")

    # 对比 Phase3 vs DualGeo
    if phase3_exps and dualgeo_exps:
        phase3_iou = max([exp['iou_data']['mean_IoU'] for exp in phase3_exps])
        dualgeo_iou = max([exp['iou_data']['mean_IoU'] for exp in dualgeo_exps])

        improvement = ((dualgeo_iou - phase3_iou) / phase3_iou) * 100

        print(f"\n  📊 Phase3_DualGeo 最佳 IoU: {dualgeo_iou:.4f}")
        print(f"  {'📈' if improvement > 0 else '📉'} 相比原 Phase3: {improvement:+.2f}%")

        if improvement > 2:
            print("  🎉 DualGeo 显著优于原模型！双流几何引导有效！")
        elif improvement > 0:
            print("  ✅ DualGeo 略优于原模型，双流机制有正面效果")
        else:
            print("  ⚠️  DualGeo 未带来提升，可能需要调整超参数")

        # 添加 Recall 和 Precision 对比
        phase3_best = max(phase3_exps, key=lambda x: x['iou_data']['mean_IoU'])
        dualgeo_best = max(dualgeo_exps, key=lambda x: x['iou_data']['mean_IoU'])

        if (phase3_best['metric_data'] and phase3_best['metric_data']['recall'] and
            dualgeo_best['metric_data'] and dualgeo_best['metric_data']['recall']):
            idx = 5 if len(phase3_best['metric_data']['recall']) > 5 else len(phase3_best['metric_data']['recall']) // 2

            phase3_recall = phase3_best['metric_data']['recall'][idx]
            dualgeo_recall = dualgeo_best['metric_data']['recall'][idx]
            phase3_precision = phase3_best['metric_data']['precision'][idx]
            dualgeo_precision = dualgeo_best['metric_data']['precision'][idx]

            recall_improvement = ((dualgeo_recall - phase3_recall) / phase3_recall) * 100
            precision_improvement = ((dualgeo_precision - phase3_precision) / phase3_precision) * 100

            print(f"\n  🎯 检测性能对比 (@ IoU 0.5):")
            print(f"    Recall:    {phase3_recall:.4f} → {dualgeo_recall:.4f} ({recall_improvement:+.2f}%)")
            print(f"    Precision: {phase3_precision:.4f} → {dualgeo_precision:.4f} ({precision_improvement:+.2f}%)")

    # 三模型对比
    if baselines and phase3_exps and dualgeo_exps:
        baseline_iou = max([exp['iou_data']['mean_IoU'] for exp in baselines])
        phase3_iou = max([exp['iou_data']['mean_IoU'] for exp in phase3_exps])
        dualgeo_iou = max([exp['iou_data']['mean_IoU'] for exp in dualgeo_exps])

        print(f"\n  🏆 总体进化路径:")
        print(f"    Baseline → Phase3 → DualGeo")
        print(f"    {baseline_iou:.4f} → {phase3_iou:.4f} (+{((phase3_iou - baseline_iou) / baseline_iou) * 100:.2f}%) → {dualgeo_iou:.4f} (+{((dualgeo_iou - phase3_iou) / phase3_iou) * 100:.2f}%)")
        print(f"    总提升: {((dualgeo_iou - baseline_iou) / baseline_iou) * 100:+.2f}%")

        # 添加 Recall 和 Precision 的三模型对比
        baseline_best = max(baselines, key=lambda x: x['iou_data']['mean_IoU'])
        phase3_best = max(phase3_exps, key=lambda x: x['iou_data']['mean_IoU'])
        dualgeo_best = max(dualgeo_exps, key=lambda x: x['iou_data']['mean_IoU'])

        if (baseline_best['metric_data'] and baseline_best['metric_data']['recall'] and
            phase3_best['metric_data'] and phase3_best['metric_data']['recall'] and
            dualgeo_best['metric_data'] and dualgeo_best['metric_data']['recall']):
            idx = 5 if len(baseline_best['metric_data']['recall']) > 5 else len(baseline_best['metric_data']['recall']) // 2

            baseline_recall = baseline_best['metric_data']['recall'][idx]
            phase3_recall = phase3_best['metric_data']['recall'][idx]
            dualgeo_recall = dualgeo_best['metric_data']['recall'][idx]
            baseline_precision = baseline_best['metric_data']['precision'][idx]
            phase3_precision = phase3_best['metric_data']['precision'][idx]
            dualgeo_precision = dualgeo_best['metric_data']['precision'][idx]

            recall_phase3_improve = ((phase3_recall - baseline_recall) / baseline_recall) * 100
            recall_dualgeo_improve = ((dualgeo_recall - phase3_recall) / phase3_recall) * 100
            recall_total_improve = ((dualgeo_recall - baseline_recall) / baseline_recall) * 100

            precision_phase3_improve = ((phase3_precision - baseline_precision) / baseline_precision) * 100
            precision_dualgeo_improve = ((dualgeo_precision - phase3_precision) / phase3_precision) * 100
            precision_total_improve = ((dualgeo_precision - baseline_precision) / baseline_precision) * 100

            print(f"\n  🎯 检测性能进化路径 (@ IoU 0.5):")
            print(f"    Recall:")
            print(f"      {baseline_recall:.4f} → {phase3_recall:.4f} (+{recall_phase3_improve:.2f}%) → {dualgeo_recall:.4f} (+{recall_dualgeo_improve:.2f}%)")
            print(f"      总提升: {recall_total_improve:+.2f}%")
            print(f"    Precision:")
            print(f"      {baseline_precision:.4f} → {phase3_precision:.4f} (+{precision_phase3_improve:.2f}%) → {dualgeo_precision:.4f} (+{precision_dualgeo_improve:.2f}%)")
            print(f"      总提升: {precision_total_improve:+.2f}%")

def main():
    print("=" * 160)
    print("🔍 训练结果分析工具 - 全量对比模式")
    print("=" * 160)

    # 查找所有结果文件夹
    result_dirs = find_all_result_dirs()

    if not result_dirs:
        print("❌ 未找到任何结果文件夹！")
        print("请确保在项目根目录运行此脚本，且 result/ 文件夹中有训练结果。")
        return

    print(f"\n📁 找到 {len(result_dirs)} 个实验结果文件夹")

    # 分析所有实验
    print("⏳ 正在分析所有实验...\n")
    experiments = []
    valid_count = 0
    for result_dir in result_dirs:
        exp_result = analyze_single_experiment(result_dir)
        experiments.append(exp_result)
        if exp_result['iou_data']:
            valid_count += 1

    print(f"✅ 成功分析 {valid_count}/{len(experiments)} 个实验（{len(experiments) - valid_count} 个未完成或缺失日志）")

    # 1. 打印基础性能对比表格
    print_comparison_table(experiments)

    # 2. 打印训练效率对比表格（NEW - 突出时间优势）
    print_efficiency_table(experiments)

    # 3. 打印稳定性分析表格
    print_stability_table(experiments)

    # 4. 打印详细分析
    print_detailed_analysis(experiments)

    print("\n" + "=" * 160)
    print("✅ 分析完成！")
    print("=" * 160)

if __name__ == "__main__":
    main()
