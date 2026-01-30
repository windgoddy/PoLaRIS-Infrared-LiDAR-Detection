#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Category 对比评估脚本：对比 MS_CAFNet 和 DNANet 在所有三个类别上的性能

⚠️ 重要配置：
    - 阈值设置：threshold=0.3（与训练时一致，见 train_Phase3.py:55）
    - ❌ 之前错误地使用 0.5 导致 mIoU 和 Recall 严重偏低
    - ✅ 现已修正为 0.3

用法：
    python scripts/eval_cat3_comparison.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse

# 添加项目根目录到 Python 路径
# 获取脚本所在目录的父目录（项目根目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 切换到项目根目录（确保相对路径正确）
os.chdir(project_root)

from model.model_Phase3 import MS_CAFNet_DualGeo
from model.model_DNANet import DNANet
from model.utils_lidar import PoLaRISTestLoader
from model.utils import TestSetLoader


def load_category_mapping(dataset_dir):
    """加载类别映射"""
    summary_file = os.path.join(dataset_dir, 'selection_summary_new.txt')
    category_map = {}
    with open(summary_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 2:
                sample_id = parts[0].strip().replace('.png', '')
                category = int(parts[1].strip())
                category_map[sample_id] = category
    return category_map


def filter_category_samples(test_txt, category_map, target_category):
    """筛选出指定类别的样本"""
    with open(test_txt, 'r') as f:
        all_samples = [line.strip() for line in f if line.strip()]

    cat_samples = []
    for sample_id in all_samples:
        if sample_id in category_map and category_map[sample_id] == target_category:
            cat_samples.append(sample_id)

    return cat_samples


def create_temp_test_file(cat3_samples, output_path):
    """创建临时的 Category 3 测试文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for sample_id in cat3_samples:
            f.write(f"{sample_id}\n")
    return output_path


class MS_CAFNet_DualGeo_Legacy(nn.Module):
    """没有 Transformer Bottleneck 的旧版本 MS_CAFNet（用于加载旧权重）"""
    def __init__(self, in_channels=2, num_classes=1):
        super(MS_CAFNet_DualGeo_Legacy, self).__init__()
        # 导入旧版本的模型定义（没有 Transformer）
        # 这里直接使用 Phase3 的模型，但不初始化 transformer_bottleneck
        from model.model_DNANet import Res_CBAM_block
        import torch.nn as nn

        nb_filter = [16, 32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.conv0_0 = Res_CBAM_block(in_channels, nb_filter[0])
        self.conv1_0 = Res_CBAM_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Res_CBAM_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Res_CBAM_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Res_CBAM_block(nb_filter[3], nb_filter[4])

        # Multi-Scale Context Block
        from model.model_Phase3 import MSBlock
        self.ms_context = MSBlock(in_channels=nb_filter[4], out_channels=nb_filter[4])

        # Decoder
        self.conv3_1 = Res_CBAM_block(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = Res_CBAM_block(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = Res_CBAM_block(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = Res_CBAM_block(nb_filter[0] + nb_filter[1], nb_filter[0])

        # Deep Supervision (从深层到浅层)
        self.final1 = nn.Conv2d(nb_filter[3], num_classes, kernel_size=1)  # 从 x3_1: 128 通道
        self.final2 = nn.Conv2d(nb_filter[2], num_classes, kernel_size=1)  # 从 x2_2: 64 通道
        self.final3 = nn.Conv2d(nb_filter[1], num_classes, kernel_size=1)  # 从 x1_3: 32 通道
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)   # 从 x0_4: 16 通道 (最终输出)

    def forward(self, input):
        # Encoder
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Multi-Scale Context (旧版本直接使用，没有 Transformer)
        x4_0_enhanced = self.ms_context(x4_0)

        # Decoder
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0_enhanced)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        # Deep Supervision outputs (上采样到原始分辨率)
        out1 = self.final1(x3_1)  # 1/8 分辨率
        out1 = F.interpolate(out1, scale_factor=8, mode='bilinear', align_corners=True)

        out2 = self.final2(x2_2)  # 1/4 分辨率
        out2 = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=True)

        out3 = self.final3(x1_3)  # 1/2 分辨率
        out3 = F.interpolate(out3, scale_factor=2, mode='bilinear', align_corners=True)

        out_final = self.final(x0_4)  # 原始分辨率

        # 返回多尺度预测列表 (从粗到细)
        return [out1, out2, out3, out_final]


def calculate_metrics(pred, label, threshold=0.5):
    """
    计算各项指标

    Args:
        pred: 预测概率图 [H, W], 值在 [0, 1]
        label: 真实标签 [H, W], 值为 0 或 1
        threshold: 二值化阈值

    Returns:
        dict: 包含 TP, FP, TN, FN, IoU, Precision, Recall, F1, FPR
    """
    pred_binary = (pred > threshold).astype(np.float32)
    label = label.astype(np.float32)

    # 计算混淆矩阵元素
    TP = np.sum((pred_binary == 1) & (label == 1))
    FP = np.sum((pred_binary == 1) & (label == 0))
    TN = np.sum((pred_binary == 0) & (label == 0))
    FN = np.sum((pred_binary == 0) & (label == 1))

    # IoU (Intersection over Union)
    intersection = TP
    union = TP + FP + FN
    iou = intersection / (union + 1e-8)

    # Precision (精确率): TP / (TP + FP)
    precision = TP / (TP + FP + 1e-8)

    # Recall (召回率): TP / (TP + FN)
    recall = TP / (TP + FN + 1e-8)

    # F1 Score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # False Positive Rate: FP / (FP + TN)
    fpr = FP / (FP + TN + 1e-8)

    return {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'IoU': iou,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'FPR': fpr
    }


def evaluate_model(model, dataloader, device, threshold=0.5, use_imagenet_norm=False):
    """
    评估模型

    Args:
        use_imagenet_norm: 是否使用 ImageNet 归一化 (DNANet 需要)

    Returns:
        dict: 累计指标
    """
    model.eval()

    # 累计统计量
    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0

    ious = []

    # ImageNet 归一化参数
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 处理不同的 DataLoader 格式
            if isinstance(batch, dict):
                # PoLaRISTestLoader 返回字典
                data = batch['image'].to(device)
                labels = batch['mask'].to(device)
            else:
                # TestSetLoader 返回元组
                data, labels = batch
                data = data.to(device)
                labels = labels.to(device)

            # 确保数据类型为 float32 (TestSetLoader 返回 uint8)
            if data.dtype == torch.uint8:
                data = data.float() / 255.0

            # 检查数据格式是否为 NHWC，如果是则转换为 NCHW
            # PyTorch 期望 [B, C, H, W]，但某些 DataLoader 返回 [B, H, W, C]
            if data.dim() == 4 and data.shape[-1] in [1, 3]:  # 最后一维是通道
                data = data.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            # DNANet 需要 ImageNet 归一化
            if use_imagenet_norm:
                imagenet_mean = imagenet_mean.to(device)
                imagenet_std = imagenet_std.to(device)
                data = (data - imagenet_mean) / imagenet_std

            # 前向传播
            outputs = model(data)

            # 处理不同模型的输出格式
            # 1. 新 MS_CAFNet (带 Transformer): 返回 tuple ([out1, out2, out3, out_final], pred_conf)
            # 2. DNANet deep_supervision: 返回 list [output1, output2, output3, output4]
            # 3. 旧 MS_CAFNet: 返回 list [out1, out2, out3, out_final]
            if isinstance(outputs, tuple):
                # 新架构返回 (list_of_outputs, confidence_map)
                outputs, pred_conf = outputs
                # 继续处理 list_of_outputs

            if isinstance(outputs, list):
                # 深度监督输出：取最后一个（最精细的输出）
                output = outputs[-1]
            else:
                output = outputs

            # Sigmoid
            pred = torch.sigmoid(output)

            # 转换为 numpy
            pred_np = pred.cpu().numpy().squeeze()  # [H, W]
            label_np = labels.cpu().numpy().squeeze()  # [H, W]

            # 计算单张图像的指标
            metrics = calculate_metrics(pred_np, label_np, threshold)

            total_TP += metrics['TP']
            total_FP += metrics['FP']
            total_TN += metrics['TN']
            total_FN += metrics['FN']
            ious.append(metrics['IoU'])

    # 计算全局指标
    precision = total_TP / (total_TP + total_FP + 1e-8)
    recall = total_TP / (total_TP + total_FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    fpr = total_FP / (total_FP + total_TN + 1e-8)
    mean_iou = np.mean(ious)

    return {
        'mIoU': mean_iou,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'FPR': fpr,
        'TP': total_TP,
        'FP': total_FP,
        'TN': total_TN,
        'FN': total_FN,
        'num_images': len(ious)
    }


def load_mscafnet_model(weights_path, device, use_transformer=False):
    """加载 MS_CAFNet 模型

    Args:
        use_transformer: 是否使用带 Transformer 的新架构
    """
    if use_transformer:
        # 新架构：带 Transformer Bottleneck
        from model.model_Phase3 import MS_CAFNet_DualGeo
        model = MS_CAFNet_DualGeo(input_channels=2, num_classes=1)
        print("  使用新架构（带 Transformer Bottleneck）")
    else:
        # 旧架构：没有 Transformer
        model = MS_CAFNet_DualGeo_Legacy(in_channels=2, num_classes=1)
        print("  使用旧架构（无 Transformer）")

    # 加载权重
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"  ⚠️  缺少的键: {len(missing_keys)} 个")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"    - {key}")

    if unexpected_keys:
        print(f"  ⚠️  多余的键: {len(unexpected_keys)} 个")
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys:
                print(f"    - {key}")

    model = model.to(device)
    model.eval()

    return model


def load_dnanet_model(weights_path, device):
    """加载 DNANet 模型"""
    from model.model_DNANet import Res_CBAM_block

    # DNANet 标准配置
    nb_filter = [16, 32, 64, 128, 256]  # channel_size='three'
    num_blocks = [2, 2, 2, 2]  # backbone='resnet_18'

    model = DNANet(
        num_classes=1,
        input_channels=3,  # DNANet 训练时用的 3 通道 8-bit RGB
        block=Res_CBAM_block,
        num_blocks=num_blocks,
        nb_filter=nb_filter,
        deep_supervision=True  # ⚠️ 训练时用的是 True！
    )

    # 加载权重
    checkpoint = torch.load(weights_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


def evaluate_category(category, dataset_dir, test_txt, category_map,
                      mscafnet_model, dnanet_model, device, threshold=0.5):
    """评估单个类别"""
    category_names = {
        1: "简单场景（开放海面）",
        2: "中等场景（部分海岸）",
        3: "复杂场景（密集海岸）"
    }

    print(f"\n{'='*100}")
    print(f"评估 Category {category}: {category_names[category]}")
    print(f"{'='*100}")

    # 筛选样本
    cat_samples = filter_category_samples(test_txt, category_map, category)
    print(f"样本数量: {len(cat_samples)}")

    if len(cat_samples) == 0:
        print(f"⚠️  Category {category} 没有样本，跳过")
        return None

    # 评估 MS_CAFNet
    print(f"\n[1/2] 评估 MS_CAFNet...")
    mscafnet_loader = PoLaRISTestLoader(
        dataset_dir=dataset_dir,
        img_id=cat_samples,
        base_size=512,
        normalize_16bit=True,
        in_channels=2
    )
    mscafnet_dataloader = torch.utils.data.DataLoader(
        mscafnet_loader, batch_size=1, shuffle=False, num_workers=4
    )
    ms_results = evaluate_model(mscafnet_model, mscafnet_dataloader, device, threshold)

    print(f"MS_CAFNet - mIoU: {ms_results['mIoU']:.4f}, Precision: {ms_results['Precision']:.4f}, "
          f"Recall: {ms_results['Recall']:.4f}, F1: {ms_results['F1']:.4f}, FPR: {ms_results['FPR']:.4f}")

    # 评估 DNANet
    print(f"[2/2] 评估 DNANet...")
    dnanet_loader = TestSetLoader(
        dataset_dir=dataset_dir,
        img_id=cat_samples,
        base_size=256,
        image_folder='images-8bit'
    )
    dnanet_dataloader = torch.utils.data.DataLoader(
        dnanet_loader, batch_size=1, shuffle=False, num_workers=4
    )
    dna_results = evaluate_model(dnanet_model, dnanet_dataloader, device, threshold, use_imagenet_norm=True)

    print(f"DNANet    - mIoU: {dna_results['mIoU']:.4f}, Precision: {dna_results['Precision']:.4f}, "
          f"Recall: {dna_results['Recall']:.4f}, F1: {dna_results['F1']:.4f}, FPR: {dna_results['FPR']:.4f}")

    return {
        'MS_CAFNet': ms_results,
        'DNANet': dna_results,
        'num_samples': len(cat_samples)
    }


def print_category_comparison(category, results):
    """打印单个类别的对比"""
    category_names = {
        1: "简单场景（开放海面）",
        2: "中等场景（部分海岸）",
        3: "复杂场景（密集海岸）"
    }

    ms_results = results['MS_CAFNet']
    dna_results = results['DNANet']

    print(f"\n{'='*100}")
    print(f"Category {category} 对比：{category_names[category]}")
    print(f"{'='*100}")
    print(f"{'指标':<15} {'MS_CAFNet':>12} {'DNANet':>12} {'差值':>12} {'变化':>12}")
    print("-" * 100)

    metrics = ['mIoU', 'Precision', 'Recall', 'F1', 'FPR']
    ms_better_count = 0

    for metric in metrics:
        ms_val = ms_results[metric]
        dna_val = dna_results[metric]
        diff = ms_val - dna_val

        if metric == 'FPR':
            improvement = -diff / (dna_val + 1e-8) * 100
            symbol = '↓' if diff < 0 else '↑'
            if diff < 0:
                ms_better_count += 1
        else:
            improvement = diff / (dna_val + 1e-8) * 100
            symbol = '↑' if diff > 0 else '↓'
            if diff > 0:
                ms_better_count += 1

        print(f"{metric:<15} {ms_val:>12.4f} {dna_val:>12.4f} {diff:>+12.4f} {improvement:>+11.2f}% {symbol}")

    print()
    if ms_better_count >= 3:
        print(f"✅ MS_CAFNet 在 Category {category} 上表现更好 ({ms_better_count}/5 项指标优于 DNANet)")
    elif ms_better_count == 0:
        print(f"❌ DNANet 在 Category {category} 上全面优于 MS_CAFNet")
    else:
        print(f"⚖️  两个模型在 Category {category} 上各有优劣 (MS_CAFNet: {ms_better_count}/5, DNANet: {5-ms_better_count}/5)")

    return ms_better_count


def main():
    # 配置
    dataset_root = 'dataset'
    dataset_name = 'Pohang-Canal-3k'
    split_method = '50_50_2k_new'
    threshold = 0.3  # ✅ 修正：与训练时保持一致（原来错误使用了 0.5）

    # 权重文件路径
    # 旧权重（无 Transformer，在旧分割 50_50_2k 上训练）
    mscafnet_weights_old = 'result/PoLaRIS_16bit_full_Pohang-Canal-3k_MS_CAFNet_DualGeo_28_01_2026_11_45_53_wDS/best_model_epoch1441_mIoU0.7838.pth.tar'
    # 新权重（带 Transformer，在新分割 50_50_2k_new 上训练）
    mscafnet_weights_new = 'result/PoLaRIS_16bit_full_Pohang-Canal-3k_MS_CAFNet_DualGeo_28_01_2026_22_47_15_wDS/latest_best_model.pth.tar'

    # 选择使用哪个权重（默认使用新权重）
    use_new_weights = True  # 改为 False 使用旧权重
    mscafnet_weights = mscafnet_weights_new if use_new_weights else mscafnet_weights_old
    use_transformer = use_new_weights  # 新权重使用 Transformer

    dnanet_weights = 'result/DNANet_baseline_8bit_Pohang-Canal-3k_DNANet_27_01_2026_21_39_26_wDS/best_model_epoch1982_mIoU0.7774.pth.tar'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print()

    # 加载类别映射
    dataset_dir = os.path.join(dataset_root, dataset_name)
    category_map = load_category_mapping(dataset_dir)
    print(f"✅ 加载了 {len(category_map)} 个图像的类别标签")

    # 加载模型
    print(f"\n{'='*100}")
    print("加载模型")
    print(f"{'='*100}")
    print(f"MS_CAFNet: {mscafnet_weights}")
    mscafnet_model = load_mscafnet_model(mscafnet_weights, device, use_transformer=use_transformer)
    print(f"DNANet: {dnanet_weights}")
    dnanet_model = load_dnanet_model(dnanet_weights, device)

    # 评估所有类别
    test_txt = os.path.join(dataset_dir, split_method, 'test.txt')
    all_results = {}

    for category in [1, 2, 3]:
        results = evaluate_category(category, dataset_dir, test_txt, category_map,
                                   mscafnet_model, dnanet_model, device, threshold)
        if results:
            all_results[category] = results
            print_category_comparison(category, results)

    # 综合总结
    if len(all_results) > 0:
        print(f"\n{'='*100}")
        print("综合总结")
        print(f"{'='*100}\n")

        print(f"{'类别':<8} {'样本数':>8} {'MS_CAFNet mIoU':>18} {'DNANet mIoU':>15} "
              f"{'MS_CAFNet FPR':>18} {'DNANet FPR':>15} {'优势':<10}")
        print("-" * 100)

        for category in [1, 2, 3]:
            if category in all_results:
                ms = all_results[category]['MS_CAFNet']
                dna = all_results[category]['DNANet']
                n = all_results[category]['num_samples']

                # 计算优势指标数
                better = 0
                if ms['mIoU'] > dna['mIoU']: better += 1
                if ms['Precision'] > dna['Precision']: better += 1
                if ms['Recall'] > dna['Recall']: better += 1
                if ms['F1'] > dna['F1']: better += 1
                if ms['FPR'] < dna['FPR']: better += 1

                winner = "MS_CAFNet" if better >= 3 else "DNANet"
                cat_name = f"Cat{category}"

                print(f"{cat_name:<8} {n:>8} {ms['mIoU']:>18.4f} {dna['mIoU']:>15.4f} "
                      f"{ms['FPR']:>18.4f} {dna['FPR']:>15.4f} {winner:<10}")

        # 保存结果
        output_file = f'result/ALL_CATEGORIES_COMPARISON_{split_method}.txt'
        os.makedirs('result', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("全类别对比评估结果\n")
            f.write("="*100 + "\n\n")
            f.write(f"测试集: {split_method}\n\n")

            for category in [1, 2, 3]:
                if category in all_results:
                    ms = all_results[category]['MS_CAFNet']
                    dna = all_results[category]['DNANet']
                    n = all_results[category]['num_samples']

                    f.write(f"\nCategory {category} (样本数: {n}):\n")
                    f.write(f"  MS_CAFNet: mIoU={ms['mIoU']:.4f}, Precision={ms['Precision']:.4f}, "
                           f"Recall={ms['Recall']:.4f}, F1={ms['F1']:.4f}, FPR={ms['FPR']:.4f}\n")
                    f.write(f"  DNANet:    mIoU={dna['mIoU']:.4f}, Precision={dna['Precision']:.4f}, "
                           f"Recall={dna['Recall']:.4f}, F1={dna['F1']:.4f}, FPR={dna['FPR']:.4f}\n")

        print(f"\n📊 详细结果已保存到: {output_file}")

    print(f"\n✅ 评估完成！测试集: {split_method}")


if __name__ == '__main__':
    main()
