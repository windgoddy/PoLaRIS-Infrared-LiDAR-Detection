#!/usr/bin/env python3
"""
eval_pdfa_fixed.py
===================
评估指定 checkpoint 在固定阈值（默认 0.5）下的 Pd/Fa，
同时输出 mIoU，用于修正 best_pd_idx 策略下 Fa 异常的实验。

用法示例（IRSTD ACM A 组）：
    python eval_pdfa_fixed.py \
        --model ACM \
        --dataset IRSTD-1k \
        --image_folder IRSTD1k_Img \
        --model_dir P2_A_gt_sirst3_acm_IRSTD-1k_ACM_.../best_model_epoch1XXX_mIoU*.pth.tar \
        --threshold 0.5

支持模型：DNANet, ACM, ALCNet
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.utils import *
from model.metric import mIoU, PD_FA
from model.load_param_data import load_dataset, load_param
from model.model_DNANet import Res_CBAM_block, DNANet
from model.model_ACM import ASKCResUNet
from model.model_ALCNet import ASKCResNetFPN


def parse_args():
    parser = argparse.ArgumentParser(description='Fixed-threshold Pd/Fa evaluator')
    parser.add_argument('--model', type=str, default='ACM',
                        choices=['DNANet', 'ACM', 'ALCNet'])
    parser.add_argument('--dataset', type=str, default='IRSTD-1k')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--split_method', type=str, default='50_50')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--image_folder', type=str, default='images')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--base_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model_dir', type=str, required=True,
                        help='相对路径（从项目根目录），如 P2_A_gt.../best_model_epoch*.pth.tar；'
                             '不含 result/ 前缀')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='固定推理阈值（默认 0.5）')
    # DNANet specific
    parser.add_argument('--channel_size', type=str, default='three')
    parser.add_argument('--backbone', type=str, default='resnet_18')
    parser.add_argument('--deep_supervision', type=str, default='False')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = os.path.join(args.root, args.dataset)
    _, val_img_ids, _ = load_dataset(args.root, args.dataset, args.split_method)

    if args.in_channels == 1:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
    else:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    testset = TestSetLoader(
        dataset_dir, img_id=val_img_ids,
        base_size=args.base_size, crop_size=args.crop_size,
        transform=input_transform, suffix=args.suffix,
        in_channels=args.in_channels, image_folder=args.image_folder)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                             num_workers=args.workers, drop_last=False)

    # Build model
    if args.model == 'DNANet':
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        model = DNANet(num_classes=1, input_channels=args.in_channels,
                       block=Res_CBAM_block, num_blocks=num_blocks,
                       nb_filter=nb_filter, deep_supervision=args.deep_supervision)
    elif args.model == 'ACM':
        model = ASKCResUNet(in_channels=args.in_channels, layers=[3, 3, 3],
                            channels=[8, 16, 32, 64], fuse_mode='AsymBi',
                            tiny=False, classes=1)
    elif args.model == 'ALCNet':
        model = ASKCResNetFPN(in_channels=args.in_channels, layers=[4, 4, 4],
                              channels=[8, 16, 32, 64], fuse_mode='AsymBi',
                              act_dilation=16, classes=1)

    model = model.cuda()
    model.apply(weights_init_xavier)

    ckpt_path = os.path.join('result', args.model_dir)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Metrics: PD_FA with bins=10, same as train.py
    pd_fa = PD_FA(1, bins=10, img_size=args.base_size)
    miou_metric = mIoU(1, threshold=args.threshold)

    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc='Evaluating'):
            data = data.cuda()
            labels = labels.cuda()

            if args.model == 'DNANet' and args.deep_supervision == 'True':
                preds = model(data)
                pred = preds[-1]
            else:
                pred = model(data)

            # PD_FA: sigmoid(pred)*255, matching train.py
            pd_fa.update(torch.sigmoid(pred) * 255, labels)
            miou_metric.update(pred, labels)

    FA, PD = pd_fa.get(len(val_img_ids))
    _, mean_iou = miou_metric.get()

    # iBin=5 → score_thresh = 5 * (255/10) = 127.5  →  sigmoid(pred) ≈ 0.5
    pd5 = float(PD[5]) if not np.isnan(PD[5]) else 0.0
    fa5 = float(FA[5])

    # Also report best Pd across all thresholds
    valid_pd = np.where(np.isnan(PD), 0, PD)
    best_idx = int(np.argmax(valid_pd))
    best_pd = float(valid_pd[best_idx])
    fa_best = float(FA[best_idx])

    print(f"\n{'='*55}")
    print(f"  Model    : {args.model}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Ckpt     : {args.model_dir}")
    print(f"{'='*55}")
    print(f"  mIoU                    : {mean_iou:.4f}  ({mean_iou*100:.2f}%)")
    print(f"  Pd @ threshold=0.5      : {pd5:.4f}  ({pd5*100:.2f}%)")
    print(f"  Fa @ threshold=0.5      : {fa5:.2e}")
    print(f"  Best Pd (any threshold) : {best_pd:.4f}  ({best_pd*100:.2f}%)  [iBin={best_idx}]")
    print(f"  Fa @ Best Pd threshold  : {fa_best:.2e}")
    print(f"{'='*55}")

    print(f"\nAll thresholds (iBin → score_thresh → Pd, Fa):")
    for i in range(11):
        thr = i * (255 / 10)
        pd_i = float(PD[i]) if not np.isnan(PD[i]) else float('nan')
        fa_i = float(FA[i])
        marker = " ◄ threshold≈0.5" if i == 5 else (" ◄ best Pd" if i == best_idx else "")
        print(f"  iBin={i:2d}  thresh={thr:6.1f}  Pd={pd_i:.4f}  Fa={fa_i:.2e}{marker}")


if __name__ == '__main__':
    main()
