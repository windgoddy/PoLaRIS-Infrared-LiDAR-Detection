#!/usr/bin/env python3
"""
run_inference.py
================
对测试集跑推理，将每张图的 sigmoid(pred) 保存为灰度 PNG，
供 fig3_qualitative_hardcases.py 的 --mode plot 使用。

用法示例：
    # Box Fill B组，NUAA
    python paper/run_inference.py \
        --model DNANet --dataset NUAA-SIRST \
        --model_dir result/P1_B_box_nuaa_NUAA-SIRST_DNANet_13_03_2026_21_11_33_wDS/best_model_epoch0003_mIoU0.3013_BoxIoU0.2957.pth.tar \
        --out_dir paper/figures/preds/boxfill_nuaa

    # HALO D组，NUAA
    python paper/run_inference.py \
        --model DNANet --dataset NUAA-SIRST \
        --model_dir result/P1_D_pag_nuaa_NUAA-SIRST_DNANet_13_03_2026_21_12_40_wDS/best_model_epoch0346_mIoU0.7069_BoxIoU0.6681.pth.tar \
        --out_dir paper/figures/preds/halo_nuaa

支持模型：DNANet（P1）, ACM, ALCNet（P2）
"""

import os, sys, argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.utils import *
from model.load_param_data import load_dataset, load_param
from model.model_DNANet import Res_CBAM_block, DNANet
from model.model_ACM import ASKCResUNet
from model.model_ALCNet import ASKCResNetFPN


IMAGE_FOLDERS = {
    'NUAA-SIRST': 'images',
    'NUDT-SIRST': 'images',
    'IRSTD-1k':   'IRSTD1k_Img',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   default='DNANet', choices=['DNANet', 'ACM', 'ALCNet'])
    parser.add_argument('--dataset', default='NUAA-SIRST',
                        choices=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k'])
    parser.add_argument('--root',    default='dataset/')
    parser.add_argument('--split_method', default='50_50')
    parser.add_argument('--in_channels',  type=int, default=3)
    parser.add_argument('--base_size',    type=int, default=256)
    parser.add_argument('--crop_size',    type=int, default=256)
    parser.add_argument('--workers',      type=int, default=4)
    parser.add_argument('--channel_size', default='three',
                        help='DNANet only: one/two/three/four')
    parser.add_argument('--backbone',     default='resnet_18',
                        help='DNANet only')
    parser.add_argument('--deep_supervision', default='True',
                        help='DNANet only: True/False')
    parser.add_argument('--model_dir', required=True,
                        help='checkpoint 路径（含 result/ 前缀的完整相对路径）')
    parser.add_argument('--out_dir',   required=True,
                        help='预测图输出目录（每张图保存为 {sample_name}.png）')
    return parser.parse_args()


def build_model(args):
    if args.model == 'DNANet':
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        return DNANet(num_classes=1, input_channels=args.in_channels,
                      block=Res_CBAM_block, num_blocks=num_blocks,
                      nb_filter=nb_filter, deep_supervision=args.deep_supervision)
    elif args.model == 'ACM':
        return ASKCResUNet(in_channels=args.in_channels, layers=[3, 3, 3],
                           channels=[8, 16, 32, 64], fuse_mode='AsymBi',
                           tiny=False, classes=1)
    elif args.model == 'ALCNet':
        return ASKCResNetFPN(in_channels=args.in_channels, layers=[4, 4, 4],
                             channels=[8, 16, 32, 64], fuse_mode='AsymBi',
                             act_dilation=16, classes=1)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 数据集 ──
    image_folder = IMAGE_FOLDERS.get(args.dataset, 'images')
    dataset_dir  = os.path.join(args.root, args.dataset)
    _, val_img_ids, _ = load_dataset(args.root, args.dataset, args.split_method)

    if args.in_channels == 1:
        tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([0.5], [0.5])])
    else:
        tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([.485, .456, .406],
                                                       [.229, .224, .225])])

    testset = TestSetLoader(
        dataset_dir, img_id=val_img_ids,
        base_size=args.base_size, crop_size=args.crop_size,
        transform=tf, suffix='.png',
        in_channels=args.in_channels, image_folder=image_folder)
    loader = DataLoader(testset, batch_size=1, num_workers=args.workers,
                        drop_last=False)

    # ── 模型 ──
    model = build_model(args).cuda()
    model.apply(weights_init_xavier)

    ckpt = torch.load(args.model_dir, map_location='cuda')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Loaded: {args.model_dir}")
    print(f"Saving {len(val_img_ids)} predictions → {args.out_dir}")

    # ── 推理 ──
    with torch.no_grad():
        for idx, (data, _) in enumerate(tqdm(loader)):
            data = data.cuda()

            if args.model == 'DNANet' and args.deep_supervision == 'True':
                preds = model(data)
                pred  = preds[-1]
            else:
                pred = model(data)

            # sigmoid → [0,1] → [0,255] uint8
            prob = torch.sigmoid(pred).squeeze().cpu().numpy()  # (H, W)
            img_u8 = (prob * 255).clip(0, 255).astype(np.uint8)

            name = val_img_ids[idx]
            save_path = os.path.join(args.out_dir, name + '.png')
            cv2.imwrite(save_path, img_u8)

    print("Done.")


if __name__ == '__main__':
    main()
