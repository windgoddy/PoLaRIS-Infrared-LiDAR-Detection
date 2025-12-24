import argparse
import os
from glob import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_DNANet import DNANet, Res_CBAM_block
from model.model_Phase3 import MS_CAFNet
from model.metric import mIoU, ROCMetric
from model.load_param_data import load_param

class GoldenSetLoader(torch.utils.data.Dataset):
    def __init__(self, root_dir, golden_dir, img_ids, transform=None, in_channels=3):
        self.root_dir = root_dir
        self.golden_dir = golden_dir
        self.img_ids = img_ids
        self.transform = transform
        self.in_channels = in_channels

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        # Image and Mask from Golden Set
        img_path = os.path.join(self.golden_dir, 'images', img_id + '.png')
        mask_path = os.path.join(self.golden_dir, 'masks', img_id + '.png')
        
        # Depth from Main Dataset (if needed)
        depth_path = os.path.join(self.root_dir, 'depth_maps', img_id + '.npy')

        mask = Image.open(mask_path)
        
        depth = None
        if self.in_channels == 1:
            img = Image.open(img_path).convert('L')
        elif self.in_channels == 2:
            img = Image.open(img_path).convert('L')
            if os.path.exists(depth_path):
                depth_arr = np.load(depth_path)
            else:
                depth_arr = np.zeros((img.size[1], img.size[0]), dtype=np.float32)
            depth = Image.fromarray(depth_arr, mode='F')
        else:
            img = Image.open(img_path).convert('RGB')

        # Resize to 256x256 (Standard for this project)
        base_size = 256
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)
        if depth is not None:
            depth = depth.resize((base_size, base_size), Image.BILINEAR)

        # Transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        if depth is not None:
            depth = np.array(depth, dtype=np.float32)
        
        # Normalize
        if self.in_channels == 2:
            img = img.astype(np.float32) / 255.0
            depth = depth.astype(np.float32) / 80.0
            combined = np.stack([img, depth], axis=0)
            img = torch.from_numpy(combined).float()
        elif self.transform is not None:
            img = self.transform(img)
            
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0
        
        return img, torch.from_numpy(mask)

    def __len__(self):
        return len(self.img_ids)

def evaluate(args):
    # 1. Prepare Data
    golden_dir = 'dataset/Pohang-Canal/golden_set'
    root_dir = 'dataset/Pohang-Canal'
    
    # Get IDs
    # Only use images that have corresponding masks
    mask_files = glob(os.path.join(golden_dir, 'masks', '*.png'))
    mask_ids = set([os.path.basename(f).replace('.png', '') for f in mask_files])
    
    img_files = glob(os.path.join(golden_dir, 'images', '*.png'))
    all_img_ids = [os.path.basename(f).replace('.png', '') for f in img_files]
    
    # Filter IDs
    img_ids = [img_id for img_id in all_img_ids if img_id in mask_ids]
    
    print(f"Found {len(all_img_ids)} images, but only {len(img_ids)} have masks in Golden Set.")

    # Transform
    if args.in_channels == 1:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
    else:
        input_transform = None # Handled in loader for 2-channel

    dataset = GoldenSetLoader(root_dir, golden_dir, img_ids, transform=input_transform, in_channels=args.in_channels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # 2. Load Model
    print(f"Loading model: {args.model_name} from {args.checkpoint}")
    if args.model_name == 'DNANet':
        # Baseline 1 & 2 used resnet_34, not resnet_18
        nb_filter, num_blocks = load_param('three', 'resnet_34')
        # Note: deep_supervision=True matches training config, output will be list
        model = DNANet(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=True)
    elif args.model_name == 'MS_CAFNet':
        model = MS_CAFNet(num_classes=1, input_channels=args.in_channels)
    
    # Load weights
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    # 3. Metrics
    metric_iou = mIoU(1)
    metric_roc = ROCMetric(1, 10)
    
    # 4. Inference
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data = data.cuda()
            labels = labels.cuda()
            
            if args.model_name == 'MS_CAFNet':
                preds, _ = model(data) # Ignore confidence map
            else:
                preds = model(data)
                if isinstance(preds, list) or isinstance(preds, tuple):
                    preds = preds[-1] # Deep supervision output
            
            metric_iou.update(preds, labels)
            metric_roc.update(preds, labels)

    # 5. Report
    _, mean_iou = metric_iou.get()
    _, _, recall, precision = metric_roc.get()
    
    # Get metrics at threshold 0.5 (bin 5 if bins=10)
    # ROCMetric bins are 0.0, 0.1, ... 1.0
    # Bin 5 corresponds to threshold 0.5
    idx = 5 
    
    print("="*40)
    print(f"Model:      {args.model_name}")
    print(f"Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"Golden Set: {len(img_ids)} images")
    print("-" * 40)
    print(f"mIoU:       {mean_iou:.4f}")
    print(f"Recall:     {recall[idx]:.4f} (Threshold=0.5)")
    print(f"Precision:  {precision[idx]:.4f} (Threshold=0.5)")
    print("="*40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=['DNANet', 'MS_CAFNet'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--in_channels', type=int, default=1)
    args = parser.parse_args()
    
    evaluate(args)
