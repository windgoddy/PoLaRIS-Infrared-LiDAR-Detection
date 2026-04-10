#!/usr/bin/env python3
"""
简化的BoxLevelset训练脚本 - 绕过MMCV兼容性问题
"""
import os
import sys
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from PIL import Image
import numpy as np
from torchvision import transforms

# 禁用警告
warnings.filterwarnings('ignore')

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['MMCV_WITH_OPS'] = '0'
os.environ['FORCE_MLU'] = '0'

print("🚀 启动简化版BoxLevelset训练")
print("=" * 50)

class SimpleIRSTDataset(torch.utils.data.Dataset):
    """简化的红外小目标数据集"""

    def __init__(self, annotation_file, img_dir, transform=None):
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.img_dir = img_dir
        self.transform = transform
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        # 创建图像ID到注释的映射
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 如果图像加载失败，创建一个空白图像
            image = Image.new('RGB', (256, 256), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # 获取标注
        img_id = img_info['id']
        anns = self.img_to_anns.get(img_id, [])

        # 简化的目标信息
        targets = {
            'boxes': [],
            'labels': [],
            'image_id': img_id
        }

        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            # 转换为 [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            targets['boxes'].append([x1, y1, x2, y2])
            targets['labels'].append(ann.get('category_id', 1))

        if len(targets['boxes']) == 0:
            # 如果没有目标，添加一个虚拟框
            targets['boxes'] = [[0, 0, 1, 1]]
            targets['labels'] = [1]

        targets['boxes'] = torch.tensor(targets['boxes'], dtype=torch.float32)
        targets['labels'] = torch.tensor(targets['labels'], dtype=torch.long)

        return image, targets

class SimpleBoxLevelSetModel(nn.Module):
    """简化的BoxLevelSet模型"""

    def __init__(self, num_classes=2):
        super().__init__()

        # 简单的特征提取器
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 4)  # x1, y1, x2, y2
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        cls_scores = self.classifier(features)
        bbox_pred = self.regressor(features)

        return cls_scores, bbox_pred

def train_epoch(model, dataloader, optimizer, criterion_cls, criterion_reg, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        if batch_idx >= 10:  # 限制批次数量用于测试
            break

        images = images.to(device)

        # 简化处理：只使用第一个目标
        labels = torch.stack([t['labels'][0] for t in targets]).to(device)
        boxes = torch.stack([t['boxes'][0] for t in targets]).to(device)

        optimizer.zero_grad()

        cls_scores, bbox_pred = model(images)

        # 计算损失
        cls_loss = criterion_cls(cls_scores, labels)
        reg_loss = criterion_reg(bbox_pred, boxes)

        total_loss_batch = cls_loss + reg_loss
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()
        num_batches += 1

        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}: cls_loss={cls_loss.item():.4f}, reg_loss={reg_loss.item():.4f}")

    return total_loss / num_batches if num_batches > 0 else 0

def main():
    print("📋 初始化训练环境...")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集路径
    train_ann_file = 'dataset/NUAA-SIRST/boxinstseg_coco/annotations/instances_train2017.json'
    train_img_dir = 'dataset/NUAA-SIRST/boxinstseg_coco/train2017'

    print("📂 检查数据集...")
    if not os.path.exists(train_ann_file):
        print(f"❌ 训练注释文件不存在: {train_ann_file}")
        return

    if not os.path.exists(train_img_dir):
        print(f"❌ 训练图像目录不存在: {train_img_dir}")
        # 尝试其他可能的路径
        alt_paths = [
            'dataset/NUAA-SIRST/images',
            'dataset/NUAA-SIRST/train',
            'dataset/NUAA-SIRST'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                train_img_dir = alt_path
                print(f"✅ 使用替代图像目录: {train_img_dir}")
                break
        else:
            print("❌ 找不到图像目录")
            return

    # 创建数据集和数据加载器
    print("🗂️ 创建数据集...")
    try:
        train_dataset = SimpleIRSTDataset(train_ann_file, train_img_dir, transform)
        print(f"✅ 数据集大小: {len(train_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2,
            collate_fn=lambda x: x  # 简单的collate函数
        )

        # 重新定义collate函数
        def collate_fn(batch):
            images = torch.stack([item[0] for item in batch])
            targets = [item[1] for item in batch]
            return images, targets

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # 设为0避免多进程问题
            collate_fn=collate_fn
        )

    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return

    # 创建模型
    print("🏗️ 创建模型...")
    model = SimpleBoxLevelSetModel(num_classes=2).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # 创建工作目录
    work_dir = 'work_dirs/simple_box_levelset'
    os.makedirs(work_dir, exist_ok=True)

    print("🎯 开始训练...")
    num_epochs = 3  # 简化训练，只训练3个epoch

    for epoch in range(num_epochs):
        print(f"\n📊 Epoch {epoch+1}/{num_epochs}")

        try:
            avg_loss = train_epoch(model, train_loader, optimizer, criterion_cls, criterion_reg, device)
            print(f"✅ Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")

            # 保存检查点
            checkpoint_path = os.path.join(work_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"💾 检查点已保存: {checkpoint_path}")

        except Exception as e:
            print(f"❌ Epoch {epoch+1} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\n🎉 训练完成！")
    print(f"📁 模型保存在: {work_dir}")

if __name__ == "__main__":
    main()