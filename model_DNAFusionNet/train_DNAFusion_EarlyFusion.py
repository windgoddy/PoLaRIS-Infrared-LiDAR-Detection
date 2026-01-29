"""
DNA-FusionNet Early Fusion 训练脚本

极简版训练脚本，快速验证 Early Fusion 效果。

使用方法：
    python model_DNAFusionNet/train_DNAFusion_EarlyFusion.py \\
        --dataset Pohang-Canal-3k \\
        --split_method 50_50_2k_new \\
        --gpu 0 \\
        --epochs 50

预期效果：
    - 相比单模态 DNANet，mIoU 预期提升 1-3%
    - 训练速度与 DNANet 相当
    - 50 epochs 应该能看到明显效果
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np

# 导入模型和工具
from model_DNAFusionNet.model_DNAFusion_EarlyFusion import dnafusion_early_resnet18, Res_CBAM_block
from model.utils_lidar import PoLaRISTrainLoader, PoLaRISTestLoader, CombinedSoftLoss, polaris_collate_fn
from model.metric import mIoU, ROCMetric
from model.utils import save_model, save_last_epoch
from model.load_param_data import load_dataset


def set_seed(seed=42):
    """设置随机种子确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 随机种子已设置: {seed}")


def parse_args():
    parser = argparse.ArgumentParser(description='DNA-FusionNet Early Fusion Training')

    # 数据集配置
    parser.add_argument('--dataset', type=str, default='Pohang-Canal-3k',
                        help='数据集名称')
    parser.add_argument('--split_method', type=str, default='50_50_2k_new',
                        help='数据集分割方式')
    parser.add_argument('--data_root', type=str, default='dataset',
                        help='数据集根目录')

    # 训练配置
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数（默认 50，快速验证）')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--base_size', type=int, default=512,
                        help='图像尺寸')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率（默认 1e-4，与 DNANet 一致）')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='推理阈值（默认 0.3）')

    # 模型配置
    parser.add_argument('--deep_supervision', action='store_true', default=True,
                        help='是否使用深度监督')

    # GPU 配置
    parser.add_argument('--gpu', type=str, default='0',
                        help='使用的 GPU 编号')

    # 保存配置
    parser.add_argument('--save_dir', type=str, default=None,
                        help='模型保存目录（自动生成）')

    # 随机种子
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

        # 评估指标
        self.mIoU = mIoU(1, threshold=args.threshold)
        self.ROC = ROCMetric(1, 10)

        # 设置保存目录
        if args.save_dir is None:
            timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
            self.save_dir = f'result/DNAFusion_EarlyFusion_{args.dataset}_{args.split_method}_{timestamp}'
        else:
            self.save_dir = args.save_dir

        os.makedirs(self.save_dir, exist_ok=True)
        self.save_prefix = f'DNAFusion_EarlyFusion_{args.dataset}'

        # 初始化最佳指标
        self.best_iou = 0
        self.last_mean_IOU = 0
        self.last_recall = 0
        self.last_precision = 0

        print(f"\n{'='*80}")
        print(f"初始化训练器")
        print(f"{'='*80}")
        print(f"设备: {self.device}")
        print(f"保存目录: {self.save_dir}")
        print(f"深度监督: {args.deep_supervision}")
        print(f"学习率: {args.lr}")
        print(f"阈值: {args.threshold}")

        # 加载数据集
        self.load_data()

        # 创建模型
        self.create_model()

        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

        # 创建损失函数
        self.criterion = CombinedSoftLoss()

        print(f"✅ 训练器初始化完成")

    def load_data(self):
        """加载数据集"""
        print(f"\n加载数据集...")

        # 加载训练/测试图像 ID
        train_img_ids, val_img_ids, _ = load_dataset(
            self.args.data_root,
            self.args.dataset,
            self.args.split_method
        )

        print(f"  训练集: {len(train_img_ids)} 张")
        print(f"  测试集: {len(val_img_ids)} 张")

        # 创建 DataLoader
        dataset_dir = os.path.join(self.args.data_root, self.args.dataset)

        # 训练集
        train_dataset = PoLaRISTrainLoader(
            dataset_dir=dataset_dir,
            img_id=train_img_ids,
            base_size=self.args.base_size,
            normalize_16bit=True,
            in_channels=2  # 红外 + LiDAR
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=polaris_collate_fn,
            pin_memory=True
        )

        # 测试集
        test_dataset = PoLaRISTestLoader(
            dataset_dir=dataset_dir,
            img_id=val_img_ids,
            base_size=self.args.base_size,
            normalize_16bit=True,
            in_channels=2  # 红外 + LiDAR
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        print(f"✅ 数据集加载完成")

    def create_model(self):
        """创建模型"""
        print(f"\n创建模型...")

        self.model = dnafusion_early_resnet18(
            num_classes=1,
            input_channels=2,  # 红外 + LiDAR
            deep_supervision=self.args.deep_supervision
        )

        self.model = self.model.to(self.device)

        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  模型参数量: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
        print(f"✅ 模型创建完成")

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        train_loss = 0
        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in tbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            oracle_masks = batch['oracle_mask'].to(self.device)

            # 前向传播
            outputs = self.model(images)

            # 计算损失
            if self.args.deep_supervision:
                # 深度监督：所有输出都参与损失计算
                loss = sum([self.criterion(out, masks, oracle_masks) for out in outputs]) / len(outputs)
            else:
                loss = self.criterion(outputs, masks, oracle_masks)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_postfix(loss=f'{loss.item():.4f}')

        return train_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        """验证一个 epoch"""
        self.model.eval()
        self.mIoU.reset()
        self.ROC.reset()
        test_loss = 0

        tbar = tqdm(self.test_loader, desc='Validation')

        with torch.no_grad():
            for batch in tbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                depths = batch['depth'].to(self.device) if 'depth' in batch else None

                # 前向传播
                outputs = self.model(images)

                # 取最终输出
                if self.args.deep_supervision:
                    output = outputs[-1]  # 取最后一个输出
                else:
                    output = outputs

                # 更新指标
                self.mIoU.update(output, masks, depths, use_adaptive_threshold=True)
                self.ROC.update(output, masks)

        # 获取指标
        _, mean_IOU = self.mIoU.get()
        _, _, recall, precision = self.ROC.get()

        self.last_mean_IOU = mean_IOU
        self.last_recall = recall[5]  # threshold=0.5 的 recall
        self.last_precision = precision[5]

        return mean_IOU

    def train(self):
        """完整训练流程"""
        print(f"\n{'='*80}")
        print(f"开始训练")
        print(f"{'='*80}\n")

        for epoch in range(1, self.args.epochs + 1):
            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            mean_IOU = self.validate_epoch(epoch)

            # 更新学习率
            self.scheduler.step()

            # 打印信息
            print(f"Epoch {epoch}/{self.args.epochs} - "
                  f"Loss: {train_loss:.4f}, "
                  f"mIoU: {mean_IOU:.4f}, "
                  f"Recall: {self.last_recall:.4f}, "
                  f"Precision: {self.last_precision:.4f}")

            # 保存模型
            old_best_iou = self.best_iou
            self.best_iou = save_model(
                mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                train_loss, 0, self.last_recall, self.last_precision,
                epoch, self.model
            )

            # 保存最新模型
            if epoch % 10 == 0 or epoch == self.args.epochs:
                save_last_epoch(
                    self.save_dir, train_loss, 0, mean_IOU,
                    self.last_recall, self.last_precision, epoch, self.model
                )

        print(f"\n{'='*80}")
        print(f"训练完成！")
        print(f"{'='*80}")
        print(f"最佳 mIoU: {self.best_iou:.4f}")
        print(f"模型保存路径: {self.save_dir}")


def main():
    args = parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 设置随机种子
    set_seed(args.seed)

    # 创建训练器并训练
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
