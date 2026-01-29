# torch and visulization
from tqdm             import tqdm
import torch
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args
import random
import numpy as np
import os
import csv
from collections import defaultdict

# metric, loss .etc
from model.utils import *
from model.utils_lidar import PoLaRISTrainLoader, PoLaRISTestLoader, CombinedSoftLoss, polaris_collate_fn
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet
from model.model_Phase3 import MS_CAFNet, MS_CAFNet_DualGeo

def set_seed(seed=42):
    """
    设置随机种子以确保实验可复现

    Args:
        seed: 随机种子值，默认 42

    说明：
        - random: Python 内置随机数生成器
        - numpy: NumPy 随机数生成器
        - torch: PyTorch CPU 随机数生成器
        - torch.cuda: PyTorch GPU 随机数生成器
        - cudnn.deterministic: 使用确定性算法（可能稍慢，但可复现）
        - cudnn.benchmark: 关闭自动寻找最优算法（确保每次运行一致）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 随机种子已设置: {seed} (确保实验可复现)")

class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1, threshold=getattr(args, 'thres', 0.3))  # Use threshold from args, default 0.3
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        # Check if using LiDAR DataLoader (supports 16-bit, LiDAR, soft labels)
        use_lidar_loader = args.use_lidar_dataloader == 'True'

        if use_lidar_loader:
            # Use new PoLaRIS LiDAR DataLoader (no transform needed - handled internally)
            print(f"✅ Using PoLaRIS LiDAR DataLoader (16-bit: {args.normalize_16bit}, Soft Labels: {args.use_soft_labels})")
            trainset = PoLaRISTrainLoader(
                dataset_dir=dataset_dir,
                img_id=train_img_ids,
                base_size=args.base_size,
                crop_size=args.crop_size,
                transform=None,  # DataLoader handles normalization internally
                suffix=args.suffix,
                normalize_16bit=(args.normalize_16bit == 'True'),
                in_channels=args.in_channels,  # Pass in_channels for depth map support
                image_folder=args.image_folder,
                oracle_masks_folder=args.oracle_masks_folder
            )
            testset = PoLaRISTestLoader(
                dataset_dir=dataset_dir,
                img_id=val_img_ids,
                base_size=args.base_size,
                crop_size=args.crop_size,
                transform=None,
                suffix=args.suffix,
                normalize_16bit=(args.normalize_16bit == 'True'),
                in_channels=args.in_channels,
                image_folder=args.image_folder
            )
        else:
            # Use legacy DataLoader (8-bit only)
            print("⚠️  Using legacy DataLoader (8-bit images only, no LiDAR support)")
            if args.in_channels == 1:
                input_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])])
            else:
                input_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

            trainset = TrainSetLoader(dataset_dir, img_id=train_img_ids, base_size=args.base_size,
                                     crop_size=args.crop_size, transform=input_transform,
                                     suffix=args.suffix, in_channels=args.in_channels)
            testset = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=args.base_size,
                                   crop_size=args.crop_size, transform=input_transform,
                                   suffix=args.suffix, in_channels=args.in_channels)

        # Use custom collate function for PoLaRIS loader to handle variable-length LiDAR point clouds
        if use_lidar_loader:
            self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True, collate_fn=polaris_collate_fn)
            self.test_data = DataLoader(dataset=testset, batch_size=args.test_batch_size,
                                       num_workers=args.workers, drop_last=False, collate_fn=polaris_collate_fn)
        else:
            self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True,
                                        num_workers=args.workers, drop_last=True)
            self.test_data = DataLoader(dataset=testset, batch_size=args.test_batch_size,
                                       num_workers=args.workers, drop_last=False)

        # Store flag for later use in training/testing loops
        self.use_lidar_loader = use_lidar_loader
        self.use_soft_labels = (args.use_soft_labels == 'True')

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model == 'MS_CAFNet':
            model       = MS_CAFNet(num_classes=1, input_channels=args.in_channels)
        elif args.model == 'MS_CAFNet_DualGeo':
            model       = MS_CAFNet_DualGeo(num_classes=1, input_channels=args.in_channels)

        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                                                 lr=args.lr, weight_decay=args.weight_decay)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        # self.scheduler.step()

        # Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Last epoch metrics (for saving final model)
        self.last_epoch = 0
        self.last_test_loss = 0
        self.last_mean_IOU = 0
        self.last_recall = [0,0,0,0,0,0,0,0,0,0,0]
        self.last_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Loss
        self.conf_loss = ConfidenceLoss()

        # [NEW] 加载类别映射用于分类别统计
        self.dataset_dir = dataset_dir
        self.val_img_ids = val_img_ids
        self.category_map = self._load_category_mapping(dataset_dir)
        self.category_stats_file = os.path.join('result', self.save_dir, self.save_prefix + '_category_stats.csv')
        self._init_category_stats_file()

    # Training
    def training(self,epoch):

        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()

        for i, batch_data in enumerate(tbar):
            # Parse batch data based on DataLoader type
            if self.use_lidar_loader:
                # New PoLaRIS LiDAR DataLoader (dict format)
                data = batch_data['image'].cuda()
                mask_hard = batch_data['mask_hard'].cuda()  # Hard Label (Binary GT)
                mask_soft = batch_data['mask_soft'].cuda()  # Soft Label (Oracle Mask)
                lidar_points = batch_data['lidar']  # List of tensors (N_i, 4)
            else:
                # Legacy DataLoader (tuple format)
                data, labels, oracle_masks = batch_data
                data = data.cuda()
                mask_hard = labels.cuda()
                mask_soft = oracle_masks.cuda()
                lidar_points = None

            # === Dual-Supervision Strategy ===
            # Instead of normalizing soft labels (which destroyed uncertainty modeling),
            # we use DIFFERENT supervision signals for DIFFERENT model components:
            #
            # 1. Segmentation Branch -> Hard Label (Binary GT: 0 or 1)
            #    - Forces model to output clear boundaries and high confidence (1.0)
            #    - Improves Recall and IoU (solves the 4% mIoU drop issue)
            #
            # 2. Confidence/Geometry Branch -> Soft Label (Oracle Mask: 0~0.6~1.0)
            #    - Preserves LiDAR geometric prior (center=1.0, edge=0.6, no-LiDAR=0.6)
            #    - Guides attention modules (DualGeo) to focus on correct regions
            #    - Preserves UNCERTAINTY modeling (the core value of soft labels!)
            #
            # This is the KEY to solving the Recall/Precision trade-off!

            # Forward pass
            if self.args.model == 'MS_CAFNet' or self.args.model == 'MS_CAFNet_DualGeo':
                # MS_CAFNet models (with confidence branch)
                # Note: Models expect 2-channel input [IR, Depth] not separate LiDAR points
                outputs, pred_conf = self.model(data)

                # Handle both single output and list of outputs (deep supervision)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                # === Dual-Supervision Loss Calculation ===
                # Loss 1: Segmentation Loss -> Hard Label (Binary GT)
                loss_seg = 0
                for pred in outputs:
                    loss_seg += SoftIoULoss(pred, mask_hard)  # Use HARD label!
                loss_seg /= len(outputs)

                # Loss 2: Confidence Loss -> Soft Label (Preserve uncertainty!)
                loss_conf = self.conf_loss(pred_conf, mask_soft)  # Use SOFT label!

                # Total Loss (weighted combination)
                loss = loss_seg + 0.5 * loss_conf

                # Use final output for batch size
                pred = outputs[-1]

            elif self.args.deep_supervision == 'True':
                # DNANet with deep supervision (uses hard labels only)
                preds = self.model(data)
                loss = 0
                for pred in preds:
                    loss += SoftIoULoss(pred, mask_hard)
                loss /= len(preds)
            else:
                # Standard models (no deep supervision, uses hard labels)
                pred = self.model(data)
                loss = SoftIoULoss(pred, mask_hard)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            if self.args.model == 'MS_CAFNet' or self.args.model == 'MS_CAFNet_DualGeo':
                losses.update(loss.item(), pred.size(0))
            elif self.args.deep_supervision == 'True':
                losses.update(loss.item(), preds[-1].size(0))
            else:
                losses.update(loss.item(), pred.size(0))

            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.train_loss = losses.avg

    def _load_category_mapping(self, dataset_dir):
        """加载类别映射 从 selection_summary_new.txt"""
        summary_file = os.path.join(dataset_dir, 'selection_summary_new.txt')

        if not os.path.exists(summary_file):
            print(f"⚠️  警告: 找不到类别映射文件 {summary_file}")
            print("   将无法按类别统计")
            return {}

        category_map = {}
        with open(summary_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue

                parts = line.split('|')
                if len(parts) == 2:
                    sample_id = parts[0].strip().replace('.png', '')
                    category = int(parts[1].strip())
                    category_map[sample_id] = category

        print(f"✅ 加载类别映射: {len(category_map)} 个样本")
        return category_map

    def _init_category_stats_file(self):
        """初始化类别统计CSV文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.category_stats_file), exist_ok=True)

        # 写入CSV表头
        with open(self.category_stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Epoch', 'SaveType',
                'Overall_mIoU', 'Overall_Precision', 'Overall_Recall', 'Overall_F1',
                'Cat1_Count', 'Cat1_mIoU', 'Cat1_Precision', 'Cat1_Recall', 'Cat1_F1', 'Cat1_FPR',
                'Cat2_Count', 'Cat2_mIoU', 'Cat2_Precision', 'Cat2_Recall', 'Cat2_F1', 'Cat2_FPR',
                'Cat3_Count', 'Cat3_mIoU', 'Cat3_Precision', 'Cat3_Recall', 'Cat3_F1', 'Cat3_FPR'
            ])
        print(f"✅ 初始化类别统计文件: {self.category_stats_file}")

    def _calculate_category_metrics(self):
        """计算按类别的统计指标"""
        # 初始化统计
        category_stats = defaultdict(lambda: {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'iou_sum': 0.0, 'count': 0
        })
        overall_stats = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'iou_sum': 0.0, 'count': 0}

        # 评估每个样本
        self.model.eval()
        threshold = self.args.thres

        with torch.no_grad():
            for batch_data in self.test_data:
                # 解析数据
                if self.use_lidar_loader:
                    data = batch_data['image'].cuda()
                    labels = batch_data['mask'].cuda()
                    sample_ids = batch_data['id']
                else:
                    data, labels = batch_data
                    data = data.cuda()
                    labels = labels.cuda()
                    sample_ids = [self.val_img_ids[i] for i in range(len(data))]

                # 前向传播
                if self.args.model == 'MS_CAFNet' or self.args.model == 'MS_CAFNet_DualGeo':
                    outputs, pred_conf = self.model(data)
                    pred = outputs[-1] if isinstance(outputs, list) else outputs
                elif self.args.deep_supervision == 'True':
                    preds = self.model(data)
                    pred = preds[-1]
                else:
                    pred = self.model(data)

                # 计算每个样本的指标
                pred_sigmoid = torch.sigmoid(pred).cpu().numpy()
                labels_np = labels.cpu().numpy()

                for i, sample_id in enumerate(sample_ids):
                    pred_np = pred_sigmoid[i, 0]  # [H, W]
                    target_np = labels_np[i, 0]  # [H, W]

                    # 二值化
                    pred_binary = (pred_np >= threshold).astype(np.float32)
                    target_binary = target_np.astype(np.float32)

                    # 计算混淆矩阵
                    tp = np.sum((pred_binary == 1) & (target_binary == 1))
                    fp = np.sum((pred_binary == 1) & (target_binary == 0))
                    tn = np.sum((pred_binary == 0) & (target_binary == 0))
                    fn = np.sum((pred_binary == 0) & (target_binary == 1))

                    # IoU
                    iou = tp / (tp + fp + fn + 1e-10)

                    # 获取类别
                    category = self.category_map.get(sample_id, None)

                    # 更新类别统计
                    if category is not None:
                        stats = category_stats[category]
                        stats['tp'] += tp
                        stats['fp'] += fp
                        stats['tn'] += tn
                        stats['fn'] += fn
                        stats['iou_sum'] += iou
                        stats['count'] += 1

                    # 更新整体统计
                    overall_stats['tp'] += tp
                    overall_stats['fp'] += fp
                    overall_stats['tn'] += tn
                    overall_stats['fn'] += fn
                    overall_stats['iou_sum'] += iou
                    overall_stats['count'] += 1

        # 计算最终指标
        results = {'overall': self._compute_final_metrics(overall_stats)}
        for cat in [1, 2, 3]:
            if cat in category_stats:
                results[f'category_{cat}'] = self._compute_final_metrics(category_stats[cat])
            else:
                results[f'category_{cat}'] = None

        return results

    def _compute_final_metrics(self, stats):
        """计算最终指标"""
        if stats['count'] == 0:
            return None

        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        tn = stats['tn']

        iou = tp / (tp + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        mean_iou = stats['iou_sum'] / stats['count']
        fpr = fp / (fp + tn + 1e-10)

        return {
            'count': stats['count'],
            'iou': iou,
            'mean_iou': mean_iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fp_rate': fpr
        }

    def _save_category_stats(self, epoch, save_type, results):
        """保存类别统计到CSV"""
        with open(self.category_stats_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # 整体指标
            overall = results['overall']

            # 各类别指标
            row = [
                epoch,
                save_type,
                f"{overall['mean_iou']:.4f}",
                f"{overall['precision']:.4f}",
                f"{overall['recall']:.4f}",
                f"{overall['f1']:.4f}"
            ]

            for cat in [1, 2, 3]:
                key = f'category_{cat}'
                if results[key] is not None:
                    cat_metrics = results[key]
                    row.extend([
                        cat_metrics['count'],
                        f"{cat_metrics['mean_iou']:.4f}",
                        f"{cat_metrics['precision']:.4f}",
                        f"{cat_metrics['recall']:.4f}",
                        f"{cat_metrics['f1']:.4f}",
                        f"{cat_metrics['fp_rate']:.4f}"
                    ])
                else:
                    row.extend([0, '0.0000', '0.0000', '0.0000', '0.0000', '0.0000'])

            writer.writerow(row)

        # 同时打印到控制台
        print("\n" + "="*80)
        print(f"📊 按类别统计 (Epoch {epoch}, {save_type})")
        print("="*80)
        print(f"整体: mIoU={overall['mean_iou']:.4f}, Prec={overall['precision']:.4f}, Rec={overall['recall']:.4f}, F1={overall['f1']:.4f}")

        cat_names = {1: 'Cat1 (简单海面)', 2: 'Cat2 (一般海岸)', 3: 'Cat3 (复杂海岸)'}
        for cat in [1, 2, 3]:
            key = f'category_{cat}'
            if results[key] is not None:
                m = results[key]
                print(f"{cat_names[cat]}: mIoU={m['mean_iou']:.4f}, Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1']:.4f}, FPR={m['fp_rate']:.4f} (n={m['count']})")
        print("="*80 + "\n")

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, batch_data in enumerate(tbar):
                # Parse batch data based on DataLoader type
                if self.use_lidar_loader:
                    # New PoLaRIS LiDAR DataLoader (dict format)
                    data = batch_data['image'].cuda()
                    labels = batch_data['mask'].cuda()  # Always use GT mask for evaluation
                    lidar_points = batch_data['lidar']  # List of tensors
                else:
                    # Legacy DataLoader (tuple format)
                    data, labels = batch_data
                    data = data.cuda()
                    labels = labels.cuda()
                    lidar_points = None

                # Forward pass
                if self.args.model == 'MS_CAFNet' or self.args.model == 'MS_CAFNet_DualGeo':
                    # MS_CAFNet models (expect 2-channel input [IR, Depth])
                    outputs, pred_conf = self.model(data)

                    # Handle both single output and list of outputs (deep supervision)
                    if not isinstance(outputs, list):
                        outputs = [outputs]

                    # Loss calculation with deep supervision
                    loss = 0
                    for pred in outputs:
                        loss += SoftIoULoss(pred, labels)  # Only seg loss for validation
                    loss /= len(outputs)

                    # Use final output for evaluation
                    pred = outputs[-1]
                elif self.args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)

                # Update metrics (always evaluate against GT labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC.update(pred, labels)
                self.mIoU.update(pred, labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU))
            test_loss = losses.avg

        # Store last epoch metrics
        self.last_epoch = epoch
        self.last_test_loss = test_loss
        self.last_mean_IOU = mean_IOU
        self.last_recall = recall
        self.last_precision = precision

        # [NEW] 计算按类别统计（在保存模型前）
        category_results = None
        if len(self.category_map) > 0:
            print(f"\n⏳ 计算按类别统计 (Epoch {epoch})...")
            category_results = self._calculate_category_metrics()

        # save high-performance model and update best_iou
        old_best_iou = self.best_iou
        self.best_iou = save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                                    self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())

        # [NEW] 如果保存了最佳模型，记录类别统计
        if category_results is not None and mean_IOU > old_best_iou:
            self._save_category_stats(epoch, 'BestModel', category_results)

def main(args):
    # 设置随机种子以确保实验可复现
    set_seed(args.seed)

    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)
        trainer.scheduler.step()

    # Save the last epoch model
    print("\n" + "="*60)
    print("Training completed! Saving last epoch model...")
    print("="*60)
    save_last_epoch(trainer.save_dir, trainer.train_loss, trainer.last_test_loss,
                    trainer.last_mean_IOU, trainer.last_recall, trainer.last_precision,
                    trainer.last_epoch, trainer.model.state_dict())

    # [NEW] 计算并保存最后epoch的类别统计
    if len(trainer.category_map) > 0:
        print(f"\n⏳ 计算最后epoch的按类别统计...")
        final_category_results = trainer._calculate_category_metrics()
        trainer._save_category_stats(trainer.last_epoch, 'LastEpoch', final_category_results)
        print(f"✅ 类别统计已保存到: {trainer.category_stats_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
