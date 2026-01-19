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
        self.mIoU = mIoU(1)
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
                image_folder=args.image_folder
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
                labels = batch_data['mask'].cuda()  # GT mask (hard labels)
                oracle_masks = batch_data['oracle_mask'].cuda()  # Soft labels
                lidar_points = batch_data['lidar']  # List of tensors (N_i, 4)
            else:
                # Legacy DataLoader (tuple format)
                data, labels, oracle_masks = batch_data
                data = data.cuda()
                labels = labels.cuda()
                oracle_masks = oracle_masks.cuda()
                lidar_points = None

            # Choose training target: soft labels (oracle_masks) or hard labels (labels)
            if self.use_soft_labels:
                train_target = oracle_masks  # Use soft labels (0.6, 1.0, etc.)
            else:
                train_target = labels  # Use hard labels (0.0, 1.0)

            # Forward pass
            if self.args.model == 'MS_CAFNet' or self.args.model == 'MS_CAFNet_DualGeo':
                # MS_CAFNet models (with confidence branch)
                # Note: Models expect 2-channel input [IR, Depth] not separate LiDAR points
                pred, pred_conf = self.model(data)

                # Loss calculation
                loss_seg = SoftIoULoss(pred, train_target)  # Use train_target (soft or hard)
                loss_conf = self.conf_loss(pred_conf, oracle_masks)  # Confidence always uses oracle_masks
                loss = loss_seg + 0.5 * loss_conf

            elif self.args.deep_supervision == 'True':
                # DNANet with deep supervision
                preds = self.model(data)
                loss = 0
                for pred in preds:
                    loss += SoftIoULoss(pred, train_target)
                loss /= len(preds)
            else:
                # Standard models (no deep supervision)
                pred = self.model(data)
                loss = SoftIoULoss(pred, train_target)

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
                    pred, pred_conf = self.model(data)
                    loss = SoftIoULoss(pred, labels)  # Only seg loss for validation
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

        # save high-performance model and update best_iou
        self.best_iou = save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                                    self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())

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


if __name__ == "__main__":
    args = parse_args()
    main(args)
