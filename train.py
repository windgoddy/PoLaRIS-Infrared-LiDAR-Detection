# torch and visulization
from tqdm             import tqdm
import torch
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.metric import calculate_mask_to_box_iou  # 2026-02-03: Added Mask-to-Box IoU
from model.loss import *
from model.load_param_data import  load_dataset, load_param
from model_Mamba.core.loss_advanced import LossFactory  # 2026-03-17: projection/hybrid loss support

# model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet
from model.model_DNANet_SNR import DNANet_SNR
from model.model_ACM import ASKCResUNet
from model.model_ALCNet import ASKCResNetFPN

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
        if args.in_channels == 1:
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])
        else:
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
                
        mask_folder = getattr(args, 'mask_folder', 'masks')
        trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix, in_channels=args.in_channels, image_folder=args.image_folder, mask_folder=mask_folder)
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix, in_channels=args.in_channels, image_folder=args.image_folder, mask_folder='masks')
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model == 'DNANet_SNR':
            model       = DNANet_SNR(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model == 'ACM':
            model       = ASKCResUNet(in_channels=args.in_channels, layers=[3,3,3], channels=[8,16,32,64], fuse_mode='AsymBi', tiny=False, classes=1)
        elif args.model == 'ALCNet':
            model       = ASKCResNetFPN(in_channels=args.in_channels, layers=[4,4,4], channels=[8,16,32,64], fuse_mode='AsymBi', act_dilation=16, classes=1)

        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)

        # Loss function (2026-03-17)
        self.loss_type = getattr(args, 'loss_type', 'soft_iou')
        if self.loss_type != 'soft_iou':
            self.criterion = LossFactory.create(
                self.loss_type,
                projection_weight=getattr(args, 'projection_weight', 1.0),
            )
            print(f"[Loss] Using {self.loss_type} loss")

        # Evaluation metrics
        self.best_iou       = 0
        self.best_box_iou   = 0  # Track best Mask-to-Box IoU (independent from best Seg IoU)
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Pd/Fa metrics (P0: 红外领域标准评估指标)
        # bins=10: 阈值从0到255，步长25.5；pred*255后iBin=5对应阈值≈0.5
        self.PD_FA    = PD_FA(1, bins=10, img_size=args.base_size)
        self.test_len = len(val_img_ids)

        # Last epoch metrics
        self.last_epoch = 0
        self.last_mean_IOU = 0
        self.last_box_IOU = 0  # 2026-02-03: Track last Mask-to-Box IoU

    # Training
    def training(self,epoch):

        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, ( data, labels, _) in enumerate(tbar):
            data   = data.cuda()
            labels = labels.cuda()
            if args.deep_supervision == 'True':
                preds= self.model(data)
                loss = 0
                for pred in preds:
                    if self.loss_type == 'soft_iou':
                        loss += SoftIoULoss(pred, labels)
                    else:
                        loss += self.criterion(torch.sigmoid(pred), labels)
                loss /= len(preds)
            else:
               pred = self.model(data)
               if self.loss_type == 'soft_iou':
                   loss = SoftIoULoss(pred, labels)
               else:
                   loss = self.criterion(torch.sigmoid(pred), labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.train_loss = losses.avg
        self.scheduler.step()

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        losses = AverageMeter()
        box_iou_sum = 0.0  # 2026-02-03: Accumulate Mask-to-Box IoU
        box_iou_count = 0

        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        if self.loss_type == 'soft_iou':
                            loss += SoftIoULoss(pred, labels)
                        else:
                            loss += self.criterion(torch.sigmoid(pred), labels)
                    loss /= len(preds)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                    if self.loss_type == 'soft_iou':
                        loss = SoftIoULoss(pred, labels)
                    else:
                        loss = self.criterion(torch.sigmoid(pred), labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC .update(pred, labels)
                self.mIoU.update(pred, labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()

                # 2026-02-03: Calculate Mask-to-Box IoU (使用与 segmentation 一致的阈值)
                batch_box_iou = calculate_mask_to_box_iou(pred, labels, threshold=self.mIoU.threshold)
                box_iou_sum += batch_box_iou
                box_iou_count += 1

                # P0: Pd/Fa 指标（红外领域标准），sigmoid(pred)*255 映射到 0-255 阈值扫描
                self.PD_FA.update(torch.sigmoid(pred) * 255, labels)

                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f, Box_IoU: %.4f' %
                                    (epoch, losses.avg, mean_IOU, batch_box_iou))
            test_loss=losses.avg

        # 2026-02-03: Calculate average Mask-to-Box IoU
        mean_box_IOU = box_iou_sum / max(box_iou_count, 1)

        # P0: 计算 Pd/Fa（bins=10: iBin=3→thresh≈0.3, iBin=5→thresh≈0.5）
        FA, PD = self.PD_FA.get(self.test_len)
        pd_at_t3 = float(PD[3]) if not np.isnan(PD[3]) else 0.0  # threshold≈0.3
        fa_at_t3 = float(FA[3])
        pd_at_t5 = float(PD[5]) if not np.isnan(PD[5]) else 0.0  # threshold≈0.5
        fa_at_t5 = float(FA[5])
        # 固定使用 threshold≈0.5 作为 Pd/Fa 报告口径（iBin=5）
        # 原 best_pd_idx=argmax 策略会选到 iBin=0（threshold=0），导致 Fa≈1 的假象
        best_pd = pd_at_t5
        fa_at_best = fa_at_t5

        # Print test results with all metrics
        print(f"\n[Epoch {epoch}] Test Results:")
        print(f"  Loss              : {test_loss:.4f}")
        print(f"  Segmentation IoU  : {mean_IOU:.4f}")
        print(f"  Mask-to-Box IoU   : {mean_box_IOU:.4f}")
        print(f"  Precision         : {precision[5]:.4f}")
        print(f"  Recall            : {recall[5]:.4f}")
        print(f"  Pd (thresh≈0.3)  : {pd_at_t3:.4f}   Fa: {fa_at_t3:.2e}")
        print(f"  Pd (thresh≈0.5)  : {pd_at_t5:.4f}   Fa: {fa_at_t5:.2e}")
        print(f"  Best Pd           : {best_pd:.4f}   Fa: {fa_at_best:.2e}")

        # Store last epoch metrics
        self.last_epoch = epoch
        self.last_mean_IOU = mean_IOU
        self.last_box_IOU = mean_box_IOU  # 2026-02-03: Store Box IoU
        # 保存 Seg IoU 最佳模型
        self.best_iou = save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                                    self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict(),
                                    mean_box_IOU, best_pd=best_pd, fa_at_best=fa_at_best)

        # 保存 Mask-to-Box IoU 最佳模型（独立追踪）
        self.best_box_iou = save_model_box_iou(mean_box_IOU, self.best_box_iou, self.save_dir, self.save_prefix,
                                                self.train_loss, test_loss, recall, precision, epoch,
                                                self.model.state_dict(), seg_iou=mean_IOU,
                                                best_pd=best_pd, fa_at_best=fa_at_best)

def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)

    # 2026-02-03: Print final summary with Box IoU
    print(f"\n{'=' * 60}")
    print(f"Training Summary")
    print(f"{'=' * 60}")
    print(f"Best Segmentation IoU : {trainer.best_iou:.4f}")
    print(f"Best Mask-to-Box IoU  : {trainer.best_box_iou:.4f}")
    print(f"Final Segmentation IoU: {trainer.last_mean_IOU:.4f}")
    print(f"Final Mask-to-Box IoU : {trainer.last_box_IOU:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)





