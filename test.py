# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio
import re
from collections import defaultdict

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# Model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet

def load_image_categories(dataset_dir):
    """
    Load image categories from selection_summary_new.txt
    Returns: dict mapping image_name (without .png) to category
    """
    selection_file = os.path.join(dataset_dir, 'selection_summary_new.txt')
    image_categories = {}

    if not os.path.exists(selection_file):
        print(f"Warning: {selection_file} not found. Category-specific evaluation disabled.")
        return image_categories

    with open(selection_file, 'r') as f:
        for line in f:
            if '|' in line:
                parts = line.strip().split('|')
                img = parts[0].strip().replace('.png', '')
                cat = int(parts[1].strip())
                if cat in [1, 2, 3]:
                    image_categories[img] = cat

    print(f"✅ Loaded {len(image_categories)} image categories")
    return image_categories

class CategoryMetrics:
    """Track metrics for each category separately"""
    def __init__(self, nclass, threshold=0.3):
        self.nclass = nclass
        self.threshold = threshold
        self.categories = [1, 2, 3]

        # Per-category metrics
        self.cat_mIoU = {cat: mIoU(nclass, threshold) for cat in self.categories}
        self.cat_count = {cat: 0 for cat in self.categories}

        # False alarm tracking (prediction > threshold but GT is 0)
        self.cat_false_alarms = {cat: 0 for cat in self.categories}
        self.cat_total_pixels = {cat: 0 for cat in self.categories}

    def update(self, pred, labels, category):
        """Update metrics for a specific category"""
        if category not in self.categories:
            return

        # Update mIoU
        self.cat_mIoU[category].update(pred, labels)
        self.cat_count[category] += 1

        # Track false alarms: pixels where pred > threshold but GT == 0
        pred_binary = (torch.sigmoid(pred) > self.threshold).float()
        gt_binary = (labels > 0).float()

        false_alarm_pixels = ((pred_binary == 1) & (gt_binary == 0)).sum().item()
        total_pixels = labels.numel()

        self.cat_false_alarms[category] += false_alarm_pixels
        self.cat_total_pixels[category] += total_pixels

    def get_results(self):
        """Get category-specific results"""
        results = {}
        for cat in self.categories:
            if self.cat_count[cat] > 0:
                _, mean_iou = self.cat_mIoU[cat].get()
                fa_rate = self.cat_false_alarms[cat] / (self.cat_total_pixels[cat] + 1e-10)
                results[cat] = {
                    'count': self.cat_count[cat],
                    'mIoU': mean_iou,
                    'false_alarm_rate': fa_rate,
                    'false_alarm_pixels': self.cat_false_alarms[cat],
                    'total_pixels': self.cat_total_pixels[cat]
                }
        return results

    def reset(self):
        for cat in self.categories:
            self.cat_mIoU[cat].reset()
            self.cat_count[cat] = 0
            self.cat_false_alarms[cat] = 0
            self.cat_total_pixels[cat] = 0

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr)
        self.mIoU  = mIoU(1, threshold=getattr(args, 'thres', 0.3))  # Use threshold from args, default 0.3
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        if args.in_channels == 1:
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])
        else:
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix, in_channels=args.in_channels, image_folder=args.image_folder)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load image categories for category-specific evaluation
        self.image_categories = load_image_categories(dataset_dir)
        self.category_metrics = CategoryMetrics(1, threshold=getattr(args, 'thres', 0.3))

        # Load trained model
        checkpoint        = torch.load('result/' + args.model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)

                # Category-specific evaluation
                img_id = val_img_ids[i]
                if img_id in self.image_categories:
                    category = self.image_categories[img_id]
                    self.category_metrics.update(pred, labels, category)

                ture_positive_rate, false_positive_rate, recall, precision= self.ROC.get()
                _, mean_IOU = self.mIoU.get()
            FA, PD = self.PD_FA.get(len(val_img_ids))
            scio.savemat(dataset_dir + '/' +  'value_result'+ '/' +args.st_model  + '_PD_FA_' + str(255),
                         {'number_record1': FA, 'number_record2': PD})

            # Print overall results
            save_result_for_test(dataset_dir, args.st_model,args.epochs, mean_IOU, recall, precision)

            # Print category-specific results
            print("\n" + "="*80)
            print("Category-Specific Evaluation Results")
            print("="*80)
            cat_results = self.category_metrics.get_results()
            for cat in sorted(cat_results.keys()):
                result = cat_results[cat]
                print(f"\nCategory {cat} ({result['count']} images):")
                print(f"  mIoU: {result['mIoU']:.4f}")
                print(f"  False Alarm Rate: {result['false_alarm_rate']:.6f}")
                print(f"  False Alarm Pixels: {result['false_alarm_pixels']:,} / {result['total_pixels']:,}")

            # Save category-specific results
            cat_result_file = os.path.join(dataset_dir, 'value_result', f"{args.st_model}_category_results.txt")
            with open(cat_result_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("Category-Specific Evaluation Results\n")
                f.write("="*80 + "\n\n")
                for cat in sorted(cat_results.keys()):
                    result = cat_results[cat]
                    f.write(f"Category {cat} ({result['count']} images):\n")
                    f.write(f"  mIoU: {result['mIoU']:.4f}\n")
                    f.write(f"  False Alarm Rate: {result['false_alarm_rate']:.6f}\n")
                    f.write(f"  False Alarm Pixels: {result['false_alarm_pixels']:,} / {result['total_pixels']:,}\n\n")
            print(f"\n✅ Category-specific results saved to: {cat_result_file}")


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





