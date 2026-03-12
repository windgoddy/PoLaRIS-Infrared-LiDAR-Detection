from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from torch.nn import init
from datetime import datetime
import argparse
import shutil
from  matplotlib import pyplot as plt
import glob

class TrainSetLoader(Dataset):


    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png', in_channels=3, image_folder='images', mask_folder='masks'):
        super(TrainSetLoader, self).__init__()

        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir+'/'+mask_folder
        self.images = dataset_dir+'/'+image_folder
        self.depth_maps = dataset_dir+'/'+'depth_maps'
        self.oracle_masks = dataset_dir+'/'+'oracle_masks'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix
        self.in_channels = in_channels

    def _sync_transform(self, img, mask, depth=None, oracle_mask=None):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if depth is not None:
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            if oracle_mask is not None:
                oracle_mask = oracle_mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if depth is not None:
            depth = depth.resize((ow, oh), Image.BILINEAR)
        if oracle_mask is not None:
            oracle_mask = oracle_mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            if oracle_mask is not None:
                oracle_mask = ImageOps.expand(oracle_mask, border=(0, 0, padw, padh), fill=0)
            if depth is not None:
                # ImageOps.expand might not work for float images directly or fill value issues
                # But let's try. Fill 0 is correct for depth.
                # ImageOps.expand converts to RGB if not careful? No, it respects mode.
                # But 'F' mode support in ImageOps is limited.
                # Fallback to paste
                new_depth = Image.new('F', (ow + padw, oh + padh), 0.0)
                new_depth.paste(depth, (0, 0))
                depth = new_depth

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        if depth is not None:
            depth = depth.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        if oracle_mask is not None:
            oracle_mask = oracle_mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            # Don't blur depth? Or maybe yes? Let's skip blurring depth for now.

        # final transform
        # [FIX 2026-03-02] Don't convert to numpy if transform is provided (it expects PIL Image)
        mask = np.array(mask, dtype=np.float32)
        if self.transform is None:
            img = np.array(img)  # Only convert if no transform
        if depth is not None:
            depth = np.array(depth, dtype=np.float32)
        if oracle_mask is not None:
            oracle_mask = np.array(oracle_mask, dtype=np.float32)

        return img, mask, depth, oracle_mask

    def __getitem__(self, idx):

        img_id     = self._items[idx]                        # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix   # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix
        oracle_path = self.oracle_masks + '/' + img_id + self.suffix

        depth = None
        if self.in_channels == 1:
            img = Image.open(img_path).convert('L')
        elif self.in_channels == 2:
            img = Image.open(img_path).convert('L')
            depth_path = self.depth_maps + '/' + img_id + '.npy'
            if os.path.exists(depth_path):
                depth_arr = np.load(depth_path)
            else:
                # Fallback if not generated
                depth_arr = np.zeros((img.size[1], img.size[0]), dtype=np.float32)
            depth = Image.fromarray(depth_arr, mode='F')
        else:
            img = Image.open(img_path).convert('RGB')         ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        
        mask = Image.open(label_path)
        
        # Load Oracle Mask if exists, else zeros
        if os.path.exists(oracle_path):
            oracle_mask = Image.open(oracle_path).convert('L') # Load as grayscale
        else:
            oracle_mask = Image.new('L', mask.size, 0)

        # synchronized transform
        img, mask, depth, oracle_mask = self._sync_transform(img, mask, depth, oracle_mask)

        # general resize, normalize and toTensor
        if self.in_channels == 2:
            # Manual handling for 2-channel
            img = img.astype(np.float32) / 255.0
            depth = depth.astype(np.float32) / 80.0 # Normalize depth (approx max 80m)
            combined = np.stack([img, depth], axis=0) # (2, H, W)
            img = torch.from_numpy(combined).float()
            # Normalize? Let's apply simple normalization
            # IR: (x - 0.485)/0.229, Depth: (x - 0.1)/0.1 (Sparse)
            # Using standard ImageNet for IR
            norm = torch.nn.Sequential(
                # transforms.Normalize([0.485, 0.0], [0.229, 1.0]) # Don't normalize depth further?
                # Let's just subtract mean.
            )
            # For simplicity in Naive Fusion, let's just return the scaled tensors.
            # The BN layers in DNANet will handle distribution shift.
        elif self.transform is not None:
            img = self.transform(img)
            
        mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0
        oracle_mask = np.expand_dims(oracle_mask, axis=0).astype('float32') / 255.0

        return img, torch.from_numpy(mask), torch.from_numpy(oracle_mask) #img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id,transform=None,base_size=512,crop_size=480,suffix='.png', in_channels=3, image_folder='images', mask_folder='masks'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+mask_folder
        self.images    = dataset_dir+'/'+image_folder
        self.depth_maps = dataset_dir+'/'+'depth_maps'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix
        self.in_channels = in_channels

    def _testval_sync_transform(self, img, mask, depth=None):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)
        if depth is not None:
            depth = depth.resize((base_size, base_size), Image.BILINEAR)

        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        if depth is not None:
            depth = np.array(depth, dtype=np.float32)
            return img, mask, depth
        return img, mask, None

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix
        
        depth = None
        if self.in_channels == 1:
            img = Image.open(img_path).convert('L')
        elif self.in_channels == 2:
            img = Image.open(img_path).convert('L')
            depth_path = self.depth_maps + '/' + img_id + '.npy'
            if os.path.exists(depth_path):
                depth_arr = np.load(depth_path)
            else:
                depth_arr = np.zeros((img.size[1], img.size[0]), dtype=np.float32)
            depth = Image.fromarray(depth_arr, mode='F')
        else:
            img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
            
        mask = Image.open(label_path)
        # synchronized transform
        img, mask, depth = self._testval_sync_transform(img, mask, depth)

        # general resize, normalize and toTensor
        if self.in_channels == 2:
            img = img.astype(np.float32) / 255.0
            depth = depth.astype(np.float32) / 80.0
            combined = np.stack([img, depth], axis=0)
            img = torch.from_numpy(combined).float()
        elif self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0


        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)

class DemoLoader (Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(DemoLoader, self).__init__()
        self.transform = transform
        self.images    = dataset_dir
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _demo_sync_transform(self, img):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)

        # final transform
        img = np.array(img)
        return img

    def img_preprocess(self):
        img_path   =  self.images
        img  = Image.open(img_path).convert('RGB')

        # synchronized transform
        img  = self._demo_sync_transform(img)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img



def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))

def save_train_log(args, save_dir):
    dict_args=vars(args)
    args_key=list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('result/%s/train_log.txt'%save_dir ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return

def save_model_and_result(dt_string, epoch,train_loss, test_loss, best_iou, recall, precision, save_mIoU_dir, save_other_metric_dir, box_iou=None):
    # Ensure directory exists before writing
    os.makedirs(os.path.dirname(save_mIoU_dir), exist_ok=True)

    with open(save_mIoU_dir, 'a') as f:
        # 2026-02-03: Include Box IoU in log
        if box_iou is not None:
            f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t mIoU {:.4f}\t Box_IoU {:.4f}\n' .format(dt_string, epoch,train_loss, test_loss, best_iou, box_iou))
        else:
            f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t mIoU {:.4f}\n' .format(dt_string, epoch,train_loss, test_loss, best_iou))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')

def save_model(mean_IOU, best_iou, save_dir, save_prefix, train_loss, test_loss, recall, precision, epoch, net, box_iou=None):
    """
    保存最佳模型

    Args:
        box_iou: 2026-02-03: Mask-to-Box IoU (optional)

    Returns:
        best_iou: 更新后的最佳 mIoU（如果当前 mIoU 更好则返回新值，否则返回原值）
    """
    if mean_IOU > best_iou:
        # Ensure result directory exists
        result_path = 'result/' + save_dir
        os.makedirs(result_path, exist_ok=True)

        save_mIoU_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_IoU.log'
        save_other_metric_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_other_metric.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        best_iou = mean_IOU
        save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou,
                              recall, precision, save_mIoU_dir, save_other_metric_dir, box_iou)

        # 删除旧的最佳模型文件（只保留最新的最佳模型）
        result_path = 'result/' + save_dir
        old_best_models = glob.glob(os.path.join(result_path, 'best_model_epoch*_mIoU*.pth.tar'))
        for old_model in old_best_models:
            try:
                os.remove(old_model)
                print(f'✅ 删除旧模型: {os.path.basename(old_model)}')
            except Exception as e:
                print(f'❌ 删除失败 {old_model}: {e}')

        # 保存最佳模型（包含 epoch、Seg IoU、Box IoU 信息）
        if box_iou is not None:
            best_model_filename = f'best_model_epoch{epoch:04d}_mIoU{mean_IOU:.4f}_BoxIoU{box_iou:.4f}.pth.tar'
        else:
            best_model_filename = f'best_model_epoch{epoch:04d}_mIoU{mean_IOU:.4f}.pth.tar'
        save_dict = {
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'mean_IOU': mean_IOU,
            'recall': recall,
            'precision': precision,
        }
        # 2026-02-03: Add Box IoU to checkpoint
        if box_iou is not None:
            save_dict['box_IOU'] = box_iou

        save_ckpt(save_dict, save_path='result/' + save_dir, filename=best_model_filename)

        # Print with Box IoU if available
        if box_iou is not None:
            print(f'🎉 保存新的最佳模型: {best_model_filename} (Epoch {epoch}, mIoU {mean_IOU:.4f}, Box_IoU {box_iou:.4f})')
        else:
            print(f'🎉 保存新的最佳模型: {best_model_filename} (Epoch {epoch}, mIoU {mean_IOU:.4f})')

        # 同时保存一个 latest_best.pth.tar（方便加载）
        save_ckpt(save_dict, save_path='result/' + save_dir, filename='latest_best_model.pth.tar')

    return best_iou


def save_model_box_iou(mean_box_IOU, best_box_iou, save_dir, save_prefix, train_loss, test_loss, recall, precision, epoch, net, seg_iou=None):
    """
    保存 Mask-to-Box IoU 最佳模型（与 save_model 独立追踪）

    Returns:
        best_box_iou: 更新后的最佳 Box IoU
    """
    if mean_box_IOU > best_box_iou:
        result_path = 'result/' + save_dir
        os.makedirs(result_path, exist_ok=True)

        # 删除旧的 Box IoU 最佳模型
        old_best_models = glob.glob(os.path.join(result_path, 'best_model_box_epoch*_BoxIoU*.pth.tar'))
        for old_model in old_best_models:
            try:
                os.remove(old_model)
                print(f'✅ 删除旧 Box IoU 模型: {os.path.basename(old_model)}')
            except Exception as e:
                print(f'❌ 删除失败 {old_model}: {e}')

        best_box_iou = mean_box_IOU

        # 写 log 文件（与 Seg IoU log 独立，前缀 _best_BoxIoU_）
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        save_box_iou_dir = 'result/' + save_dir + '/' + save_prefix + '_best_BoxIoU_IoU.log'
        save_box_other_dir = 'result/' + save_dir + '/' + save_prefix + '_best_BoxIoU_other_metric.log'
        save_model_and_result(dt_string, epoch, train_loss, test_loss,
                              seg_iou if seg_iou is not None else 0.0,
                              recall, precision,
                              save_box_iou_dir, save_box_other_dir,
                              box_iou=mean_box_IOU)

        if seg_iou is not None:
            best_model_filename = f'best_model_box_epoch{epoch:04d}_BoxIoU{mean_box_IOU:.4f}_mIoU{seg_iou:.4f}.pth.tar'
        else:
            best_model_filename = f'best_model_box_epoch{epoch:04d}_BoxIoU{mean_box_IOU:.4f}.pth.tar'
        save_dict = {
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'box_IOU': mean_box_IOU,
            'recall': recall,
            'precision': precision,
        }
        if seg_iou is not None:
            save_dict['mean_IOU'] = seg_iou

        save_ckpt(save_dict, save_path='result/' + save_dir, filename=best_model_filename)

        if seg_iou is not None:
            print(f'🎉 保存新的最佳 Box IoU 模型: {best_model_filename} (Epoch {epoch}, BoxIoU {mean_box_IOU:.4f}, mIoU {seg_iou:.4f})')
        else:
            print(f'🎉 保存新的最佳 Box IoU 模型: {best_model_filename} (Epoch {epoch}, BoxIoU {mean_box_IOU:.4f})')

    return best_box_iou


def save_last_epoch(save_dir, train_loss, test_loss, mean_IOU, recall, precision, epoch, net, box_iou=None):
    """保存最后一个 epoch 的模型

    Args:
        box_iou: 2026-02-03: Mask-to-Box IoU (optional)
    """
    # 删除旧的最后一个 epoch 模型（如果存在）
    result_path = 'result/' + save_dir
    old_last_models = glob.glob(os.path.join(result_path, 'last_epoch*.pth.tar'))
    for old_model in old_last_models:
        try:
            os.remove(old_model)
        except Exception as e:
            print(f'Failed to delete {old_model}: {e}')

    # 保存最后一个 epoch 的模型
    last_model_filename = f'last_epoch{epoch:04d}.pth.tar'
    save_dict = {
        'epoch': epoch,
        'state_dict': net,
        'loss': test_loss,
        'mean_IOU': mean_IOU,
        'recall': recall,
        'precision': precision,
    }
    # 2026-02-03: Add Box IoU to checkpoint
    if box_iou is not None:
        save_dict['box_IOU'] = box_iou

    save_ckpt(save_dict, save_path='result/' + save_dir, filename=last_model_filename)

    if box_iou is not None:
        print(f'Saved last epoch model: {last_model_filename} (mIoU {mean_IOU:.4f}, Box_IoU {box_iou:.4f})')
    else:
        print(f'Saved last epoch model: {last_model_filename} (mIoU {mean_IOU:.4f})')

def save_result_for_test(dataset_dir, st_model, epochs, best_iou, recall, precision ):
    with open(dataset_dir + '/' + 'value_result'+'/' + st_model +'_best_IoU.log', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epochs, best_iou))

    with open(dataset_dir + '/' +'value_result'+'/'+ st_model + '_best_other_metric.log', 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
    return

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def make_dir(deep_supervision, dataset, model, experiment_name=None):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    # 如果提供了实验名称，将其作为前缀
    if experiment_name:
        prefix = f"{experiment_name}_"
    else:
        prefix = ""

    if deep_supervision:
        save_dir = "%s%s_%s_%s_wDS" % (prefix, dataset, model, dt_string)
    else:
        save_dir = "%s%s_%s_%s_woDS" % (prefix, dataset, model, dt_string)

    os.makedirs('result/%s' % save_dir, exist_ok=True)
    return save_dir

def total_visulization_generation(dataset_dir, mode, test_txt, suffix, target_image_path, target_dir):
    source_image_path = dataset_dir + '/images'

    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)

        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')



def make_visulization_dir(target_image_path, target_dir):
    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_image_path)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_dir)

def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    if predsss.ndim >= 2:
        h, w = predsss.shape[-2], predsss.shape[-1]
    else:
        raise ValueError("save_Pred_GT: invalid pred shape")

    img = Image.fromarray(predsss.reshape(h, w))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(h, w))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)


def save_Pred_GT_visulize(pred, img_demo_dir, img_demo_index, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)

    if predsss.ndim >= 2:
        h, w = predsss.shape[-2], predsss.shape[-1]
    else:
        raise ValueError("save_Pred_GT_visulize: invalid pred shape")

    img = Image.fromarray(predsss.reshape(h, w))
    img.save(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    img = plt.imread(img_demo_dir + '/' + img_demo_index + suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Raw Imamge", size=11)

    plt.subplot(1, 2, 2)
    img = plt.imread(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Predicts", size=11)


    plt.savefig(img_demo_dir + '/' + img_demo_index + "_fuse" + suffix, facecolor='w', edgecolor='red')
    plt.show()



def save_and_visulize_demo(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    if predsss.ndim >= 2:
        h, w = predsss.shape[-2], predsss.shape[-1]
    else:
        raise ValueError("save_and_visulize_demo: invalid pred shape")

    img = Image.fromarray(predsss.reshape(h, w))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(h, w))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)

    return


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
