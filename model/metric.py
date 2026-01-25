import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy

def adaptive_threshold_binarization(output, depth_map=None, threshold_with_lidar=0.5, threshold_without_lidar=0.35):
    """
    动态自适应阈值二值化
    
    Args:
        output: (B, 1, H, W) 模型输出概率图
        depth_map: (B, 1, H, W) 深度图，None表示无LiDAR数据
        threshold_with_lidar: float 有LiDAR区域的阈值（默认0.5）
        threshold_without_lidar: float 无LiDAR区域的阈值（默认0.35）
    
    Returns:
        binary_mask: (B, 1, H, W) 二值化结果
    
    逻辑：
        - 有LiDAR覆盖的像素（depth > 0）→ 使用threshold_with_lidar=0.5
        - 无LiDAR覆盖的像素（depth == 0）→ 使用threshold_without_lidar=0.35
    """
    # 确保output经过sigmoid
    if output.min() < 0 or output.max() > 1:
        prob = torch.sigmoid(output)
    else:
        prob = output
    
    # 如果没有depth map，使用统一阈值（保守：使用较高阈值）
    if depth_map is None:
        return (prob > threshold_with_lidar).float()
    
    # 创建LiDAR掩码：depth > 0 的区域有LiDAR
    if isinstance(depth_map, np.ndarray):
        depth_map = torch.from_numpy(depth_map).to(output.device)
    
    if depth_map.dim() == 3:  # (B, H, W)
        depth_map = depth_map.unsqueeze(1)  # (B, 1, H, W)
    
    lidar_mask = (depth_map > 0).float()  # 1=有LiDAR, 0=无LiDAR
    
    # 动态阈值：有LiDAR用高阈值，无LiDAR用低阈值
    binary_with_lidar = (prob > threshold_with_lidar).float()
    binary_without_lidar = (prob > threshold_without_lidar).float()
    
    # 组合结果
    binary_mask = lidar_mask * binary_with_lidar + (1 - lidar_mask) * binary_without_lidar
    
    return binary_mask

class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])



class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            predits  = np.reshape (predits,  (256,256))
            labelss = np.array((labels).cpu()).astype('int64') # P
            labelss = np.reshape (labelss , (256,256))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.FA / ((256 * 256) * img_num)
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

class mIoU():

    def __init__(self, nclass, threshold=0.3):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.threshold = threshold  # Inference threshold (default 0.3 for Soft Labels)
        self.reset()

    def update(self, preds, labels, depth_map=None, use_adaptive_threshold=True):
        """
        Args:
            preds: 模型预测
            labels: Ground Truth
            depth_map: 深度图（可选，用于动态阈值）
            use_adaptive_threshold: 是否使用动态自适应阈值
        """
        correct, labeled = batch_pix_accuracy(preds, labels, depth_map, use_adaptive_threshold, self.threshold)
        inter, union = batch_intersection_union(preds, labels, self.nclass, depth_map, use_adaptive_threshold, self.threshold)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target, depth_map=None, use_adaptive_threshold=True, threshold=0.3):
    """
    Args:
        output: 模型输出
        target: Ground Truth
        depth_map: 深度图（可选）
        use_adaptive_threshold: 是否使用动态自适应阈值
        threshold: 推理阈值（默认0.3，适配Soft Label max=0.6）
    """
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"

    # 使用动态自适应阈值
    if use_adaptive_threshold and depth_map is not None:
        predict = adaptive_threshold_binarization(output, depth_map,
                                                   threshold_with_lidar=threshold,
                                                   threshold_without_lidar=threshold * 0.7)
    else:
        # 传统固定阈值
        predict = (output > threshold).float()

    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass, depth_map=None, use_adaptive_threshold=True, threshold=0.3):
    """
    Args:
        output: 模型输出
        target: Ground Truth
        nclass: 类别数
        depth_map: 深度图（可选）
        use_adaptive_threshold: 是否使用动态自适应阈值
        threshold: 推理阈值（默认0.3，适配Soft Label max=0.6）
    """
    mini = 1
    maxi = 1
    nbins = 1

    # 使用动态自适应阈值
    if use_adaptive_threshold and depth_map is not None:
        predict = adaptive_threshold_binarization(output, depth_map,
                                                   threshold_with_lidar=threshold,
                                                   threshold_without_lidar=threshold * 0.7)
    else:
        # 传统固定阈值
        predict = (output > threshold).float()

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

