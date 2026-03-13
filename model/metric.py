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
    def __init__(self, nclass, bins, img_size=None):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.img_size = img_size
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):
        preds_np  = np.array(preds.cpu()).astype('float32')
        labels_np = np.array(labels.cpu()).astype('int64')

        # Flatten batch/channel dims → (N, H, W)
        h, w = preds_np.shape[-2], preds_np.shape[-1]
        preds_np  = preds_np.reshape(-1, h, w)
        labels_np = labels_np.reshape(-1, h, w)
        self._last_area = h * w

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)

            for predits, labelss in zip(preds_np, labels_np):
                predits = (predits > score_thresh).astype('int64')

                image = measure.label(predits, connectivity=2)
                coord_image = measure.regionprops(image)
                label = measure.label(labelss, connectivity=2)
                coord_label = measure.regionprops(label)

                self.target[iBin] += len(coord_label)
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
                self.FA[iBin] += np.sum(self.dismatch)
                self.PD[iBin] += len(self.distance_match)

    def get(self,img_num):

        if self.img_size is not None:
            area = self.img_size * self.img_size
        else:
            # Use inferred size from last update (if available)
            area = getattr(self, '_last_area', None)
            if area is None:
                raise ValueError("PD_FA: image area is unknown. Provide img_size or call update first.")
        Final_FA = self.FA / (area * img_num)
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


# ======================== Mask-to-Box IoU ========================
# 2026-02-03: Added for fair evaluation with Box-annotated datasets

def calculate_mask_to_box_iou(pred_mask, gt_mask, threshold=0.5):
    """
    计算 Mask-to-Box IoU（检测评估指标）。

    原理：
    1. 将预测 mask 和 GT mask 转换为二值图
    2. 计算它们的外接矩形 (Bounding Box)
    3. 计算这两个矩形的 IoU

    这个指标解决了"弱监督评估不公平"问题：
    - Segmentation IoU 会因为 GT 是 box/ellipse 而产生差异
    - Mask-to-Box IoU 统一用外接矩形评估，更公平
    - 更接近真实的检测性能（类似 YOLO 的评估方式）

    Args:
        pred_mask: (B, 1, H, W) 预测概率图 或 Torch Tensor
        gt_mask: (B, 1, H, W) 真实标签 mask (box 或 ellipse 均可) 或 Torch Tensor
        threshold: 二值化阈值（建议与 segmentation 评估一致）

    Returns:
        avg_iou: 当前 batch 的平均 Box IoU

    Example:
        >>> pred = torch.rand(4, 1, 512, 640)
        >>> gt = torch.randint(0, 2, (4, 1, 512, 640)).float()
        >>> box_iou = calculate_mask_to_box_iou(pred, gt, threshold=0.5)
        >>> print(f"Mask-to-Box IoU: {box_iou:.4f}")
    """
    # Convert to numpy if needed
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = (pred_mask > threshold).float()
        pred_np = pred_mask.detach().cpu().numpy()
    else:
        pred_np = (pred_mask > threshold).astype(np.float32)

    if isinstance(gt_mask, torch.Tensor):
        gt_mask = (gt_mask > 0.5).float()
        gt_np = gt_mask.detach().cpu().numpy()
    else:
        gt_np = (gt_mask > 0.5).astype(np.float32)

    batch_size = pred_np.shape[0]
    total_iou = 0.0
    valid_count = 0

    for i in range(batch_size):
        # 提取当前样本
        p_m = pred_np[i, 0]
        g_m = gt_np[i, 0]

        # 获取外接矩形 [x1, y1, x2, y2]
        pred_box = _get_bounding_box(p_m)
        gt_box = _get_bounding_box(g_m)

        if gt_box is None:
            # 如果 GT 是空的（理论不应发生），跳过
            continue

        if pred_box is None:
            # 如果预测是空的（漏检），IoU 为 0
            iou = 0.0
        else:
            # 计算 Box IoU
            iou = _compute_box_iou(pred_box, gt_box)

        total_iou += iou
        valid_count += 1

    # 返回平均值
    if valid_count == 0:
        return 0.0
    return total_iou / valid_count


def _get_bounding_box(binary_mask):
    """
    从二值 mask 提取外接矩形 [x_min, y_min, x_max, y_max]。

    Args:
        binary_mask: (H, W) numpy array, binary {0, 1}

    Returns:
        box: [x_min, y_min, x_max, y_max] 或 None (如果 mask 为空)
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [x_min, y_min, x_max, y_max]


def _compute_box_iou(box1, box2):
    """
    计算两个矩形的 IoU。

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        iou: Intersection over Union
    """
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2

    # Intersection
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h

    # Union
    area_a = (x2_a - x1_a + 1) * (y2_a - y1_a + 1)
    area_b = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)
    union_area = area_a + area_b - inter_area

    return inter_area / (union_area + 1e-7)

