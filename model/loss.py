import torch.nn as nn
import numpy as np
import  torch

def SoftIoULoss( pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 0.8

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss

def focal_loss_per_pixel(pred, target, alpha=2.0, gamma=4.0):
    """Per-pixel Focal Loss without reduction (same formulation as LESPS utils.py)."""
    pred = torch.sigmoid(pred).clamp(min=1e-6, max=1-1e-6)  # float16-safe: avoids log(0)=nan under AMP
    eps = 1e-6
    pos_weights = target
    neg_weights = (1 - target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss

def focal_loss_ohem(pred, target, ohem_ratio=0.5, alpha=2.0, gamma=4.0):
    """Focal Loss + OHEM: keep the hardest (1 - ohem_ratio) fraction of pixels."""
    per_pixel = focal_loss_per_pixel(pred, target, alpha=alpha, gamma=gamma)
    loss_flat = per_pixel.reshape(-1)
    n_keep = max(int(loss_flat.numel() * (1.0 - ohem_ratio)), 1)
    topk_loss, _ = loss_flat.topk(n_keep)
    avg_factor = max(1, int((target > 0.5).sum().item()))
    return topk_loss.sum() / avg_factor

class ConfidenceLoss(nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce(pred, target)

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