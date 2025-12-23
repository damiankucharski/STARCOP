import torch
import torch.nn as nn
import torch.nn.functional as F


def l1(input, target):
    return F.l1_loss(input, target)


def mse(input, target):
    return F.mse_loss(input, target)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2*intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class TverskyLoss(nn.Module):
    """
    Tversky loss for imbalanced segmentation.

    alpha=0.5, beta=0.5 → Dice
    alpha=0.3, beta=0.7 → Recall-focused (penalize FN)
    alpha=0.7, beta=0.3 → Precision-focused (penalize FP)
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return 1 - tversky


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        return (alpha_weight * focal_weight * bce).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=torch.tensor([self.pos_weight], device=pred.device)
        )
        return self.bce_weight * bce + self.dice_weight * self.dice(pred, target)
