
import torch
import numpy as np
from typing import Union

Tensor = Union[torch.Tensor, np.array]

def user_accuracy(cm:Tensor) -> float:
    """ TP / (TP + FP) """
    return precision(cm)

def producer_accuracy(cm:Tensor) -> float:
    """ TP / (TP + FN) """
    return recall(cm)

def TPR(cm:Tensor) -> float:
    """ TP / (TP + FN) """
    return recall(cm)

def precision(cm:Tensor) -> float:
    """ TP / (TP + FP) """
    assert cm.shape == (2, 2), f"Expected binary found {cm.shape}"

    return cm[1,1] / (cm[1,1] + cm[0,1])

def recall(cm:Tensor) -> float:
    """ TP / (TP + FN) """
    assert cm.shape == (2, 2), f"Expected binary found {cm.shape}"

    return cm[1, 1] / (cm[1, 1] + cm[1, 0])

def f1score(cm:Tensor) -> float:
    prec = precision(cm)
    rec = recall(cm)
    return 2 * (prec * rec) / (prec + rec)

def fbeta_score(cm:Tensor, beta:float) -> float:
    """F-beta score: weights precision vs recall.

    beta < 1: precision-focused (F0.5 weights precision 2x)
    beta = 1: balanced (F1)
    beta > 1: recall-focused (F2 weights recall 2x)
    """
    prec = precision(cm)
    rec = recall(cm)
    beta_sq = beta ** 2
    return (1 + beta_sq) * (prec * rec) / (beta_sq * prec + rec)

def f05score(cm:Tensor) -> float:
    """F0.5 score: precision-focused (weights precision 2x over recall)."""
    return fbeta_score(cm, beta=0.5)

def f2score(cm:Tensor) -> float:
    """F2 score: recall-focused (weights recall 2x over precision)."""
    return fbeta_score(cm, beta=2.0)

def FPR(cm:Tensor) -> float:
    """ FP / (FP + TN)"""
    return cm[0, 1] / (cm[0, 1] + cm[0, 0])

def iou(cm:Tensor) -> float:
    """ TP / (TP + FN + FP) """
    assert cm.shape == (2, 2), f"Expected binary found {cm.shape}"

    return cm[1, 1] / (cm[1, 1] + cm[1, 0] + cm[0,1])

def accuracy(cm:Tensor) -> float:
    """ (TP + TN) / (TP + FN + FP + TN) """
    assert cm.shape == (2, 2), f"Expected binary found {cm.shape}"

    return (cm[1, 1] + cm[0, 0]) / cm.sum()

def cohen_kappa(cm:Tensor) -> float:
    confmat = cm.float() if not cm.is_floating_point() else cm
    sum0 = confmat.sum(dim=0, keepdim=True)
    sum1 = confmat.sum(dim=1, keepdim=True)
    expected = sum1 @ sum0 / sum0.sum()

    w_mat = torch.ones_like(confmat).flatten()
    w_mat[:: 2 + 1] = 0
    w_mat = w_mat.reshape(2, 2)

    k = torch.sum(w_mat * confmat) / torch.sum(w_mat * expected)
    return 1 - k

def balanced_accuracy(cm:Tensor) -> float:
    """ 0.5 (PA + TN /(TN + FP))"""

    PA = recall(cm)
    TNR = cm[0, 0] /(cm[0, 0] + cm[0, 1])
    return 0.5 * (PA + TNR)

def TP(cm:Tensor) -> float:
    return cm[1, 1]

def TN(cm:Tensor) -> float:
    return cm[0, 0]

def FP(cm:Tensor) -> float:
    return cm[0, 1]

def FN(cm:Tensor) -> float:
    return cm[1, 0]

METRICS_CONFUSION_MATRIX = [precision, recall, f1score, f05score, f2score, iou, accuracy, cohen_kappa, balanced_accuracy]

