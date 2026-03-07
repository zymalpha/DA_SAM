"""
Loss functions for DA-SAM training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, eps=1e-6):
    """
    Dice Loss for binary segmentation

    Args:
        pred: Predicted logits (B, 1, H, W)
        target: Ground truth (B, 1, H, W)
        eps: Small epsilon for numerical stability
    Returns:
        Dice loss value
    """
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1. - dice.mean()


def compute_metrics(pred_logits, gt, threshold=0.5):
    """
    Compute segmentation metrics

    Args:
        pred_logits: Predicted logits (B, 1, H, W)
        gt: Ground truth (B, 1, H, W)
        threshold: Binarization threshold
    Returns:
        Dictionary with IoU, Dice, Precision, Recall
    """
    pred = (torch.sigmoid(pred_logits) >= threshold).float()
    pred_flat = pred.view(pred.size(0), -1)
    gt_flat = gt.view(gt.size(0), -1)

    inter = (pred_flat * gt_flat).sum(1)
    pred_sum, gt_sum = pred_flat.sum(1), gt_flat.sum(1)
    union = pred_sum + gt_sum - inter

    return {
        "iou": (inter / (union + 1e-8)).mean().item(),
        "dice": (2 * inter / (pred_sum + gt_sum + 1e-8)).mean().item(),
        "prec": (inter / (pred_sum + 1e-8)).mean().item(),
        "rec": (inter / (gt_sum + 1e-8)).mean().item()
    }


class CombinedLoss(nn.Module):
    """
    Combined loss for DA-SAM training

    Losses:
    - BCE: Binary Cross Entropy
    - Dice: Dice Loss
    - MSE: Mean Squared Error (auxiliary)
    """

    def __init__(self, lambda_bce=0.8, lambda_dice=2.0, lambda_aux=0.5, target_mod=0.3):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.lambda_aux = lambda_aux
        self.target_mod = target_mod

    def forward(self, pred, target, aux=None):
        """
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth (B, 1, H, W)
            aux: AIA modulation map delta (optional)
        Returns:
            Combined loss value
        """
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dl = dice_loss(pred, target)

        loss = self.lambda_bce * bce + self.lambda_dice * dl

        if aux is not None:
            mse = F.mse_loss(aux, target * self.target_mod)
            loss = loss + self.lambda_aux * mse

        return loss


def bce_dice_loss(pred, target, lambda_bce=0.5, lambda_dice=0.5):
    """
    Combined BCE + Dice loss

    Args:
        pred: Predicted logits
        target: Ground truth
        lambda_bce: Weight for BCE
        lambda_dice: Weight for Dice
    Returns:
        Combined loss
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dl = dice_loss(pred, target)
    return lambda_bce * bce + lambda_dice * dl


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
