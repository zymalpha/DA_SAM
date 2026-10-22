"""
Metrics for evaluation
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


def _binary_mask_pair(pred, gt):
    """Validate matching finite 2D masks and normalize positive labels to bool."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("Prediction and ground truth must be matching 2D masks")
    if pred.size == 0:
        raise ValueError("Masks must contain at least one pixel")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Masks must contain finite values")
    return pred > 0, gt > 0


def compute_hd95(pred, gt):
    """
    Compute the pooled bidirectional 95th percentile boundary distance.

    Boundaries use SciPy's default binary erosion connectivity. Euclidean
    distances are measured in pixels of the supplied 2D grid. Distances from
    both directions are concatenated before taking the 95th percentile.
    Two empty masks return 0; exactly one empty mask returns the existing
    finite penalty of 50 pixels. Positive label values are treated as foreground.

    Args:
        pred: Binary prediction mask (H, W)
        gt: Binary ground truth mask (H, W)
    Returns:
        HD95 value
    """
    pred_bin, gt_bin = _binary_mask_pair(pred, gt)

    # Handle edge cases
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 50.0

    # Get borders
    pred_border = pred_bin ^ binary_erosion(pred_bin)
    gt_border = gt_bin ^ binary_erosion(gt_bin)

    # Compute distance transforms
    gt_dist = distance_transform_edt(~gt_border)
    pred_dist = distance_transform_edt(~pred_border)

    # Get distances from each border to the other
    pred_to_gt = gt_dist[pred_border]
    gt_to_pred = pred_dist[gt_border]

    if len(pred_to_gt) == 0 or len(gt_to_pred) == 0:
        return 50.0

    all_dist = np.concatenate([pred_to_gt, gt_to_pred])
    return float(np.percentile(all_dist, 95))


def get_full_metrics(pred, gt):
    """
    Compute full evaluation metrics

    Args:
        pred: Binary prediction (H, W); positive labels are foreground
        gt: Binary ground truth (H, W); positive labels are foreground
    Returns:
        Tuple of (dice, iou, hd95, precision, recall)
    """
    pred, gt = _binary_mask_pair(pred, gt)

    # Handle edge cases
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 1.0, 1.0, 0.0, 1.0, 1.0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return 0.0, 0.0, 50.0, 0.0, 0.0

    # Compute basic metrics
    inter = (pred * gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    union = pred_sum + gt_sum - inter

    dice = 2 * inter / (pred_sum + gt_sum + 1e-8)
    iou = inter / (union + 1e-8)
    hd95 = compute_hd95(pred, gt)
    precision = inter / (pred_sum + 1e-8)
    recall = inter / (gt_sum + 1e-8)

    return float(dice), float(iou), float(hd95), float(precision), float(recall)


def compute_psnr(pred, gt, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio

    Args:
        pred: Predicted image
        gt: Ground truth image
        max_val: Maximum possible value
    Returns:
        PSNR value
    """
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_mae(pred, gt):
    """
    Compute Mean Absolute Error

    Args:
        pred: Predicted image
        gt: Ground truth image
    Returns:
        MAE value
    """
    return np.mean(np.abs(pred - gt))


class MetricTracker:
    """Track metrics during training/evaluation"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            'dice': [],
            'iou': [],
            'hd95': [],
            'precision': [],
            'recall': []
        }

    def update(self, dice=None, iou=None, hd95=None, precision=None, recall=None):
        if dice is not None:
            self.metrics['dice'].append(dice)
        if iou is not None:
            self.metrics['iou'].append(iou)
        if hd95 is not None:
            self.metrics['hd95'].append(hd95)
        if precision is not None:
            self.metrics['precision'].append(precision)
        if recall is not None:
            self.metrics['recall'].append(recall)

    def get_avg(self):
        avg = {}
        for k, v in self.metrics.items():
            if len(v) > 0:
                avg[k] = np.mean(v)
            else:
                avg[k] = 0.0
        return avg

    def __str__(self):
        avg = self.get_avg()
        return f"Dice: {avg['dice']:.4f}, IoU: {avg['iou']:.4f}, HD95: {avg['hd95']:.2f}"
