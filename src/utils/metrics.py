"""
Evaluation metrics for anomaly detection.

Primary metric: AUROC (image-level and pixel-level).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def image_auroc(scores: list[float], labels: list[int]) -> float:
    """
    Image-level AUROC.

    Args:
        scores: anomaly score per image (higher = more anomalous)
        labels: 0=normal, 1=anomaly
    """
    return roc_auc_score(labels, scores)


def pixel_auroc(score_maps: np.ndarray, gt_masks: np.ndarray) -> float:
    """
    Pixel-level AUROC.

    Args:
        score_maps: (N, H, W) float array of anomaly scores
        gt_masks:   (N, H, W) binary ground-truth masks
    """
    return roc_auc_score(gt_masks.flatten(), score_maps.flatten())


def pixel_ap(score_maps: np.ndarray, gt_masks: np.ndarray) -> float:
    """Average Precision at pixel level."""
    return average_precision_score(gt_masks.flatten(), score_maps.flatten())


def compute_best_threshold(score_maps: np.ndarray, gt_masks: np.ndarray) -> float:
    """
    Find the threshold that maximises F1 on the pixel-level predictions.
    """
    from sklearn.metrics import f1_score

    flat_scores = score_maps.flatten()
    flat_labels = gt_masks.flatten().astype(int)

    thresholds = np.percentile(flat_scores, np.linspace(50, 99, 50))
    best_f1, best_thr = 0.0, thresholds[0]

    for thr in thresholds:
        preds = (flat_scores >= thr).astype(int)
        f1 = f1_score(flat_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr
