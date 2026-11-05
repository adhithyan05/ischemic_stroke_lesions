"""
metrics.py
==========
Evaluation metrics for binary segmentation.

All functions accept numpy arrays or torch tensors (CPU only).

Metrics implemented:
  - Dice coefficient (F1 score over voxels)
  - IoU (Jaccard index)
  - Precision
  - Recall
  - Sensitivity  (== Recall)
  - Specificity
  - Hausdorff Distance 95 (requires scipy)
"""

from __future__ import annotations

import logging
from typing import Dict, Union

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _to_numpy(x) -> np.ndarray:
    """Accept numpy array or torch Tensor and return numpy."""
    if hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _binary(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5):
    """Binarise float prediction at threshold."""
    pred   = (_to_numpy(pred).ravel()   >= threshold).astype(np.uint8)
    target = (_to_numpy(target).ravel() >= threshold).astype(np.uint8)
    return pred, target


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample metrics
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(pred, target, threshold: float = 0.5) -> float:
    """Dice coefficient (2 * TP / (2*TP + FP + FN))."""
    p, t = _binary(pred, target, threshold)
    tp = (p & t).sum()
    fp = (p & ~t).sum()
    fn = (~p & t).sum()
    return float(2 * tp / (2 * tp + fp + fn + _EPS))


def iou_score(pred, target, threshold: float = 0.5) -> float:
    """Intersection over Union (Jaccard index)."""
    p, t = _binary(pred, target, threshold)
    tp = (p & t).sum()
    fp = (p & ~t).sum()
    fn = (~p & t).sum()
    return float(tp / (tp + fp + fn + _EPS))


def precision_score(pred, target, threshold: float = 0.5) -> float:
    """Precision = TP / (TP + FP)."""
    p, t = _binary(pred, target, threshold)
    tp = (p & t).sum()
    fp = (p & ~t).sum()
    return float(tp / (tp + fp + _EPS))


def recall_score(pred, target, threshold: float = 0.5) -> float:
    """Recall = TP / (TP + FN)."""
    p, t = _binary(pred, target, threshold)
    tp = (p & t).sum()
    fn = (~p & t).sum()
    return float(tp / (tp + fn + _EPS))


def sensitivity_score(pred, target, threshold: float = 0.5) -> float:
    """Sensitivity == Recall (alias for medical nomenclature)."""
    return recall_score(pred, target, threshold)


def specificity_score(pred, target, threshold: float = 0.5) -> float:
    """Specificity = TN / (TN + FP)."""
    p, t = _binary(pred, target, threshold)
    tn = (~p & ~t).sum()
    fp = (p & ~t).sum()
    return float(tn / (tn + fp + _EPS))


def hausdorff95(pred, target, threshold: float = 0.5) -> float:
    """
    95th-percentile Hausdorff distance (requires scipy).
    Returns NaN if either mask is empty.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        logger.warning("scipy not available ? skipping HD95")
        return float("nan")

    p = (_to_numpy(pred).squeeze() >= threshold).astype(bool)
    t = (_to_numpy(target).squeeze() >= threshold).astype(bool)

    if not p.any() or not t.any():
        return float("nan")

    dt_p = distance_transform_edt(~p)
    dt_t = distance_transform_edt(~t)

    h_pt = dt_t[p]
    h_tp = dt_p[t]
    return float(np.percentile(np.concatenate([h_pt, h_tp]), 95))


# ─────────────────────────────────────────────────────────────────────────────
# Batch / epoch-level accumulator
# ─────────────────────────────────────────────────────────────────────────────

class MetricAccumulator:
    """
    Running average of all segmentation metrics over an epoch.

    Usage:
        acc = MetricAccumulator(threshold=0.5)
        for logits, targets in loader:
            probs = torch.sigmoid(logits)
            acc.update(probs.cpu(), targets.cpu())
        results = acc.compute()
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._reset()

    def _reset(self):
        self._vals: Dict[str, list] = {
            "dice":        [],
            "iou":         [],
            "precision":   [],
            "recall":      [],
            "sensitivity": [],
            "specificity": [],
        }

    def update(self, probs, targets):
        """
        probs:   [B, 1, H, W] float (sigmoid-activated)
        targets: [B, 1, H, W] binary float
        """
        probs   = _to_numpy(probs)
        targets = _to_numpy(targets)

        for b in range(probs.shape[0]):
            p = probs[b]
            t = targets[b]
            self._vals["dice"].append(dice_score(p, t, self.threshold))
            self._vals["iou"].append(iou_score(p, t, self.threshold))
            self._vals["precision"].append(precision_score(p, t, self.threshold))
            self._vals["recall"].append(recall_score(p, t, self.threshold))
            self._vals["sensitivity"].append(sensitivity_score(p, t, self.threshold))
            self._vals["specificity"].append(specificity_score(p, t, self.threshold))

    def compute(self) -> Dict[str, float]:
        """Return mean of all accumulated metrics."""
        return {
            k: float(np.mean(v)) if v else 0.0
            for k, v in self._vals.items()
        }

    def reset(self):
        self._reset()
