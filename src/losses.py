"""
losses.py
=========
Loss functions for binary medical image segmentation.

Provides:
  - DiceLoss
  - BinaryFocalLoss
  - CombinedLoss (Dice + BCE / Focal)
  - Lovász Softmax approximation (optional)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    L_dice = 1 - (2 * |P ∩ T| + ε) / (|P| + |T| + ε)

    Args:
        smooth:   Laplace smoothing term to avoid division by zero.
        sigmoid:  If True, applies sigmoid to logits before computing loss.
        per_image: If True, compute Dice per image and average.
    """

    def __init__(
        self,
        smooth:    float = 1.0,
        sigmoid:   bool  = True,
        per_image: bool  = True,
    ):
        super().__init__()
        self.smooth    = smooth
        self.sigmoid   = sigmoid
        self.per_image = per_image

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, 1, H, W]  raw model output
            targets: [B, 1, H, W]  binary ground truth
        """
        if self.sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = logits

        if self.per_image:
            # Compute per image, then average
            dims = (1, 2, 3)
        else:
            dims = (0, 1, 2, 3)

        intersection = (probs * targets).sum(dim=dims)
        cardinality  = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary segmentation – addresses class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha:  Weight for positive class.
        gamma:  Focusing parameter.
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        fl    = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return fl.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of Dice loss and BCE or Focal loss.

    Args:
        dice_weight: Weight for Dice term.
        bce_weight:  Weight for BCE term.
        use_focal:   If True, use BinaryFocalLoss instead of plain BCE.
        focal_alpha: Passed to BinaryFocalLoss.
        focal_gamma: Passed to BinaryFocalLoss.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight:  float = 0.5,
        use_focal:   bool  = False,
        focal_alpha: float = 0.8,
        focal_gamma: float = 2.0,
        smooth:      float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight

        self.dice_loss = DiceLoss(smooth=smooth)

        if use_focal:
            self.bce_loss = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce  = self.bce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce


def build_loss(cfg: dict) -> CombinedLoss:
    """Build CombinedLoss from config dict."""
    tcfg = cfg.get("training", {})
    return CombinedLoss(
        dice_weight = tcfg.get("dice_weight", 0.5),
        bce_weight  = tcfg.get("bce_weight",  0.5),
    )
