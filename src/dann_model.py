"""
dann_model.py
=============
Domain-Adversarial Neural Network (DANN) for Adult-to-Neonatal Brain Lesion Segmentation.

Implements the standard DANN framework (Ganin & Lempitsky, 2015; Ganin et al., JMLR 2016):
  1. Feature Extractor (G_f): Encoder of the Attention U-Net extracting deep representation.
  2. Label Predictor  (G_y): Decoder and segmentation head predicting binary lesion masks.
  3. Domain Classifier(G_d): Binary discriminator classifying source (adult) vs target (neonatal).
  4. Gradient Reversal Layer (GRL): Reverses gradients during backprop to learn domain-invariant features.

Domain convention:
  - Source domain (Adult ISLES 2022)   : label = 0
  - Target domain (Neonatal BONBID-HIE): label = 1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UNet, build_unet

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gradient Reversal Layer (GRL)
# ─────────────────────────────────────────────────────────────────────────────

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) autograd function.
    In the forward pass, it acts as an identity operator:
        y = x
    In the backward pass, it reverses and scales the incoming gradients:
        d/dx = -lambda_ * d/dy
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Reverse and scale gradient
        reversed_grad = grad_output.neg() * ctx.lambda_
        return reversed_grad, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Convenience functional wrapper for GradientReversalFunction."""
    return GradientReversalFunction.apply(x, lambda_)


class GradientReversal(nn.Module):
    """
    Module wrapper for GradientReversalFunction.

    Args:
        lambda_: Initial gradient reversal scaling factor (default: 1.0).
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor, lambda_: Optional[float] = None) -> torch.Tensor:
        lam = self.lambda_ if lambda_ is None else float(lambda_)
        return grad_reverse(x, lam)


def compute_dann_lambda(progress: float, gamma: float = 10.0) -> float:
    """
    Ganin & Lempitsky gradual adaptation schedule:
        lambda_p = 2 / (1 + exp(-gamma * p)) - 1
    where p in [0, 1] is the training progress.
    Ramps smoothly from 0.0 at p=0 to ~1.0 at p=1.
    """
    p = float(max(0.0, min(1.0, progress)))
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)).item()) - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Domain Classifier
# ─────────────────────────────────────────────────────────────────────────────

class DomainClassifier(nn.Module):
    """
    Discriminative domain classifier network.
    Takes bottleneck feature maps from the U-Net feature extractor,
    pools them to a compact feature vector, and classifies the domain.

    Args:
        in_channels: Channel count of the input feature map (e.g., 256 for depth=4, base=16).
        hidden_dim:  Dimension of intermediate fully connected layers.
        dropout:     Dropout probability for regularization.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim:  int = 128,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Global average + max pooling for robust representation
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        pool_dim = in_channels * 2

        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] feature tensor from encoder bottleneck.
        Returns:
            domain_logits: [B, 1] raw logits (source=0, target=1).
        """
        avg_f = self.avg_pool(features).flatten(1)
        max_f = self.max_pool(features).flatten(1)
        f = torch.cat([avg_f, max_f], dim=1)  # [B, 2 * in_channels]
        return self.classifier(f)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DANN U-Net Architecture
# ─────────────────────────────────────────────────────────────────────────────

class DANNUNet(nn.Module):
    """
    Complete Domain-Adversarial U-Net Architecture.

    Decomposes the Attention U-Net into:
      1. Feature Extractor G_f : unet.inc + unet.downs (produces bottleneck & skips)
      2. Label Predictor  G_y : unet.ups + unet.attention_gates + unet.out_conv
      3. Domain Classifier G_d : GRL + DomainClassifier

    Args:
        base_unet: Optional pre-constructed UNet instance.
        cfg:       Configuration dictionary used if base_unet is not provided.
        dropout:   Dropout rate for the domain classifier.
    """

    def __init__(
        self,
        base_unet: Optional[UNet] = None,
        cfg: Optional[dict] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Feature extractor & label predictor
        if base_unet is not None:
            self.unet = base_unet
        elif cfg is not None:
            self.unet = build_unet(cfg)
        else:
            self.unet = UNet(in_channels=3, out_channels=1, base_features=16, depth=4, bilinear=True)

        bottleneck_ch = self.unet.bottleneck_channels  # 256 for depth=4, base=16

        # Gradient Reversal Layer
        self.grl = GradientReversal(lambda_=1.0)

        # Domain Classifier
        self.domain_classifier = DomainClassifier(
            in_channels=bottleneck_ch,
            hidden_dim=128,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run feature extractor (encoder) on input images."""
        return self.unet.encode(x)

    def decode(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Run label predictor (decoder + head) on features."""
        feat = self.unet.decode(bottleneck, skips)
        return self.unet.out_conv(feat)

    def classify_domain(self, bottleneck: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        """Run domain classifier on bottleneck features with gradient reversal."""
        reversed_features = self.grl(bottleneck, lambda_=lambda_)
        return self.domain_classifier(reversed_features)

    def forward(
        self,
        x: torch.Tensor,
        lambda_: float = 1.0,
        return_domain: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through DANN.

        Args:
            x:              [B, 3, H, W] input image batch.
            lambda_:        Current gradient reversal weight.
            return_domain:  If True, returns (seg_logits, domain_logits).
                            If False, returns seg_logits only (standard inference mode).
        """
        bottleneck, skips = self.encode(x)
        seg_logits = self.decode(bottleneck, skips)

        if not return_domain:
            return seg_logits

        domain_logits = self.classify_domain(bottleneck, lambda_=lambda_)
        return seg_logits, domain_logits

    def load_pretrained_baseline(self, checkpoint_path: Union[str, Path], device: torch.device):
        """
        Load weights from the adult baseline checkpoint into G_f and G_y.
        Initializes domain classifier randomly with Kaiming normal.
        Ensures checkpoint file is loaded read-only without modification.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt

        # Load weights into self.unet
        missing, unexpected = self.unet.load_state_dict(state_dict, strict=True)
        logger.info(
            f"Successfully loaded baseline weights into DANN from: {ckpt_path.name} "
            f"(epoch: {ckpt.get('epoch', '?')}, val_dice: {ckpt.get('val_dice', '?'):.4f})"
        )


def build_dann(cfg: dict, checkpoint_path: Optional[str | Path] = None, device: Optional[torch.device] = None) -> DANNUNet:
    """Build and optionally initialize a DANN model from config."""
    model = DANNUNet(cfg=cfg)
    if device is not None:
        model = model.to(device)

    if checkpoint_path is not None:
        dev = device if device is not None else torch.device("cpu")
        model.load_pretrained_baseline(checkpoint_path, dev)

    return model

