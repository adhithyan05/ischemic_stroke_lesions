"""
model.py
========
U-Net architecture for 2-D binary lesion segmentation.

Design choices:
  - Configurable depth (number of encoder/decoder stages)
  - Residual double-conv blocks for improved gradient flow
  - Dropout for regularisation (important with small medical datasets)
  - Supports both transposed-conv and bilinear upsampling
  - Encoder output features accessible for the DANN domain classifier (Phase 2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Two 3×3 Conv -> BN -> ReLU blocks with optional residual connection.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        dropout:      float = 0.0,
        residual:     bool  = True,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Residual projection if channel count changes
        self.residual = residual
        if residual and in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.residual:
            if self.proj is not None:
                x = self.proj(x)
            out = out + x
        return out


class Down(nn.Module):
    """MaxPool -> DoubleConv (encoder stage)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample -> DoubleConv (decoder stage)."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, 1, bias=False),
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Pad to match skip if sizes differ (due to odd dimensions)
        dH = skip.size(2) - x.size(2)
        dW = skip.size(3) - x.size(3)
        x = F.pad(x, [dW // 2, dW - dW // 2, dH // 2, dH - dH // 2])

        return self.conv(torch.cat([skip, x], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Attention Gate (SE-style channel attention on skip connections)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Soft-attention gate that weighs skip-connection feature maps.
    Based on Oktay et al., "Attention U-Net", 2018.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample g1 to match x1 spatial size
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        psi = F.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Configurable Attention U-Net for 2-D medical image segmentation.

    Args:
        in_channels:   Number of input image channels.
        out_channels:  Number of output channels (1 for binary segmentation).
        base_features: Feature map count at the first encoder level.
        depth:         Number of encoder / decoder stages.
        bilinear:      Use bilinear upsampling instead of transposed convolution.
        dropout:       Dropout rate (applied in DoubleConv blocks).
        use_attention: Enable attention gates on skip connections.
    """

    def __init__(
        self,
        in_channels:   int   = 3,
        out_channels:  int   = 1,
        base_features: int   = 32,
        depth:         int   = 4,
        bilinear:      bool  = False,
        dropout:       float = 0.1,
        use_attention: bool  = True,
    ):
        super().__init__()
        self.depth         = depth
        self.use_attention = use_attention

        # ── Encoder ────────────────────────────────────────────────────────
        # inc: first double-conv (no pooling)
        # down_*: pool + double-conv
        features = [base_features * (2 ** i) for i in range(depth + 1)]

        self.inc   = DoubleConv(in_channels, features[0], dropout=dropout)
        self.downs = nn.ModuleList([
            Down(features[i], features[i + 1], dropout=dropout)
            for i in range(depth)
        ])

        # ── Bottleneck feature channel count ──────────────────────────────
        self.bottleneck_channels = features[depth]

        # ── Decoder ────────────────────────────────────────────────────────
        self.ups = nn.ModuleList([
            Up(features[depth - i], features[depth - i - 1], bilinear=bilinear, dropout=dropout)
            for i in range(depth)
        ])

        # ── Attention gates ────────────────────────────────────────────────
        if use_attention:
            self.attention_gates = nn.ModuleList([
                AttentionGate(
                    F_g   = features[depth - i],      # gating signal channels
                    F_l   = features[depth - i - 1],  # skip channels
                    F_int = features[depth - i - 1] // 2,
                )
                for i in range(depth)
            ])

        # ── Output head ───────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(features[0], out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run encoder only.

        Returns:
            bottleneck: [B, C_bottle, H_bottle, W_bottle]
            skips:      list of skip feature maps (shallow -> deep)
        """
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        return x, skips   # x is bottleneck; skips[-1] is also bottleneck

    def decode(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Run decoder from bottleneck + skip features."""
        x = bottleneck
        # skips: [s0(inc), s1, s2, ..., s_depth-1(deepest encoder), s_depth(bottleneck)]
        # We use skips[depth-1] down to skips[0]
        for i, up in enumerate(self.ups):
            skip = skips[self.depth - 1 - i]  # corresponding encoder skip
            if self.use_attention:
                skip = self.attention_gates[i](g=x, x=skip)
            x = up(x, skip)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encode(x)
        x = self.decode(bottleneck, skips)
        return self.out_conv(x)   # [B, out_channels, H, W] (logits)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_unet(cfg: dict) -> UNet:
    """Build a UNet from a configuration dict (from config.yaml)."""
    ucfg = cfg.get("unet", {})
    return UNet(
        in_channels   = ucfg.get("in_channels",   3),
        out_channels  = ucfg.get("out_channels",  1),
        base_features = ucfg.get("base_features", 32),
        depth         = ucfg.get("depth",         4),
        bilinear      = ucfg.get("bilinear",      False),
        dropout       = ucfg.get("dropout",       0.1),
        use_attention = True,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
