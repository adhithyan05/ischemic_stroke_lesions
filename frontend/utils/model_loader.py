"""
model_loader.py
===============
Safe, cached model loading utilities for the Streamlit web frontend.
Loads both the Baseline U-Net and Domain-Adversarial Neural Network (DANN).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import streamlit as st
import yaml

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import build_unet, UNet
from dann_model import build_dann, DANNUNet


def get_inference_device() -> torch.device:
    """Detect and return appropriate torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config() -> dict:
    """Load primary project configuration."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner=False)
def load_baseline_unet(checkpoint_path: Optional[str] = None) -> Tuple[UNet, dict]:
    """
    Load the adult-trained baseline U-Net in evaluation mode.
    Cached across Streamlit sessions to prevent redundant disk I/O.
    """
    if checkpoint_path is None:
        checkpoint_path = str(PROJECT_ROOT / "outputs/checkpoints/baseline_unet_best.pth")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at: {ckpt_file}")

    device = get_inference_device()
    cfg = load_config()

    model = build_unet(cfg).to(device)
    ckpt = torch.load(ckpt_file, map_location=device)

    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Freeze gradients
    for p in model.parameters():
        p.requires_grad = False

    meta = {
        "name": "Baseline Attention U-Net",
        "domain": "Adult ISLES 2022 Trained",
        "checkpoint_name": ckpt_file.name,
        "epoch": ckpt.get("epoch", 82),
        "val_dice": ckpt.get("val_dice", 0.2673),
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, meta


@st.cache_resource(show_spinner=False)
def load_dann_model(checkpoint_path: Optional[str] = None) -> Tuple[DANNUNet, dict]:
    """
    Load the Domain-Adversarial Neural Network (DANN) model in evaluation mode.
    Specifically loads the best validation checkpoint from Epoch 6.
    """
    if checkpoint_path is None:
        checkpoint_path = str(PROJECT_ROOT / "outputs/checkpoints/dann/dann_unet_best.pth")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"DANN checkpoint not found at: {ckpt_file}")

    device = get_inference_device()
    cfg = load_config()

    model = build_dann(cfg=cfg, device=device)
    ckpt = torch.load(ckpt_file, map_location=device)

    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    meta = {
        "name": "Domain-Adversarial Neural Network (DANN)",
        "domain": "Adult-to-Neonatal Adapted (ISLES 2022 -> BONBID-HIE)",
        "checkpoint_name": ckpt_file.name,
        "epoch": ckpt.get("epoch", 6),
        "best_epoch": ckpt.get("best_epoch", 6),
        "best_val_dice": ckpt.get("best_val_dice", 0.2652),
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, meta

