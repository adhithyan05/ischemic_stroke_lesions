"""
run_phase2_e1_e4.py
===================
Phase 2, Step 1: Implementation and Comparative Evaluation of:
  - E1: Reference Benchmark (Existing D1 input [-Z-ADC, ADC, 0] with 128x128 full-volume pad)
  - E4-A: Brain Bounding-Box Cropping without Resizing (Pad to 128x128 only if necessary)
  - E4-B: Brain Bounding-Box Cropping with Resizing (Resize cropped brain to 128x128)

Models Evaluated:
  1. Baseline Attention U-Net (outputs/checkpoints/baseline_unet_best.pth)
  2. DANN Attention U-Net (outputs/checkpoints/dann/dann_unet_best.pth)

Cohort:
  4 Official Neonatal Validation Subjects (MGHNICU_001, MGHNICU_002, MGHNICU_302, MGHNICU_436)

Metrics Computed:
  - Voxel: Dice, IoU, Precision, Recall, Specificity, HD95 (mm)
  - Lesion: 3D 26-connectivity connected components matching (>=10% overlap),
            Lesion-wise Precision, Recall, and F1
  - Volumetric: GT volume (mm3), Pred volume (mm3), Absolute Error (mm3), Relative Error
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import distance_transform_edt, label, zoom
import torch
import yaml

# Path setup
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from model import build_unet
from dann_model import build_dann
from data_utils import (
    discover_bonbid,
    load_nifti,
    normalize_volume,
    resample_to_target_spacing,
    _crop_or_pad,
)
from scientific_evaluation import compute_hd95, compute_file_hash


# ─────────────────────────────────────────────────────────────────────────────
# 1. Lesion-Level & Volumetric Metrics (3D 26-Connectivity)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lesion_and_volume_metrics(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    voxel_vol_mm3: float = 8.0,
    min_overlap: float = 0.10,
) -> Dict[str, Any]:
    """
    Compute 3D lesion-level connected component metrics (26-connectivity)
    and volumetric errors.
    """
    p_bin = (pred_vol >= 0.5)
    g_bin = (gt_vol >= 0.5)

    gt_vol_mm3 = float(g_bin.sum() * voxel_vol_mm3)
    pred_vol_mm3 = float(p_bin.sum() * voxel_vol_mm3)
    abs_vol_err = float(abs(pred_vol_mm3 - gt_vol_mm3))
    rel_vol_err = float(abs_vol_err / max(1.0, gt_vol_mm3))

    # 3D 26-connectivity structure
    structure = np.ones((3, 3, 3), dtype=int)
    pred_labeled, num_pred = label(p_bin, structure)
    gt_labeled, num_gt = label(g_bin, structure)

    if num_gt == 0 and num_pred == 0:
        return {
            "gt_volume_mm3": gt_vol_mm3,
            "pred_volume_mm3": pred_vol_mm3,
            "abs_volume_error_mm3": abs_vol_err,
            "rel_volume_error": rel_vol_err,
            "num_gt_lesions": 0,
            "num_pred_lesions": 0,
            "lesion_precision": 1.0,
            "lesion_recall": 1.0,
            "lesion_f1": 1.0,
        }

    if num_gt == 0 or num_pred == 0:
        return {
            "gt_volume_mm3": gt_vol_mm3,
            "pred_volume_mm3": pred_vol_mm3,
            "abs_volume_error_mm3": abs_vol_err,
            "rel_volume_error": rel_vol_err,
            "num_gt_lesions": int(num_gt),
            "num_pred_lesions": int(num_pred),
            "lesion_precision": 0.0,
            "lesion_recall": 0.0,
            "lesion_f1": 0.0,
        }

    # Match GT lesions: A GT lesion is detected if >= min_overlap fraction is overlapped by prediction
    tp_gt = 0
    for g_id in range(1, num_gt + 1):
        g_mask = (gt_labeled == g_id)
        overlap = (pred_labeled[g_mask] > 0).sum() / max(1, g_mask.sum())
        if overlap >= min_overlap:
            tp_gt += 1

    # Match Pred lesions: A predicted lesion is a True Positive if >= min_overlap overlaps GT
    tp_pred = 0
    for p_id in range(1, num_pred + 1):
        p_mask = (pred_labeled == p_id)
        overlap = (gt_labeled[p_mask] > 0).sum() / max(1, p_mask.sum())
        if overlap >= min_overlap:
            tp_pred += 1

    lesion_rec = float(tp_gt / max(1, num_gt))
    lesion_prec = float(tp_pred / max(1, num_pred))
    denom = lesion_prec + lesion_rec
    lesion_f1 = float((2 * lesion_prec * lesion_rec) / denom) if denom > 1e-8 else 0.0

    return {
        "gt_volume_mm3": gt_vol_mm3,
        "pred_volume_mm3": pred_vol_mm3,
        "abs_volume_error_mm3": abs_vol_err,
        "rel_volume_error": rel_vol_err,
        "num_gt_lesions": int(num_gt),
        "num_pred_lesions": int(num_pred),
        "lesion_precision": lesion_prec,
        "lesion_recall": lesion_rec,
        "lesion_f1": lesion_f1,
    }


def compute_comprehensive_metrics(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute all voxel, lesion, and volumetric metrics."""
    p = (pred_vol.ravel() >= threshold).astype(bool)
    t = (gt_vol.ravel() >= threshold).astype(bool)

    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())

    gt_voxels = tp + fn
    pred_voxels = tp + fp

    eps = 1e-8
    dice = float(2 * tp / (2 * tp + fp + fn + eps)) if (gt_voxels + pred_voxels) > 0 else 1.0
    iou = float(tp / (tp + fp + fn + eps)) if (gt_voxels + pred_voxels) > 0 else 1.0
    prec = float(tp / (tp + fp + eps)) if pred_voxels > 0 else (1.0 if gt_voxels == 0 else 0.0)
    rec = float(tp / (tp + fn + eps)) if gt_voxels > 0 else (1.0 if pred_voxels == 0 else 0.0)
    spec = float(tn / (tn + fp + eps))

    voxel_vol = float(np.prod(voxel_spacing))
    hd_dict = compute_hd95(pred_vol, gt_vol, voxel_spacing=voxel_spacing, threshold=threshold)
    lesion_vol_dict = compute_lesion_and_volume_metrics(pred_vol, gt_vol, voxel_vol_mm3=voxel_vol)

    return {
        "dice": dice,
        "iou": iou,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "hd95_mm": hd_dict["hd95_mm"],
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "tn_voxels": tn,
        "gt_voxels": gt_voxels,
        "pred_voxels": pred_voxels,
        **lesion_vol_dict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Automated Brain Mask & Coordinate Transformation Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_brain_bounding_box(
    adc_vol: np.ndarray,
    margin: int = 5,
) -> Tuple[int, int, int, int]:
    """
    Extract 2D in-plane bounding box enclosing the 3D intracranial brain mask
    from the skull-stripped ADC volume with a safety margin.
    """
    # Background in skull-stripped BONBID is <= 0 or normalized noise < -1.5
    brain_mask = (adc_vol > 10.0) if adc_vol.max() > 50.0 else (adc_vol > -1.5)
    pos = np.where(brain_mask)

    if len(pos[0]) == 0:
        # Fallback to full matrix if mask empty
        H, W = adc_vol.shape[0], adc_vol.shape[1]
        return 0, H, 0, W

    ymin, ymax = pos[0].min(), pos[0].max()
    xmin, xmax = pos[1].min(), pos[1].max()

    H, W = adc_vol.shape[0], adc_vol.shape[1]
    y1 = max(0, int(ymin - margin))
    y2 = min(H, int(ymax + 1 + margin))
    x1 = max(0, int(xmin - margin))
    x2 = min(W, int(xmax + 1 + margin))

    return y1, y2, x1, x2


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inference Engines for E1, E4-A, E4-B
# ─────────────────────────────────────────────────────────────────────────────

def run_e1_inference(
    model: torch.nn.Module,
    adc_res: np.ndarray,
    z_res: np.ndarray,
    is_dann: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    E1: Reference benchmark with standard 128x128 full-volume center padding/cropping.
    Input channels: D1 [-Z-ADC, ADC, 0]
    Returns restored (H0, W0, D0) probability and binary prediction volumes.
    """
    H0, W0, D0 = adc_res.shape
    inv_z = -z_res
    zeros = np.zeros_like(adc_res)
    channels_3d = [inv_z, adc_res, zeros]

    prob_full = np.zeros((H0, W0, D0), dtype=np.float32)

    with torch.no_grad():
        for z in range(D0):
            # Crop or pad slice to (128, 128)
            ch_slices = [_crop_or_pad(ch[:, :, z], 128, 128) for ch in channels_3d]
            inp = torch.from_numpy(np.stack(ch_slices, axis=0)).unsqueeze(0).float().to(device)

            logits = model(inp, return_domain=False) if is_dann else model(inp)
            p_128 = torch.sigmoid(logits).squeeze().cpu().numpy()

            # Invert crop_or_pad back to (H0, W0)
            if H0 <= 128 and W0 <= 128:
                pad_top = (128 - H0) // 2
                pad_left = (128 - W0) // 2
                prob_full[:, :, z] = p_128[pad_top : pad_top + H0, pad_left : pad_left + W0]
            else:
                # If H0 or W0 was > 128, restore center slice
                sl_restored = np.zeros((H0, W0), dtype=np.float32)
                y_start = (H0 - 128) // 2 if H0 > 128 else 0
                x_start = (W0 - 128) // 2 if W0 > 128 else 0
                sl_restored[y_start:y_start+128, x_start:x_start+128] = p_128
                prob_full[:, :, z] = sl_restored

    pred_full = (prob_full >= 0.5).astype(np.float32)
    return prob_full, pred_full


def run_e4a_inference(
    model: torch.nn.Module,
    adc_res: np.ndarray,
    z_res: np.ndarray,
    bbox: Tuple[int, int, int, int],
    is_dann: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    E4-A: Brain Bounding-Box Cropping without Resizing.
    Crops to brain bbox, preserves pixel scale, pads to 128x128 only if necessary.
    Inverts padding and crop to restore into (H0, W0, D0).
    """
    H0, W0, D0 = adc_res.shape
    y1, y2, x1, x2 = bbox
    H_crop = y2 - y1
    W_crop = x2 - x1

    inv_z = -z_res
    zeros = np.zeros_like(adc_res)

    # Crop channels
    ch_crops = [ch[y1:y2, x1:x2, :] for ch in [inv_z, adc_res, zeros]]

    prob_full = np.zeros((H0, W0, D0), dtype=np.float32)

    pad_top = max(0, (128 - H_crop) // 2)
    pad_left = max(0, (128 - W_crop) // 2)

    with torch.no_grad():
        for z in range(D0):
            # Pad cropped slice to 128x128
            sl_128 = []
            for ch in ch_crops:
                sl = ch[:, :, z]
                if H_crop <= 128 and W_crop <= 128:
                    p = np.zeros((128, 128), dtype=np.float32)
                    p[pad_top : pad_top + H_crop, pad_left : pad_left + W_crop] = sl
                else:
                    p = _crop_or_pad(sl, 128, 128)
                sl_128.append(p)

            inp = torch.from_numpy(np.stack(sl_128, axis=0)).unsqueeze(0).float().to(device)
            logits = model(inp, return_domain=False) if is_dann else model(inp)
            p_128 = torch.sigmoid(logits).squeeze().cpu().numpy()

            # Invert padding back to (H_crop, W_crop)
            if H_crop <= 128 and W_crop <= 128:
                p_crop = p_128[pad_top : pad_top + H_crop, pad_left : pad_left + W_crop]
            else:
                p_crop = p_128

            # Invert crop back to (H0, W0)
            prob_full[y1:y2, x1:x2, z] = p_crop

    pred_full = (prob_full >= 0.5).astype(np.float32)
    return prob_full, pred_full


def run_e4b_inference(
    model: torch.nn.Module,
    adc_res: np.ndarray,
    z_res: np.ndarray,
    bbox: Tuple[int, int, int, int],
    is_dann: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    E4-B: Brain Bounding-Box Cropping with Resizing.
    Crops to brain bbox, resizes in-plane to (128, 128) via linear interpolation,
    performs inference, and applies inverse spatial transformation back to (H0, W0, D0).
    """
    H0, W0, D0 = adc_res.shape
    y1, y2, x1, x2 = bbox
    H_crop = y2 - y1
    W_crop = x2 - x1

    inv_z = -z_res
    zeros = np.zeros_like(adc_res)

    # Crop channels
    ch_crops = [ch[y1:y2, x1:x2, :] for ch in [inv_z, adc_res, zeros]]

    # Scaling factors to 128x128
    sy = 128.0 / H_crop
    sx = 128.0 / W_crop

    # In-plane resize using order=1 (continuous linear)
    ch_resized = [zoom(ch, (sy, sx, 1.0), order=1) for ch in ch_crops]

    prob_128 = np.zeros((128, 128, D0), dtype=np.float32)

    with torch.no_grad():
        for z in range(D0):
            sls = [ch[:, :, z] for ch in ch_resized]
            inp = torch.from_numpy(np.stack(sls, axis=0)).unsqueeze(0).float().to(device)
            logits = model(inp, return_domain=False) if is_dann else model(inp)
            p = torch.sigmoid(logits).squeeze().cpu().numpy()
            prob_128[:, :, z] = p

    # Invert spatial transformation: resize probability map back to (H_crop, W_crop)
    inv_sy = H_crop / 128.0
    inv_sx = W_crop / 128.0
    prob_crop = zoom(prob_128, (inv_sy, inv_sx, 1.0), order=1)

    # Fix potential 1-pixel rounding offset
    prob_crop = prob_crop[:H_crop, :W_crop, :]

    # Invert crop back into full (H0, W0, D0)
    prob_full = np.zeros((H0, W0, D0), dtype=np.float32)
    prob_full[y1:y2, x1:x2, :] = prob_crop
    pred_full = (prob_full >= 0.5).astype(np.float32)

    return prob_full, pred_full


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Panel Diagnostic Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def create_rgb_error_overlay(
    pred_slice: np.ndarray,
    gt_slice: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """RGB: Green=TP, Red=FP, Blue=FN."""
    H, W = pred_slice.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    p = (pred_slice >= 0.5)
    g = (gt_slice >= 0.5)
    tp = p & g
    fp = p & ~g
    fn = ~p & g
    overlay[tp, 1] = 1.0  # Green
    overlay[tp, 3] = alpha
    overlay[fp, 0] = 1.0  # Red
    overlay[fp, 3] = alpha
    overlay[fn, 2] = 1.0  # Blue
    overlay[fn, 3] = alpha
    return overlay


def plot_e1_e4_subject_comparison(
    subj: str,
    adc_res: np.ndarray,
    z_res: np.ndarray,
    gt_vol: np.ndarray,
    e1_prob: np.ndarray,
    e1_pred: np.ndarray,
    e4a_prob: np.ndarray,
    e4a_pred: np.ndarray,
    e4b_prob: np.ndarray,
    e4b_pred: np.ndarray,
    out_file: Path,
    n_slices: int = 4,
):
    """
    Generate comprehensive 8-row diagnostic figure comparing E1, E4-A, and E4-B:
      Row 1: ADC Map
      Row 2: Inverted Z-ADC (-Z) Map
      Row 3: Ground Truth Lesion Mask
      Row 4: E1 (Pad) Probability Heatmap
      Row 5: E4-A (Crop, No Resize) Probability Heatmap
      Row 6: E4-B (Crop + Resize) Probability Heatmap
      Row 7: E1 Error Map (TP=Green, FP=Red, FN=Blue)
      Row 8: E4-B Error Map (TP=Green, FP=Red, FN=Blue)
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    H, W, D = gt_vol.shape

    slice_lesion_counts = [int(gt_vol[:, :, z].sum()) for z in range(D)]
    ranked = np.argsort(slice_lesion_counts)[::-1]

    if max(slice_lesion_counts) > 0:
        top_slices = [int(s) for s in ranked if slice_lesion_counts[s] > 0]
        if len(top_slices) >= n_slices:
            selected_slices = sorted(top_slices[:n_slices])
        else:
            extras = [int(s) for s in ranked if s not in top_slices]
            selected_slices = sorted((top_slices + extras)[:n_slices])
    else:
        selected_slices = [int(x) for x in np.linspace(D * 0.2, D * 0.8, n_slices)]

    fig, axes = plt.subplots(8, n_slices, figsize=(3.3 * n_slices, 19.5), facecolor="#0d1117")

    row_names = [
        "1. ADC Map",
        "2. Inverted Z-ADC (-Z)",
        "3. Ground Truth Mask",
        "4. E1 Prob Heatmap",
        "5. E4-A Prob Heatmap",
        "6. E4-B Prob Heatmap",
        "7. E1 Error (TP/FP/FN)",
        "8. E4-B Error (TP/FP/FN)",
    ]

    for col, z in enumerate(selected_slices):
        adc_sl = adc_res[:, :, z]
        invz_sl = -z_res[:, :, z]
        gt_sl = gt_vol[:, :, z]

        # 1. ADC
        ax = axes[0][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        ax.set_title(f"Slice z={z} ({int(gt_sl.sum())} vx)", color="#58a6ff", fontsize=10, weight="bold")
        ax.axis("off")

        # 2. Inverted Z
        ax = axes[1][col]
        ax.imshow(invz_sl, cmap="magma", interpolation="bilinear")
        ax.axis("off")

        # 3. Ground Truth
        ax = axes[2][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        m_gt = np.zeros((H, W, 4), dtype=np.float32)
        m_gt[gt_sl >= 0.5, 0] = 1.0  # Red
        m_gt[gt_sl >= 0.5, 3] = 0.7
        ax.imshow(m_gt)
        ax.axis("off")

        # 4. E1 Prob
        ax = axes[3][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        ax.imshow(e1_prob[:, :, z], cmap="hot", alpha=0.65, vmin=0.0, vmax=1.0)
        ax.axis("off")

        # 5. E4-A Prob
        ax = axes[4][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        ax.imshow(e4a_prob[:, :, z], cmap="hot", alpha=0.65, vmin=0.0, vmax=1.0)
        ax.axis("off")

        # 6. E4-B Prob
        ax = axes[5][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        ax.imshow(e4b_prob[:, :, z], cmap="hot", alpha=0.65, vmin=0.0, vmax=1.0)
        ax.axis("off")

        # 7. E1 Error
        ax = axes[6][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        ax.imshow(create_rgb_error_overlay(e1_pred[:, :, z], gt_sl))
        ax.axis("off")

        # 8. E4-B Error
        ax = axes[7][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        ax.imshow(create_rgb_error_overlay(e4b_pred[:, :, z], gt_sl))
        ax.axis("off")

    for r, label_str in enumerate(row_names):
        axes[r][0].set_ylabel(label_str, color="#c9d1d9", fontsize=10, weight="bold")

    fig.suptitle(
        f"Phase 2 Preprocessing Comparison: Subject {subj}\n"
        f"E1 (Full Pad) vs. E4-A (Brain Crop, No Resize) vs. E4-B (Brain Crop + Resized)",
        color="white", fontsize=12, weight="bold", y=0.99
    )

    patches = [
        mpatches.Patch(color="green", alpha=0.8, label="True Positive (TP)"),
        mpatches.Patch(color="red",   alpha=0.8, label="False Positive (FP)"),
        mpatches.Patch(color="blue",  alpha=0.8, label="False Negative (FN)"),
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=3,
        facecolor="#161b22", edgecolor="#30363d", labelcolor="white",
        fontsize=10, bbox_to_anchor=(0.5, 0.005)
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_file, dpi=130, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Execution Workflow
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("PHASE 2, STEP 1: E1 (REFERENCE) VS E4-A (CROP) VS E4-B (CROP+RESIZE)")
    print("=" * 80)

    config_path = PROJECT_ROOT / "configs/config.yaml"
    base_ckpt_path = PROJECT_ROOT / "outputs/checkpoints/baseline_unet_best.pth"
    dann_ckpt_path = PROJECT_ROOT / "outputs/checkpoints/dann/dann_unet_best.pth"
    bonbid_root = PROJECT_ROOT / "data/BONBID-HIE"

    out_phase2 = PROJECT_ROOT / "outputs/phase2"
    log_dir = out_phase2 / "logs"
    viz_dir = out_phase2 / "visualizations"
    log_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")

    # Load frozen models
    model_base = build_unet(cfg).to(device)
    model_base.load_state_dict(torch.load(base_ckpt_path, map_location=device)["model_state"])
    model_base.eval()
    for p in model_base.parameters():
        p.requires_grad = False

    model_dann = build_dann(cfg=cfg, device=device)
    model_dann.load_state_dict(torch.load(dann_ckpt_path, map_location=device)["model_state"])
    model_dann.eval()
    for p in model_dann.parameters():
        p.requires_grad = False

    print("Checkpoints loaded and verified in strict evaluation mode.")

    val_samples = discover_bonbid(str(bonbid_root), split="val")
    print(f"Discovered {len(val_samples)} validation subjects.")

    target_sp = (2.0, 2.0, 2.0)

    all_subject_rows = []
    comparison_summary = []

    models = [
        ("Baseline_UNet", model_base, False),
        ("DANN_UNet", model_dann, True),
    ]

    experiments = [
        ("E1_Reference", "E1: Full-Volume Center Pad (128x128)"),
        ("E4A_Crop_NoResize", "E4-A: Brain Bounding-Box Crop (Preserve Scale)"),
        ("E4B_Crop_Resize", "E4-B: Brain Bounding-Box Crop (Resize to 128x128)"),
    ]

    for model_name, model_obj, is_dann in models:
        print(f"\n==================== Model: {model_name} ====================")
        for exp_key, exp_desc in experiments:
            print(f"\n--- Running Experiment: {exp_desc} ---")
            exp_metrics_list = []

            for s in val_samples:
                subj = s["subject"]
                adc_raw, adc_meta = load_nifti(s["image_paths"]["adc"])
                z_raw, _ = load_nifti(s["image_paths"]["z-adc"])
                msk_raw, _ = load_nifti(s["mask_path"])
                native_sp = adc_meta.get_zooms()

                # Preprocessing to reference space (2.0 mm isotropic)
                adc_norm = normalize_volume(adc_raw, method="zscore")
                z_norm = normalize_volume(z_raw, method="zscore")

                adc_res = resample_to_target_spacing(adc_norm, native_sp, target_sp, order=1)
                z_res   = resample_to_target_spacing(z_norm, native_sp, target_sp, order=1)
                msk_res = resample_to_target_spacing(msk_raw, native_sp, target_sp, order=0)

                # Brain bounding box with 5-voxel safety margin
                bbox = get_brain_bounding_box(adc_res, margin=5)

                if exp_key == "E1_Reference":
                    prob_vol, pred_vol = run_e1_inference(model_obj, adc_res, z_res, is_dann=is_dann, device=device)
                elif exp_key == "E4A_Crop_NoResize":
                    prob_vol, pred_vol = run_e4a_inference(model_obj, adc_res, z_res, bbox=bbox, is_dann=is_dann, device=device)
                elif exp_key == "E4B_Crop_Resize":
                    prob_vol, pred_vol = run_e4b_inference(model_obj, adc_res, z_res, bbox=bbox, is_dann=is_dann, device=device)

                # Compute comprehensive metrics
                m = compute_comprehensive_metrics(pred_vol, msk_res, voxel_spacing=target_sp, threshold=0.5)

                row = {
                    "Model": model_name,
                    "Experiment": exp_key,
                    "Subject": subj,
                    **m,
                    "bbox_y": f"{bbox[0]}:{bbox[1]}",
                    "bbox_x": f"{bbox[2]}:{bbox[3]}",
                }
                all_subject_rows.append(row)
                exp_metrics_list.append(m)

                print(
                    f"  [{subj}] Dice={m['dice']:.4f} | IoU={m['iou']:.4f} | Prec={m['precision']:.4f} | "
                    f"Rec={m['recall']:.4f} | HD95={m['hd95_mm']:.1f} mm | LesionF1={m['lesion_f1']:.4f} | "
                    f"RelVolErr={m['rel_volume_error']:.2f} | GT={m['gt_voxels']} vx | Pred={m['pred_voxels']} vx"
                )

                # Save probability maps & prediction masks for each subject
                npz_out = viz_dir / f"{model_name}_{exp_key}_{subj}_masks.npz"
                np.savez_compressed(
                    npz_out,
                    prob=prob_vol.astype(np.float32),
                    pred=pred_vol.astype(np.uint8),
                    gt=msk_res.astype(np.uint8),
                )

            # Summarize experiment
            def agg(key):
                vals = [x[key] for x in exp_metrics_list if not np.isnan(x[key])]
                return float(np.mean(vals)), float(np.std(vals))

            d_m, d_s = agg("dice")
            i_m, i_s = agg("iou")
            p_m, p_s = agg("precision")
            r_m, r_s = agg("recall")
            s_m, s_s = agg("specificity")
            h_m, h_s = agg("hd95_mm")
            f1_m, f1_s = agg("lesion_f1")
            v_m, v_s = agg("rel_volume_error")

            comparison_summary.append({
                "Model": model_name,
                "Experiment": exp_key,
                "Description": exp_desc,
                "Mean_Dice": f"{d_m:.4f}", "Std_Dice": f"{d_s:.4f}",
                "Mean_IoU": f"{i_m:.4f}",
                "Mean_Precision": f"{p_m:.4f}",
                "Mean_Recall": f"{r_m:.4f}",
                "Mean_Specificity": f"{s_m:.4f}",
                "Mean_HD95_mm": f"{h_m:.1f}",
                "Mean_Lesion_F1": f"{f1_m:.4f}",
                "Mean_Rel_Vol_Error": f"{v_m:.2f}",
            })

    # Save per-subject CSV
    subj_csv = log_dir / "e1_e4_subject_metrics.csv"
    with open(subj_csv, "w", newline="", encoding="utf-8") as f:
        keys = list(all_subject_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_subject_rows)
    print(f"\nSaved subject metrics CSV: {subj_csv.name}")

    # Save comparison CSV
    comp_csv = log_dir / "e1_e4_comparison.csv"
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        keys = list(comparison_summary[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(comparison_summary)
    print(f"Saved comparison summary CSV: {comp_csv.name}")

    # Generate visual comparison panels for all subjects under Baseline
    print("\nGenerating visual comparison panels...")
    for s in val_samples:
        subj = s["subject"]
        adc_raw, adc_meta = load_nifti(s["image_paths"]["adc"])
        z_raw, _ = load_nifti(s["image_paths"]["z-adc"])
        msk_raw, _ = load_nifti(s["mask_path"])
        native_sp = adc_meta.get_zooms()

        adc_res = resample_to_target_spacing(normalize_volume(adc_raw), native_sp, target_sp, order=1)
        z_res   = resample_to_target_spacing(normalize_volume(z_raw), native_sp, target_sp, order=1)
        msk_res = resample_to_target_spacing(msk_raw, native_sp, target_sp, order=0)
        bbox = get_brain_bounding_box(adc_res, margin=5)

        # Baseline
        e1_p, e1_b = run_e1_inference(model_base, adc_res, z_res, is_dann=False, device=device)
        e4a_p, e4a_b = run_e4a_inference(model_base, adc_res, z_res, bbox=bbox, is_dann=False, device=device)
        e4b_p, e4b_b = run_e4b_inference(model_base, adc_res, z_res, bbox=bbox, is_dann=False, device=device)

        panel_path = viz_dir / f"baseline_e1_e4_comparison_{subj}.png"
        plot_e1_e4_subject_comparison(
            subj=subj, adc_res=adc_res, z_res=z_res, gt_vol=msk_res,
            e1_prob=e1_p, e1_pred=e1_b, e4a_prob=e4a_p, e4a_pred=e4a_b,
            e4b_prob=e4b_p, e4b_pred=e4b_b, out_file=panel_path, n_slices=4
        )
        print(f"  -> Generated panel: {panel_path.name}")

    print("\n" + "=" * 80)
    print("PHASE 2 STEP 1 COMPLETE: E1, E4-A, AND E4-B EVALUATED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

