"""
scientific_evaluation.py
========================
Rigorous Scientific Evaluation and Comparison of Adult Baseline U-Net vs DANN U-Net
on the Neonatal BONBID-HIE Validation Cohort (4 Official Subjects).

Authored from the perspective of a Senior Deep Learning Researcher in Medical Imaging & UDA.

Key Principles:
  1. Strict Checkpoint Safety: Baseline and DANN checkpoints are verified via SHA-256 before
     and after inference; weights and batch-norm running statistics remain strictly frozen.
  2. Identical Preprocessing: Both models use identical isotropic 2.0 mm resampling, Z-score
     normalization, center crop/pad to 128x128, and primary mapping [ADC, Z-ADC, Z-ADC].
  3. Metric Rigor: Calculates Dice, IoU, Precision, Recall/Sensitivity, Specificity, and HD95 (mm and voxels).
  4. Statistical Rigor: Paired t-test and Wilcoxon signed-rank test comparing Baseline vs DANN.
  5. Multi-Threshold Sensitivity: Sweeps tau in [0.1, 0.9] to inspect operating curves.
  6. Modality Shift Ablation: Evaluates both primary mapping and surrogate inverted-Z mapping.
  7. High-Fidelity Scientific Visualizations: Multi-panel comparisons, error maps, bar charts, and PR curves.
"""

from __future__ import annotations

import argparse
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
from scipy.ndimage import distance_transform_edt
from scipy import stats
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
from evaluate_neonatal_zeroshot import adapt_channels, compute_volume_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("scientific_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

logger = setup_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Verification
# ─────────────────────────────────────────────────────────────────────────────

def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file for integrity verification."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# HD95 Calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_hd95(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute 95th-percentile Hausdorff Distance (HD95) in both millimeters and voxels.
    Uses distance_transform_edt on surface boundary voxels.
    """
    p = (pred_vol >= threshold).astype(bool)
    t = (gt_vol >= threshold).astype(bool)

    if not p.any() or not t.any():
        return {"hd95_mm": float("nan"), "hd95_voxels": float("nan")}

    # HD95 in physical millimeters
    dt_p_mm = distance_transform_edt(~p, sampling=voxel_spacing)
    dt_t_mm = distance_transform_edt(~t, sampling=voxel_spacing)
    h_pt_mm = dt_t_mm[p]
    h_tp_mm = dt_p_mm[t]
    hd95_mm = float(np.percentile(np.concatenate([h_pt_mm, h_tp_mm]), 95))

    # HD95 in voxel units
    dt_p_vox = distance_transform_edt(~p)
    dt_t_vox = distance_transform_edt(~t)
    h_pt_vox = dt_t_vox[p]
    h_tp_vox = dt_p_vox[t]
    hd95_vox = float(np.percentile(np.concatenate([h_pt_vox, h_tp_vox]), 95))

    return {"hd95_mm": hd95_mm, "hd95_voxels": hd95_vox}


# ─────────────────────────────────────────────────────────────────────────────
# Volumetric Inference Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_model_inference(
    model: torch.nn.Module,
    sample: Dict,
    target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    image_size: Tuple[int, int] = (128, 128),
    channel_mapping: str = "primary_adc_z_z",
    threshold: float = 0.5,
    is_dann: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Run 3D volumetric slice-by-slice inference.
    Returns predicted volumes, probabilities, ground-truth, and metrics.
    """
    subj = sample["subject"]
    img_paths = sample.get("image_paths", {})
    mask_path = sample.get("mask_path")

    adc_raw, adc_meta = load_nifti(img_paths["adc"])
    z_raw, _ = load_nifti(img_paths["z-adc"])
    mask_raw, _ = load_nifti(mask_path) if mask_path else (None, None)

    native_spacing = adc_meta.get_zooms()
    native_shape = adc_raw.shape

    # Preprocessing
    adc_norm = normalize_volume(adc_raw, method="zscore")
    z_norm = normalize_volume(z_raw, method="zscore")

    adc_res = resample_to_target_spacing(adc_norm, native_spacing, target_spacing, order=1)
    z_res = resample_to_target_spacing(z_norm, native_spacing, target_spacing, order=1)

    mask_res = None
    if mask_raw is not None:
        m = resample_to_target_spacing(mask_raw, native_spacing, target_spacing, order=0)
        mask_res = (m > 0.5).astype(np.float32)

    channels_3d = adapt_channels(adc_res, z_res, mapping_name=channel_mapping)

    H, W = image_size
    depth = adc_res.shape[2]

    pred_vol = np.zeros((H, W, depth), dtype=np.float32)
    prob_vol = np.zeros((H, W, depth), dtype=np.float32)
    gt_vol = np.zeros((H, W, depth), dtype=np.float32) if mask_res is not None else None
    input_slices = []

    with torch.no_grad():
        for z in range(depth):
            ch_slices = [_crop_or_pad(ch[:, :, z], H, W) for ch in channels_3d]
            slice_tensor = np.stack(ch_slices, axis=0)
            input_slices.append(slice_tensor)

            inp = torch.from_numpy(slice_tensor).unsqueeze(0).float().to(device)

            if is_dann:
                logits = model(inp, return_domain=False)
            else:
                logits = model(inp)

            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (prob >= threshold).astype(np.float32)

            prob_vol[:, :, z] = prob
            pred_vol[:, :, z] = pred
            if mask_res is not None:
                gt_vol[:, :, z] = _crop_or_pad(mask_res[:, :, z], H, W)

    # Metrics
    metrics = {}
    if gt_vol is not None:
        metrics = compute_volume_metrics(pred_vol, gt_vol, threshold=0.5)
        hd_dict = compute_hd95(pred_vol, gt_vol, voxel_spacing=target_spacing, threshold=0.5)
        metrics.update(hd_dict)
        # Physical volume calculation (mm3 = voxels * spacing_x * spacing_y * spacing_z)
        voxel_vol_mm3 = float(np.prod(target_spacing))
        metrics["gt_volume_mm3"] = metrics["gt_voxels"] * voxel_vol_mm3
        metrics["pred_volume_mm3"] = metrics["pred_voxels"] * voxel_vol_mm3

    return {
        "subject": subj,
        "depth": depth,
        "native_shape": list(native_shape),
        "native_spacing": [float(x) for x in native_spacing],
        "target_shape": [H, W, depth],
        "target_spacing": list(target_spacing),
        "channel_mapping": channel_mapping,
        "threshold": threshold,
        "adc_res": adc_res,
        "z_res": z_res,
        "prob_vol": prob_vol,
        "pred_vol": pred_vol,
        "gt_vol": gt_vol,
        "input_slices": input_slices,
        "metrics": metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Utilities
# ─────────────────────────────────────────────────────────────────────────────

def create_rgb_error_overlay(
    pred_slice: np.ndarray,
    gt_slice: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Create RGB overlay:
      Green: True Positive (TP)
      Red:   False Positive (FP)
      Blue:  False Negative (FN)
    """
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


def plot_scientific_comparison_panel(
    base_res: Dict,
    dann_res: Dict,
    out_file: Path,
    n_slices: int = 4,
):
    """
    Generate publication-ready 7-row comparative panel:
      Row 1: ADC slice (grayscale)
      Row 2: Z-ADC slice (grayscale)
      Row 3: Ground Truth mask (overlay)
      Row 4: Baseline U-Net prediction
      Row 5: DANN U-Net prediction
      Row 6: Baseline Error Map (TP=Green, FP=Red, FN=Blue)
      Row 7: DANN Error Map (TP=Green, FP=Red, FN=Blue)
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    subj = base_res["subject"]
    depth = base_res["depth"]
    gt_vol = base_res["gt_vol"]
    adc_res = base_res["adc_res"]
    z_res = base_res["z_res"]
    base_pred = base_res["pred_vol"]
    dann_pred = dann_res["pred_vol"]
    H, W, _ = base_pred.shape

    # Select representative slices with maximal lesion presence
    slice_lesion_counts = [int(gt_vol[:, :, z].sum()) for z in range(depth)]
    ranked_slices = np.argsort(slice_lesion_counts)[::-1]

    if max(slice_lesion_counts) > 0:
        top_slices = [int(s) for s in ranked_slices if slice_lesion_counts[s] > 0]
        if len(top_slices) >= n_slices:
            selected_slices = sorted(top_slices[:n_slices])
        else:
            extras = [int(s) for s in ranked_slices if s not in top_slices]
            selected_slices = sorted((top_slices + extras)[:n_slices])
    else:
        selected_slices = [int(x) for x in np.linspace(depth * 0.2, depth * 0.8, n_slices)]

    fig, axes = plt.subplots(7, n_slices, figsize=(3.2 * n_slices, 17.5), facecolor="#0d1117")

    row_titles = [
        "1. ADC Map",
        "2. Z-ADC Map",
        "3. Ground Truth",
        "4. Baseline U-Net",
        "5. DANN Adapted",
        "6. Baseline Error",
        "7. DANN Error",
    ]

    for col, sl_idx in enumerate(selected_slices):
        adc_sl = _crop_or_pad(adc_res[:, :, sl_idx], H, W)
        z_sl   = _crop_or_pad(z_res[:, :, sl_idx], H, W)
        gt_sl  = gt_vol[:, :, sl_idx]
        b_sl   = base_pred[:, :, sl_idx]
        d_sl   = dann_pred[:, :, sl_idx]

        # Row 1: ADC
        ax = axes[0][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        ax.set_title(f"Slice z={sl_idx} (Lesion: {int(gt_sl.sum())} vx)", color="#58a6ff", fontsize=11, weight="bold")
        ax.axis("off")

        # Row 2: Z-ADC
        ax = axes[1][col]
        ax.imshow(z_sl, cmap="coolwarm", interpolation="bilinear")
        ax.axis("off")

        # Row 3: Ground Truth
        ax = axes[2][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        gt_mask = np.zeros((H, W, 4), dtype=np.float32)
        gt_mask[gt_sl >= 0.5, 0] = 1.0  # Red
        gt_mask[gt_sl >= 0.5, 3] = 0.7
        ax.imshow(gt_mask)
        ax.axis("off")

        # Row 4: Baseline Pred
        ax = axes[3][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        b_mask = np.zeros((H, W, 4), dtype=np.float32)
        b_mask[b_sl >= 0.5, 0] = 1.0  # Orange-yellow
        b_mask[b_sl >= 0.5, 1] = 0.6
        b_mask[b_sl >= 0.5, 3] = 0.7
        ax.imshow(b_mask)
        ax.axis("off")

        # Row 5: DANN Pred
        ax = axes[4][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        d_mask = np.zeros((H, W, 4), dtype=np.float32)
        d_mask[d_sl >= 0.5, 1] = 0.9  # Cyan-green
        d_mask[d_sl >= 0.5, 2] = 0.9
        d_mask[d_sl >= 0.5, 3] = 0.7
        ax.imshow(d_mask)
        ax.axis("off")

        # Row 6: Baseline Error Overlay
        ax = axes[5][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        base_err = create_rgb_error_overlay(b_sl, gt_sl, alpha=0.7)
        ax.imshow(base_err)
        ax.axis("off")

        # Row 7: DANN Error Overlay
        ax = axes[6][col]
        ax.imshow(adc_sl, cmap="gray", interpolation="bilinear")
        dann_err = create_rgb_error_overlay(d_sl, gt_sl, alpha=0.7)
        ax.imshow(dann_err)
        ax.axis("off")

    for r, label in enumerate(row_titles):
        axes[r][0].set_ylabel(label, color="#c9d1d9", fontsize=11, weight="bold")

    b_m = base_res["metrics"]
    d_m = dann_res["metrics"]
    fig.suptitle(
        f"Scientific Lesion Segmentation Comparison: Subject {subj}\n"
        f"Baseline U-Net: Dice={b_m['dice']:.4f}, IoU={b_m['iou']:.4f}, Prec={b_m['precision']:.4f}, Rec={b_m['recall']:.4f}, HD95={b_m['hd95_mm']:.1f} mm\n"
        f"DANN Adapted:   Dice={d_m['dice']:.4f}, IoU={d_m['iou']:.4f}, Prec={d_m['precision']:.4f}, Rec={d_m['recall']:.4f}, HD95={d_m['hd95_mm']:.1f} mm",
        color="white", fontsize=12, weight="bold", y=0.99
    )

    patches = [
        mpatches.Patch(color="green", alpha=0.8, label="True Positive (TP) - Hit"),
        mpatches.Patch(color="red",   alpha=0.8, label="False Positive (FP) - False Alarm"),
        mpatches.Patch(color="blue",  alpha=0.8, label="False Negative (FN) - Missed Lesion"),
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=3,
        facecolor="#161b22", edgecolor="#30363d", labelcolor="white",
        fontsize=11, bbox_to_anchor=(0.5, 0.005)
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_file, dpi=130, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison_barchart(
    base_summary: Dict,
    dann_summary: Dict,
    out_file: Path,
):
    """Grouped bar chart comparing Baseline vs DANN across all key segmentation metrics."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = ["dice", "iou", "precision", "recall", "specificity"]
    labels = ["Dice", "IoU", "Precision", "Recall (Sens)", "Specificity"]

    base_means = [base_summary[m]["mean"] for m in metric_keys]
    base_stds  = [base_summary[m]["std"] for m in metric_keys]
    dann_means = [dann_summary[m]["mean"] for m in metric_keys]
    dann_stds  = [dann_summary[m]["std"] for m in metric_keys]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2, base_means, width, yerr=base_stds,
        capsize=5, label="Baseline U-Net (Zero-Shot)",
        color="#ff7b72", alpha=0.85, edgecolor="#f85149"
    )
    rects2 = ax.bar(
        x + width / 2, dann_means, width, yerr=dann_stds,
        capsize=5, label="DANN Adapted U-Net (Epoch 6)",
        color="#58a6ff", alpha=0.85, edgecolor="#1f6feb"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11, weight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (0.0 to 1.0)", color="white", fontsize=11, weight="bold")
    ax.set_title(
        "Baseline U-Net vs DANN U-Net on Neonatal Cohort (BONBID-HIE Val)\n"
        "Adult-to-Neonatal Domain Adaptation Performance Comparison",
        color="white", fontsize=12, weight="bold", pad=12
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")
    ax.grid(axis="y", color="#30363d", linestyle="--", alpha=0.6)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=11)

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.03, f"{h:.3f}",
                ha="center", va="bottom", color="#ff7b72", fontsize=9, weight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.03, f"{h:.3f}",
                ha="center", va="bottom", color="#58a6ff", fontsize=9, weight="bold")

    plt.tight_layout()
    plt.savefig(out_file, dpi=130, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity_curves(
    base_thresh_results: Dict[float, Dict],
    dann_thresh_results: Dict[float, Dict],
    out_file: Path,
):
    """Plot Dice, Precision, Recall curves across threshold sweeps tau in [0.1, 0.9]."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    thresholds = sorted(base_thresh_results.keys())

    base_dice = [base_thresh_results[t]["dice"] for t in thresholds]
    base_prec = [base_thresh_results[t]["precision"] for t in thresholds]
    base_rec  = [base_thresh_results[t]["recall"] for t in thresholds]

    dann_dice = [dann_thresh_results[t]["dice"] for t in thresholds]
    dann_prec = [dann_thresh_results[t]["precision"] for t in thresholds]
    dann_rec  = [dann_thresh_results[t]["recall"] for t in thresholds]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor="#0d1117")

    titles = ["Dice vs Threshold", "Precision vs Threshold", "Recall vs Threshold"]
    y_labels = ["Mean Dice Score", "Mean Precision", "Mean Recall"]

    curves = [
        (base_dice, dann_dice),
        (base_prec, dann_prec),
        (base_rec, dann_rec),
    ]

    for ax, title, ylabel, (b_curve, d_curve) in zip(axes, titles, y_labels, curves):
        ax.set_facecolor("#161b22")
        ax.plot(thresholds, b_curve, marker="o", color="#ff7b72", linewidth=2.2, label="Baseline U-Net")
        ax.plot(thresholds, d_curve, marker="s", color="#58a6ff", linewidth=2.2, label="DANN Adapted")
        ax.set_title(title, color="white", fontsize=11, weight="bold")
        ax.set_xlabel("Sigmoid Binarization Threshold (τ)", color="white", fontsize=10)
        ax.set_ylabel(ylabel, color="white", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(thresholds)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#30363d")
        ax.grid(color="#30363d", linestyle="--", alpha=0.6)
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    fig.suptitle(
        "Threshold Sensitivity Analysis (τ ∈ [0.1, 0.9]): Precision-Recall Trade-off Dynamics",
        color="white", fontsize=12, weight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=130, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)


def plot_lesion_volume_vs_dice(
    base_results: List[Dict],
    dann_results: List[Dict],
    out_file: Path,
):
    """Scatter plot demonstrating the Small-Lesion Bias and volume dependence."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    subjects = [r["subject"] for r in base_results]
    gt_vols = [r["metrics"]["gt_voxels"] for r in base_results]
    base_dices = [r["metrics"]["dice"] for r in base_results]
    dann_dices = [r["metrics"]["dice"] for r in dann_results]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    ax.scatter(gt_vols, base_dices, color="#ff7b72", s=120, edgecolors="white", linewidth=1.5, label="Baseline U-Net", zorder=3)
    ax.scatter(gt_vols, dann_dices, color="#58a6ff", s=120, edgecolors="white", linewidth=1.5, label="DANN Adapted", zorder=3)

    for i, subj in enumerate(subjects):
        ax.annotate(f"{subj}\n({gt_vols[i]} vx)", (gt_vols[i], max(base_dices[i], dann_dices[i]) + 0.02),
                    color="white", fontsize=9, ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Ground-Truth Lesion Volume (Voxel Count, Log Scale)", color="white", fontsize=11, weight="bold")
    ax.set_ylabel("Dice Coefficient", color="white", fontsize=11, weight="bold")
    ax.set_ylim(-0.05, 0.6)
    ax.set_title(
        "Impact of Lesion Volume on Segmentation Performance\n"
        "(Demonstrating Extreme Class-Imbalance Penalty in Neonatal HIE)",
        color="white", fontsize=12, weight="bold", pad=12
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")
    ax.grid(color="#30363d", linestyle="--", alpha=0.6)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_file, dpi=130, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main Scientific Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scientific Evaluation of Baseline U-Net vs DANN")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/config.yaml"))
    parser.add_argument("--base-checkpoint", default=str(PROJECT_ROOT / "outputs/checkpoints/baseline_unet_best.pth"))
    parser.add_argument("--dann-checkpoint", default=str(PROJECT_ROOT / "outputs/checkpoints/dann/dann_unet_best.pth"))
    parser.add_argument("--bonbid-root", default=str(PROJECT_ROOT / "data/BONBID-HIE"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config_path = Path(args.config)
    base_ckpt_path = Path(args.base_checkpoint)
    dann_ckpt_path = Path(args.dann_checkpoint)
    bonbid_root = Path(args.bonbid_root)

    log_dir = PROJECT_ROOT / "outputs/logs"
    viz_dir = PROJECT_ROOT / "outputs/visualizations/scientific_evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "scientific_evaluation.log"
    global logger
    logger = setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("SCIENTIFIC EVALUATION & BENCHMARK: BASELINE U-NET VS DANN ADAPTATION")
    logger.info("Adult-to-Neonatal Ischemic Stroke Lesion Segmentation using Domain Adaptation")
    logger.info("=" * 80)

    # 1. Verify Checkpoints & Integrity
    assert base_ckpt_path.exists(), f"Baseline checkpoint missing: {base_ckpt_path}"
    assert dann_ckpt_path.exists(), f"DANN checkpoint missing: {dann_ckpt_path}"

    base_hash = compute_file_hash(base_ckpt_path)
    dann_hash = compute_file_hash(dann_ckpt_path)

    logger.info("--- CHECKPOINT INTEGRITY VERIFICATION ---")
    logger.info(f"Baseline Checkpoint: {base_ckpt_path.name}")
    logger.info(f"  • Size:   {base_ckpt_path.stat().st_size:,} bytes")
    logger.info(f"  • SHA256: {base_hash}")
    logger.info(f"DANN Checkpoint:     {dann_ckpt_path.name}")
    logger.info(f"  • Size:   {dann_ckpt_path.stat().st_size:,} bytes")
    logger.info(f"  • SHA256: {dann_hash}")

    # 2. Load Config & Device
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    logger.info(f"Execution Device: {device} (Deterministic CPU Evaluation)")

    # 3. Instantiate Models in Strict Evaluation Mode
    model_base = build_unet(cfg).to(device)
    ckpt_b = torch.load(base_ckpt_path, map_location=device)
    model_base.load_state_dict(ckpt_b["model_state"])
    model_base.eval()
    for p in model_base.parameters():
        p.requires_grad = False

    model_dann = build_dann(cfg=cfg, device=device)
    ckpt_d = torch.load(dann_ckpt_path, map_location=device)
    model_dann.load_state_dict(ckpt_d["model_state"])
    model_dann.eval()
    for p in model_dann.parameters():
        p.requires_grad = False

    logger.info("Models loaded successfully in eval mode (torch.no_grad, requires_grad=False).")

    # 4. Discover Validation Cohort
    val_samples = discover_bonbid(str(bonbid_root), split="val")
    logger.info(f"\nDiscovered {len(val_samples)} validation subjects in {bonbid_root}/BONBID2023_Val:")
    for s in val_samples:
        logger.info(f"  • {s['subject']}: ADC={Path(s['image_paths']['adc']).name}, Z-ADC={Path(s['image_paths']['z-adc']).name}")

    target_sp = tuple(cfg.get("data", {}).get("target_spacing", [2.0, 2.0, 2.0]))
    image_size = tuple(cfg.get("data", {}).get("image_size", [128, 128]))

    # 5. Evaluate Both Models Under Primary Mapping [ADC, Z-ADC, Z-ADC]
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION 1: PRIMARY CHANNEL MAPPING [ADC, Z-ADC, Z-ADC]")
    logger.info("=" * 80)

    base_results = []
    dann_results = []

    for s in val_samples:
        subj = s["subject"]
        logger.info(f"\nEvaluating Subject: {subj} ...")

        # Baseline
        b_res = run_model_inference(
            model=model_base, sample=s, target_spacing=target_sp,
            image_size=image_size, channel_mapping="primary_adc_z_z",
            threshold=args.threshold, is_dann=False, device=device,
        )
        base_results.append(b_res)
        bm = b_res["metrics"]
        logger.info(
            f"  [Baseline] GT={bm['gt_voxels']:5d} vx ({bm['gt_volume_mm3']:.0f} mm³) | "
            f"Pred={bm['pred_voxels']:5d} vx | Dice={bm['dice']:.4f} | IoU={bm['iou']:.4f} | "
            f"Prec={bm['precision']:.4f} | Rec={bm['recall']:.4f} | Spec={bm['specificity']:.4f} | "
            f"HD95={bm['hd95_mm']:.1f} mm"
        )

        # DANN
        d_res = run_model_inference(
            model=model_dann, sample=s, target_spacing=target_sp,
            image_size=image_size, channel_mapping="primary_adc_z_z",
            threshold=args.threshold, is_dann=True, device=device,
        )
        dann_results.append(d_res)
        dm = d_res["metrics"]
        logger.info(
            f"  [DANN    ] GT={dm['gt_voxels']:5d} vx ({dm['gt_volume_mm3']:.0f} mm³) | "
            f"Pred={dm['pred_voxels']:5d} vx | Dice={dm['dice']:.4f} | IoU={dm['iou']:.4f} | "
            f"Prec={dm['precision']:.4f} | Rec={dm['recall']:.4f} | Spec={dm['specificity']:.4f} | "
            f"HD95={dm['hd95_mm']:.1f} mm"
        )

        # Generate comparative visualization panel
        panel_file = viz_dir / f"comparison_panel_{subj}.png"
        plot_scientific_comparison_panel(b_res, d_res, panel_file, n_slices=4)
        logger.info(f"  -> Generated visualization panel: {panel_file.name}")

    # Compute Summary Statistics
    def summarize_metrics(results: List[Dict]) -> Dict[str, Dict[str, float]]:
        keys = ["dice", "iou", "precision", "recall", "sensitivity", "specificity", "hd95_mm", "hd95_voxels"]
        summary = {}
        for k in keys:
            vals = [r["metrics"][k] for r in results if not np.isnan(r["metrics"][k])]
            if vals:
                summary[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            else:
                summary[k] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return summary

    base_summary = summarize_metrics(base_results)
    dann_summary = summarize_metrics(dann_results)

    logger.info("\n" + "=" * 80)
    logger.info("COHORT SUMMARY METRICS (PRIMARY MAPPING: [ADC, Z-ADC, Z-ADC])")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<14} | {'Baseline U-Net (Mean ± Std)':<30} | {'DANN U-Net (Mean ± Std)':<30} | {'Delta (DANN - Base)':<18}")
    logger.info("-" * 98)
    for k in ["dice", "iou", "precision", "recall", "specificity", "hd95_mm"]:
        bm = base_summary[k]
        dm = dann_summary[k]
        delta = dm["mean"] - bm["mean"]
        sign = "+" if delta >= 0 else ""
        logger.info(f"{k:<14} | {bm['mean']:.4f} ± {bm['std']:.4f} ({bm['min']:.4f} - {bm['max']:.4f})  | {dm['mean']:.4f} ± {dm['std']:.4f} ({dm['min']:.4f} - {dm['max']:.4f})  | {sign}{delta:.4f}")

    # 6. Statistical Significance Testing (Paired t-test and Wilcoxon signed-rank)
    logger.info("\n" + "=" * 80)
    logger.info("STATISTICAL SIGNIFICANCE TESTING (N=4 PAIRED SUBJECTS)")
    logger.info("=" * 80)

    stat_tests = {}
    for k in ["dice", "iou", "precision", "recall", "specificity", "hd95_mm"]:
        b_vals = np.array([r["metrics"][k] for r in base_results])
        d_vals = np.array([r["metrics"][k] for r in dann_results])
        # Mask out NaNs
        valid = ~(np.isnan(b_vals) | np.isnan(d_vals))
        b_v = b_vals[valid]
        d_v = d_vals[valid]

        t_stat, t_pval = stats.ttest_rel(d_v, b_v)
        try:
            w_stat, w_pval = stats.wilcoxon(d_v, b_v)
        except Exception:
            w_stat, w_pval = float("nan"), float("nan")

        stat_tests[k] = {
            "t_statistic": float(t_stat),
            "t_pvalue": float(t_pval),
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_pvalue": float(w_pval),
        }
        logger.info(
            f"{k:<12}: Paired t-stat = {t_stat:+.4f} (p = {t_pval:.4f}) | "
            f"Wilcoxon stat = {w_stat:.1f} (p = {w_pval:.4f})"
        )

    # 7. Threshold Sensitivity Sweep tau in [0.1, 0.9]
    logger.info("\n" + "=" * 80)
    logger.info("THRESHOLD SENSITIVITY SWEEP (τ ∈ [0.1, 0.9])")
    logger.info("=" * 80)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    base_thresh_res = {}
    dann_thresh_res = {}

    for t in thresholds:
        b_dices, b_precs, b_recs = [], [], []
        d_dices, d_precs, d_recs = [], [], []

        for b_r, d_r in zip(base_results, dann_results):
            # Compute metrics for baseline at threshold t
            bm_t = compute_volume_metrics(b_r["prob_vol"], b_r["gt_vol"], threshold=t)
            b_dices.append(bm_t["dice"])
            b_precs.append(bm_t["precision"])
            b_recs.append(bm_t["recall"])

            # Compute metrics for DANN at threshold t
            dm_t = compute_volume_metrics(d_r["prob_vol"], d_r["gt_vol"], threshold=t)
            d_dices.append(dm_t["dice"])
            d_precs.append(dm_t["precision"])
            d_recs.append(dm_t["recall"])

        base_thresh_res[t] = {
            "dice": float(np.mean(b_dices)),
            "precision": float(np.mean(b_precs)),
            "recall": float(np.mean(b_recs)),
        }
        dann_thresh_res[t] = {
            "dice": float(np.mean(d_dices)),
            "precision": float(np.mean(d_precs)),
            "recall": float(np.mean(d_recs)),
        }

        logger.info(
            f"  τ = {t:.1f} | Base Dice={np.mean(b_dices):.4f} (P={np.mean(b_precs):.4f}, R={np.mean(b_recs):.4f}) | "
            f"DANN Dice={np.mean(d_dices):.4f} (P={np.mean(d_precs):.4f}, R={np.mean(d_recs):.4f})"
        )

    # 8. Modality Shift Ablation: Surrogate Mapping [-Z_ADC, ADC, 0]
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY: SURROGATE INVERTED-Z MAPPING [-Z_ADC, ADC, 0]")
    logger.info("=" * 80)

    base_surr_results = []
    dann_surr_results = []
    for s in val_samples:
        b_s = run_model_inference(
            model=model_base, sample=s, target_spacing=target_sp,
            image_size=image_size, channel_mapping="surrogate_z_adc_dwi",
            threshold=args.threshold, is_dann=False, device=device
        )
        base_surr_results.append(b_s)

        d_s = run_model_inference(
            model=model_dann, sample=s, target_spacing=target_sp,
            image_size=image_size, channel_mapping="surrogate_z_adc_dwi",
            threshold=args.threshold, is_dann=True, device=device
        )
        dann_surr_results.append(d_s)

    base_surr_summary = summarize_metrics(base_surr_results)
    dann_surr_summary = summarize_metrics(dann_surr_results)

    logger.info(f"Baseline (Surrogate): Mean Dice = {base_surr_summary['dice']['mean']:.4f} ± {base_surr_summary['dice']['std']:.4f}")
    logger.info(f"DANN     (Surrogate): Mean Dice = {dann_surr_summary['dice']['mean']:.4f} ± {dann_surr_summary['dice']['std']:.4f}")

    # 9. Generate Aggregate Figures
    bar_chart_file = viz_dir / "baseline_vs_dann_metrics_barchart.png"
    plot_metrics_comparison_barchart(base_summary, dann_summary, bar_chart_file)
    logger.info(f"\nGenerated metrics bar chart: {bar_chart_file.name}")

    thresh_curve_file = viz_dir / "threshold_sensitivity_curve.png"
    plot_threshold_sensitivity_curves(base_thresh_res, dann_thresh_res, thresh_curve_file)
    logger.info(f"Generated threshold sensitivity curves: {thresh_curve_file.name}")

    vol_scatter_file = viz_dir / "lesion_volume_vs_dice_scatter.png"
    plot_lesion_volume_vs_dice(base_results, dann_results, vol_scatter_file)
    logger.info(f"Generated lesion volume scatter plot: {vol_scatter_file.name}")

    # 10. Save Detailed CSV and JSON Reports
    csv_file = log_dir / "scientific_evaluation_metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Subject", "Dice", "IoU", "Precision", "Recall",
            "Sensitivity", "Specificity", "HD95_mm", "HD95_voxels",
            "TP_voxels", "FP_voxels", "FN_voxels", "TN_voxels",
            "GT_voxels", "Pred_voxels", "GT_Volume_mm3", "Pred_Volume_mm3"
        ])
        for r in base_results:
            m = r["metrics"]
            writer.writerow([
                "Baseline_UNet", r["subject"], m["dice"], m["iou"], m["precision"],
                m["recall"], m["sensitivity"], m["specificity"], m["hd95_mm"], m["hd95_voxels"],
                m["tp"], m["fp"], m["fn"], m["tn"], m["gt_voxels"], m["pred_voxels"],
                m["gt_volume_mm3"], m["pred_volume_mm3"]
            ])
        for r in dann_results:
            m = r["metrics"]
            writer.writerow([
                "DANN_UNet", r["subject"], m["dice"], m["iou"], m["precision"],
                m["recall"], m["sensitivity"], m["specificity"], m["hd95_mm"], m["hd95_voxels"],
                m["tp"], m["fp"], m["fn"], m["tn"], m["gt_voxels"], m["pred_voxels"],
                m["gt_volume_mm3"], m["pred_volume_mm3"]
            ])

    logger.info(f"Saved comprehensive CSV metrics: {csv_file.name}")

    json_file = log_dir / "scientific_evaluation_metrics.json"
    full_report = {
        "title": "Scientific Evaluation: Adult-to-Neonatal Ischemic Stroke Lesion Segmentation",
        "cohort": {
            "dataset": "BONBID-HIE (BONBID2023_Val)",
            "num_subjects": len(val_samples),
            "subjects": [s["subject"] for s in val_samples],
        },
        "checkpoints": {
            "baseline_unet": {
                "path": str(base_ckpt_path),
                "sha256": base_hash,
                "size_bytes": base_ckpt_path.stat().st_size,
            },
            "dann_unet": {
                "path": str(dann_ckpt_path),
                "sha256": dann_hash,
                "size_bytes": dann_ckpt_path.stat().st_size,
                "best_epoch": 6,
                "adult_val_dice": 0.2652,
            },
        },
        "primary_mapping_summary": {
            "baseline_unet": base_summary,
            "dann_unet": dann_summary,
        },
        "statistical_tests": stat_tests,
        "threshold_sensitivity": {
            "thresholds": thresholds,
            "baseline": base_thresh_res,
            "dann": dann_thresh_res,
        },
        "surrogate_mapping_ablation": {
            "baseline_unet": base_surr_summary,
            "dann_unet": dann_surr_summary,
        },
        "per_subject_metrics": {
            "baseline_unet": [{ "subject": r["subject"], **r["metrics"] } for r in base_results],
            "dann_unet": [{ "subject": r["subject"], **r["metrics"] } for r in dann_results],
        },
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Saved full JSON report: {json_file.name}")

    # Final Checkpoint Integrity Re-verification
    base_hash_after = compute_file_hash(base_ckpt_path)
    dann_hash_after = compute_file_hash(dann_ckpt_path)
    assert base_hash == base_hash_after, "CRITICAL: Baseline checkpoint was altered!"
    assert dann_hash == dann_hash_after, "CRITICAL: DANN checkpoint was altered!"
    logger.info("\nCheckpoint safety confirmed: No checkpoint files modified during evaluation.")
    logger.info("=" * 80)
    logger.info("SCIENTIFIC EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

