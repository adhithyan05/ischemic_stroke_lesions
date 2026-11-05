"""
evaluate.py
===========
Evaluation of the trained baseline U-Net on the BONBID-HIE neonatal
validation subset.

Outputs:
  1. JSON report with Dice, IoU, Precision, Recall, Sensitivity per subject
     and overall mean +/- std.
  2. Matplotlib visualisations: original MRI slices + predicted masks
     overlaid in colour.

Usage (standalone):
    python evaluate.py --config ../configs/config.yaml \
                       --checkpoint ../outputs/checkpoints/baseline_unet_best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.amp import autocast

from model import build_unet
from metrics import (
    dice_score, iou_score, precision_score,
    recall_score, sensitivity_score, specificity_score,
)
from data_utils import discover_bonbid, load_nifti, normalize_volume, get_voxel_spacing, \
    resample_to_target_spacing, _crop_or_pad

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Single-volume inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_volume(
    model:           torch.nn.Module,
    sample:          Dict,
    image_size:      tuple,
    target_spacing:  tuple,
    device:          torch.device,
    threshold:       float = 0.5,
) -> Dict:
    """
    Run inference on a single BONBID-HIE subject volume.

    Returns dict with per-slice images, predictions, and masks.
    """
    model.eval()

    image_paths = sample.get("image_paths", {})
    mask_path   = sample.get("mask_path")

    # Load and normalise all modality volumes
    volumes: Dict[str, np.ndarray] = {}
    spacing = None
    for mod, path in image_paths.items():
        vol, img = load_nifti(path)
        volumes[mod] = normalize_volume(vol)
        if spacing is None:
            spacing = get_voxel_spacing(img)

    if not volumes:
        return {}

    # Ensure 3 channels
    vol_keys = list(volumes.keys())
    while len(vol_keys) < 3:
        vol_keys.append(vol_keys[-1])
    vols_3 = [volumes[k] for k in vol_keys[:3]]

    # Resample
    if spacing is not None:
        vols_3 = [resample_to_target_spacing(v, spacing, target_spacing) for v in vols_3]

    # Load mask
    mask_vol = None
    if mask_path is not None:
        m, _ = load_nifti(mask_path)
        if spacing is not None:
            m = resample_to_target_spacing(m, spacing, target_spacing, order=0)
        mask_vol = (m > 0.5).astype(np.float32)

    H, W = image_size
    depth = min(v.shape[2] for v in vols_3)

    slice_results = []

    with torch.no_grad():
        for z in range(depth):
            channels = []
            for vol in vols_3:
                sl = _crop_or_pad(vol[:, :, z], H, W)
                channels.append(sl)

            img_arr = np.stack(channels, axis=0)   # [3, H, W]
            inp = torch.from_numpy(img_arr).unsqueeze(0).float().to(device)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(inp)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()   # [H, W]
            pred = (prob >= threshold).astype(np.float32)

            gt_sl = None
            if mask_vol is not None:
                gt_sl = _crop_or_pad(mask_vol[:, :, z], H, W)

            slice_results.append({
                "z":       z,
                "image":   img_arr,   # [3, H, W]
                "prob":    prob,
                "pred":    pred,
                "gt":      gt_sl,
            })

    return {
        "subject":  sample["subject"],
        "slices":   slice_results,
        "has_mask": mask_vol is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_subject_metrics(result: Dict, threshold: float = 0.5) -> Optional[Dict]:
    """Aggregate all slices of a subject into volume-level metrics."""
    if not result or not result["has_mask"]:
        return None

    all_pred = []
    all_gt   = []
    for sl in result["slices"]:
        all_pred.append(sl["pred"].ravel())
        all_gt.append(sl["gt"].ravel())

    pred_flat = np.concatenate(all_pred)
    gt_flat   = np.concatenate(all_gt)

    return {
        "subject":     result["subject"],
        "dice":        dice_score(pred_flat, gt_flat, threshold),
        "iou":         iou_score(pred_flat, gt_flat, threshold),
        "precision":   precision_score(pred_flat, gt_flat, threshold),
        "recall":      recall_score(pred_flat, gt_flat, threshold),
        "sensitivity": sensitivity_score(pred_flat, gt_flat, threshold),
        "specificity": specificity_score(pred_flat, gt_flat, threshold),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _select_representative_slices(result: Dict, n: int = 4) -> List[Dict]:
    """Select n slices that best show the lesion (or evenly distributed)."""
    slices = result["slices"]
    if result["has_mask"]:
        # Sort by lesion area in GT
        scored = [(sl, sl["gt"].sum() if sl["gt"] is not None else 0) for sl in slices]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in scored[:n]]
    else:
        step = max(1, len(slices) // n)
        selected = slices[::step][:n]
    return selected


def visualise_predictions(
    result:      Dict,
    save_dir:    str | Path,
    n_slices:    int = 4,
    alpha:       float = 0.45,
):
    """
    Save a multi-panel figure showing:
      Row 1: DWI channel (grayscale)
      Row 2: Prediction overlay (green = TP, red = FP, blue = FN)
      Row 3: Ground truth (if available)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    subj    = result["subject"]
    slices  = _select_representative_slices(result, n=n_slices)
    has_gt  = result["has_mask"]

    n_rows = 3 if has_gt else 2
    fig, axes = plt.subplots(n_rows, n_slices, figsize=(n_slices * 3.5, n_rows * 3.5),
                             facecolor="#0d1117")
    fig.suptitle(f"Subject: {subj}", fontsize=13, color="white", weight="bold", y=0.98)

    row_labels = ["DWI (ch0)", "Prediction", "Ground Truth"]

    for col, sl in enumerate(slices):
        img  = sl["image"]   # [3, H, W]
        pred = sl["pred"]    # [H, W]
        gt   = sl["gt"]      # [H, W] or None
        z    = sl["z"]

        # ── Row 0: DWI channel ─────────────────────────────────────────
        ax = axes[0][col] if n_rows > 1 else axes[col]
        ax.imshow(img[0], cmap="gray", interpolation="bilinear")
        ax.set_title(f"z={z}", color="#aaaaaa", fontsize=9)
        ax.axis("off")

        # ── Row 1: Prediction overlay ─────────────────────────────────
        ax = axes[1][col]
        ax.imshow(img[0], cmap="gray", interpolation="bilinear")
        if has_gt and gt is not None:
            # TP green, FP red, FN blue
            tp_mask = ((pred > 0) & (gt > 0)).astype(float)
            fp_mask = ((pred > 0) & (gt == 0)).astype(float)
            fn_mask = ((pred == 0) & (gt > 0)).astype(float)

            H, W = pred.shape
            overlay = np.zeros((H, W, 4), dtype=np.float32)
            overlay[..., 1] = tp_mask * alpha  # green TP
            overlay[..., 0] = fp_mask * alpha  # red FP
            overlay[..., 2] = fn_mask * alpha  # blue FN
            overlay[..., 3] = (tp_mask + fp_mask + fn_mask) * alpha
            ax.imshow(overlay)
        else:
            # Just show prediction without comparison
            overlay = np.zeros((*pred.shape, 4), dtype=np.float32)
            overlay[..., 1] = pred * alpha
            overlay[..., 3] = pred * alpha
            ax.imshow(overlay)
        ax.axis("off")

        # ── Row 2: Ground truth ────────────────────────────────────────
        if has_gt and gt is not None:
            ax = axes[2][col]
            ax.imshow(img[0], cmap="gray", interpolation="bilinear")
            gt_overlay = np.zeros((*gt.shape, 4), dtype=np.float32)
            gt_overlay[..., 0] = gt * alpha   # red
            gt_overlay[..., 3] = gt * alpha
            ax.imshow(gt_overlay)
            ax.axis("off")

    # ── Row labels ────────────────────────────────────────────────────
    for r, label in enumerate(row_labels[:n_rows]):
        axes[r][0].set_ylabel(label, color="#aaaaaa", fontsize=9)

    # ── Legend ────────────────────────────────────────────────────────
    if has_gt:
        patches = [
            mpatches.Patch(color="green", alpha=0.6, label="TP"),
            mpatches.Patch(color="red",   alpha=0.6, label="FP"),
            mpatches.Patch(color="blue",  alpha=0.6, label="FN"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=3,
                   facecolor="#1c2128", labelcolor="white", fontsize=10,
                   framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    save_path = save_dir / f"pred_{subj}.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info(f"  Saved visualisation -> {save_path}")


def plot_metrics_summary(metrics_list: List[Dict], save_dir: str | Path):
    """Bar chart of mean metrics across all subjects."""
    save_dir = Path(save_dir)
    if not metrics_list:
        return

    keys = ["dice", "iou", "precision", "recall", "sensitivity", "specificity"]
    means = {k: np.mean([m[k] for m in metrics_list]) for k in keys}
    stds  = {k: np.std([m[k] for m in metrics_list])  for k in keys}

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    x = np.arange(len(keys))
    bars = ax.bar(
        x, [means[k] for k in keys],
        yerr=[stds[k] for k in keys],
        color=["#58a6ff", "#3fb950", "#ff7b72", "#ffa657", "#d2a8ff", "#79c0ff"],
        alpha=0.85, capsize=5, width=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([k.capitalize() for k in keys], color="white", fontsize=11)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_tick_params(labelcolor="white")
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title(
        "Baseline U-Net: Neonatal Validation Metrics\n(Adult-to-Neonatal Domain Shift)",
        color="white", fontsize=13, weight="bold",
    )
    ax.spines[:].set_color("#30363d")
    ax.grid(axis="y", color="#30363d", linestyle="--", alpha=0.5)

    # Value labels on bars
    for bar, k in zip(bars, keys):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stds[k] + 0.02,
            f"{means[k]:.3f}",
            ha="center", va="bottom", color="white", fontsize=9,
        )

    plt.tight_layout()
    path = save_dir / "baseline_metrics_summary.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info(f"Metrics summary plot saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_neonatal(
    model:          torch.nn.Module,
    cfg:            dict,
    device:         torch.device,
    bonbid_root:    str,
    checkpoint_path: str,
    viz_dir:        str,
    split:          str = "val",
    save_n_vis:     int = 10,
) -> Dict:
    """
    Load best checkpoint, run inference on BONBID-HIE validation split,
    compute metrics, save visualisations, and return results dict.
    """
    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    threshold    = cfg.get("evaluation", {}).get("threshold", 0.5)
    image_size   = tuple(cfg.get("data", {}).get("image_size", [256, 256]))
    target_sp    = tuple(cfg.get("data", {}).get("target_spacing", [1.0, 1.0, 1.0]))

    samples = discover_bonbid(bonbid_root, split)
    if not samples:
        logger.error(f"No BONBID-HIE samples found in '{bonbid_root}/{split}'")
        return {}

    all_metrics: List[Dict] = []
    results_for_viz:  List[Dict] = []

    for idx, sample in enumerate(samples):
        logger.info(f"Evaluating {sample['subject']} ({idx+1}/{len(samples)}) ...")
        result = predict_volume(model, sample, image_size, target_sp, device, threshold)

        if not result:
            continue

        subject_metrics = compute_subject_metrics(result, threshold)
        if subject_metrics:
            all_metrics.append(subject_metrics)
            logger.info(
                f"  Dice={subject_metrics['dice']:.4f} | "
                f"IoU={subject_metrics['iou']:.4f} | "
                f"Prec={subject_metrics['precision']:.4f} | "
                f"Recall={subject_metrics['recall']:.4f}"
            )

        if idx < save_n_vis:
            results_for_viz.append(result)

    # ── Aggregate ─────────────────────────────────────────────────────────
    if all_metrics:
        keys = ["dice", "iou", "precision", "recall", "sensitivity", "specificity"]
        summary = {}
        for k in keys:
            vals = [m[k] for m in all_metrics]
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        logger.info("\n" + "=" * 60)
        logger.info("BASELINE EVALUATION RESULTS (Adult -> Neonatal)")
        logger.info("=" * 60)
        for k, v in summary.items():
            logger.info(f"  {k.upper():12s}: {v['mean']:.4f} +/- {v['std']:.4f}")
        logger.info("=" * 60)
    else:
        summary = {}
        logger.warning("No labelled samples found ? cannot compute metrics.")

    # ── Save reports ──────────────────────────────────────────────────────
    viz_path = Path(viz_dir)
    viz_path.mkdir(parents=True, exist_ok=True)

    report = {
        "split":           split,
        "checkpoint":      checkpoint_path,
        "threshold":       threshold,
        "num_subjects":    len(samples),
        "num_evaluated":   len(all_metrics),
        "summary":         summary,
        "per_subject":     all_metrics,
    }

    report_path = viz_path / "baseline_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved -> {report_path}")

    # ── Visualisations ────────────────────────────────────────────────────
    for result in results_for_viz:
        try:
            visualise_predictions(result, viz_path, n_slices=4)
        except Exception as e:
            logger.warning(f"Visualisation failed for {result.get('subject')}: {e}")

    if all_metrics:
        plot_metrics_summary(all_metrics, viz_path)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import yaml

    from trainer import get_device

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Evaluate baseline U-Net on BONBID-HIE")
    parser.add_argument("--config",     required=True,  help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pth checkpoint")
    parser.add_argument("--split",      default="val",  help="val | test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg)
    model  = build_unet(cfg).to(device)

    evaluate_on_neonatal(
        model           = model,
        cfg             = cfg,
        device          = device,
        bonbid_root     = cfg["paths"]["bonbid_root"],
        checkpoint_path = args.checkpoint,
        viz_dir         = cfg["paths"]["viz_dir"],
        split           = args.split,
        save_n_vis      = cfg["evaluation"]["save_n_vis"],
    )
