"""
evaluate_dann.py
================
Phase 4: Evaluation Support for Domain-Adversarial Neural Network (DANN).

Evaluates the DANN-adapted U-Net model on the 4 official neonatal BONBID-HIE
validation subjects (MGHNICU_001, MGHNICU_002, MGHNICU_302, MGHNICU_436) to assess
how well domain adaptation reduces the domain gap compared to the Phase 3 zero-shot baseline.

Safety & Governance Constraints:
  1. Final evaluation remains strictly separate from training.
  2. Evaluates strictly in evaluation mode (model.eval(), torch.no_grad()).
  3. Uses the exact same preprocessing and metrics as Phase 3.
  4. Preserves all baseline and DANN checkpoints without modification.

Usage:
    python src/evaluate_dann.py
    python src/evaluate_dann.py --checkpoint outputs/checkpoints/dann/dann_unet_best.pth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml

# Path setup
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from dann_model import DANNUNet, build_dann
from data_utils import (
    discover_bonbid,
    load_nifti,
    normalize_volume,
    resample_to_target_spacing,
    _crop_or_pad,
)
from evaluate_neonatal_zeroshot import compute_volume_metrics, adapt_channels

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("evaluate_dann")
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
# Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

def run_dann_inference(
    model: DANNUNet,
    sample: Dict,
    target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    image_size: Tuple[int, int] = (128, 128),
    channel_mapping: str = "primary_adc_z_z",
    threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Run volumetric inference using the DANN model."""
    subj = sample["subject"]
    img_paths = sample.get("image_paths", {})
    mask_path = sample.get("mask_path")

    adc_raw, adc_meta = load_nifti(img_paths["adc"])
    z_raw, _ = load_nifti(img_paths["z-adc"])
    mask_raw, _ = load_nifti(mask_path) if mask_path else (None, None)

    native_spacing = adc_meta.get_zooms()

    # Preprocessing identical to Phase 2 & Phase 3
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

            # In evaluation mode: return_domain=False runs through feature extractor + label predictor
            logits = model(inp, return_domain=False)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (prob >= threshold).astype(np.float32)

            prob_vol[:, :, z] = prob
            pred_vol[:, :, z] = pred

            if mask_res is not None:
                gt_sl = _crop_or_pad(mask_res[:, :, z], H, W)
                gt_vol[:, :, z] = gt_sl

    metrics = None
    if gt_vol is not None:
        metrics = compute_volume_metrics(pred_vol, gt_vol, threshold=threshold)

    return {
        "subject": subj,
        "depth": depth,
        "pred_vol": pred_vol,
        "prob_vol": prob_vol,
        "gt_vol": gt_vol,
        "input_slices": input_slices,
        "metrics": metrics,
    }


def plot_dann_comparison(
    summary_dann: Dict,
    baseline_summary: Dict,
    out_file: Path,
):
    """Plot bar chart comparing Phase 3 Baseline vs Phase 4 DANN."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = ["dice", "iou", "precision", "recall", "specificity"]
    labels = ["Dice", "IoU", "Precision", "Recall", "Specificity"]

    base_means = [baseline_summary[k]["mean"] for k in metric_keys]
    dann_means = [summary_dann[k]["mean"] for k in metric_keys]
    dann_stds  = [summary_dann[k]["std"]  for k in metric_keys]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    x = np.arange(len(labels))
    width = 0.35

    b1 = ax.bar(x - width/2, base_means, width, label="Phase 3: Baseline (Zero-Shot)", color="#8b949e", alpha=0.85, edgecolor="#30363d")
    b2 = ax.bar(x + width/2, dann_means, width, yerr=dann_stds, capsize=4, label="Phase 4: DANN Adapted", color="#58a6ff", alpha=0.9, edgecolor="#30363d")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0.0 to 1.0)", color="white", fontsize=11)
    ax.set_title(
        "Domain Adaptation Evaluation on Neonatal Cohort\nPhase 3 Adult Baseline vs Phase 4 DANN",
        color="white", fontsize=12, weight="bold", pad=12
    )
    ax.tick_params(colors="white")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", loc="upper right")
    ax.spines[:].set_color("#30363d")
    ax.grid(axis="y", color="#30363d", linestyle="--", alpha=0.5)

    for bar, val in zip(b1, base_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", color="#8b949e", fontsize=8.5, weight="bold")
    for bar, val in zip(b2, dann_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", color="#58a6ff", fontsize=8.5, weight="bold")

    plt.tight_layout()
    plt.savefig(out_file, dpi=130, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate DANN on Neonatal BONBID-HIE Validation Set")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"), help="Config file")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "outputs/checkpoints/dann/dann_unet_best.pth"), help="DANN checkpoint")
    parser.add_argument("--baseline-metrics", default=str(PROJECT_ROOT / "outputs/logs/neonatal_zeroshot_metrics.json"), help="Phase 3 metrics for comparison")
    parser.add_argument("--split", default="val", help="Cohort split ('val')")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization cutoff")
    args = parser.parse_args()

    config_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    log_dir = PROJECT_ROOT / "outputs/logs/dann"
    viz_dir = PROJECT_ROOT / "outputs/visualizations/dann"

    log_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "dann_evaluation.log"
    global logger
    logger = setup_logger(log_file)

    logger.info("=" * 78)
    logger.info("PHASE 4: DANN EVALUATION ON NEONATAL VALIDATION COHORT")
    logger.info("=" * 78)

    if not ckpt_path.exists():
        logger.error(f"DANN Checkpoint not found at: {ckpt_path}")
        logger.info("Train DANN first using: python src/train_dann.py")
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    model = build_dann(cfg=cfg, device=device)

    # Load DANN weights
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logger.info(f"Loaded DANN Checkpoint: {ckpt_path.name} (Epoch: {ckpt.get('epoch', '?')})")

    samples = discover_bonbid(cfg["paths"]["bonbid_root"], split=args.split)
    target_sp = tuple(cfg.get("data", {}).get("target_spacing", [2.0, 2.0, 2.0]))
    image_size = tuple(cfg.get("data", {}).get("image_size", [128, 128]))

    results = []
    for s in samples:
        res = run_dann_inference(
            model=model, sample=s, target_spacing=target_sp,
            image_size=image_size, threshold=args.threshold, device=device
        )
        results.append(res)
        m = res["metrics"]
        logger.info(
            f"  • {res['subject']}: GT={m['gt_voxels']:5d} | Pred={m['pred_voxels']:5d} | "
            f"Dice={m['dice']:.4f} | IoU={m['iou']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f}"
        )

    # Summary
    metric_keys = ["dice", "iou", "precision", "recall", "sensitivity", "specificity"]
    summary = {
        k: {
            "mean": float(np.mean([r["metrics"][k] for r in results])),
            "std":  float(np.std([r["metrics"][k] for r in results])),
        }
        for k in metric_keys
    }

    logger.info("-" * 78)
    logger.info(f"DANN Cohort Mean Dice: {summary['dice']['mean']:.4f} ± {summary['dice']['std']:.4f} | Mean IoU: {summary['iou']['mean']:.4f} ± {summary['iou']['std']:.4f}")
    logger.info("-" * 78)

    # Compare with Phase 3 baseline if available
    base_metrics_path = Path(args.baseline_metrics)
    if base_metrics_path.exists():
        with open(base_metrics_path) as f:
            base_data = json.load(f)
        base_summary = base_data.get("summary_metrics", {})
        if base_summary:
            base_dice = base_summary.get("dice", {}).get("mean", 0.1451)
            dann_dice = summary["dice"]["mean"]
            diff = dann_dice - base_dice
            logger.info(f"Comparison with Phase 3 Baseline (Zero-Shot):")
            logger.info(f"  • Baseline Mean Dice : {base_dice:.4f}")
            logger.info(f"  • DANN Mean Dice     : {dann_dice:.4f}")
            logger.info(f"  • Absolute Gain (Delta): {diff:+.4f} ({diff/max(1e-5, base_dice)*100:+.1f}%)")

            # Plot comparison
            plot_dann_comparison(summary, base_summary, viz_dir / "dann_vs_baseline_comparison.png")
            logger.info(f"Saved comparison figure -> dann_vs_baseline_comparison.png")

    # Save reports
    json_path = log_dir / "dann_evaluation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase": "Phase 4: DANN Evaluation",
            "checkpoint": str(ckpt_path),
            "summary_metrics": summary,
            "per_subject_metrics": [
                {"subject": r["subject"], "metrics": r["metrics"]} for r in results
            ],
        }, f, indent=2)

    logger.info(f"Saved DANN evaluation report -> {json_path.name}")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()

