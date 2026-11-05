"""
evaluate_input_representations.py
=================================
Systematic Controlled Evaluation of MRI Input Representations
for Adult-to-Neonatal Brain Lesion Segmentation.

Evaluates Baseline Attention U-Net and DANN Attention U-Net on the
4 official held-out neonatal validation subjects across multiple
channel configurations:
  Config A: [ADC, Z-ADC, Z-ADC]
  Config B: [ADC, -Z-ADC, -Z-ADC]
  Config C: [ADC, Z-ADC, -Z-ADC]
  Config D1: [-Z-ADC, ADC, 0] (DWI-aligned surrogate with zero FLAIR)
  Config D2: [-Z-ADC, ADC, Z-ADC] (DWI-aligned with Z-ADC in FLAIR slot)
  Config D3: [-Z-ADC, ADC, -Z-ADC] (DWI-aligned with inverted Z-ADC in FLAIR slot)

Measures:
  - Dice
  - IoU
  - Precision
  - Recall (Sensitivity)
  - Specificity
  - HD95 (mm and voxels)
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
from evaluate_neonatal_zeroshot import compute_volume_metrics
from scientific_evaluation import compute_hd95, compute_file_hash


def get_channel_representation(
    adc_res: np.ndarray,
    z_res: np.ndarray,
    config_name: str,
) -> List[np.ndarray]:
    """
    Construct 3-channel input representation based on specified configuration.
    Adult model input expectation: [Ch0=DWI, Ch1=ADC, Ch2=FLAIR]
    """
    inv_z = -z_res
    zeros = np.zeros_like(adc_res)

    if config_name == "config_A":
        # [ADC, Z-ADC, Z-ADC]
        return [adc_res, z_res, z_res]
    elif config_name == "config_B":
        # [ADC, inverted Z-ADC, inverted Z-ADC]
        return [adc_res, inv_z, inv_z]
    elif config_name == "config_C":
        # [ADC, Z-ADC, inverted Z-ADC]
        return [adc_res, z_res, inv_z]
    elif config_name == "config_D1_dwi_aligned_zero":
        # [-Z-ADC, ADC, 0] (DWI-aligned surrogate)
        return [inv_z, adc_res, zeros]
    elif config_name == "config_D2_dwi_aligned_z":
        # [-Z-ADC, ADC, Z-ADC]
        return [inv_z, adc_res, z_res]
    elif config_name == "config_D3_dwi_aligned_invz":
        # [-Z-ADC, ADC, -Z-ADC]
        return [inv_z, adc_res, inv_z]
    else:
        raise ValueError(f"Unknown config name: {config_name}")


def run_single_inference(
    model: torch.nn.Module,
    sample: Dict,
    config_name: str,
    target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    image_size: Tuple[int, int] = (128, 128),
    threshold: float = 0.5,
    is_dann: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    subj = sample["subject"]
    img_paths = sample.get("image_paths", {})
    mask_path = sample.get("mask_path")

    adc_raw, adc_meta = load_nifti(img_paths["adc"])
    z_raw, _ = load_nifti(img_paths["z-adc"])
    mask_raw, _ = load_nifti(mask_path) if mask_path else (None, None)

    native_spacing = adc_meta.get_zooms()

    # Preprocessing (consistent with pipeline)
    adc_norm = normalize_volume(adc_raw, method="zscore")
    z_norm = normalize_volume(z_raw, method="zscore")

    adc_res = resample_to_target_spacing(adc_norm, native_spacing, target_spacing, order=1)
    z_res = resample_to_target_spacing(z_norm, native_spacing, target_spacing, order=1)

    mask_res = None
    if mask_raw is not None:
        m = resample_to_target_spacing(mask_raw, native_spacing, target_spacing, order=0)
        mask_res = (m > 0.5).astype(np.float32)

    channels_3d = get_channel_representation(adc_res, z_res, config_name)

    H, W = image_size
    depth = adc_res.shape[2]

    pred_vol = np.zeros((H, W, depth), dtype=np.float32)
    prob_vol = np.zeros((H, W, depth), dtype=np.float32)
    gt_vol = np.zeros((H, W, depth), dtype=np.float32) if mask_res is not None else None

    with torch.no_grad():
        for z in range(depth):
            ch_slices = [_crop_or_pad(ch[:, :, z], H, W) for ch in channels_3d]
            slice_tensor = np.stack(ch_slices, axis=0)
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

    metrics = {}
    if gt_vol is not None:
        metrics = compute_volume_metrics(pred_vol, gt_vol, threshold=0.5)
        hd_dict = compute_hd95(pred_vol, gt_vol, voxel_spacing=target_spacing, threshold=0.5)
        metrics.update(hd_dict)

    return {
        "subject": subj,
        "metrics": metrics,
    }


def main():
    config_path = PROJECT_ROOT / "configs/config.yaml"
    base_ckpt_path = PROJECT_ROOT / "outputs/checkpoints/baseline_unet_best.pth"
    dann_ckpt_path = PROJECT_ROOT / "outputs/checkpoints/dann/dann_unet_best.pth"
    bonbid_root = PROJECT_ROOT / "data/BONBID-HIE"
    log_dir = PROJECT_ROOT / "outputs/logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")

    # Load Baseline
    model_base = build_unet(cfg).to(device)
    ckpt_b = torch.load(base_ckpt_path, map_location=device)
    model_base.load_state_dict(ckpt_b["model_state"])
    model_base.eval()
    for p in model_base.parameters():
        p.requires_grad = False

    # Load DANN
    model_dann = build_dann(cfg=cfg, device=device)
    ckpt_d = torch.load(dann_ckpt_path, map_location=device)
    model_dann.load_state_dict(ckpt_d["model_state"])
    model_dann.eval()
    for p in model_dann.parameters():
        p.requires_grad = False

    val_samples = discover_bonbid(str(bonbid_root), split="val")
    target_sp = tuple(cfg.get("data", {}).get("target_spacing", [2.0, 2.0, 2.0]))
    image_size = tuple(cfg.get("data", {}).get("image_size", [128, 128]))

    configurations = [
        ("config_A", "A. [ADC, Z-ADC, Z-ADC]"),
        ("config_B", "B. [ADC, -Z-ADC, -Z-ADC]"),
        ("config_C", "C. [ADC, Z-ADC, -Z-ADC]"),
        ("config_D1_dwi_aligned_zero", "D1. [-Z-ADC, ADC, 0] (DWI-aligned, Zero-FLAIR)"),
        ("config_D2_dwi_aligned_z", "D2. [-Z-ADC, ADC, Z-ADC] (DWI-aligned, Z-ADC FLAIR)"),
        ("config_D3_dwi_aligned_invz", "D3. [-Z-ADC, ADC, -Z-ADC] (DWI-aligned, InvZ FLAIR)"),
    ]

    all_results = {}

    print("=" * 100)
    print("CONTROLLED INPUT REPRESENTATION EXPERIMENT: NEONATAL VALIDATION COHORT")
    print("=" * 100)

    for cfg_key, cfg_desc in configurations:
        print(f"\n--- Running: {cfg_desc} ---")
        base_subj_res = []
        dann_subj_res = []

        for s in val_samples:
            b_res = run_single_inference(
                model=model_base, sample=s, config_name=cfg_key,
                target_spacing=target_sp, image_size=image_size,
                is_dann=False, device=device
            )
            base_subj_res.append(b_res)

            d_res = run_single_inference(
                model=model_dann, sample=s, config_name=cfg_key,
                target_spacing=target_sp, image_size=image_size,
                is_dann=True, device=device
            )
            dann_subj_res.append(d_res)

        def get_summary(res_list):
            summary = {}
            for k in ["dice", "iou", "precision", "recall", "specificity", "hd95_mm"]:
                vals = [r["metrics"][k] for r in res_list if not np.isnan(r["metrics"][k])]
                summary[k] = {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "std": float(np.std(vals)) if vals else 0.0,
                    "min": float(np.min(vals)) if vals else 0.0,
                    "max": float(np.max(vals)) if vals else 0.0,
                }
            return summary

        base_sum = get_summary(base_subj_res)
        dann_sum = get_summary(dann_subj_res)

        all_results[cfg_key] = {
            "description": cfg_desc,
            "baseline": {
                "summary": base_sum,
                "subjects": {r["subject"]: r["metrics"] for r in base_subj_res},
            },
            "dann": {
                "summary": dann_sum,
                "subjects": {r["subject"]: r["metrics"] for r in dann_subj_res},
            },
        }

        print(f"  [Baseline] Mean Dice: {base_sum['dice']['mean']:.4f} +/- {base_sum['dice']['std']:.4f} | Prec: {base_sum['precision']['mean']:.4f} | Rec: {base_sum['recall']['mean']:.4f} | HD95: {base_sum['hd95_mm']['mean']:.1f} mm")
        print(f"  [DANN    ] Mean Dice: {dann_sum['dice']['mean']:.4f} +/- {dann_sum['dice']['std']:.4f} | Prec: {dann_sum['precision']['mean']:.4f} | Rec: {dann_sum['recall']['mean']:.4f} | HD95: {dann_sum['hd95_mm']['mean']:.1f} mm")

    # Save to JSON
    out_json = log_dir / "controlled_input_representations_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to: {out_json}")

    # Save to CSV
    out_csv = log_dir / "controlled_input_representations_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Configuration", "Model", "Mean_Dice", "Std_Dice",
            "Mean_IoU", "Mean_Precision", "Mean_Recall",
            "Mean_Specificity", "Mean_HD95_mm"
        ])
        for cfg_key, data in all_results.items():
            b_s = data["baseline"]["summary"]
            d_s = data["dann"]["summary"]
            writer.writerow([
                cfg_key, "Baseline_UNet",
                f"{b_s['dice']['mean']:.4f}", f"{b_s['dice']['std']:.4f}",
                f"{b_s['iou']['mean']:.4f}", f"{b_s['precision']['mean']:.4f}",
                f"{b_s['recall']['mean']:.4f}", f"{b_s['specificity']['mean']:.4f}",
                f"{b_s['hd95_mm']['mean']:.1f}"
            ])
            writer.writerow([
                cfg_key, "DANN_UNet",
                f"{d_s['dice']['mean']:.4f}", f"{d_s['dice']['std']:.4f}",
                f"{d_s['iou']['mean']:.4f}", f"{d_s['precision']['mean']:.4f}",
                f"{d_s['recall']['mean']:.4f}", f"{d_s['specificity']['mean']:.4f}",
                f"{d_s['hd95_mm']['mean']:.1f}"
            ])
    print(f"Saved summary CSV to: {out_csv}")


if __name__ == "__main__":
    main()

