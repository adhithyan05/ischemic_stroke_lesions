"""
test_frontend.py
================
Unit and integration tests for the Streamlit web frontend modules.
Tests model loading, data loading, preprocessing, inference, and exports.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "frontend"))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch

from utils.model_loader import load_baseline_unet, load_dann_model, get_inference_device
from utils.data_loader import get_available_validation_samples
from utils.inference import run_model_inference, export_mask_to_npz, export_mask_to_nifti
from data_utils import load_nifti


def test_model_loading():
    print("1. Testing Model Loading...")
    device = get_inference_device()
    print(f"   Inference device: {device}")

    # Baseline
    unet, unet_meta = load_baseline_unet()
    assert unet is not None, "Failed to load Baseline U-Net!"
    assert unet_meta["epoch"] == 82, f"Unexpected baseline epoch: {unet_meta['epoch']}"
    print(f"   [OK] Loaded {unet_meta['name']} ({unet_meta['parameters']:,} params, epoch {unet_meta['epoch']})")

    # DANN
    dann, dann_meta = load_dann_model()
    assert dann is not None, "Failed to load DANN!"
    assert dann_meta["best_epoch"] == 6, f"Expected DANN best epoch 6, got {dann_meta['best_epoch']}"
    assert abs(dann_meta["best_val_dice"] - 0.2652) < 0.001, f"Unexpected DANN val dice: {dann_meta['best_val_dice']}"
    print(f"   [OK] Loaded {dann_meta['name']} ({dann_meta['parameters']:,} params, best epoch {dann_meta['best_epoch']}, val dice {dann_meta['best_val_dice']:.4f})")


def test_validation_samples():
    print("2. Testing Validation Sample Discovery...")
    samples = get_available_validation_samples()
    assert len(samples) == 4, f"Expected 4 validation samples, found {len(samples)}"
    print(f"   [OK] Discovered {len(samples)} validation subjects: {[s['subject'] for s in samples]}")


def test_inference_pipeline():
    print("3. Testing End-to-End Inference Engine...")
    samples = get_available_validation_samples()
    s = samples[0]  # MGHNICU_001
    subj = s["subject"]

    adc_raw, adc_meta = load_nifti(s["adc_path"])
    z_raw, _ = load_nifti(s["z_adc_path"])
    mask_raw, _ = load_nifti(s["mask_path"])
    native_spacing = adc_meta.get_zooms()

    print(f"   Subject {subj}: Native shape={adc_raw.shape}, Spacing={native_spacing}")

    # Test Baseline U-Net inference
    unet, _ = load_baseline_unet()
    res_unet = run_model_inference(
        model=unet,
        model_type="Baseline Attention U-Net",
        adc_raw=adc_raw,
        z_raw=z_raw,
        native_spacing=native_spacing,
        mask_raw=mask_raw,
        channel_mapping="primary_adc_z_z",
        threshold=0.5,
    )
    assert res_unet["pred_vol"].shape == (128, 128, 48), f"Unexpected shape: {res_unet['pred_vol'].shape}"
    assert res_unet["metrics"] is not None, "Metrics missing!"
    print(f"   [OK] Baseline U-Net Inference: Pred voxels={res_unet['pred_voxels']}, Dice={res_unet['metrics']['dice']:.4f}")

    # Test DANN inference
    dann, _ = load_dann_model()
    res_dann = run_model_inference(
        model=dann,
        model_type="DANN Adapted U-Net",
        adc_raw=adc_raw,
        z_raw=z_raw,
        native_spacing=native_spacing,
        mask_raw=mask_raw,
        channel_mapping="primary_adc_z_z",
        threshold=0.5,
    )
    assert res_dann["pred_vol"].shape == (128, 128, 48), f"Unexpected shape: {res_dann['pred_vol'].shape}"
    assert res_dann["metrics"] is not None, "Metrics missing!"
    print(f"   [OK] DANN Model Inference: Pred voxels={res_dann['pred_voxels']}, Dice={res_dann['metrics']['dice']:.4f}")

    # Test mask export
    npz_data = export_mask_to_npz(res_dann["pred_vol"], res_dann["prob_vol"])
    assert len(npz_data) > 0, "NPZ export failed!"
    nii_data = export_mask_to_nifti(res_dann["pred_vol"])
    assert len(nii_data) > 0, "NIfTI export failed!"
    print(f"   [OK] Export buffers generated: NPZ={len(npz_data):,} bytes, NIfTI={len(nii_data):,} bytes")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING FRONTEND MODULE INTEGRATION TESTS")
    print("=" * 70)
    test_model_loading()
    test_validation_samples()
    test_inference_pipeline()
    print("=" * 70)
    print("ALL FRONTEND INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

