"""
inference.py
============
Inference and preprocessing engine for the Streamlit web frontend.
Connects directly to the existing model code and preprocessing pipeline.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import normalize_volume, resample_to_target_spacing, _crop_or_pad
from evaluate_neonatal_zeroshot import compute_volume_metrics, adapt_channels


def run_model_inference(
    model: torch.nn.Module,
    model_type: str,
    adc_raw: np.ndarray,
    z_raw: np.ndarray,
    native_spacing: Tuple[float, float, float],
    mask_raw: Optional[np.ndarray] = None,
    channel_mapping: str = "primary_adc_z_z",
    threshold: float = 0.5,
    target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    image_size: Tuple[int, int] = (128, 128),
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Run full 3D volumetric lesion segmentation inference.

    Args:
        model: Loaded PyTorch U-Net or DANN model.
        model_type: 'Baseline U-Net' or 'DANN'.
        adc_raw: Native 3D ADC volume.
        z_raw: Native 3D Z-ADC volume.
        native_spacing: Original voxel dimensions (dx, dy, dz) in mm.
        mask_raw: Optional ground-truth mask volume for quantitative validation.
        channel_mapping: 'primary_adc_z_z' or 'surrogate_z_adc_dwi'.
        threshold: Sigmoid probability binarization threshold (default: 0.5).
        target_spacing: Resampling spacing (default: (2.0, 2.0, 2.0)).
        image_size: Target slice matrix dimensions (default: (128, 128)).
        device: Torch device (CPU or CUDA).

    Returns:
        Dictionary containing slices, 3D predicted mask, probabilities,
        lesion volume statistics, and metrics if ground-truth is available.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()

    # 1. Intensity Normalization (Z-score standardisation)
    adc_norm = normalize_volume(adc_raw, method="zscore")
    z_norm   = normalize_volume(z_raw, method="zscore")

    # 2. Continuous interpolation (order=1) for MRI volumes
    adc_res = resample_to_target_spacing(adc_norm, native_spacing, target_spacing, order=1)
    z_res   = resample_to_target_spacing(z_norm, native_spacing, target_spacing, order=1)

    # 3. Nearest-neighbour interpolation (order=0) strictly for segmentation masks
    mask_res = None
    if mask_raw is not None:
        m = resample_to_target_spacing(mask_raw, native_spacing, target_spacing, order=0)
        mask_res = (m > 0.5).astype(np.float32)

    # 4. Input channel adaptation
    channels_3d = adapt_channels(adc_res, z_res, mapping_name=channel_mapping)

    H, W = image_size
    depth = adc_res.shape[2]

    pred_vol = np.zeros((H, W, depth), dtype=np.float32)
    prob_vol = np.zeros((H, W, depth), dtype=np.float32)
    adc_slices = []
    z_slices = []
    gt_vol = np.zeros((H, W, depth), dtype=np.float32) if mask_res is not None else None

    with torch.no_grad():
        for z in range(depth):
            ch_slices = [_crop_or_pad(ch[:, :, z], H, W) for ch in channels_3d]
            adc_slices.append(ch_slices[0])
            z_slices.append(ch_slices[1])

            slice_tensor = np.stack(ch_slices, axis=0)  # [3, H, W]
            inp = torch.from_numpy(slice_tensor).unsqueeze(0).float().to(device)

            if "dann" in model_type.lower():
                logits = model(inp, return_domain=False)
            else:
                logits = model(inp)

            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (prob >= threshold).astype(np.float32)

            prob_vol[:, :, z] = prob
            pred_vol[:, :, z] = pred

            if mask_res is not None:
                gt_sl = _crop_or_pad(mask_res[:, :, z], H, W)
                gt_vol[:, :, z] = gt_sl

    # 5. Volume and lesion quantification
    voxel_vol_mm3 = float(target_spacing[0] * target_spacing[1] * target_spacing[2])  # 8.0 mm³
    pred_voxels = int((pred_vol >= threshold).sum())
    pred_volume_mm3 = float(pred_voxels * voxel_vol_mm3)

    # Find slices with lesions
    slices_with_lesions = [z for z in range(depth) if pred_vol[:, :, z].sum() > 0]

    # Metrics if GT available
    metrics = None
    if gt_vol is not None:
        metrics = compute_volume_metrics(pred_vol, gt_vol, threshold=threshold)

    return {
        "model_type": model_type,
        "depth": depth,
        "image_size": image_size,
        "target_spacing": target_spacing,
        "pred_vol": pred_vol,
        "prob_vol": prob_vol,
        "adc_slices": adc_slices,
        "z_slices": z_slices,
        "gt_vol": gt_vol,
        "pred_voxels": pred_voxels,
        "pred_volume_mm3": pred_volume_mm3,
        "slices_with_lesions": slices_with_lesions,
        "metrics": metrics,
    }


def export_mask_to_npz(pred_vol: np.ndarray, prob_vol: np.ndarray) -> bytes:
    """Export 3D prediction and probability volume to compressed NPZ bytes."""
    buf = io.BytesIO()
    np.savez_compressed(buf, pred=pred_vol.astype(np.uint8), prob=prob_vol.astype(np.float32))
    buf.seek(0)
    return buf.getvalue()


def export_mask_to_nifti(pred_vol: np.ndarray, spacing=(2.0, 2.0, 2.0)) -> bytes:
    """Export 3D prediction to NIfTI (.nii.gz) bytes using nibabel."""
    import nibabel as nib
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    nii = nib.Nifti1Image(pred_vol.astype(np.uint8), affine=affine)
    buf = io.BytesIO()
    file_map = nii.make_file_map()
    file_map["image"].fileobj = buf
    nii.to_file_map(file_map)
    buf.seek(0)
    return buf.getvalue()

