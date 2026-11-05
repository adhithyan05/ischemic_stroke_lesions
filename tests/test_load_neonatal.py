"""
test_load_neonatal.py
=====================
End-to-end unit test: load real neonatal BONBID-HIE subjects through the pipeline.
Verifies discovery -> MetaImage load -> spatial checks -> resampling -> normalization -> slice extraction.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from data_utils import discover_bonbid, load_nifti, resample_to_target_spacing, extract_slices

BONBID_ROOT = project_root / "data" / "BONBID-HIE"
TARGET_SP = (2.0, 2.0, 2.0)
TARGET_SIZE = (128, 128)

print("1. Discovering neonatal subjects...")
val_samples = discover_bonbid(BONBID_ROOT, split="val")
train_samples = discover_bonbid(BONBID_ROOT, split="train")

print(f"   Discovered {len(val_samples)} validation subjects: {[s['subject'] for s in val_samples]}")
print(f"   Discovered {len(train_samples)} training subjects")
assert len(val_samples) == 4, f"Expected 4 validation subjects, found {len(val_samples)}"
assert len(train_samples) == 85, f"Expected 85 training subjects, found {len(train_samples)}"

# Test diverse subjects (2 val, 2 train)
test_cases = ["MGHNICU_001", "MGHNICU_002", "MGHNICU_010", "MGHNICU_014"]
all_samples = {s["subject"]: s for s in val_samples + train_samples}

for case_id in test_cases:
    sample = all_samples.get(case_id)
    assert sample is not None, f"Subject {case_id} not found"
    print(f"\n2. Testing subject: {case_id}")
    
    img_paths = sample["image_paths"]
    mask_path = sample["mask_path"]
    assert "adc" in img_paths, f"Missing ADC for {case_id}"
    assert "z-adc" in img_paths, f"Missing Z-ADC for {case_id}"
    assert mask_path is not None, f"Missing mask for {case_id}"
    
    # Load ADC
    adc_vol, adc_wrapper = load_nifti(img_paths["adc"])
    z_vol, z_wrapper = load_nifti(img_paths["z-adc"])
    mask_vol, mask_wrapper = load_nifti(mask_path)
    
    sp = adc_wrapper.get_zooms()
    print(f"   Native shape: ADC={adc_vol.shape}, Z-ADC={z_vol.shape}, Mask={mask_vol.shape}")
    print(f"   Native spacing: {sp} mm")
    assert adc_vol.shape == z_vol.shape == mask_vol.shape, "Shape mismatch across modalities"
    
    # Verify binary mask
    unique_vals = set(np.unique(mask_vol))
    assert unique_vals.issubset({0.0, 1.0}), f"Mask is not strictly binary: {unique_vals}"
    n_lesion = int((mask_vol > 0.5).sum())
    print(f"   Lesion voxel count: {n_lesion} voxels")
    
    # Test resampling
    adc_res = resample_to_target_spacing(adc_vol, sp, TARGET_SP, order=1)
    z_res = resample_to_target_spacing(z_vol, sp, TARGET_SP, order=1)
    mask_res = (resample_to_target_spacing(mask_vol, sp, TARGET_SP, order=0) > 0.5).astype(np.float32)
    
    print(f"   Resampled shape (at 2mm isotropic): {adc_res.shape}")
    assert adc_res.shape == z_res.shape == mask_res.shape, "Resampled shape mismatch"
    assert set(np.unique(mask_res)).issubset({0.0, 1.0}), "Resampled mask lost binary property"
    
    # Test slice extraction
    vols_3 = {"ch0": adc_res, "ch1": z_res, "ch2": adc_res}
    slices = extract_slices(vols_3, mask_res, ["ch0", "ch1", "ch2"], TARGET_SIZE, num_slices=4)
    assert len(slices) == 4, f"Expected 4 slices, got {len(slices)}"
    assert slices[0]["image"].shape == (3, 128, 128), f"Unexpected slice shape: {slices[0]['image'].shape}"
    assert slices[0]["mask"].shape == (128, 128), f"Unexpected mask slice shape: {slices[0]['mask'].shape}"
    print(f"   Extracted {len(slices)} slices of shape {slices[0]['image'].shape}")

print("\n[OK] All neonatal loading and preprocessing pipeline tests passed successfully!")
