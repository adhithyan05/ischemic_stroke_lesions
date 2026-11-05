"""
Quick end-to-end test: load one real ISLES 2022 patient through the full pipeline.
Verifies discovery -> load -> align -> resample -> normalize -> slice extraction.
"""
import sys
sys.path.insert(0, r"C:\ADHITHYAN\ischemic stroke lesions\src")

from data_utils import discover_isles2022, load_isles_sample, extract_slices
import numpy as np

ISLES_ROOT = r"C:\ADHITHYAN\ischemic stroke lesions\ISLES-2022\ISLES-2022"
MODALITIES  = ["dwi", "adc", "flair"]
TARGET_SP   = (2.0, 2.0, 2.0)

print("1. Discovering patients...")
samples = discover_isles2022(ISLES_ROOT, MODALITIES)
valid   = [s for s in samples if s["valid"]]
print(f"   Found {len(valid)} valid patients\n")

# Test 3 diverse patients
for case_id in ["sub-strokecase0001", "sub-strokecase0034", "sub-strokecase0200"]:
    sample = next((s for s in valid if s["subject"] == case_id), None)
    if not sample:
        print(f"   {case_id}: NOT FOUND")
        continue

    print(f"2. Loading {case_id}...")
    result = load_isles_sample(sample, MODALITIES, TARGET_SP)
    if result is None:
        print(f"   ERROR: load returned None")
        continue

    if result["warnings"]:
        for w in result["warnings"]:
            print(f"   WARN: {w}")

    vols  = result["volumes"]
    mask  = result["mask"]
    sp    = result["spacing"]

    print(f"   Spacing  : {sp}")
    for mod, vol in vols.items():
        print(f"   {mod:6s}: shape={vol.shape}  mean={vol.mean():.3f}  std={vol.std():.3f}")
    if mask is not None:
        n_lesion = int((mask > 0.5).sum())
        print(f"   mask    : shape={mask.shape}  lesion voxels={n_lesion}")
    else:
        print(f"   mask    : None (no mask)")

    # Test slice extraction
    slices = extract_slices(vols, mask, MODALITIES, (256, 256), num_slices=4)
    print(f"   slices  : {len(slices)} extracted")
    s0 = slices[0]
    print(f"   slice[0]: image={s0['image'].shape}  mask={s0['mask'].shape}\n")

print("✓ Pipeline test complete.")
