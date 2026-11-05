"""
sanity_check.py
===============
Quick smoke test to verify the project code is importable and
core components work correctly, WITHOUT requiring actual datasets.

Run this after installing dependencies:
    python sanity_check.py
"""

from __future__ import annotations

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

errors = []


def check(name: str, fn):
    try:
        fn()
        log.info(f"  ✓ {name}")
    except Exception as e:
        log.error(f"  ✗ {name}: {e}")
        errors.append((name, str(e)))


# ── Imports ────────────────────────────────────────────────────────────────
check("import torch",       lambda: __import__("torch"))
check("import nibabel",     lambda: __import__("nibabel"))
check("import yaml",        lambda: __import__("yaml"))
check("import numpy",       lambda: __import__("numpy"))
check("import scipy",       lambda: __import__("scipy"))
check("import matplotlib",  lambda: __import__("matplotlib"))


# ── Model architecture ────────────────────────────────────────────────────
def test_model():
    import torch
    sys.path.insert(0, "src")
    from model import UNet, count_parameters

    model = UNet(in_channels=3, out_channels=1, base_features=16, depth=3)
    n = count_parameters(model)
    log.info(f"    UNet params: {n:,}")

    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128), f"Expected (2,1,128,128), got {y.shape}"

    # Test encode / decode split (for Phase 2 DANN)
    bottleneck, skips = model.encode(x)
    out = model.decode(bottleneck, skips)
    logits = model.out_conv(out)
    assert logits.shape == (2, 1, 128, 128)


check("UNet forward pass",          test_model)


# ── Losses ────────────────────────────────────────────────────────────────
def test_losses():
    import torch
    sys.path.insert(0, "src")
    from losses import DiceLoss, CombinedLoss

    dice = DiceLoss()
    logits  = torch.randn(4, 1, 64, 64)
    targets = (torch.rand(4, 1, 64, 64) > 0.7).float()

    l1 = dice(logits, targets)
    assert 0 <= l1.item() <= 2.0, f"Unexpected dice loss: {l1.item()}"

    combo = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    l2 = combo(logits, targets)
    assert l2.item() > 0


check("Dice + Combined loss",       test_losses)


# ── Metrics ───────────────────────────────────────────────────────────────
def test_metrics():
    import numpy as np
    sys.path.insert(0, "src")
    from metrics import dice_score, iou_score, precision_score, recall_score, \
        sensitivity_score, MetricAccumulator

    pred   = np.array([1, 1, 0, 1, 0])
    target = np.array([1, 0, 1, 1, 0])

    d = dice_score(pred, target)
    assert 0 <= d <= 1.0

    acc = MetricAccumulator(threshold=0.5)
    import torch
    probs   = torch.rand(4, 1, 64, 64)
    targets = (torch.rand(4, 1, 64, 64) > 0.5).float()
    acc.update(probs, targets)
    res = acc.compute()
    assert all(0 <= v <= 1.0 for v in res.values()), f"Out of range: {res}"


check("Metrics accumulator",        test_metrics)


# ── Data utilities (no NIfTI files needed) ────────────────────────────────
def test_data_utils():
    import numpy as np
    sys.path.insert(0, "src")
    from data_utils import normalize_volume, _crop_or_pad, Augmentor

    vol = np.random.rand(50, 50, 20).astype(np.float32)
    norm = normalize_volume(vol, method="zscore")
    assert abs(norm.mean()) < 0.1

    sl = _crop_or_pad(vol[:, :, 0], 64, 64)
    assert sl.shape == (64, 64)

    aug = Augmentor(prob=1.0)
    img  = np.random.rand(3, 128, 128).astype(np.float32)
    mask = np.random.rand(128, 128).astype(np.float32)
    img2, mask2 = aug(img, mask)
    assert img2.shape == (3, 128, 128)
    assert mask2.shape == (128, 128)


check("Data utilities (no files)", test_data_utils)


# ── Summary ───────────────────────────────────────────────────────────────
print()
if errors:
    log.error(f"✗ {len(errors)} check(s) failed:")
    for name, err in errors:
        log.error(f"    [{name}] {err}")
    sys.exit(1)
else:
    log.info(f"✓ All checks passed! The codebase is ready.")
    log.info("  Next step: Place ISLES 2022 + BONBID-HIE data, then run:")
    log.info("    cd src && python inspect_data.py --config ../configs/config.yaml")
