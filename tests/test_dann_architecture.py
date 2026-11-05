"""
test_dann_architecture.py
=========================
Unit and integration tests for Phase 4: Domain-Adversarial Neural Network (DANN).

Tests:
  1. Gradient Reversal Layer (GRL) forward identity and backward negation.
  2. Ganin schedule lambda progression.
  3. DANN U-Net model forward pass (segmentation + domain classification shapes).
  4. Feature extractor, label predictor, and domain classifier gradient flow.
  5. Safety verification: Baseline checkpoints remain strictly untouched.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import hashlib
import torch
import torch.nn as nn
from dann_model import DANNUNet, GradientReversal, grad_reverse, compute_dann_lambda, build_dann


def test_grl_forward_backward():
    print("1. Testing Gradient Reversal Layer (GRL)...")
    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    lambda_val = 0.75
    y = grad_reverse(x, lambda_val)

    # Forward must be identity
    assert torch.allclose(x, y), "GRL forward pass altered values!"

    loss = (y * torch.tensor([2.0, 3.0, -1.0])).sum()
    loss.backward()

    # Backward gradient should be -lambda * incoming_grad
    expected_grad = -lambda_val * torch.tensor([2.0, 3.0, -1.0])
    assert torch.allclose(x.grad, expected_grad), f"GRL backward failed! Got {x.grad}, expected {expected_grad}"
    print("   [OK] GRL forward is identity, backward strictly negates and scales gradients.")


def test_dann_lambda_schedule():
    print("2. Testing Ganin Lambda Schedule...")
    l0 = compute_dann_lambda(0.0)
    l5 = compute_dann_lambda(0.5)
    l1 = compute_dann_lambda(1.0)

    assert abs(l0 - 0.0) < 1e-5, f"Lambda at p=0 should be 0.0, got {l0}"
    assert 0.98 < l5 < 0.99, f"Lambda at p=0.5 unexpected: {l5}"
    assert 0.999 < l1 <= 1.0, f"Lambda at p=1.0 unexpected: {l1}"
    print(f"   [OK] Ganin schedule: p=0.0 -> {l0:.3f}, p=0.5 -> {l5:.3f}, p=1.0 -> {l1:.4f}")


def test_dann_model_dimensions():
    print("3. Testing DANN Model Dimensions and Tensor Shapes...")
    model = DANNUNet(dropout=0.2)
    model.train()

    B = 2
    dummy_input = torch.randn(B, 3, 128, 128)

    # Forward with domain classification
    seg_logits, domain_logits = model(dummy_input, lambda_=1.0, return_domain=True)

    assert seg_logits.shape == (B, 1, 128, 128), f"Unexpected seg_logits shape: {seg_logits.shape}"
    assert domain_logits.shape == (B, 1), f"Unexpected domain_logits shape: {domain_logits.shape}"

    # Forward without domain classification (inference mode)
    inf_logits = model(dummy_input, return_domain=False)
    assert inf_logits.shape == (B, 1, 128, 128), f"Unexpected inference shape: {inf_logits.shape}"
    print("   [OK] All output tensor dimensions are correct: seg=(B, 1, 128, 128), domain=(B, 1).")


def test_adversarial_gradient_flow():
    print("4. Testing Adversarial Gradient Flow...")
    model = DANNUNet()
    model.train()

    B = 2
    x_s = torch.randn(B, 3, 128, 128)
    y_s = torch.zeros(B, 1, 128, 128)
    x_t = torch.randn(B, 3, 128, 128)

    # 1. Forward source
    seg_s, dom_s = model(x_s, lambda_=1.0, return_domain=True)
    # 2. Forward target (strictly features only)
    f_t, _ = model.encode(x_t)
    dom_t = model.classify_domain(f_t, lambda_=1.0)

    # Losses
    l_seg = nn.BCEWithLogitsLoss()(seg_s, y_s)
    l_dom_s = nn.BCEWithLogitsLoss()(dom_s, torch.zeros_like(dom_s))
    l_dom_t = nn.BCEWithLogitsLoss()(dom_t, torch.ones_like(dom_t))
    l_total = l_seg + 0.5 * (l_dom_s + l_dom_t)

    l_total.backward()

    # Verify gradients exist across feature extractor, label predictor, and domain classifier
    assert model.unet.inc.conv[0].weight.grad is not None, "Feature extractor received no gradient!"
    assert model.unet.out_conv.weight.grad is not None, "Label predictor received no gradient!"
    assert model.domain_classifier.classifier[0].weight.grad is not None, "Domain classifier received no gradient!"

    print("   [OK] Gradient flow verified across feature extractor, segmentation head, and domain classifier.")


def test_baseline_checkpoint_integrity():
    print("5. Verifying Baseline Checkpoint Protection...")
    ckpt_path = project_root / "outputs/checkpoints/baseline_unet_best.pth"
    assert ckpt_path.exists(), f"Baseline checkpoint not found: {ckpt_path}"
    size = ckpt_path.stat().st_size
    assert size == 23373919, f"Baseline checkpoint size altered! Expected 23373919, got {size}"

    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    h = hasher.hexdigest()
    assert h.startswith("6f9469aa01629ce2"), f"Baseline checkpoint hash altered! Got: {h}"
    print(f"   [OK] Baseline checkpoint {ckpt_path.name} is 100% untouched (SHA-256 verified).")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PHASE 4 DANN ARCHITECTURE & UNIT TESTS")
    print("=" * 70)
    test_grl_forward_backward()
    test_dann_lambda_schedule()
    test_dann_model_dimensions()
    test_adversarial_gradient_flow()
    test_baseline_checkpoint_integrity()
    print("=" * 70)
    print("ALL PHASE 4 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

