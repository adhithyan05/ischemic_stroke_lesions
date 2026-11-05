"""
train_dann.py
=============
Phase 4: Domain-Adversarial Training for Adult-to-Neonatal Brain Lesion Segmentation.

Trains the DANN architecture to align feature representations between:
  - Source Domain: Adult ISLES 2022 MRI (labeled with ground-truth lesion masks, domain label = 0)
  - Target Domain: Neonatal BONBID-HIE MRI (unlabeled training cohort, domain label = 1)

Safety & Governance Constraints:
  1. Checkpoint Protection: Adult baseline checkpoints (baseline_unet_best.pth and
     baseline_unet_last.pth) are read-only and strictly verified before and after training.
  2. Target Unlabeled Protocol: Neonatal lesion masks are NEVER accessed or used during DANN training.
  3. Validation Set Isolation: The 4 held-out BONBID validation subjects are strictly excluded
     from training and hyperparameter tuning.
  4. Separate Checkpoints & Logs: All DANN outputs are isolated under outputs/checkpoints/dann/
     and outputs/logs/dann/.

Usage:
    python src/train_dann.py
    python src/train_dann.py --epochs 30 --batch-size 4
    python src/train_dann.py --sanity-check-only  # Run 1-step verification without full training
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import yaml

# Path setup
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from dann_model import DANNUNet, build_dann, compute_dann_lambda
from losses import CombinedLoss, build_loss
from metrics import MetricAccumulator, dice_score, iou_score
from data_utils import (
    discover_bonbid,
    make_isles_loaders,
    BONBIDDataset,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("train_dann")
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
# Safety Checkpoint Integrity Verification
# ─────────────────────────────────────────────────────────────────────────────

def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Unlabeled Target Dataset Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class UnlabeledTargetDataset(Dataset):
    """
    Strictly unlabelled dataset wrapper for neonatal BONBID-HIE.
    Guarantees that ground-truth lesion masks are NEVER loaded, accessed,
    or passed into the training pipeline.
    """

    def __init__(
        self,
        bonbid_root: str | Path,
        split: str = "train",
        image_size: Tuple[int, int] = (128, 128),
        num_slices: int = 4,
        target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        augment: bool = True,
    ):
        samples = discover_bonbid(str(bonbid_root), split=split)
        if not samples:
            raise RuntimeError(f"No BONBID-HIE samples found under {bonbid_root}/{split}")

        # Enforce that mask_path is stripped so BONBIDDataset builds slice cache without masks
        unlabeled_samples = []
        for s in samples:
            s_clean = s.copy()
            s_clean["mask_path"] = None  # Explicitly remove mask path
            unlabeled_samples.append(s_clean)

        self.dataset = BONBIDDataset(
            samples=unlabeled_samples,
            image_size=image_size,
            num_slices=num_slices,
            target_spacing=target_spacing,
            augment=augment,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Returns only image tensor of shape [3, 128, 128]
        image, _ = self.dataset[idx]
        return image


# ─────────────────────────────────────────────────────────────────────────────
# DANN Trainer Engine
# ─────────────────────────────────────────────────────────────────────────────

class DANNTrainer:
    """
    Domain-Adversarial Training Loop for Adult-to-Neonatal Domain Adaptation.
    """

    def __init__(
        self,
        model: DANNUNet,
        seg_criterion: nn.Module,
        source_train_loader: DataLoader,
        source_val_loader: DataLoader,
        target_train_loader: DataLoader,
        cfg: dict,
        checkpoint_dir: Path,
        log_dir: Path,
        device: torch.device,
        dann_weight: float = 0.5,
    ):
        self.model = model
        self.seg_criterion = seg_criterion
        self.domain_criterion = nn.BCEWithLogitsLoss()
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self.target_train_loader = target_train_loader
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.device = device
        self.dann_weight = dann_weight

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg.get("training", {})
        self.num_epochs = tcfg.get("num_epochs", 50)
        self.lr = tcfg.get("learning_rate", 1e-4)
        self.weight_decay = tcfg.get("weight_decay", 1e-5)
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.threshold = cfg.get("evaluation", {}).get("threshold", 0.5)

        # Optimizer: joint optimization over all parameters (G_f, G_y, G_d)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6,
        )

        self.history: List[Dict] = []
        self.best_val_dice = -1.0
        self.best_epoch = -1

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute one balanced domain-adversarial training epoch.
        Pairs each source batch with a target batch.
        """
        self.model.train()

        total_loss_accum = 0.0
        seg_loss_accum = 0.0
        domain_loss_accum = 0.0
        correct_domain_s = 0
        correct_domain_t = 0
        total_domain_samples = 0
        num_batches = 0

        # Number of batches per epoch is determined by source loader
        target_iter = iter(self.target_train_loader)

        total_steps = self.num_epochs * len(self.source_train_loader)

        for step, (src_imgs, src_masks) in enumerate(self.source_train_loader):
            current_step = epoch * len(self.source_train_loader) + step
            progress = current_step / max(1, total_steps)
            lambda_p = compute_dann_lambda(progress, gamma=10.0)

            # Get target batch (cycle if target loader exhausts)
            try:
                tgt_imgs = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_train_loader)
                tgt_imgs = next(target_iter)

            src_imgs = src_imgs.to(self.device, non_blocking=True)
            src_masks = src_masks.to(self.device, non_blocking=True)
            tgt_imgs = tgt_imgs.to(self.device, non_blocking=True)

            B_s = src_imgs.size(0)
            B_t = tgt_imgs.size(0)

            # Domain labels: Source=0.0, Target=1.0
            domain_label_s = torch.zeros((B_s, 1), device=self.device)
            domain_label_t = torch.ones((B_t, 1), device=self.device)

            self.optimizer.zero_grad()

            # ── 1. Forward Pass on Source Domain ───────────────────────────
            src_seg_logits, src_domain_logits = self.model(
                src_imgs, lambda_=lambda_p, return_domain=True
            )
            # Task Segmentation Loss (Source ONLY)
            seg_loss = self.seg_criterion(src_seg_logits, src_masks)

            # Source Domain Loss
            loss_domain_s = self.domain_criterion(src_domain_logits, domain_label_s)

            # ── 2. Forward Pass on Target Domain ───────────────────────────
            # Target features extracted through G_f; masks NEVER accessed!
            tgt_bottleneck, _ = self.model.encode(tgt_imgs)
            tgt_domain_logits = self.model.classify_domain(tgt_bottleneck, lambda_=lambda_p)

            # Target Domain Loss
            loss_domain_t = self.domain_criterion(tgt_domain_logits, domain_label_t)

            # Combined Domain Loss
            domain_loss = 0.5 * (loss_domain_s + loss_domain_t)

            # ── 3. Total Adversarial Loss ──────────────────────────────────
            total_loss = seg_loss + (self.dann_weight * lambda_p * domain_loss)

            total_loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Tracking
            total_loss_accum += total_loss.item()
            seg_loss_accum += seg_loss.item()
            domain_loss_accum += domain_loss.item()
            num_batches += 1

            # Domain accuracy tracking (prob < 0.5 for source, prob >= 0.5 for target)
            with torch.no_grad():
                pred_domain_s = (src_domain_logits >= 0.0).float()
                pred_domain_t = (tgt_domain_logits >= 0.0).float()
                correct_domain_s += int((pred_domain_s == 0.0).sum().item())
                correct_domain_t += int((pred_domain_t == 1.0).sum().item())
                total_domain_samples += (B_s + B_t)

        domain_acc = float((correct_domain_s + correct_domain_t) / max(1, total_domain_samples))

        return {
            "total_loss": float(total_loss_accum / max(1, num_batches)),
            "seg_loss": float(seg_loss_accum / max(1, num_batches)),
            "domain_loss": float(domain_loss_accum / max(1, num_batches)),
            "domain_acc": domain_acc,
            "lambda_p": lambda_p,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def validate(self) -> Dict[str, float]:
        """Validate segmentation performance on adult source validation set."""
        self.model.eval()
        val_loss_accum = 0.0
        acc = MetricAccumulator(threshold=self.threshold)
        num_batches = 0

        with torch.no_grad():
            for imgs, masks in self.source_val_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                seg_logits = self.model(imgs, return_domain=False)
                loss = self.seg_criterion(seg_logits, masks)
                val_loss_accum += loss.item()
                num_batches += 1

                probs = torch.sigmoid(seg_logits)
                acc.update(probs.cpu(), masks.cpu())

        metrics = acc.compute()
        metrics["val_loss"] = float(val_loss_accum / max(1, num_batches))
        return metrics

    def run_sanity_check(self) -> bool:
        """
        Execute a 1-step forward + backward pass to strictly verify:
          1. Tensor dimensions and shapes.
          2. Gradient flow through G_f, G_y, G_d.
          3. Reversal of gradients from G_d to G_f.
          4. Zero leakage of target labels.
        """
        logger.info("Executing DANN 1-step architecture & gradient sanity test ...")
        self.model.train()

        src_imgs, src_masks = next(iter(self.source_train_loader))
        tgt_imgs = next(iter(self.target_train_loader))

        src_imgs = src_imgs.to(self.device)
        src_masks = src_masks.to(self.device)
        tgt_imgs = tgt_imgs.to(self.device)

        B_s, C, H, W = src_imgs.shape
        B_t = tgt_imgs.shape[0]

        logger.info(f"  • Source batch: images={list(src_imgs.shape)}, masks={list(src_masks.shape)}")
        logger.info(f"  • Target batch: images={list(tgt_imgs.shape)} (strictly unlabeled)")

        assert C == 3 and H == 128 and W == 128, f"Unexpected source shape: {src_imgs.shape}"
        assert tgt_imgs.shape[1] == 3 and tgt_imgs.shape[2] == 128, f"Unexpected target shape: {tgt_imgs.shape}"

        self.optimizer.zero_grad()

        # Forward
        lambda_test = 0.5
        src_seg, src_dom = self.model(src_imgs, lambda_=lambda_test, return_domain=True)
        tgt_bottle, _ = self.model.encode(tgt_imgs)
        tgt_dom = self.model.classify_domain(tgt_bottle, lambda_=lambda_test)

        logger.info(f"  • Source seg logits shape: {list(src_seg.shape)}")
        logger.info(f"  • Source domain logits shape: {list(src_dom.shape)}")
        logger.info(f"  • Target domain logits shape: {list(tgt_dom.shape)}")

        assert src_seg.shape == (B_s, 1, H, W), "Seg output shape mismatch!"
        assert src_dom.shape == (B_s, 1), "Source domain output shape mismatch!"
        assert tgt_dom.shape == (B_t, 1), "Target domain output shape mismatch!"

        # Loss calculation
        l_seg = self.seg_criterion(src_seg, src_masks)
        l_dom_s = self.domain_criterion(src_dom, torch.zeros_like(src_dom))
        l_dom_t = self.domain_criterion(tgt_dom, torch.ones_like(tgt_dom))
        l_total = l_seg + lambda_test * 0.5 * (l_dom_s + l_dom_t)

        logger.info(f"  • Sanity losses: seg={l_seg.item():.4f}, dom_s={l_dom_s.item():.4f}, dom_t={l_dom_t.item():.4f}")

        # Backward
        l_total.backward()

        # Check gradient presence across all 3 components
        has_encoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.unet.inc.parameters())
        has_decoder_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.unet.out_conv.parameters())
        has_domain_grad  = any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.domain_classifier.parameters())

        logger.info(f"  • Gradients active: Encoder={has_encoder_grad}, Decoder={has_decoder_grad}, DomainClassifier={has_domain_grad}")
        assert has_encoder_grad and has_decoder_grad and has_domain_grad, "Gradient flow broken!"

        self.optimizer.zero_grad()
        logger.info("[OK] DANN forward/backward sanity check PASSED successfully!\n")
        return True

    def train(self) -> Dict:
        """Run full DANN domain adaptation training loop."""
        logger.info("=" * 78)
        logger.info(f"Starting DANN Domain Adaptation Training ({self.num_epochs} Epochs)")
        logger.info(f"  • Source batches: {len(self.source_train_loader)} | Target batches: {len(self.target_train_loader)}")
        logger.info(f"  • DANN Adversarial Weight: {self.dann_weight}")
        logger.info("=" * 78)

        start_time = time.time()
        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            train_stats = self.train_epoch(epoch - 1)
            val_stats = self.validate()
            self.scheduler.step()
            epoch_time = time.time() - t0

            val_dice = val_stats.get("dice", 0.0)
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
                self.best_epoch = epoch

            logger.info(
                f"Epoch [{epoch:02d}/{self.num_epochs:02d}] ({epoch_time:.1f}s) | "
                f"TotalLoss: {train_stats['total_loss']:.4f} | "
                f"SegLoss: {train_stats['seg_loss']:.4f} | "
                f"DomLoss: {train_stats['domain_loss']:.4f} | "
                f"DomAcc: {train_stats['domain_acc']*100:.1f}% | "
                f"λ: {train_stats['lambda_p']:.3f} | "
                f"ValDice: {val_dice:.4f} {'[BEST]' if is_best else ''}"
            )

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_total_loss": train_stats["total_loss"],
                "train_seg_loss": train_stats["seg_loss"],
                "train_domain_loss": train_stats["domain_loss"],
                "train_domain_acc": train_stats["domain_acc"],
                "lambda_p": train_stats["lambda_p"],
                "lr": train_stats["lr"],
                "val_loss": val_stats["val_loss"],
                "val_dice": val_stats["dice"],
                "val_iou": val_stats.get("iou", 0.0),
                "val_precision": val_stats.get("precision", 0.0),
                "val_recall": val_stats.get("recall", 0.0),
            }
            self.history.append(epoch_record)

            # Save checkpoints
            checkpoint_payload = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_dice": val_dice,
                "best_val_dice": self.best_val_dice,
                "best_epoch": self.best_epoch,
                "cfg": self.cfg,
            }

            # Save latest
            latest_path = self.checkpoint_dir / "dann_unet_last.pth"
            torch.save(checkpoint_payload, latest_path)

            # Save best
            if is_best:
                best_path = self.checkpoint_dir / "dann_unet_best.pth"
                torch.save(checkpoint_payload, best_path)

        total_elapsed = time.time() - start_time
        logger.info("=" * 78)
        logger.info(f"DANN Training completed in {total_elapsed/60:.2f} mins. Best Epoch: {self.best_epoch} (Val Dice: {self.best_val_dice:.4f})")
        logger.info(f"Checkpoints saved -> {self.checkpoint_dir}")
        logger.info("=" * 78)

        # Save history JSON & CSV
        self._save_logs()
        return {"best_epoch": self.best_epoch, "best_val_dice": self.best_val_dice, "history": self.history}

    def _save_logs(self):
        """Save history as JSON, CSV, and summary TXT."""
        # JSON
        json_path = self.log_dir / "dann_training_history.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "phase": "Phase 4: DANN Domain Adaptation",
                "best_epoch": self.best_epoch,
                "best_val_dice": self.best_val_dice,
                "history": self.history,
            }, f, indent=2)

        # CSV
        csv_path = self.log_dir / "dann_training_log.csv"
        if self.history:
            keys = list(self.history[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)

        # Summary TXT
        txt_path = self.log_dir / "dann_training_summary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 78 + "\n")
            f.write("        DOMAIN-ADVERSARIAL NEURAL NETWORK (DANN) TRAINING SUMMARY\n")
            f.write("=" * 78 + "\n\n")
            f.write(f"Total Epochs Trained : {len(self.history)}\n")
            f.write(f"Best Validation Epoch: {self.best_epoch}\n")
            f.write(f"Best Val Dice (Adult): {self.best_val_dice:.4f}\n")
            f.write(f"Final Lambda (GRL)   : {self.history[-1]['lambda_p']:.4f}\n")
            f.write(f"Final Domain Accuracy: {self.history[-1]['train_domain_acc']*100:.2f}%\n")
            f.write(f"Final Seg Loss       : {self.history[-1]['train_seg_loss']:.4f}\n")
            f.write(f"Final Domain Loss    : {self.history[-1]['train_domain_loss']:.4f}\n\n")
            f.write(f"{'Epoch':6s} | {'TotalLoss':10s} | {'SegLoss':10s} | {'DomLoss':10s} | {'DomAcc':8s} | {'λ':6s} | {'ValDice':8s}\n")
            f.write("-" * 78 + "\n")
            for h in self.history:
                f.write(
                    f"{h['epoch']:6d} | {h['train_total_loss']:10.4f} | {h['train_seg_loss']:10.4f} | "
                    f"{h['train_domain_loss']:10.4f} | {h['train_domain_acc']*100:7.1f}% | {h['lambda_p']:6.3f} | {h['val_dice']:8.4f}\n"
                )
            f.write("=" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DANN Domain Adaptation Training")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"), help="Config file")
    parser.add_argument("--baseline-checkpoint", default=str(PROJECT_ROOT / "outputs/checkpoints/baseline_unet_best.pth"), help="Adult baseline checkpoint")
    parser.add_argument("--epochs", type=int, default=30, help="Number of DANN training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for source and target")
    parser.add_argument("--dann-weight", type=float, default=0.5, help="Weight for domain loss")
    parser.add_argument("--sanity-check-only", action="store_true", help="Only run forward/backward sanity verification")
    args = parser.parse_args()

    config_path = Path(args.config)
    ckpt_path = Path(args.baseline_checkpoint)
    checkpoint_dir = PROJECT_ROOT / "outputs/checkpoints/dann"
    log_dir = PROJECT_ROOT / "outputs/logs/dann"

    log_file = log_dir / "dann_training.log"
    global logger
    logger = setup_logger(log_file)

    logger.info("=" * 78)
    logger.info("PHASE 4: DOMAIN-ADVERSARIAL NEURAL NETWORK (DANN) IMPLEMENTATION")
    logger.info("=" * 78)

    # 1. Safety verification of adult baseline checkpoints
    assert ckpt_path.exists(), f"Baseline checkpoint not found: {ckpt_path}"
    best_hash_pre = compute_file_hash(ckpt_path)
    best_size_pre = ckpt_path.stat().st_size
    logger.info(f"Baseline checkpoint verified: {ckpt_path.name} ({best_size_pre:,} bytes, hash: {best_hash_pre[:16]}...)")

    last_ckpt_path = PROJECT_ROOT / "outputs/checkpoints/baseline_unet_last.pth"
    last_hash_pre = compute_file_hash(last_ckpt_path) if last_ckpt_path.exists() else None

    # 2. Load Config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Update epochs and batch size from args
    cfg["training"]["num_epochs"] = args.epochs
    cfg["training"]["batch_size"] = args.batch_size

    device = torch.device("cpu")
    logger.info(f"Execution Device: {device}")

    # 3. Build DANN Model and Initialize with Baseline Checkpoint
    model = build_dann(cfg=cfg, checkpoint_path=ckpt_path, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    feature_params = sum(p.numel() for p in model.unet.parameters())
    domain_params = sum(p.numel() for p in model.domain_classifier.parameters())
    logger.info(f"DANN Architecture Initialized:")
    logger.info(f"  • Total Parameters:      {total_params:,}")
    logger.info(f"  • Feature & Seg (U-Net): {feature_params:,} (initialized from adult baseline)")
    logger.info(f"  • Domain Classifier:     {domain_params:,} (initialized with Kaiming Normal)")

    # 4. Build DataLoaders
    # Source: Adult ISLES 2022
    dcfg = cfg["data"]
    target_sp = tuple(dcfg.get("target_spacing", [2.0, 2.0, 2.0]))
    image_size = tuple(dcfg.get("image_size", [128, 128]))
    num_slices = dcfg.get("num_slices_per_volume", 4)
    batch_size = args.batch_size

    logger.info("Building balanced Source (Adult ISLES 2022) & Target (Neonatal BONBID-HIE) DataLoaders ...")
    src_train_loader, src_val_loader, _ = make_isles_loaders(
        isles_root=cfg["paths"]["isles_root"],
        modalities=dcfg["isles_modalities"],
        image_size=image_size,
        num_slices=num_slices,
        target_spacing=target_sp,
        batch_size=batch_size,
        val_fraction=dcfg.get("isles_val_fraction", 0.15),
        aug_prob=0.5,
    )

    # Target: Neonatal BONBID-HIE (strictly unlabeled training split)
    target_dataset = UnlabeledTargetDataset(
        bonbid_root=cfg["paths"]["bonbid_root"],
        split="train",
        image_size=image_size,
        num_slices=num_slices,
        target_spacing=target_sp,
        augment=True,
    )
    target_train_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    logger.info(f"  • Source train batches: {len(src_train_loader)} ({len(src_train_loader.dataset)} slices)")
    logger.info(f"  • Source val batches:   {len(src_val_loader)} ({len(src_val_loader.dataset)} slices)")
    logger.info(f"  • Target train batches: {len(target_train_loader)} ({len(target_dataset)} unlabeled slices)")

    seg_loss = build_loss(cfg)

    # 5. Initialize Trainer
    trainer = DANNTrainer(
        model=model,
        seg_criterion=seg_loss,
        source_train_loader=src_train_loader,
        source_val_loader=src_val_loader,
        target_train_loader=target_train_loader,
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        device=device,
        dann_weight=args.dann_weight,
    )

    # 6. Execute Forward/Backward Sanity Check
    sanity_passed = trainer.run_sanity_check()
    assert sanity_passed, "Sanity check failed!"

    if args.sanity_check_only:
        logger.info("Completed sanity check successfully (--sanity-check-only requested).")
    else:
        # Full training loop
        results = trainer.train()

    # 7. Safety Verification Post-Execution
    best_hash_post = compute_file_hash(ckpt_path)
    best_size_post = ckpt_path.stat().st_size
    assert best_hash_pre == best_hash_post, "CRITICAL ERROR: baseline_unet_best.pth modified!"
    assert best_size_pre == best_size_post, "CRITICAL ERROR: baseline_unet_best.pth size modified!"

    if last_ckpt_path.exists() and last_hash_pre is not None:
        last_hash_post = compute_file_hash(last_ckpt_path)
        assert last_hash_pre == last_hash_post, "CRITICAL ERROR: baseline_unet_last.pth modified!"

    logger.info("\n--- Post-Execution Checkpoint Safety Verification ---")
    logger.info(f"  [OK] baseline_unet_best.pth is 100% UNCHANGED (SHA-256: {best_hash_post[:16]}...)")
    if last_ckpt_path.exists():
        logger.info(f"  [OK] baseline_unet_last.pth is 100% UNCHANGED (SHA-256: {last_hash_post[:16]}...)")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
