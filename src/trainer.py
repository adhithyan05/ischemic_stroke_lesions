"""
trainer.py
==========
Training loop for Phase 1 (supervised baseline U-Net).

Features:
  - Mixed-precision training (torch.amp)
  - Learning-rate scheduling (cosine / step / plateau)
  - Warmup epochs
  - Early stopping
  - Best-checkpoint saving (by validation Dice)
  - CSV + console logging per epoch
  - Gradient clipping
"""

from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from metrics import MetricAccumulator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────

def get_device(cfg: dict) -> torch.device:
    setting = cfg.get("training", {}).get("device", "auto")
    if setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(setting)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class BaselineTrainer:
    """
    Supervised trainer for Phase 1 baseline U-Net.

    Args:
        model:        UNet (or any nn.Module with the same interface).
        criterion:    CombinedLoss or any criterion(logits, targets).
        train_loader: ISLES 2022 training DataLoader.
        val_loader:   ISLES 2022 internal validation DataLoader.
        cfg:          Full config dict (from config.yaml).
        checkpoint_dir: Directory to save model checkpoints.
        log_dir:       Directory to save CSV training logs.
    """

    def __init__(
        self,
        model:          nn.Module,
        criterion:      nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        cfg:            dict,
        checkpoint_dir: str,
        log_dir:        str,
    ):
        self.model        = model
        self.criterion    = criterion
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.ckpt_dir     = Path(checkpoint_dir)
        self.log_dir      = Path(log_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg.get("training", {})
        self.num_epochs    = tcfg.get("num_epochs", 100)
        self.lr            = tcfg.get("learning_rate", 1e-4)
        self.weight_decay  = tcfg.get("weight_decay", 1e-5)
        self.grad_clip     = tcfg.get("grad_clip", 1.0)
        self.patience      = tcfg.get("patience", 20)
        self.min_delta     = tcfg.get("min_delta", 1e-4)
        self.warmup_epochs = tcfg.get("warmup_epochs", 5)
        self.threshold     = cfg.get("evaluation", {}).get("threshold", 0.5)
        self.scheduler_type = tcfg.get("lr_scheduler", "cosine")

        self.device = get_device(cfg)
        self.model  = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        # Optimiser
        self.optimiser = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # LR Scheduler
        self.scheduler = self._build_scheduler()

        # AMP scaler (only for CUDA)
        self.use_amp  = (self.device.type == "cuda")
        self.scaler   = GradScaler(device=self.device.type, enabled=self.use_amp)

        # Metric accumulators
        self.train_acc = MetricAccumulator(threshold=self.threshold)
        self.val_acc   = MetricAccumulator(threshold=self.threshold)

        # Early stopping state
        self.best_val_dice   = -1.0
        self.patience_counter = 0

        # CSV log
        self.csv_path = self.log_dir / "baseline_training_log.csv"
        self._init_csv()

        logger.info(
            f"Trainer initialised | device={self.device} | "
            f"epochs={self.num_epochs} | amp={self.use_amp}"
        )

    # ── Scheduler factory ─────────────────────────────────────────────────

    def _build_scheduler(self):
        if self.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimiser,
                T_max=max(1, self.num_epochs - self.warmup_epochs),
                eta_min=self.lr * 1e-3,
            )
        elif self.scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimiser, mode="max", factor=0.5, patience=10, verbose=True
            )
        elif self.scheduler_type == "step":
            return StepLR(self.optimiser, step_size=30, gamma=0.5)
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_type}")

    # ── CSV logging ───────────────────────────────────────────────────────

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "lr",
                "train_loss", "val_loss",
                "train_dice", "val_dice",
                "val_iou", "val_precision", "val_recall", "val_sensitivity",
                "epoch_time_s",
            ])

    def _log_csv(self, epoch: int, data: dict):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                data.get("lr", 0.0),
                f'{data.get("train_loss", 0):.6f}',
                f'{data.get("val_loss", 0):.6f}',
                f'{data.get("train_dice", 0):.4f}',
                f'{data.get("val_dice", 0):.4f}',
                f'{data.get("val_iou", 0):.4f}',
                f'{data.get("val_precision", 0):.4f}',
                f'{data.get("val_recall", 0):.4f}',
                f'{data.get("val_sensitivity", 0):.4f}',
                f'{data.get("epoch_time_s", 0):.1f}',
            ])

    # ── Warmup ────────────────────────────────────────────────────────────

    def _get_warmup_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs and self.warmup_epochs > 0:
            return self.lr * (epoch + 1) / self.warmup_epochs
        return None   # use scheduler

    def _set_lr(self, lr: float):
        for pg in self.optimiser.param_groups:
            pg["lr"] = lr

    # ── Single epoch ──────────────────────────────────────────────────────

    def _train_epoch(self) -> Tuple[float, Dict]:
        self.model.train()
        self.train_acc.reset()
        total_loss = 0.0
        n_batches  = 0

        for images, masks in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            self.optimiser.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimiser)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimiser)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1

            probs = torch.sigmoid(logits.detach())
            self.train_acc.update(probs.cpu(), masks.cpu())

        avg_loss = total_loss / max(n_batches, 1)
        metrics  = self.train_acc.compute()
        return avg_loss, metrics

    @torch.no_grad()
    def _val_epoch(self) -> Tuple[float, Dict]:
        self.model.eval()
        self.val_acc.reset()
        total_loss = 0.0
        n_batches  = 0

        for images, masks in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            total_loss += loss.item()
            n_batches  += 1

            probs = torch.sigmoid(logits)
            self.val_acc.update(probs.cpu(), masks.cpu())

        avg_loss = total_loss / max(n_batches, 1)
        metrics  = self.val_acc.compute()
        return avg_loss, metrics

    # ── Checkpoint ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_dice: float, tag: str = "best"):
        path = self.ckpt_dir / f"baseline_unet_{tag}.pth"
        torch.save({
            "epoch":      epoch,
            "val_dice":   val_dice,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimiser.state_dict(),
            "cfg":         self.cfg,
        }, path)
        logger.info(f"  [OK] Checkpoint saved -> {path} (val_dice={val_dice:.4f})")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimiser.load_state_dict(ckpt["optim_state"])
        logger.info(f"Loaded checkpoint from {path} (epoch={ckpt['epoch']})")
        return ckpt

    # ── Main training loop ────────────────────────────────────────────────

    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns: dict with final best validation metrics.
        """
        logger.info("=" * 70)
        logger.info("Phase 1: Baseline U-Net Training (ISLES 2022 -> supervised)")
        logger.info("=" * 70)

        best_metrics = {}

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()

            # ── LR warmup ────────────────────────────────────────────────
            warmup_lr = self._get_warmup_lr(epoch - 1)
            if warmup_lr is not None:
                self._set_lr(warmup_lr)

            # ── Train ─────────────────────────────────────────────────────
            train_loss, train_metrics = self._train_epoch()

            # ── Validate ──────────────────────────────────────────────────
            val_loss, val_metrics = self._val_epoch()

            # ── Scheduler step ────────────────────────────────────────────
            if epoch > self.warmup_epochs:
                if self.scheduler_type == "plateau":
                    self.scheduler.step(val_metrics["dice"])
                else:
                    self.scheduler.step()

            current_lr = self.optimiser.param_groups[0]["lr"]
            elapsed    = time.time() - t0

            # ── Console logging ───────────────────────────────────────────
            logger.info(
                f"Epoch [{epoch:03d}/{self.num_epochs:03d}] "
                f"lr={current_lr:.2e} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_dice={val_metrics['dice']:.4f} | "
                f"val_iou={val_metrics['iou']:.4f} | "
                f"time={elapsed:.1f}s"
            )

            # ── CSV log ───────────────────────────────────────────────────
            self._log_csv(epoch, {
                "lr":              current_lr,
                "train_loss":      train_loss,
                "val_loss":        val_loss,
                "train_dice":      train_metrics["dice"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "epoch_time_s":    elapsed,
            })

            # ── Checkpoint ────────────────────────────────────────────────
            improvement = val_metrics["dice"] - self.best_val_dice
            if improvement > self.min_delta:
                self.best_val_dice    = val_metrics["dice"]
                self.patience_counter = 0
                best_metrics          = val_metrics.copy()
                best_metrics["epoch"] = epoch
                self._save_checkpoint(epoch, val_metrics["dice"], tag="best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

            # Always save last checkpoint
            if epoch == self.num_epochs:
                self._save_checkpoint(epoch, val_metrics["dice"], tag="last")

        logger.info("=" * 70)
        logger.info("Training complete.")
        logger.info(f"Best val Dice: {self.best_val_dice:.4f} at epoch {best_metrics.get('epoch', '?')}")
        logger.info("=" * 70)

        return best_metrics
