"""
train_baseline.py
=================
Entry point for Phase 1: Train supervised baseline U-Net on ISLES 2022.

Pipeline:
  1. Load config
  2. Inspect datasets (print summary)
  3. Build DataLoaders (ISLES 2022 only – no neonatal data)
  4. Build U-Net + loss + trainer
  5. Train with early stopping
  6. Evaluate best checkpoint on BONBID-HIE validation split
  7. Save all results

Usage:
    python train_baseline.py
    python train_baseline.py --config ../configs/config.yaml
    python train_baseline.py --no-eval   # skip neonatal evaluation after training
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

import yaml

from data_utils import (
    make_isles_loaders,
    make_bonbid_loader,
    inspect_datasets,
)
from model   import build_unet, count_parameters
from losses  import build_loss
from trainer import BaselineTrainer, get_device
from evaluate import evaluate_on_neonatal


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "baseline_training.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler - force UTF-8 encoding on Windows (avoids cp1252 crashes)
    import io
    utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    ) if hasattr(sys.stdout, 'buffer') else sys.stdout
    ch = logging.StreamHandler(utf8_stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Config ────────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = SRC_DIR.parent / "configs" / "config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    paths  = cfg["paths"]
    dcfg   = cfg["data"]
    tcfg   = cfg["training"]
    evcfg  = cfg.get("evaluation", {})

    logger = setup_logging(paths["log_dir"])
    logger.info("=" * 70)
    logger.info("Domain Adaptation for Brain Lesion Segmentation")
    logger.info("PHASE 1: Supervised Baseline U-Net")
    logger.info("=" * 70)
    logger.info(f"Config: {config_path}")

    # ── Dataset inspection ────────────────────────────────────────────────
    logger.info("\n--- Dataset Inspection ---")
    try:
        inspection = inspect_datasets(
            paths["isles_root"], paths["bonbid_root"], dcfg["isles_modalities"]
        )
        insp_path = Path(paths["log_dir"]) / "inspection_report.json"
        with open(insp_path, "w") as f:
            json.dump(inspection, f, indent=2)
        logger.info(
            f"Datasets: ISLES={inspection['ISLES2022']['total']} subjects | "
            f"BONBID-val={inspection['BONBID-HIE']['splits'].get('val', {}).get('count', '?')}"
        )
    except Exception as e:
        logger.error(f"Inspection failed: {e}")

    # ── DataLoaders ────────────────────────────────────────────────────────
    logger.info("\n--- Building DataLoaders (ISLES 2022 only) ---")
    train_loader, val_loader, _ = make_isles_loaders(
        isles_root     = paths["isles_root"],
        modalities     = dcfg["isles_modalities"],
        image_size     = tuple(dcfg["image_size"]),
        num_slices     = dcfg["num_slices_per_volume"],
        target_spacing = tuple(dcfg["target_spacing"]),
        batch_size     = tcfg["batch_size"],
        val_fraction   = dcfg["isles_val_fraction"],
        seed           = dcfg["random_seed"],
        aug_prob       = tcfg["aug_prob"] if tcfg["augment"] else 0.0,
    )
    logger.info(
        f"Train batches: {len(train_loader)} | "
        f"Val batches: {len(val_loader)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("\n--- Building U-Net ---")
    device = get_device(cfg)
    model  = build_unet(cfg)
    n_params = count_parameters(model)
    logger.info(f"Model: Attention U-Net | Params: {n_params:,} | Device: {device}")

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = build_loss(cfg)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = BaselineTrainer(
        model          = model,
        criterion      = criterion,
        train_loader   = train_loader,
        val_loader     = val_loader,
        cfg            = cfg,
        checkpoint_dir = paths["checkpoint_dir"],
        log_dir        = paths["log_dir"],
    )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("\n--- Training ---")
    best_metrics = trainer.train()

    # ── Evaluate on neonatal validation set ───────────────────────────────
    if not args.no_eval:
        logger.info("\n--- Evaluating on BONBID-HIE neonatal validation set ---")
        checkpoint_path = str(Path(paths["checkpoint_dir"]) / "baseline_unet_best.pth")

        if not Path(checkpoint_path).exists():
            logger.warning(
                f"Best checkpoint not found at {checkpoint_path}. "
                f"Skipping neonatal evaluation."
            )
        else:
            try:
                eval_report = evaluate_on_neonatal(
                    model           = model,
                    cfg             = cfg,
                    device          = device,
                    bonbid_root     = paths["bonbid_root"],
                    checkpoint_path = checkpoint_path,
                    viz_dir         = paths["viz_dir"],
                    split           = "val",
                    save_n_vis      = evcfg.get("save_n_vis", 10),
                )
                logger.info("\n[OK] Neonatal evaluation complete.")
            except Exception as e:
                logger.error(f"Neonatal evaluation failed: {e}", exc_info=True)

    logger.info("\n[OK] Phase 1 complete.")
    logger.info(f"  Checkpoints -> {paths['checkpoint_dir']}")
    logger.info(f"  Logs        -> {paths['log_dir']}")
    logger.info(f"  Visuals     -> {paths['viz_dir']}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline U-Net (Phase 1)")
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip neonatal evaluation after training",
    )
    args = parser.parse_args()
    main(args)
