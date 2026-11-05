"""
data_loader.py
==============
Data loading, parsing, and metric aggregation utilities for the web frontend.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils import load_nifti, read_mha, MetaWrapper


def get_available_validation_samples() -> List[Dict]:
    """Discover available sample subjects from the official validation cohort."""
    val_dir = PROJECT_ROOT / "data/BONBID-HIE/BONBID2023_Val"
    if not val_dir.exists():
        return []

    adc_dir = val_dir / "1ADC_ss"
    z_dir   = val_dir / "2Z_ADC"
    lbl_dir = val_dir / "3LABEL"

    samples = []
    if adc_dir.exists():
        for f in sorted(adc_dir.glob("*.mha")):
            subj = f.name.split("-")[0].split("_ADC")[0]
            z_files = list(z_dir.glob(f"*{subj}*.mha")) if z_dir.exists() else []
            lbl_files = list(lbl_dir.glob(f"*{subj}*.mha")) if lbl_dir.exists() else []

            if z_files:
                samples.append({
                    "subject": subj,
                    "adc_path": str(f),
                    "z_adc_path": str(z_files[0]),
                    "mask_path": str(lbl_files[0]) if lbl_files else None,
                    "split": "Validation Cohort",
                })
    return samples


def load_mha_from_upload(uploaded_file) -> Tuple[np.ndarray, MetaWrapper]:
    """
    Load a MetaImage (.mha) file from a Streamlit UploadedFile object.
    Writes temporarily to disk to reuse robust file-based header parsing.
    """
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        vol, meta = load_nifti(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return vol, meta


def load_dann_training_history() -> Optional[Dict]:
    """Load DANN training metrics from outputs/logs/dann/."""
    json_path = PROJECT_ROOT / "outputs/logs/dann/dann_training_history.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_zeroshot_baseline_metrics() -> Optional[Dict]:
    """Load Phase 3 zero-shot metrics from outputs/logs/."""
    json_path = PROJECT_ROOT / "outputs/logs/neonatal_zeroshot_metrics.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_dann_evaluation_metrics() -> Optional[Dict]:
    """Load Phase 4 DANN evaluation metrics on neonatal validation cohort."""
    json_path = PROJECT_ROOT / "outputs/logs/dann/dann_evaluation_metrics.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

