"""
data_utils.py
=============
Dataset loading, preprocessing and DataLoader creation.

ISLES 2022 – Verified layout (250 patients)
============================================
  <root>/sub-strokecaseXXXX/ses-0001/
      anat/sub-strokecaseXXXX_ses-0001_FLAIR.nii       <- actual NIfTI file
      dwi/sub-strokecaseXXXX_ses-0001_dwi.nii/          <- DIRECTORY; real DWI inside
           └── <anything>_dwi.nii                       <- actual NIfTI file
      dwi/sub-strokecaseXXXX_ses-0001_adc.nii           <- DIRECTORY (240 cases)
           └── <anything>_adc.nii                       <- actual NIfTI file
                             OR
      dwi/sub-strokecaseXXXX_ses-0001_adc.nii           <- flat NIfTI file (10 cases)

  <root>/derivatives/sub-strokecaseXXXX/ses-0001/
      sub-strokecaseXXXX_ses-0001_msk.nii              <- actual NIfTI mask

Key rule: a path ending in .nii that is a DIRECTORY is a pseudo-directory;
          the actual NIfTI is the single *.nii file inside it.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Low-level NIfTI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_nii_path(p: Path) -> Optional[Path]:
    """
    Resolve a path that may be:
      (a) a real .nii / .nii.gz file  -> return it as-is
      (b) a directory whose name ends in .nii (pseudo-dir) -> return the
          single *.nii file found inside
      (c) non-existent                -> return None
    """
    if not p.exists():
        return None
    if p.is_file():
        return p
    if p.is_dir():
        # Pseudo-directory: look for the actual NIfTI inside
        candidates = sorted(p.glob("*.nii")) + sorted(p.glob("*.nii.gz"))
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            # Prefer the one whose stem contains a distinguishing keyword
            logger.debug(f"Multiple .nii files in pseudo-dir {p}: {[c.name for c in candidates]}")
            return candidates[0]   # first alphabetically
    return None


class MetaWrapper:
    """Lightweight wrapper so .mha metadata behaves like nib.Nifti1Image."""
    def __init__(self, meta: Dict[str, Any]):
        self.affine = meta.get("affine", np.eye(4, dtype=np.float64))
        self.header = self
        self._zooms = tuple(float(x) for x in meta.get("spacing_xyz", (1.0, 1.0, 1.0)))

    def get_zooms(self) -> Tuple[float, float, float]:
        return self._zooms


def read_mha(filepath: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read an uncompressed MetaImage (.mha) file into (data [H,W,D], meta)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MHA file not found: {filepath}")

    meta: Dict[str, Any] = {}
    with open(filepath, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_str = line.decode("ascii", errors="ignore").strip()
            if not line_str or line_str.startswith("#"):
                continue
            if "=" in line_str:
                k, v = [x.strip() for x in line_str.split("=", 1)]
                meta[k] = v
                if k == "ElementDataFile" and v == "LOCAL":
                    break
        raw_bytes = f.read()

    dims = [int(x) for x in meta.get("DimSize", "").split()]
    elem_type = meta.get("ElementType", "MET_FLOAT")
    dtype_map = {
        "MET_FLOAT": np.float32, "MET_DOUBLE": np.float64,
        "MET_UCHAR": np.uint8, "MET_CHAR": np.int8,
        "MET_USHORT": np.uint16, "MET_SHORT": np.int16,
        "MET_UINT": np.uint32, "MET_INT": np.int32,
    }
    dtype = dtype_map.get(elem_type, np.float32)
    raw_arr = np.frombuffer(raw_bytes, dtype=dtype)
    vol_zyx = raw_arr.reshape(dims[::-1])
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0)).astype(np.float32)

    spacing = [float(x) for x in meta.get("ElementSpacing", "1.0 1.0 1.0").split()]
    meta["spacing_xyz"] = tuple(spacing)
    meta["shape_xyz"] = vol_xyz.shape

    tf_vals = [float(x) for x in meta.get("TransformMatrix", "1 0 0 0 1 0 0 0 1").split()]
    rot_mat = np.array(tf_vals, dtype=np.float64).reshape((3, 3)) if len(tf_vals) == 9 else np.eye(3)
    offset_vals = [float(x) for x in meta.get("Offset", "0 0 0").split()]

    affine = np.eye(4, dtype=np.float64)
    if "srow_x" in meta and "srow_y" in meta and "srow_z" in meta:
        try:
            affine[0, :] = [float(x) for x in meta["srow_x"].split()]
            affine[1, :] = [float(x) for x in meta["srow_y"].split()]
            affine[2, :] = [float(x) for x in meta["srow_z"].split()]
        except Exception:
            affine[:3, :3] = rot_mat @ np.diag(spacing)
            affine[:3, 3] = offset_vals
    else:
        affine[:3, :3] = rot_mat @ np.diag(spacing)
        affine[:3, 3] = offset_vals

    meta["affine"] = affine
    return vol_xyz, meta


def load_nifti(path: Path | str) -> Tuple[np.ndarray, Any]:
    """Load a NIfTI or MetaImage (.mha) file; returns (float32 array [H,W,D], img_or_wrapper)."""
    p = Path(path)
    if p.name.lower().endswith(".mha"):
        data, meta = read_mha(p)
        return data, MetaWrapper(meta)
    img  = nib.load(str(p))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img


def get_voxel_spacing(img: nib.Nifti1Image) -> np.ndarray:
    """Return voxel spacing (mm) as float32 array of length 3."""
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    zooms[zooms <= 0] = 1.0          # guard against corrupt headers
    return zooms


def get_affine(img: nib.Nifti1Image) -> np.ndarray:
    """Return the 4×4 affine matrix."""
    return img.affine.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial alignment checks
# ─────────────────────────────────────────────────────────────────────────────

def _affines_compatible(a1: np.ndarray, a2: np.ndarray,
                         tol_mm: float = 0.5, tol_cos: float = 0.02) -> bool:
    """
    Check whether two 4×4 affines describe the same orientation and origin
    (within tolerance).  We do NOT require identical voxel size.
    """
    # Normalise rotation columns
    def normalise_rotation(aff: np.ndarray) -> np.ndarray:
        R = aff[:3, :3]
        norms = np.linalg.norm(R, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        return R / norms

    R1 = normalise_rotation(a1)
    R2 = normalise_rotation(a2)

    # Frobenius norm of rotation difference
    rot_diff = np.linalg.norm(R1 - R2, ord='fro')
    # Translation difference in mm
    trans_diff = np.linalg.norm(a1[:3, 3] - a2[:3, 3])

    return (rot_diff < tol_cos) and (trans_diff < tol_mm)


def check_alignment(imgs: Dict[str, nib.Nifti1Image],
                    subject: str) -> Tuple[bool, List[str]]:
    """
    Check pairwise affine compatibility of all modality images.

    Returns (all_ok, list_of_warning_strings).
    """
    warnings: List[str] = []
    items = list(imgs.items())
    ok = True
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            n1, img1 = items[i]
            n2, img2 = items[j]
            if not _affines_compatible(img1.affine, img2.affine):
                warnings.append(
                    f"{subject}: {n1} vs {n2} affines diverge ? resampling required"
                )
                ok = False
    return ok, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_reference(
    data:      np.ndarray,
    src_img:   nib.Nifti1Image,
    ref_img:   nib.Nifti1Image,
    is_mask:   bool = False,
) -> np.ndarray:
    """
    Resample `data` (from src_img space) into ref_img voxel space using
    scipy.ndimage.zoom with correct zoom factors derived from voxel spacings.

    For masks: nearest-neighbour (order=0).
    For images: linear (order=1).
    """
    src_spacing = get_voxel_spacing(src_img)
    ref_spacing = get_voxel_spacing(ref_img)

    zoom_factors = src_spacing / ref_spacing

    # Also account for different image sizes
    src_shape = np.array(data.shape[:3], dtype=float)
    ref_shape = np.array(ref_img.shape[:3],  dtype=float)
    size_factors = ref_shape / src_shape

    # Combined factor
    factors = zoom_factors * size_factors

    order = 0 if is_mask else 1
    resampled = zoom(data, factors[:data.ndim], order=order)

    # Clip to reference shape
    target = tuple(ref_img.shape[:3]) + data.shape[3:]
    slices = tuple(slice(0, min(resampled.shape[i], target[i]))
                   for i in range(len(target)))
    out = np.zeros(target, dtype=np.float32)
    dst_slices = tuple(slice(0, min(resampled.shape[i], target[i]))
                       for i in range(len(target)))
    out[dst_slices] = resampled[dst_slices]
    return out


def resample_to_target_spacing(
    volume:           np.ndarray,
    current_spacing:  np.ndarray,
    target_spacing:   Tuple[float, float, float] = (1.0, 1.0, 1.0),
    order:            int = 1,
) -> np.ndarray:
    """Resample a volume to isotropic target_spacing using scipy zoom."""
    factors = current_spacing / np.array(target_spacing, dtype=np.float32)
    return zoom(volume, factors, order=order).astype(np.float32)


def normalize_volume(volume: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalise a 3-D volume."""
    if method == "zscore":
        mu, sd = volume.mean(), volume.std()
        return np.zeros_like(volume) if sd < 1e-8 else ((volume - mu) / sd).astype(np.float32)
    elif method == "minmax":
        lo, hi = volume.min(), volume.max()
        rng = hi - lo
        return np.zeros_like(volume) if rng < 1e-8 else ((volume - lo) / rng).astype(np.float32)
    raise ValueError(f"Unknown normalisation: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# ISLES 2022 – exact discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_isles_nii(case_id: str, ses_dir: Path, modality: str) -> Optional[Path]:
    """
    Locate the actual NIfTI file for `modality` (flair|adc|dwi) in
    one ISLES 2022 subject/session directory.

    Handles both layouts:
      - FLAIR: anat/<case_id>_ses-0001_FLAIR.nii
      - DWI:   dwi/<case_id>_ses-0001_dwi.nii/   (pseudo-dir)  -> nii inside
      - ADC:   dwi/<case_id>_ses-0001_adc.nii/   (pseudo-dir)  -> nii inside
               OR dwi/<case_id>_ses-0001_adc.nii  (flat file)
    """
    mod_lower = modality.lower()

    if mod_lower == "flair":
        # anat sub-folder
        anat_dir = ses_dir / "anat"
        if anat_dir.exists():
            for f in anat_dir.iterdir():
                if f.name.endswith("_FLAIR.nii") or f.name.endswith("_FLAIR.nii.gz"):
                    if f.is_file():
                        return f
        return None

    elif mod_lower in ("adc", "dwi"):
        dwi_dir = ses_dir / "dwi"
        if not dwi_dir.exists():
            return None

        # The pseudo-directory name pattern:  <case_id>_ses-0001_<mod>.nii
        # Search for an entry whose name ends with _<mod>.nii (case-insensitive)
        suffix = f"_{mod_lower}.nii"
        candidates = [
            p for p in dwi_dir.iterdir()
            if p.name.lower().endswith(suffix) or p.name.lower().endswith(f"_{mod_lower}.nii.gz")
        ]
        if not candidates:
            return None

        # Prefer exact match first
        exact = [p for p in candidates if case_id in p.name]
        target = exact[0] if exact else candidates[0]

        return _resolve_nii_path(target)

    return None


def discover_isles2022(root: str | Path, modalities: List[str]) -> List[Dict]:
    """
    Discover all ISLES 2022 subjects and return a list of sample dicts.

    Each dict:
      {
        "subject":        str,              e.g. "sub-strokecase0001"
        "modality_paths": {"flair": Path, "adc": Path, "dwi": Path},
        "mask_path":      Path | None,
        "missing_mods":   [str, ...],       modalities not found
        "valid":          bool,             True iff all 3 modalities + mask present
      }
    """
    root = Path(root)
    if not root.exists():
        logger.error(f"ISLES root does not exist: {root}")
        return []

    deriv_root = root / "derivatives"

    # Collect all sub-strokecaseXXXX directories
    case_dirs = sorted(
        [d for d in root.iterdir()
         if d.is_dir() and d.name.startswith("sub-strokecase")],
        key=lambda d: d.name
    )

    samples: List[Dict] = []
    n_valid = 0

    for case_dir in case_dirs:
        case_id = case_dir.name          # e.g. sub-strokecase0001

        # Session directory (always ses-0001)
        ses_dir = case_dir / "ses-0001"
        if not ses_dir.exists():
            logger.warning(f"{case_id}: ses-0001 not found ? skipping")
            continue

        # ── Locate each modality ──────────────────────────────────────────
        mod_paths: Dict[str, Path] = {}
        missing:   List[str]       = []

        for mod in modalities:
            p = _find_isles_nii(case_id, ses_dir, mod)
            if p is not None:
                mod_paths[mod] = p
            else:
                missing.append(mod)

        # ── Locate mask in derivatives ────────────────────────────────────
        mask_path: Optional[Path] = None
        if deriv_root.exists():
            msk_candidate = (
                deriv_root / case_id / "ses-0001" /
                f"{case_id}_ses-0001_msk.nii"
            )
            if msk_candidate.exists():
                mask_path = msk_candidate
            else:
                # Fallback: recursive search
                hits = list((deriv_root / case_id).rglob("*_msk.nii")) if \
                       (deriv_root / case_id).exists() else []
                if hits:
                    mask_path = hits[0]

        valid = (len(missing) == 0) and (mask_path is not None)
        if valid:
            n_valid += 1
        elif missing:
            logger.warning(f"{case_id}: missing modalities {missing}")
        if mask_path is None:
            logger.warning(f"{case_id}: no mask found in derivatives")

        samples.append({
            "subject":        case_id,
            "modality_paths": mod_paths,
            "mask_path":      mask_path,
            "missing_mods":   missing,
            "valid":          valid,
        })

    logger.info(
        f"[ISLES2022] {len(samples)} subjects discovered, "
        f"{n_valid} fully valid (all modalities + mask)"
    )
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# BONBID-HIE discovery (unchanged – generic)
# ─────────────────────────────────────────────────────────────────────────────

def discover_bonbid(root: str | Path, split: str = "val") -> List[Dict]:
    """
    Walk a BONBID-HIE directory for the given split (train|val|test).
    Supports both MetaImage (.mha) and NIfTI formats, handling official
    BONBID2023_Train, BONBID2023_Val, and top-level directory layouts.
    Returns list of dicts with image_paths and optional mask_path.
    """
    root = Path(root)
    import re
    val_set = {"MGHNICU_001", "MGHNICU_002", "MGHNICU_302", "MGHNICU_436"}

    # Discover candidate subdirectories matching split
    candidates = []
    for d in root.iterdir():
        if d.is_dir():
            d_name = d.name.lower()
            if split.lower() in d_name:
                candidates.append(d)

    all_files: List[Path] = []
    if candidates:
        for c in candidates:
            all_files.extend(list(c.rglob("*.mha")) + list(c.rglob("*.nii*")))
    else:
        # Fall back to root or parent scan
        all_files = list(root.glob("*.mha")) + list(root.glob("*/*.mha")) + list(root.glob("*.nii*"))

    subject_map: Dict[str, Dict] = {}
    for p in all_files:
        if p.name.startswith("._") or p.is_dir():
            continue
        rel = str(p.relative_to(root)).lower()
        if "atlas" in rel:
            continue
        m = re.search(r"(MGHNICU_\d+)", p.name, re.IGNORECASE)
        if not m:
            continue
        subj = m.group(1).upper()
        # Enforce split partition
        if split == "val" and subj not in val_set and "val" not in rel:
            continue
        if split == "train" and subj in val_set and "train" not in rel:
            continue

        if subj not in subject_map:
            subject_map[subj] = {"image_paths": {}, "mask_path": None}

        lower = p.name.lower()
        if any(kw in lower for kw in ["mask", "lesion", "label", "seg", "msk"]):
            subject_map[subj]["mask_path"] = p
        elif "zmap" in lower or "z-adc" in lower or "zadc" in lower:
            subject_map[subj]["image_paths"].setdefault("z-adc", p)
        elif "adc" in lower:
            subject_map[subj]["image_paths"].setdefault("adc", p)
        elif "dwi" in lower:
            subject_map[subj]["image_paths"].setdefault("dwi", p)
        elif "flair" in lower:
            subject_map[subj]["image_paths"].setdefault("flair", p)
        else:
            subject_map[subj]["image_paths"].setdefault(p.stem, p)

    samples: List[Dict] = []
    for subj, info in sorted(subject_map.items()):
        if info["image_paths"]:
            samples.append({"subject": subj, **info})

    labelled = sum(1 for s in samples if s["mask_path"])
    logger.info(
        f"[BONBID] split='{split}': {len(samples)} subjects, "
        f"{labelled} labelled at {root}"
    )
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Volume loading + alignment pipeline (ISLES)
# ─────────────────────────────────────────────────────────────────────────────

def load_isles_sample(
    sample:         Dict,
    modalities:     List[str],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Optional[Dict]:
    """
    Load, align, and normalise one ISLES 2022 subject.

    Steps:
      1. Load each modality NIfTI as float32.
      2. Check affines for pairwise compatibility.
      3. Choose FLAIR as reference space (highest resolution / full brain).
      4. Resample ADC and DWI to FLAIR space (linear interpolation).
      5. Resample mask to FLAIR space (nearest-neighbour).
      6. Resample all to target_spacing.
      7. Z-score normalise each channel independently.

    Returns dict with keys:
      volumes  : {mod: np.ndarray [H, W, D] float32}
      mask     : np.ndarray [H, W, D] float32  (or None)
      spacing  : np.ndarray [3]  (target spacing after resampling)
      affine   : np.ndarray [4,4]  (FLAIR reference affine)
      subject  : str
      warnings : List[str]
    """
    subject = sample["subject"]
    mod_paths = sample["modality_paths"]
    mask_path = sample["mask_path"]

    # ── Load all available modalities ─────────────────────────────────────
    raw_vols: Dict[str, Tuple[np.ndarray, nib.Nifti1Image]] = {}
    for mod in modalities:
        if mod not in mod_paths:
            continue
        try:
            data, img = load_nifti(mod_paths[mod])
            if data.ndim == 4:
                data = data[..., 0]          # take first volume if 4-D
            raw_vols[mod] = (data, img)
        except Exception as e:
            logger.error(f"{subject}/{mod}: load failed ? {e}")

    if not raw_vols:
        logger.error(f"{subject}: no volumes loaded ? skipping")
        return None

    # ── Load mask ─────────────────────────────────────────────────────────
    raw_mask:     Optional[np.ndarray]       = None
    raw_mask_img: Optional[nib.Nifti1Image]  = None
    if mask_path:
        try:
            raw_mask, raw_mask_img = load_nifti(mask_path)
            raw_mask = (raw_mask > 0.5).astype(np.float32)
        except Exception as e:
            logger.error(f"{subject}/mask: load failed ? {e}")

    warnings: List[str] = []

    # ── Reference space: DWI (mask is annotated in DWI space) ─────────────
    # DWI and mask share 2mm isotropic space; FLAIR is high-res and must be
    # downsampled to DWI space (NOT the other way around).
    ref_mod = "dwi" if "dwi" in raw_vols else (
        "adc" if "adc" in raw_vols else list(raw_vols.keys())[0]
    )
    ref_data, ref_img = raw_vols[ref_mod]

    # ── Affine compatibility check ─────────────────────────────────────────
    imgs_for_check = {m: iv[1] for m, iv in raw_vols.items()}
    aligned, align_warns = check_alignment(imgs_for_check, subject)
    warnings.extend(align_warns)

    # ── Align all modalities to FLAIR space ───────────────────────────────
    aligned_vols: Dict[str, np.ndarray] = {ref_mod: ref_data}

    for mod, (data, img) in raw_vols.items():
        if mod == ref_mod:
            continue
        if not _affines_compatible(ref_img.affine, img.affine, tol_mm=2.0, tol_cos=0.05):
            warnings.append(f"{subject}/{mod}: resampling to {ref_mod} space")
            data = resample_to_reference(data, img, ref_img, is_mask=False)
        else:
            # Shapes may still differ slightly
            if data.shape != ref_data.shape:
                data = resample_to_reference(data, img, ref_img, is_mask=False)
        aligned_vols[mod] = data

    # Fill missing modalities with zeros (keeps 3-channel input intact)
    ref_shape = ref_data.shape
    for mod in modalities:
        if mod not in aligned_vols:
            warnings.append(f"{subject}/{mod}: MISSING ? zero-filled")
            aligned_vols[mod] = np.zeros(ref_shape, dtype=np.float32)

    # ── Align mask to DWI reference space ────────────────────────────────
    # The mask is annotated in DWI space; check and resample if needed.
    aligned_mask: Optional[np.ndarray] = None
    if raw_mask is not None and raw_mask_img is not None:
        if not _affines_compatible(ref_img.affine, raw_mask_img.affine, tol_mm=2.0, tol_cos=0.05):
            warnings.append(f"{subject}/mask: affine differs from {ref_mod} ? resampling (NN)")
            aligned_mask = resample_to_reference(raw_mask, raw_mask_img, ref_img, is_mask=True)
        elif raw_mask.shape[:3] != ref_data.shape[:3]:
            warnings.append(f"{subject}/mask: shape {raw_mask.shape} != {ref_data.shape[:3]} ? resampling")
            aligned_mask = resample_to_reference(raw_mask, raw_mask_img, ref_img, is_mask=True)
        else:
            aligned_mask = raw_mask
        aligned_mask = (aligned_mask > 0.5).astype(np.float32)

    # ── Resample all to target isotropic spacing ──────────────────────────
    spacing = get_voxel_spacing(ref_img)
    resampled_vols: Dict[str, np.ndarray] = {}
    for mod, vol in aligned_vols.items():
        resampled_vols[mod] = resample_to_target_spacing(vol, spacing, target_spacing, order=1)

    if aligned_mask is not None:
        aligned_mask = resample_to_target_spacing(
            aligned_mask, spacing, target_spacing, order=0
        )
        aligned_mask = (aligned_mask > 0.5).astype(np.float32)

    # ── Z-score normalise ─────────────────────────────────────────────────
    for mod in resampled_vols:
        resampled_vols[mod] = normalize_volume(resampled_vols[mod])

    return {
        "subject":  subject,
        "volumes":  resampled_vols,   # {mod: [H,W,D] float32}
        "mask":     aligned_mask,
        "spacing":  np.array(target_spacing, dtype=np.float32),
        "affine":   ref_img.affine,
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Slice extraction
# ─────────────────────────────────────────────────────────────────────────────

def _crop_or_pad(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Centre-crop or zero-pad a 2-D array to (target_h, target_w)."""
    h, w = arr.shape
    ph = max(0, target_h - h); pw = max(0, target_w - w)
    if ph or pw:
        arr = np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
        h, w = arr.shape
    sh = (h - target_h) // 2; sw = (w - target_w) // 2
    return arr[sh: sh + target_h, sw: sw + target_w].astype(np.float32)


def extract_slices(
    volumes:     Dict[str, np.ndarray],   # {mod: [H,W,D]}
    mask:        Optional[np.ndarray],    # [H,W,D] or None
    modalities:  List[str],               # channel order
    image_size:  Tuple[int, int],
    num_slices:  int,
    lesion_bias: float = 0.7,
) -> List[Dict]:
    """
    Extract num_slices axial 2-D slices, biased toward lesion slices.

    Returns list of {"image": [C,H,W], "mask": [H,W]}.
    """
    H, W = image_size
    # Use first available volume to determine depth
    ref_vol = next(iter(volumes.values()))
    depth = ref_vol.shape[2]

    # Identify lesion-containing slices
    if mask is not None and mask.max() > 0:
        lesion_z = [z for z in range(depth) if mask[:, :, z].max() > 0]
    else:
        lesion_z = []
    non_lesion_z = [z for z in range(depth) if z not in lesion_z]

    n_lesion = int(num_slices * lesion_bias) if lesion_z else 0
    n_non    = num_slices - n_lesion

    chosen: List[int] = []
    if lesion_z:
        chosen += random.choices(lesion_z, k=min(n_lesion, len(lesion_z)))
    if non_lesion_z:
        chosen += random.choices(non_lesion_z, k=min(n_non, len(non_lesion_z)))
    while len(chosen) < num_slices:
        chosen.append(random.randint(0, depth - 1))

    slices = []
    for z in chosen:
        channels = []
        for mod in modalities:
            vol = volumes.get(mod)
            if vol is None:
                sl = np.zeros((H, W), dtype=np.float32)
            else:
                sl = _crop_or_pad(vol[:, :, z], H, W)
            channels.append(sl)

        img_arr = np.stack(channels, axis=0)   # [C, H, W]

        if mask is not None:
            msk_sl = _crop_or_pad(mask[:, :, z], H, W)
            msk_sl = (msk_sl > 0.5).astype(np.float32)
        else:
            msk_sl = np.zeros((H, W), dtype=np.float32)

        slices.append({"image": img_arr, "mask": msk_sl})

    return slices


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class Augmentor:
    """Spatial + intensity augmentations applied on 2-D slices [C,H,W]."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if random.random() < self.prob:
            image = image[:, :, ::-1].copy(); mask = mask[:, ::-1].copy()
        if random.random() < self.prob:
            image = image[:, ::-1, :].copy(); mask = mask[::-1, :].copy()
        if random.random() < self.prob:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask  = np.rot90(mask,  k=k, axes=(0, 1)).copy()
        if random.random() < self.prob:
            for c in range(image.shape[0]):
                image[c] = image[c] * random.uniform(0.85, 1.15) + random.uniform(-0.1, 0.1)
        if random.random() < self.prob:
            image = image + np.random.normal(0, 0.05, image.shape).astype(np.float32)
        return image, mask


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Datasets
# ─────────────────────────────────────────────────────────────────────────────

class ISLESDataset(Dataset):
    """
    ISLES 2022 slice dataset.

    Loads all valid samples during __init__, extracts 2-D slices and
    caches them in memory.  Augmentation is applied on-the-fly in __getitem__.
    """

    def __init__(
        self,
        samples:        List[Dict],
        modalities:     List[str],
        image_size:     Tuple[int, int],
        num_slices:     int,
        target_spacing: Tuple[float, float, float],
        augment:        bool  = False,
        aug_prob:       float = 0.5,
    ):
        self.modalities  = modalities
        self.augmentor   = Augmentor(prob=aug_prob) if augment else None
        self.slice_cache: List[Dict] = []

        skipped = 0
        for sample in samples:
            try:
                loaded = load_isles_sample(sample, modalities, target_spacing)
            except Exception as e:
                logger.error(f"{sample['subject']}: loading error ? {e}")
                skipped += 1
                continue

            if loaded is None:
                skipped += 1
                continue

            for w in loaded["warnings"]:
                logger.warning(w)

            slices = extract_slices(
                loaded["volumes"], loaded["mask"],
                modalities, image_size, num_slices,
            )
            self.slice_cache.extend(slices)

        logger.info(
            f"[ISLESDataset] {len(self.slice_cache)} slices from "
            f"{len(samples) - skipped}/{len(samples)} subjects"
        )

    def __len__(self) -> int:
        return len(self.slice_cache)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.slice_cache[idx]
        img  = item["image"].copy()
        msk  = item["mask"].copy()
        if self.augmentor:
            img, msk = self.augmentor(img, msk)
        return (
            torch.from_numpy(img).float(),
            torch.from_numpy(msk).unsqueeze(0).float(),
        )


class BONBIDDataset(Dataset):
    """BONBID-HIE neonatal MRI slice dataset (generic, 3-channel)."""

    def __init__(
        self,
        samples:        List[Dict],
        image_size:     Tuple[int, int],
        num_slices:     int,
        target_spacing: Tuple[float, float, float],
        augment:        bool  = False,
        aug_prob:       float = 0.3,
    ):
        self.augmentor   = Augmentor(prob=aug_prob) if augment else None
        self.slice_cache: List[Dict] = []

        for sample in samples:
            image_paths = sample.get("image_paths", {})
            mask_path   = sample.get("mask_path")
            vols: Dict[str, np.ndarray] = {}
            spacing = None

            for mod, path in image_paths.items():
                try:
                    data, img = load_nifti(path)
                    vols[mod] = normalize_volume(data)
                    if spacing is None:
                        spacing = get_voxel_spacing(img)
                except Exception as e:
                    logger.error(f"{sample['subject']}/{mod}: {e}")

            if not vols:
                continue

            # Pad to 3 channels
            keys = list(vols.keys())
            while len(keys) < 3:
                keys.append(keys[-1])
            vols_3 = {f"ch{i}": vols[k] for i, k in enumerate(keys[:3])}

            if spacing is not None:
                vols_3 = {m: resample_to_target_spacing(v, spacing, target_spacing)
                          for m, v in vols_3.items()}

            mask = None
            if mask_path:
                try:
                    m, mimg = load_nifti(mask_path)
                    if spacing is not None:
                        m = resample_to_target_spacing(m, get_voxel_spacing(mimg),
                                                       target_spacing, order=0)
                    mask = (m > 0.5).astype(np.float32)
                except Exception as e:
                    logger.error(f"{sample['subject']}/mask: {e}")

            slices = extract_slices(
                vols_3, mask,
                list(vols_3.keys()), image_size, num_slices,
            )
            self.slice_cache.extend(slices)

        logger.info(f"[BONBIDDataset] {len(self.slice_cache)} slices")

    def __len__(self) -> int:
        return len(self.slice_cache)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.slice_cache[idx]
        img  = item["image"].copy()
        msk  = item["mask"].copy()
        if self.augmentor:
            img, msk = self.augmentor(img, msk)
        return (
            torch.from_numpy(img).float(),
            torch.from_numpy(msk).unsqueeze(0).float(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────────────────────────────

def make_isles_loaders(
    isles_root:     str,
    modalities:     List[str],
    image_size:     Tuple[int, int],
    num_slices:     int,
    target_spacing: Tuple[float, float, float],
    batch_size:     int,
    val_fraction:   float = 0.15,
    seed:           int   = 42,
    aug_prob:       float = 0.5,
    num_workers:    int   = 0,
) -> Tuple[DataLoader, DataLoader, List[Dict]]:
    """Return (train_loader, val_loader, all_samples)."""
    all_samples = discover_isles2022(isles_root, modalities)
    # Only train on fully valid samples (all modalities + mask)
    valid = [s for s in all_samples if s["valid"]]
    invalid_count = len(all_samples) - len(valid)
    if invalid_count:
        logger.warning(f"{invalid_count} subjects excluded (missing modalities/mask)")
    if not valid:
        raise RuntimeError(
            f"No valid ISLES 2022 samples found under '{isles_root}'.\n"
            "Check dataset path and layout in config.yaml."
        )

    rng = random.Random(seed)
    rng.shuffle(valid)
    n_val   = max(1, int(len(valid) * val_fraction))
    val_s   = valid[:n_val]
    train_s = valid[n_val:]
    logger.info(f"[ISLES] Train={len(train_s)} Val={len(val_s)} subjects")

    train_ds = ISLESDataset(train_s, modalities, image_size, num_slices,
                             target_spacing, augment=True,  aug_prob=aug_prob)
    val_ds   = ISLESDataset(val_s,   modalities, image_size, num_slices,
                             target_spacing, augment=False)

    import torch
    _pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=_pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=_pin)
    return train_loader, val_loader, all_samples


def make_bonbid_loader(
    bonbid_root:    str,
    split:          str,
    image_size:     Tuple[int, int],
    num_slices:     int,
    target_spacing: Tuple[float, float, float],
    batch_size:     int,
    augment:        bool = False,
    num_workers:    int  = 0,
) -> DataLoader:
    samples = discover_bonbid(bonbid_root, split)
    if not samples:
        raise RuntimeError(
            f"No BONBID-HIE data found under '{bonbid_root}/{split}'."
        )
    ds = BONBIDDataset(samples, image_size, num_slices, target_spacing,
                        augment=augment)
    import torch
    _pin = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=(split == "train"),
                      num_workers=num_workers, pin_memory=_pin)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset inspection helper (called by inspect_data.py)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_datasets(isles_root: str, bonbid_root: str, modalities: List[str]) -> Dict:
    """Return a JSON-serialisable inspection report for both datasets."""
    report: Dict = {
        "ISLES2022":  {"root": isles_root,  "subjects": []},
        "BONBID-HIE": {"root": bonbid_root, "splits": {}},
    }
    isles_samples = discover_isles2022(isles_root, modalities)
    for s in isles_samples:
        report["ISLES2022"]["subjects"].append({
            "subject":        s["subject"],
            "modalities_found": list(s["modality_paths"].keys()),
            "missing":        s["missing_mods"],
            "has_mask":       s["mask_path"] is not None,
            "valid":          s["valid"],
        })
    report["ISLES2022"]["total"]   = len(isles_samples)
    report["ISLES2022"]["valid"]   = sum(s["valid"] for s in isles_samples)
    report["ISLES2022"]["invalid"] = sum(not s["valid"] for s in isles_samples)

    for split in ["train", "val", "test"]:
        bonbid_s = discover_bonbid(bonbid_root, split)
        report["BONBID-HIE"]["splits"][split] = {
            "count":   len(bonbid_s),
            "labelled": sum(1 for s in bonbid_s if s.get("mask_path")),
        }
    return report
