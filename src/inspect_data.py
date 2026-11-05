"""
inspect_data.py
===============
Comprehensive dataset integrity inspector for ISLES 2022.

Checks performed for each of the 250 patients:
  1.  FLAIR, ADC, DWI file discovery (handles pseudo-dir layout)
  2.  Mask discovery in derivatives/
  3.  NIfTI loadability (can nibabel open the file?)
  4.  Volume shape (H × W × D)
  5.  Voxel spacing (mm)
  6.  Affine matrix (printed for first patient; checked pairwise for rest)
  7.  Cross-modality affine compatibility (orientation + translation)
  8.  Mask-to-image affine compatibility
  9.  Lesion voxel count in mask
  10. Missing / anomalous modality report

Produces:
  outputs/logs/inspection_report.json   – machine-readable full report
  outputs/logs/inspection_summary.txt   – human-readable summary table

Usage:
    cd "C:\\ADHITHYAN\\ischemic stroke lesions\\src"
    python inspect_data.py --config ../configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import nibabel as nib
import numpy as np
import yaml

from data_utils import (
    _find_isles_nii,
    _resolve_nii_path,
    _affines_compatible,
    discover_isles2022,
    discover_bonbid,
    get_voxel_spacing,
    get_affine,
    load_nifti,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-patient inspector
# ─────────────────────────────────────────────────────────────────────────────

def inspect_one_patient(sample: Dict, modalities: List[str]) -> Dict:
    """
    Run all integrity checks for a single ISLES 2022 patient.

    Returns a detailed result dict.
    """
    subject   = sample["subject"]
    mod_paths = sample["modality_paths"]
    mask_path = sample["mask_path"]
    missing   = sample["missing_mods"]

    result: Dict[str, Any] = {
        "subject":       subject,
        "valid":         sample["valid"],
        "missing_mods":  missing,
        "modalities":    {},
        "mask":          None,
        "alignment":     {"all_ok": True, "issues": []},
        "errors":        [],
    }

    # ── Inspect each modality ─────────────────────────────────────────────
    loaded_imgs: Dict[str, nib.Nifti1Image] = {}
    ref_img: Optional[nib.Nifti1Image] = None
    ref_mod: Optional[str] = None

    for mod in modalities:
        entry: Dict[str, Any] = {
            "path":       str(mod_paths.get(mod, "MISSING")),
            "loadable":   False,
            "shape":      None,
            "spacing_mm": None,
            "affine_ok":  None,
            "dtype":      None,
            "value_range": None,
        }

        if mod not in mod_paths:
            entry["path"] = "MISSING"
            result["modalities"][mod] = entry
            continue

        path = mod_paths[mod]
        try:
            data, img = load_nifti(path)
            entry["loadable"]    = True
            entry["shape"]       = list(data.shape)
            entry["spacing_mm"]  = [round(float(x), 4) for x in get_voxel_spacing(img)]
            entry["dtype"]       = str(data.dtype)
            entry["value_range"] = [round(float(data.min()), 4),
                                     round(float(data.max()), 4)]

            loaded_imgs[mod] = img

            if ref_img is None:
                ref_img = img
                ref_mod = mod

        except Exception as e:
            entry["error"] = str(e)
            result["errors"].append(f"{mod}: {e}")

        result["modalities"][mod] = entry

    # ── Cross-modality affine check ────────────────────────────────────────
    mod_list = [m for m in modalities if m in loaded_imgs]
    for i in range(len(mod_list)):
        for j in range(i + 1, len(mod_list)):
            m1, m2 = mod_list[i], mod_list[j]
            img1, img2 = loaded_imgs[m1], loaded_imgs[m2]
            compat = _affines_compatible(img1.affine, img2.affine, tol_mm=2.0, tol_cos=0.05)
            if not compat:
                msg = (f"{m1} vs {m2}: affines differ "
                       f"(spacing {[round(float(x),2) for x in get_voxel_spacing(img1)]} "
                       f"vs {[round(float(x),2) for x in get_voxel_spacing(img2)]})")
                result["alignment"]["issues"].append(msg)
                result["alignment"]["all_ok"] = False

    # ── Shape consistency check ────────────────────────────────────────────
    shapes = {m: tuple(loaded_imgs[m].shape[:3]) for m in mod_list}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        result["alignment"]["issues"].append(
            f"Shape mismatch: { {m: list(s) for m, s in shapes.items()} }"
        )
        result["alignment"]["all_ok"] = False

    # ── Mask ──────────────────────────────────────────────────────────────
    if mask_path is not None:
        mask_entry: Dict[str, Any] = {
            "path":    str(mask_path),
            "loadable": False,
            "shape":   None,
            "spacing_mm": None,
            "lesion_voxels": None,
            "lesion_volume_mm3": None,
            "affine_compat_with_dwi": None,   # mask is annotated in DWI space
        }
        try:
            mdata, mimg = load_nifti(mask_path)
            binary = (mdata > 0.5).astype(np.uint8)
            msp = get_voxel_spacing(mimg)
            n_lesion = int(binary.sum())
            vol_mm3  = n_lesion * float(np.prod(msp))

            mask_entry["loadable"]             = True
            mask_entry["shape"]                = list(mdata.shape)
            mask_entry["spacing_mm"]           = [round(float(x), 4) for x in msp]
            mask_entry["lesion_voxels"]        = n_lesion
            mask_entry["lesion_volume_mm3"]    = round(vol_mm3, 2)

            # Mask is annotated in DWI space – compare against DWI (not FLAIR)
            ref_for_mask = loaded_imgs.get("dwi", ref_img)
            if ref_for_mask is not None:
                compat = _affines_compatible(ref_for_mask.affine, mimg.affine,
                                             tol_mm=2.0, tol_cos=0.05)
                mask_entry["affine_compat_with_dwi"] = compat
                if not compat:
                    result["alignment"]["issues"].append(
                        "Mask affine differs from DWI – resampling required (NN)"
                    )
                    result["alignment"]["all_ok"] = False

        except Exception as e:
            mask_entry["error"] = str(e)
            result["errors"].append(f"mask: {e}")

        result["mask"] = mask_entry
    else:
        result["mask"] = {"path": "MISSING"}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary table printer
# ─────────────────────────────────────────────────────────────────────────────

def _row(cols: List[str], widths: List[int]) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def print_summary_table(results: List[Dict], modalities: List[str]) -> str:
    """Print and return a compact per-patient summary table."""
    mods_upper = [m.upper() for m in modalities]
    header = ["Subject", *mods_upper, "Mask", "Alignment", "Lesion(vox)", "Errors"]
    widths  = [26, *([6] * len(modalities)), 6, 10, 12, 30]

    lines = []
    sep   = "─" * sum(widths + [2 * (len(widths) - 1)])
    lines.append(sep)
    lines.append(_row(header, widths))
    lines.append(sep)

    for r in results:
        mod_ok = []
        for mod in modalities:
            info = r["modalities"].get(mod, {})
            mod_ok.append("[OK]" if info.get("loadable") else "✗")

        mask_ok   = "[OK]" if (r["mask"] and r["mask"].get("loadable")) else "✗"
        align_ok  = "[OK]" if r["alignment"]["all_ok"] else "✗ !"
        lesion    = r["mask"].get("lesion_voxels", "-") if r["mask"] else "-"
        errs      = "; ".join(r["errors"])[:28] if r["errors"] else "—"

        row = [r["subject"], *mod_ok, mask_ok, align_ok, str(lesion), errs]
        lines.append(_row(row, widths))

    lines.append(sep)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(results: List[Dict], modalities: List[str]) -> Dict:
    """Aggregate statistics across all patients."""
    n          = len(results)
    n_valid    = sum(1 for r in results if r["valid"])
    n_missing  = {mod: sum(1 for r in results if mod in r["missing_mods"])
                  for mod in modalities}
    n_no_mask  = sum(1 for r in results if not (r["mask"] and r["mask"].get("loadable")))
    n_align_ok = sum(1 for r in results if r["alignment"]["all_ok"])
    n_errors   = sum(1 for r in results if r["errors"])

    # Spacing distributions per modality
    spacing_stats: Dict[str, Dict] = {}
    for mod in modalities:
        spacings = []
        for r in results:
            sp = r["modalities"].get(mod, {}).get("spacing_mm")
            if sp:
                spacings.append(sp)
        if spacings:
            arr = np.array(spacings)
            spacing_stats[mod] = {
                "mean": [round(float(x), 3) for x in arr.mean(axis=0)],
                "std":  [round(float(x), 3) for x in arr.std(axis=0)],
                "min":  [round(float(x), 3) for x in arr.min(axis=0)],
                "max":  [round(float(x), 3) for x in arr.max(axis=0)],
            }

    # Lesion volume distribution
    lesion_vols = [
        r["mask"]["lesion_volume_mm3"]
        for r in results
        if r["mask"] and r["mask"].get("lesion_volume_mm3") is not None
    ]
    lesion_stats: Dict = {}
    if lesion_vols:
        arr = np.array(lesion_vols)
        lesion_stats = {
            "n_subjects_with_lesion": int((arr > 0).sum()),
            "mean_mm3":   round(float(arr.mean()), 2),
            "median_mm3": round(float(np.median(arr)), 2),
            "std_mm3":    round(float(arr.std()), 2),
            "min_mm3":    round(float(arr.min()), 2),
            "max_mm3":    round(float(arr.max()), 2),
            "zero_lesion": int((arr == 0).sum()),
        }

    # Shape histogram per modality
    shape_counts: Dict[str, Dict] = {}
    for mod in modalities:
        cnt: Dict[str, int] = {}
        for r in results:
            sh = r["modalities"].get(mod, {}).get("shape")
            if sh:
                key = "×".join(str(x) for x in sh)
                cnt[key] = cnt.get(key, 0) + 1
        shape_counts[mod] = cnt

    return {
        "total_subjects":        n,
        "fully_valid":           n_valid,
        "invalid":               n - n_valid,
        "missing_per_modality":  n_missing,
        "no_mask":               n_no_mask,
        "alignment_ok":          n_align_ok,
        "alignment_issues":      n - n_align_ok,
        "subjects_with_errors":  n_errors,
        "spacing_stats":         spacing_stats,
        "lesion_volume_stats":   lesion_stats,
        "shape_counts":          shape_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_full_inspection(cfg: Dict) -> Dict:
    paths      = cfg["paths"]
    modalities = cfg["data"]["isles_modalities"]   # ["flair", "adc", "dwi"]

    isles_root  = paths["isles_root"]
    bonbid_root = paths["bonbid_root"]
    log_dir     = Path(paths["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("ISLES 2022 Dataset Integrity Inspector")
    logger.info(f"Root:       {isles_root}")
    logger.info(f"Modalities: {modalities}")
    logger.info("=" * 72)

    # ── Discover all patients ─────────────────────────────────────────────
    t0      = time.time()
    samples = discover_isles2022(isles_root, modalities)
    logger.info(f"Discovered {len(samples)} patients in {time.time()-t0:.1f}s")

    if not samples:
        logger.error("No patients found. Check isles_root in config.yaml.")
        return {}

    # ── Inspect each patient ──────────────────────────────────────────────
    logger.info(f"\nRunning per-patient checks ({len(samples)} patients) ...")
    results = []
    for i, sample in enumerate(samples, 1):
        if i % 50 == 0 or i == 1 or i == len(samples):
            logger.info(f"  Inspecting patient {i}/{len(samples)} ? {sample['subject']}")
        result = inspect_one_patient(sample, modalities)
        results.append(result)

    elapsed = time.time() - t0

    # ── Statistics ────────────────────────────────────────────────────────
    stats = compute_statistics(results, modalities)

    # ── Console report ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("ISLES 2022 INSPECTION SUMMARY")
    logger.info("=" * 72)
    logger.info(f"  Total subjects    : {stats['total_subjects']}")
    logger.info(f"  Fully valid       : {stats['fully_valid']}  "
                f"(all 3 modalities + mask, loadable)")
    logger.info(f"  Invalid / excluded: {stats['invalid']}")
    logger.info(f"  Alignment OK      : {stats['alignment_ok']}")
    logger.info(f"  Alignment issues  : {stats['alignment_issues']}")
    logger.info(f"  Subjects w/ errors: {stats['subjects_with_errors']}")
    logger.info("")

    for mod in modalities:
        n_miss = stats["missing_per_modality"].get(mod, 0)
        sp     = stats["spacing_stats"].get(mod, {})
        if sp:
            logger.info(
                f"  {mod.upper():6s}: missing={n_miss:3d} | "
                f"spacing mean={sp['mean']} +/- {sp['std']} mm"
            )
        else:
            logger.info(f"  {mod.upper():6s}: missing={n_miss:3d}")

    lv = stats.get("lesion_volume_stats", {})
    if lv:
        logger.info("")
        logger.info(f"  Lesion volume: mean={lv['mean_mm3']:.1f} mm?  "
                    f"median={lv['median_mm3']:.1f} mm?  "
                    f"range=[{lv['min_mm3']:.1f}, {lv['max_mm3']:.1f}]")
        logger.info(f"  Zero-lesion masks: {lv['zero_lesion']}")

    # ── Detail: alignment issues ──────────────────────────────────────────
    align_bad = [r for r in results if not r["alignment"]["all_ok"]]
    if align_bad:
        logger.info(f"\n  Patients with alignment issues ({len(align_bad)}):")
        for r in align_bad[:20]:
            for issue in r["alignment"]["issues"]:
                logger.info(f"    [{r['subject']}] {issue}")
        if len(align_bad) > 20:
            logger.info(f"    ... and {len(align_bad)-20} more (see JSON report)")

    # ── Detail: missing modalities ────────────────────────────────────────
    missing_cases = [r for r in results if r["missing_mods"]]
    if missing_cases:
        logger.info(f"\n  Patients with missing modalities ({len(missing_cases)}):")
        for r in missing_cases:
            logger.info(f"    [{r['subject']}] missing: {r['missing_mods']}")

    logger.info(f"\n  Inspection completed in {elapsed:.1f}s")
    logger.info("=" * 72)

    # ── Summary table ─────────────────────────────────────────────────────
    table = print_summary_table(results, modalities)

    # ── BONBID inspection ─────────────────────────────────────────────────
    logger.info("\nBONBID-HIE splits:")
    bonbid_report: Dict = {"root": bonbid_root, "splits": {}}
    for split in ["train", "val", "test"]:
        bonbid_samples = discover_bonbid(bonbid_root, split)
        labelled = sum(1 for s in bonbid_samples if s.get("mask_path"))
        bonbid_report["splits"][split] = {
            "count":   len(bonbid_samples),
            "labelled": labelled,
        }
        logger.info(f"  {split:5s}: {len(bonbid_samples)} subjects, {labelled} labelled")

    # ── Build final report ────────────────────────────────────────────────
    report = {
        "ISLES2022": {
            "root":       isles_root,
            "modalities": modalities,
            "statistics": stats,
            "patients":   results,
        },
        "BONBID-HIE": bonbid_report,
        "elapsed_s":  round(elapsed, 1),
    }

    # Save JSON
    json_path = log_dir / "inspection_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\n[OK] Full JSON report -> {json_path}")

    # Save summary table
    txt_path = log_dir / "inspection_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        # Header
        f.write("ISLES 2022 Dataset Inspection Summary\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Root       : {isles_root}\n")
        f.write(f"Modalities : {modalities}\n\n")
        # Stats block
        f.write(f"Total subjects    : {stats['total_subjects']}\n")
        f.write(f"Fully valid       : {stats['fully_valid']}\n")
        f.write(f"Invalid           : {stats['invalid']}\n")
        f.write(f"Alignment issues  : {stats['alignment_issues']}\n")
        f.write(f"Subjects w/ errors: {stats['subjects_with_errors']}\n\n")

        if lv:
            f.write(f"Lesion volume ? mean: {lv['mean_mm3']:.1f} mm?  "
                    f"median: {lv['median_mm3']:.1f} mm?  "
                    f"range: [{lv['min_mm3']:.1f}, {lv['max_mm3']:.1f}]\n")
            f.write(f"Zero-lesion masks   : {lv['zero_lesion']}\n\n")

        f.write("Per-patient table:\n")
        f.write(table + "\n")

        if align_bad:
            f.write(f"\nAlignment issues ({len(align_bad)} patients):\n")
            for r in align_bad:
                for issue in r["alignment"]["issues"]:
                    f.write(f"  [{r['subject']}] {issue}\n")

        if missing_cases:
            f.write(f"\nMissing modalities ({len(missing_cases)} patients):\n")
            for r in missing_cases:
                f.write(f"  [{r['subject']}] {r['missing_mods']}\n")

    logger.info(f"[OK] Summary table     -> {txt_path}")

    # ── Final go/no-go ────────────────────────────────────────────────────
    logger.info("")
    if stats["fully_valid"] >= 10:
        logger.info(
            f"[PASS]  PASS ? {stats['fully_valid']} valid subjects found. "
            f"Ready for training."
        )
        if stats["alignment_issues"] > 0:
            logger.info(
                f"[WARN]   {stats['alignment_issues']} subjects: DWI/ADC and FLAIR have different "
                f"resolutions (expected ? FLAIR will be resampled to DWI space automatically)."
            )
    else:
        logger.error(
            f"[FAIL]  FAIL ? only {stats['fully_valid']} fully valid subjects. "
            f"Check dataset root and layout."
        )

    return report


def main():
    parser = argparse.ArgumentParser(description="ISLES 2022 dataset integrity inspector")
    parser.add_argument("--config", default="../configs/config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    report = run_full_inspection(cfg)
    if not report:
        sys.exit(1)


if __name__ == "__main__":
    main()
