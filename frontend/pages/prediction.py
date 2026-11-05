"""
prediction.py
=============
Interactive neonatal MRI prediction page.
Supports uploading custom .mha volumes or testing preloaded validation subjects.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import streamlit as st

from components.header import render_header
from components.cards import render_metric_card
from components.visualizer import render_slice_viewer
from utils.data_loader import (
    get_available_validation_samples,
    load_mha_from_upload,
    load_nifti,
)
from utils.model_loader import (
    load_baseline_unet,
    load_dann_model,
    get_inference_device,
)
from utils.inference import (
    run_model_inference,
    export_mask_to_npz,
    export_mask_to_nifti,
)


def render_prediction_page():
    render_header(
        title="Neonatal MRI Lesion Segmentation",
        subtitle="Perform deep-learning inference on neonatal brain MRI using Baseline U-Net or Domain-Adapted DANN",
        badge_text="Inference Engine",
        badge_class="badge-green",
    )

    # ── Sidebar / Control Settings ───────────────────────────────────────────
    st.sidebar.markdown("### Model & Inference Settings")

    model_choice = st.sidebar.selectbox(
        "Select Segmentation Model:",
        options=[
            "DANN Adapted U-Net (Epoch 6 Best Checkpoint)",
            "Baseline Attention U-Net (Adult-Trained Zero-Shot)",
        ],
        index=0,
        help="DANN is the domain-adapted model from Phase 4. Baseline U-Net is the adult Phase 1 model.",
    )

    threshold = st.sidebar.slider(
        "Sigmoid Decision Threshold (τ):",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold for classifying a voxel as lesion. Standard medical cutoff is 0.50.",
    )

    channel_mapping = st.sidebar.selectbox(
        "Neonatal Channel Adaptation:",
        options=[
            "primary_adc_z_z",
            "surrogate_z_adc_dwi",
        ],
        index=0,
        format_func=lambda x: "Primary Mapping [ADC, Z-ADC, Z-ADC]" if x == "primary_adc_z_z" else "Experimental Surrogate [-Z_ADC, ADC, 0]",
        help="Primary mapping replicates available diffusion signals. Surrogate mapping inverts Z-ADC to emulate hyperintense DWI.",
    )

    alpha_overlay = st.sidebar.slider(
        "Overlay Transparency (α):",
        min_value=0.20,
        max_value=0.80,
        value=0.45,
        step=0.05,
    )

    # ── Input Selection (Preloaded vs Upload) ─────────────────────────────────
    input_mode = st.radio(
        "Select Input Source:",
        options=["Preloaded Validation Cohort", "Upload Custom .mha Files"],
        horizontal=True,
    )

    adc_raw = None
    z_raw = None
    mask_raw = None
    native_spacing = (1.0, 1.0, 1.0)
    subject_id = "Custom_Upload"

    if input_mode == "Preloaded Validation Cohort":
        samples = get_available_validation_samples()
        if not samples:
            st.error("No validation samples discovered in data/BONBID-HIE/BONBID2023_Val/")
            return

        sample_map = {s["subject"]: s for s in samples}
        selected_subj = st.selectbox(
            "Select Validation Subject:",
            options=list(sample_map.keys()),
            format_func=lambda s: f"{s} (Official Held-Out Neonatal Validation Case)",
        )
        sample = sample_map[selected_subj]
        subject_id = sample["subject"]

        # Load local .mha files
        with st.spinner(f"Loading {subject_id} MRI volumes from disk..."):
            adc_raw, adc_meta = load_nifti(sample["adc_path"])
            z_raw, _ = load_nifti(sample["z_adc_path"])
            native_spacing = adc_meta.get_zooms()

            if sample["mask_path"]:
                mask_raw, _ = load_nifti(sample["mask_path"])

        st.markdown(
            f"""
            <div class="callout-info">
                <b>Subject Loaded:</b> {subject_id} &nbsp;|&nbsp; 
                <b>Native Matrix:</b> {adc_raw.shape} &nbsp;|&nbsp; 
                <b>Native Spacing:</b> ({native_spacing[0]:.2f}, {native_spacing[1]:.2f}, {native_spacing[2]:.2f}) mm &nbsp;|&nbsp; 
                <b>Ground Truth:</b> {'Available (Binary Mask)' if mask_raw is not None else 'None'}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown("Upload native MetaImage (`.mha`) volumes for neonatal evaluation:")
        col_u1, col_u2, col_u3 = st.columns(3)

        with col_u1:
            adc_file = st.file_uploader("1. Upload ADC Volume (*_ADC_ss.mha)", type=["mha"])
        with col_u2:
            z_file = st.file_uploader("2. Upload Z-Score ADC Volume (*_Z_ADC.mha)", type=["mha"])
        with col_u3:
            mask_file = st.file_uploader("3. Optional: Ground Truth Mask (*_lesion.mha)", type=["mha", "nii", "gz"])

        if adc_file and z_file:
            with st.spinner("Parsing uploaded MetaImage files..."):
                adc_raw, adc_meta = load_mha_from_upload(adc_file)
                z_raw, _ = load_mha_from_upload(z_file)
                native_spacing = adc_meta.get_zooms()
                subject_id = Path(adc_file.name).stem.split("-")[0]

                if mask_file:
                    mask_raw, _ = load_mha_from_upload(mask_file)

            st.success(f"Uploaded files parsed successfully! Shape: {adc_raw.shape}, Spacing: {native_spacing} mm")
        else:
            st.info("Please upload both ADC and Z-ADC .mha files to begin inference.")
            return

    # ── Inference Execution ──────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("▶ Run Lesion Segmentation", use_container_width=True)

    if run_btn:
        if adc_raw is None or z_raw is None:
            st.error("Missing input volumes. Cannot execute prediction.")
            return

        with st.spinner(f"Running {model_choice} inference on {subject_id} ..."):
            # Load selected model
            device = get_inference_device()
            if "dann" in model_choice.lower():
                model, meta = load_dann_model()
            else:
                model, meta = load_baseline_unet()

            result = run_model_inference(
                model=model,
                model_type=meta["name"],
                adc_raw=adc_raw,
                z_raw=z_raw,
                native_spacing=native_spacing,
                mask_raw=mask_raw,
                channel_mapping=channel_mapping,
                threshold=threshold,
                device=device,
            )
            result["subject_id"] = subject_id
            st.session_state["prediction_result"] = result
            st.success(f"Inference completed in evaluation mode using {meta['name']}!")

    # ── Results Visualization ────────────────────────────────────────────────
    if "prediction_result" in st.session_state:
        res = st.session_state["prediction_result"]
        depth = res["depth"]

        st.markdown("---")
        st.markdown(f"### Segmentation Results: {res['subject_id']}")

        # Metric summary cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card(
                label="Predicted Lesion Volume",
                value=f"{res['pred_volume_mm3']:,.1f} mm³",
                subtext=f"{res['pred_voxels']:,} total voxels",
                color="cyan" if res["pred_voxels"] > 0 else "amber",
            )
        with m2:
            render_metric_card(
                label="Affected Slices",
                value=f"{len(res['slices_with_lesions'])} / {depth}",
                subtext=f"{(len(res['slices_with_lesions'])/depth)*100:.1f}% of volume",
                color="cyan",
            )

        if res["metrics"]:
            metrics = res["metrics"]
            with m3:
                render_metric_card(
                    label="Validation Dice",
                    value=f"{metrics['dice']:.4f}",
                    subtext=f"IoU: {metrics['iou']:.4f}",
                    color="green" if metrics["dice"] > 0.15 else "amber",
                )
            with m4:
                render_metric_card(
                    label="Recall / Sensitivity",
                    value=f"{metrics['recall']:.4f}",
                    subtext=f"Precision: {metrics['precision']:.4f}",
                    color="green" if metrics["recall"] > 0.20 else "amber",
                )
        else:
            with m3:
                render_metric_card(
                    label="Decision Threshold",
                    value=f"τ = {threshold:.2f}",
                    subtext="Sigmoid probability cutoff",
                    color="cyan",
                )
            with m4:
                render_metric_card(
                    label="Resampled Matrix",
                    value=f"{res['image_size'][0]}×{res['image_size'][1]}",
                    subtext="2.0 mm Isotropic Resolution",
                    color="cyan",
                )

        # Slice Navigator Slider
        st.markdown("#### Axial Slice Navigator")

        # Pick default slice (slice with largest lesion, or middle slice)
        default_slice = depth // 2
        if res["slices_with_lesions"]:
            lesion_per_slice = [res["pred_vol"][:, :, z].sum() for z in range(depth)]
            default_slice = int(np.argmax(lesion_per_slice))

        c_nav1, c_nav2 = st.columns([3, 1])
        with c_nav1:
            current_z = st.slider(
                "Axial Slice (Z-Plane):",
                min_value=0,
                max_value=depth - 1,
                value=default_slice,
                step=1,
                help="Slide to inspect cross-sectional slices through the brain volume.",
            )
        with c_nav2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if res["slices_with_lesions"]:
                if st.button("🔍 Jump to Peak Lesion Slice"):
                    current_z = default_slice

        # Multi-panel visualization
        render_slice_viewer(res, current_slice=current_z, alpha=alpha_overlay)

        # ── Export & Download Options ────────────────────────────────────────
        st.markdown("#### Export Predicted Lesion Mask")
        d_col1, d_col2, _ = st.columns([1.5, 1.5, 3])

        with d_col1:
            nii_bytes = export_mask_to_nifti(res["pred_vol"], spacing=res["target_spacing"])
            st.download_button(
                label="⬇ Download NIfTI Mask (.nii.gz)",
                data=nii_bytes,
                file_name=f"{res['subject_id']}_predicted_lesion_mask.nii.gz",
                mime="application/gzip",
                use_container_width=True,
            )

        with d_col2:
            npz_bytes = export_mask_to_npz(res["pred_vol"], res["prob_vol"])
            st.download_button(
                label="⬇ Download Numpy Volume (.npz)",
                data=npz_bytes,
                file_name=f"{res['subject_id']}_prediction.npz",
                mime="application/octet-stream",
                use_container_width=True,
            )

