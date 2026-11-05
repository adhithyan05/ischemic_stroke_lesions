"""
dataset.py
==========
Dataset information page detailing ISLES 2022 and BONBID-HIE cohorts.
"""

import streamlit as st

from components.header import render_header
from components.cards import render_metric_card, render_info_card


def render_dataset_page():
    render_header(
        title="Dataset Information & Preprocessing Harmonization",
        subtitle="Comparing adult source-domain imaging (ISLES 2022) with neonatal target-domain imaging (BONBID-HIE)",
        badge_text="Data Specifications",
        badge_class="badge-blue",
    )

    # ── Cohort Metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("ISLES 2022 Subjects", "250", "Adult Arterial Stroke", "cyan")
    with c2:
        render_metric_card("BONBID-HIE Cohort", "89", "85 Train + 4 Held-Out Val", "green")
    with c3:
        render_metric_card("Harmonized Spacing", "2.0 mm", "Isotropic 3D Voxel Grid", "cyan")
    with c4:
        render_metric_card("Input Matrix", "128 × 128", "Center-Cropped Slices", "cyan")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ── Comparison Tabs ───────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Dataset Comparison Matrix", "Imaging Modalities", "Preprocessing Pipeline"])

    with tab1:
        st.markdown(
            """
            <table>
                <thead>
                    <tr>
                        <th>Property</th>
                        <th>Source Domain: ISLES 2022</th>
                        <th>Target Domain: BONBID-HIE</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><b>Target Population</b></td>
                        <td>Adult stroke patients (mean age: ~68 years)</td>
                        <td>Term and near-term neonates (0–28 days)</td>
                    </tr>
                    <tr>
                        <td><b>Etiology / Injury</b></td>
                        <td>Focal thromboembolic arterial infarct (MCA, PCA, ACA)</td>
                        <td>Perinatal Hypoxic-Ischemic Encephalopathy (asphyxia)</td>
                    </tr>
                    <tr>
                        <td><b>Lesion Distribution</b></td>
                        <td>Unilateral, wedge-shaped territorial infarct</td>
                        <td>Bilateral, diffuse injury to basal ganglia & thalami</td>
                    </tr>
                    <tr>
                        <td><b>Available Modalities</b></td>
                        <td>DWI (b=1000), ADC map, anatomical FLAIR</td>
                        <td>Skull-stripped ADC (<code>1ADC_ss</code>), Z-Score ADC (<code>2Z_ADC</code>)</td>
                    </tr>
                    <tr>
                        <td><b>Missing Modalities</b></td>
                        <td>None (all 3 modalities present)</td>
                        <td>Raw DWI and anatomical FLAIR are <b>absent</b></td>
                    </tr>
                    <tr>
                        <td><b>Container Format</b></td>
                        <td>NIfTI-1 (<code>.nii</code> / <code>.nii.gz</code>)</td>
                        <td>MetaImage (<code>.mha</code>, uncompressed float32)</td>
                    </tr>
                    <tr>
                        <td><b>Native Voxel Spacing</b></td>
                        <td>Isotropic ~2.0 × 2.0 × 2.0 mm</td>
                        <td>Anisotropic: 0.625–2.0 mm in-plane, 2.0–6.0 mm slice thickness</td>
                    </tr>
                    <tr>
                        <td><b>Lesion Annotation</b></td>
                        <td>Expert consensus stroke core mask (<code>msk.nii</code>)</td>
                        <td>Consensus expert HIE lesion mask (<code>_lesion.mha</code>)</td>
                    </tr>
                </tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            render_info_card(
                title="Adult Modalities (ISLES 2022)",
                text="""
                • <b>DWI (b=1000):</b> Diffusion-Weighted Imaging captures restricted Brownian motion of intracellular water due to cytotoxic edema (bright lesion signal).<br>
                • <b>ADC Map:</b> Apparent Diffusion Coefficient quantitatively measures water diffusion (hypointense / dark lesion core).<br>
                • <b>FLAIR:</b> Fluid-Attenuated Inversion Recovery provides high-resolution anatomical context with suppressed cerebrospinal fluid (CSF).
                """,
                border_color="#38BDF8",
            )
        with m_col2:
            render_info_card(
                title="Neonatal Modalities (BONBID-HIE)",
                text="""
                • <b>1ADC_ss (Skull-Stripped ADC):</b> Native quantitative diffusion map with extracranial scalp and bone stripped away.<br>
                • <b>2Z_ADC (Normative Z-Map):</b> Voxelwise standard deviation difference relative to age-matched healthy neonatal templates. Crucial for compensating for unmyelinated neonatal brain water content.<br>
                • <b>3LABEL (Ground Truth):</b> Voxel-level binary annotations delineating hypoxic ischemic damage.
                """,
                border_color="#10B981",
            )

    with tab3:
        st.markdown(
            """
            <div class="med-card">
                <h4 style="color: #38BDF8; margin-top: 0;">Verified Multi-Step Medical Preprocessing Pipeline</h4>
                <ol style="color: #E2E8F0; line-height: 1.8; margin-bottom: 0;">
                    <li><b>Container Loading:</b> MetaImage (<code>.mha</code>) volumes and NIfTI volumes loaded seamlessly with native spatial geometry extraction.</li>
                    <li><b>Intensity Normalization:</b> Voxelwise Z-score normalization (<code>(x - mean) / std</code>) applied independently to each 3D MRI volume.</li>
                    <li><b>Isotropic Resampling:</b> Joint spatial resampling to uniform <code>(2.0, 2.0, 2.0) mm</code> isotropic grid using:
                        <ul>
                            <li><b>Continuous 1st-order spline interpolation (<code>order=1</code>)</b> strictly for MRI images (ADC, Z-ADC).</li>
                            <li><b>Nearest-neighbour interpolation (<code>order=0</code>)</b> strictly for segmentation masks (100% binary preservation).</li>
                        </ul>
                    </li>
                    <li><b>Matrix Standardization:</b> Center-cropping or zero-padding axial slices to uniform <code>128 × 128</code> matrices for network forward passes.</li>
                    <li><b>Channel Adaptation:</b> Primary mapping replicates available diffusion signals into a 3-channel tensor: <code>[Ch0: ADC, Ch1: Z-ADC, Ch2: Z-ADC]</code>.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

