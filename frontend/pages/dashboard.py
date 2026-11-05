"""
dashboard.py
============
Dashboard overview page for the adult-to-neonatal domain adaptation project.
"""

from pathlib import Path
import streamlit as st

from components.header import render_header
from components.cards import render_metric_card, render_info_card
from components.charts import create_training_curves_plot, create_model_comparison_chart
from utils.data_loader import load_dann_training_history, load_zeroshot_baseline_metrics


def render_dashboard_page():
    render_header(
        title="Adult-to-Neonatal Brain Lesion Segmentation",
        subtitle="Unsupervised Domain Adaptation for Hypoxic-Ischemic Encephalopathy (HIE) Lesion Segmentation using DANN",
        badge_text="Project Overview",
        badge_class="badge-blue",
    )

    # ── High-Level Metric Cards ───────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            label="Baseline Zero-Shot Dice",
            value="0.1451",
            subtext="Adult U-Net on Neonatal Cohort",
            color="amber",
        )

    with col2:
        render_metric_card(
            label="Best DANN Val Dice",
            value="0.2652",
            subtext="Adapted Model (Epoch 6)",
            color="green",
        )

    with col3:
        render_metric_card(
            label="Best Training Epoch",
            value="Epoch 6",
            subtext="Optimal Adversarial Balance",
            color="cyan",
        )

    with col4:
        render_metric_card(
            label="Domain Confusion Acc",
            value="48.18%",
            subtext="Ideal Invariant Confusion (~50%)",
            color="cyan",
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ── Project Context Cards ────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        render_info_card(
            title="Source Domain: Adult Stroke (ISLES 2022)",
            text="""
            • <b>Cohort:</b> 250 adult stroke patients with labeled arterial infarct lesions.<br>
            • <b>Modalities:</b> DWI (b=1000), ADC, and FLAIR.<br>
            • <b>Pathology:</b> Focal territorial ischemic infarcts (primarily middle cerebral artery [MCA]).<br>
            • <b>Baseline Model:</b> Attention U-Net trained for 100 epochs (Best Adult Val Dice: 0.2673).
            """,
            border_color="#38BDF8",
        )

    with c2:
        render_info_card(
            title="Target Domain: Neonatal HIE (BONBID-HIE)",
            text="""
            • <b>Cohort:</b> 89 accessible neonatal subjects (85 training + 4 held-out validation).<br>
            • <b>Modalities:</b> Skull-stripped ADC and age-normed Z-score ADC (DWI & FLAIR absent).<br>
            • <b>Pathology:</b> Perinatal Hypoxic-Ischemic Encephalopathy (bilateral deep gray nuclei injury).<br>
            • <b>Domain Adaptation:</b> DANN trained adversarially on unlabeled neonatal scans.
            """,
            border_color="#10B981",
        )

    # ── DANN Training Curves & Analysis ──────────────────────────────────────
    st.markdown("### Domain Adaptation Training Dynamics")
    history_data = load_dann_training_history()

    if history_data and "history" in history_data:
        history = history_data["history"]
        fig = create_training_curves_plot(history)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("DANN training log history not found or currently empty.")

    # ── Cohort Distribution & Specifications Table ───────────────────────────
    st.markdown("### Cohort & Model Architecture Summary")
    st.markdown(
        """
        <table>
            <thead>
                <tr>
                    <th>Domain</th>
                    <th>Dataset</th>
                    <th>Subjects</th>
                    <th>Input Modalities</th>
                    <th>Pathology Type</th>
                    <th>Model Representation</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>Source Domain</b></td>
                    <td>ISLES 2022</td>
                    <td>250 Labeled Adults</td>
                    <td>DWI (b=1000), ADC, FLAIR</td>
                    <td>Focal Arterial Stroke (Unilateral)</td>
                    <td>Supervised Attention U-Net (in=3, out=1)</td>
                </tr>
                <tr>
                    <td><b>Target Domain</b></td>
                    <td>BONBID-HIE</td>
                    <td>85 Unlabeled Neonates (Train) + 4 (Val)</td>
                    <td>Skull-stripped ADC, Z-ADC</td>
                    <td>Hypoxic-Ischemic Encephalopathy (Bilateral)</td>
                    <td>DANN with Gradient Reversal Layer (GRL)</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

