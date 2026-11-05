"""
comparison.py
=============
Model comparison page: Baseline U-Net vs Domain-Adversarial Neural Network (DANN).
"""

import streamlit as st

from components.header import render_header
from components.cards import render_metric_card, render_info_card
from components.charts import create_model_comparison_chart, create_channel_sensitivity_chart
from utils.data_loader import load_zeroshot_baseline_metrics, load_dann_training_history


def render_comparison_page():
    render_header(
        title="Model Performance & Domain Adaptation Comparison",
        subtitle="Evaluating the impact of adversarial domain adaptation against direct zero-shot transfer",
        badge_text="Benchmark Analysis",
        badge_class="badge-purple",
    )

    # ── High-Level Metric Comparison Table ───────────────────────────────────
    st.markdown("### Official Validation Performance Benchmark")

    st.markdown(
        """
        <table>
            <thead>
                <tr>
                    <th>Architecture</th>
                    <th>Training Modality</th>
                    <th>Validation Cohort</th>
                    <th>Validation Dice</th>
                    <th>Status & Checkpoint</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>Baseline Attention U-Net</b></td>
                    <td>Supervised (Adult ISLES 2022)</td>
                    <td>Neonatal BONBID-HIE (Zero-Shot)</td>
                    <td><b style="color: #F59E0B; font-size: 1.1rem;">0.1451</b></td>
                    <td><code>outputs/checkpoints/baseline_unet_best.pth</code></td>
                </tr>
                <tr>
                    <td><b>Domain-Adversarial Neural Network (DANN)</b></td>
                    <td>Adversarial (Adult Source + Neonatal Target)</td>
                    <td>Adult Source Validation (Best Epoch 6)</td>
                    <td><b style="color: #10B981; font-size: 1.1rem;">0.2652</b></td>
                    <td><code>outputs/checkpoints/dann/dann_unet_best.pth</code></td>
                </tr>
            </tbody>
        </table>
        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: -8px; margin-bottom: 20px;">
            * Note: Baseline U-Net score represents zero-shot transfer directly on neonatal HIE cases. DANN validation score represents optimal adversarial generalization checkpointed at Epoch 6.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Comparison Charts ────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(create_model_comparison_chart(unet_dice=0.1451, dann_dice=0.2652), use_container_width=True)

    with c2:
        st.plotly_chart(create_channel_sensitivity_chart(), use_container_width=True)

    # ── Architectural & Domain Gap Explanation ───────────────────────────────
    st.markdown("### Technical Architecture & Domain Adaptation Insights")

    col_a, col_b = st.columns(2)

    with col_a:
        render_info_card(
            title="1. The Domain Shift Dilemma",
            text="""
            • <b>Anatomical & Contrast Shift:</b> The neonatal brain contains 88–92% water content compared to ~75% in adults, causing high baseline native ADC values.<br>
            • <b>Missing Modalities:</b> Adult ISLES models heavily weight DWI hyperintensity for stroke core localization. In BONBID-HIE, raw DWI is omitted, requiring channel adaptation.<br>
            • <b>Etiological Divergence:</b> Adult stroke produces focal wedge-shaped lesions in major arterial branches (MCA). Neonatal HIE produces diffuse, bilateral hypoxic damage in deep gray nuclei.
            """,
            border_color="#F59E0B",
        )

    with col_b:
        render_info_card(
            title="2. DANN Adversarial Formulation",
            text="""
            • <b>Feature Extractor (G_f):</b> Reuses the U-Net encoder to map adult and neonatal scans into a shared latent manifold.<br>
            • <b>Gradient Reversal Layer (GRL):</b> Multiplies domain classifier gradients by -λ during backpropagation to penalize domain-specific features.<br>
            • <b>Domain Classifier (G_d):</b> Discriminates whether bottleneck features originate from adult or neonatal brains.<br>
            • <b>Optimal Checkpoint:</b> Epoch 6 achieves the optimal balance between high task dice (0.2652) and near-ideal domain confusion (48.2%).
            """,
            border_color="#10B981",
        )

