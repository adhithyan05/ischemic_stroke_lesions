"""
about.py
========
About the project, theoretical background on DANN, and research objectives.
"""

import streamlit as st

from components.header import render_header
from components.cards import render_info_card


def render_about_page():
    render_header(
        title="About the Domain Adaptation Project",
        subtitle="Bridging the adult-to-neonatal clinical divide through adversarial representation learning",
        badge_text="Research Background",
        badge_class="badge-purple",
    )

    st.markdown("### Research Motivation & Objectives")
    st.markdown(
        """
        <div class="med-card-highlight">
            <h4 style="color: #38BDF8; margin-top: 0;">Project Objective</h4>
            <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.6; margin-bottom: 0;">
                The objective of this research is to improve neonatal brain lesion segmentation by adapting a deep convolutional 
                neural network trained on adult ischemic stroke MRI to neonatal hypoxic-ischemic encephalopathy (HIE) brain injury MRI 
                using <b>Domain-Adversarial Neural Networks (DANN)</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Theoretical Components ───────────────────────────────────────────────
    st.markdown("### Theoretical Architecture & Methodology")

    c1, c2 = st.columns(2)

    with c1:
        render_info_card(
            title="1. Attention U-Net Backbone",
            text="""
            • <b>Encoder (Feature Extractor G_f):</b> A 4-stage residual convolutional encoder extracts hierarchical spatial features down to a 256-channel bottleneck representation.<br>
            • <b>Decoder (Label Predictor G_y):</b> Symmetrical decoding stages utilize transposed convolutions and Attention Gates (Oktay et al.) to weight skip connections and suppress irrelevant background activations.<br>
            • <b>Loss Function:</b> Joint Dice Loss + Binary Cross-Entropy Loss to handle extreme class imbalance in small medical lesions.
            """,
            border_color="#38BDF8",
        )

        render_info_card(
            title="2. Gradient Reversal Layer (GRL)",
            text="""
            • <b>Mechanism:</b> Functions as an identity operator in the forward pass (y = x) but multiplies incoming gradients by -λ during backpropagation.<br>
            • <b>Sheduling (Ganin Schedule):</b> Adaptation factor λ ramps from 0.0 to 1.0 according to <code>λ_p = 2 / (1 + exp(-10*p)) - 1</code>, shielding early encoder representations from discriminator noise.<br>
            • <b>Minimax Formulation:</b> Forces the encoder to learn representations that minimize segmentation loss while simultaneously maximizing domain confusion.
            """,
            border_color="#A855F7",
        )

    with c2:
        render_info_card(
            title="3. Discriminative Domain Classifier",
            text="""
            • <b>Feature Compression:</b> Bottleneck features are processed using dual Global Average Pooling and Global Max Pooling to preserve both mean and peak semantic responses.<br>
            • <b>Multi-Layer Perceptron (MLP):</b> High-capacity projection network with Batch Normalization and Dropout (p=0.3) to classify source (label=0) versus target (label=1).<br>
            • <b>Domain Invariance:</b> As training progresses, the domain classifier reaches ~50% accuracy, proving the learned representations are domain-invariant.
            """,
            border_color="#10B981",
        )

        render_info_card(
            title="4. Clinical Significance for Neonatal Care",
            text="""
            • <b>Data Scarcity:</b> Neonatal MRI datasets are scarce and difficult to annotate due to ongoing myelination and small brain structures.<br>
            • <b>Knowledge Transfer:</b> Adult ischemic stroke datasets (e.g. ISLES) contain hundreds of expert-annotated scans.<br>
            • <b>Automated Diagnosis:</b> Domain adaptation enables clinical AI tools to leverage abundant adult data to assist neonatologists in rapid HIE diagnosis and therapeutic hypothermia assessment.
            """,
            border_color="#F59E0B",
        )

    # ── Project Progression Timeline ─────────────────────────────────────────
    st.markdown("### Research & Engineering Phases")
    st.markdown(
        """
        <div class="med-card">
            <ul style="color: #E2E8F0; line-height: 1.8; margin-bottom: 0;">
                <li><b>Phase 1: Adult Baseline U-Net Training:</b> Trained Attention U-Net on 250 adult ISLES 2022 patients for 100 epochs (Adult Val Dice: 0.2673).</li>
                <li><b>Phase 2: Neonatal BONBID-HIE Inspection:</b> In-depth inspection of 89 neonatal subjects, MetaImage (.mha) support, geometry verification, and interpolation harmonization.</li>
                <li><b>Phase 3: Zero-Shot Baseline Evaluation:</b> Direct inference of adult model on neonatal validation subjects (Baseline Neonatal Dice: 0.1451).</li>
                <li><b>Phase 4: DANN Implementation & Training:</b> Adversarial domain adaptation on unlabeled neonatal training scans (Best Validation Dice: 0.2652 at Epoch 6).</li>
                <li><b>Phase 5: Interactive Web Frontend:</b> Full-stack medical AI dashboard for visualization, comparative benchmarking, and volumetric prediction.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

