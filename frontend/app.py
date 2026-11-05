"""
app.py
======
Main Streamlit application entry point for the Adult-to-Neonatal Brain Lesion Segmentation platform.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch

# Ensure frontend and project root are in sys.path
FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pages.dashboard import render_dashboard_page
from pages.prediction import render_prediction_page
from pages.comparison import render_comparison_page
from pages.dataset import render_dataset_page
from pages.about import render_about_page


def load_custom_css():
    """Load dark navy theme CSS."""
    css_path = FRONTEND_DIR / "styles" / "custom.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    # ── Page Configuration ───────────────────────────────────────────────────
    st.set_page_config(
        page_title="Adult-to-Neonatal Brain Lesion Segmentation",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_custom_css()

    # ── Sidebar Navigation ───────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 12px 0 16px 0;">
            <div style="font-size: 2.2rem; margin-bottom: 4px;">🧠</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em;">
                Neonatal NeuroSeg
            </div>
            <div style="font-size: 0.75rem; color: #38BDF8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">
                DANN Medical AI Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    nav_selection = st.sidebar.radio(
        "Navigation Menu",
        options=[
            "1. Dashboard",
            "2. MRI Prediction",
            "3. Model Comparison",
            "4. Dataset Information",
            "5. About Project",
        ],
        index=0,
    )

    # ── System Status in Sidebar Footer ──────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.8rem; color: #94A3B8; padding: 0 4px;">
            <div style="font-weight: 600; color: #E2E8F0; margin-bottom: 6px;">System Environment</div>
            <div>• <b>Compute Device:</b> <span style="color: #38BDF8;">"""
        + ("CUDA (GPU)" if torch.cuda.is_available() else "CPU (Standard)")
        + """</span></div>
            <div>• <b>Baseline Checkpoint:</b> <span style="color: #10B981;">Loaded (Ep 82)</span></div>
            <div>• <b>DANN Checkpoint:</b> <span style="color: #10B981;">Loaded (Ep 6 Best)</span></div>
            <div>• <b>PyTorch Version:</b> """
        + torch.__version__.split("+")[0]
        + """</div>
        </div>
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 14px; text-align: center;">
            Adult-to-Neonatal Domain Adaptation<br>Google DeepMind Pair Programming
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Router ───────────────────────────────────────────────────────────────
    if nav_selection == "1. Dashboard":
        render_dashboard_page()
    elif nav_selection == "2. MRI Prediction":
        render_prediction_page()
    elif nav_selection == "3. Model Comparison":
        render_comparison_page()
    elif nav_selection == "4. Dataset Information":
        render_dataset_page()
    elif nav_selection == "5. About Project":
        render_about_page()


if __name__ == "__main__":
    main()

