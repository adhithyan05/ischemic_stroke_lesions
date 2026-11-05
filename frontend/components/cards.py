"""
cards.py
========
Metric cards and informational widgets for the web frontend.
"""

import streamlit as st


def render_metric_card(label: str, value: str, subtext: str = "", color: str = "cyan"):
    """
    Render a modern metric card.
    color: 'cyan', 'green', or 'amber'.
    """
    val_class = f"metric-value-{color}"
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="{val_class}">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title: str, text: str, border_color: str = "#38BDF8"):
    """Render an informational container card."""
    st.markdown(
        f"""
        <div class="med-card" style="border-left: 4px solid {border_color};">
            <h4 style="margin: 0 0 8px 0; color: #F8FAFC; font-size: 1.1rem;">{title}</h4>
            <p style="margin: 0; color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

