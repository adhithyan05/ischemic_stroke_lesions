"""
header.py
=========
Standard header and banner component for the web frontend.
"""

import streamlit as st


def render_header(title: str, subtitle: str, badge_text: str = "Medical AI", badge_class: str = "badge-blue"):
    """Render a styled title banner with badges and description."""
    st.markdown(
        f"""
        <div class="title-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span class="{badge_class}">{badge_text}</span>
                <span style="font-size: 0.8rem; color: #94A3B8; font-weight: 500;">
                    Domain Adaptation • ISLES 2022 ➔ BONBID-HIE
                </span>
            </div>
            <h1 class="title-text">{title}</h1>
            <p class="subtitle-text">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

