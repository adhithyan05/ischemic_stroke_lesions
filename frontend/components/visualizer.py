"""
visualizer.py
=============
Multi-panel medical MRI slice visualization component with lesion overlays.
"""

from typing import Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st


def render_slice_viewer(
    result: Dict,
    current_slice: int,
    alpha: float = 0.45,
):
    """
    Render multi-column slice view for the specified axial slice z.
    Displays ADC, Z-ADC, Predicted Mask, and Color Overlay.
    """
    z = current_slice
    adc_sl  = result["adc_slices"][z]
    z_sl    = result["z_slices"][z]
    pred_sl = result["pred_vol"][:, :, z]
    prob_sl = result["prob_vol"][:, :, z]
    gt_vol  = result["gt_vol"]
    has_gt  = gt_vol is not None
    gt_sl   = gt_vol[:, :, z] if has_gt else None

    # Determine number of columns
    n_cols = 5 if has_gt else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.4, 3.6), facecolor="#0F172A")

    # 1. ADC Slice
    axes[0].imshow(adc_sl, cmap="gray", interpolation="bilinear")
    axes[0].set_title("1. ADC Native", color="#38BDF8", fontsize=11, weight="bold")
    axes[0].axis("off")

    # 2. Z-ADC Slice
    axes[1].imshow(z_sl, cmap="gray", interpolation="bilinear")
    axes[1].set_title("2. Z-Score ADC", color="#38BDF8", fontsize=11, weight="bold")
    axes[1].axis("off")

    # 3. Probability Heatmap
    im = axes[2].imshow(prob_sl, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("3. Prediction Prob", color="#F59E0B", fontsize=11, weight="bold")
    axes[2].axis("off")

    # 4. Lesion Overlay on ADC
    axes[3].imshow(adc_sl, cmap="gray", interpolation="bilinear")
    H, W = pred_sl.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    # Coral/Red mask for predicted lesion
    overlay[..., 0] = pred_sl * 1.0   # Red
    overlay[..., 1] = pred_sl * 0.25  # Slight green
    overlay[..., 3] = pred_sl * alpha # Alpha
    axes[3].imshow(overlay)
    n_lesion_px = int(pred_sl.sum())
    axes[3].set_title(f"4. Mask Overlay ({n_lesion_px} px)", color="#F43F5E", fontsize=11, weight="bold")
    axes[3].axis("off")

    # 5. Ground Truth & Error Map (if available)
    if has_gt and gt_sl is not None:
        axes[4].imshow(adc_sl, cmap="gray", interpolation="bilinear")
        tp = ((pred_sl > 0) & (gt_sl > 0)).astype(float)
        fp = ((pred_sl > 0) & (gt_sl == 0)).astype(float)
        fn = ((pred_sl == 0) & (gt_sl > 0)).astype(float)

        err_overlay = np.zeros((H, W, 4), dtype=np.float32)
        err_overlay[..., 1] = tp * 0.95   # Green = TP
        err_overlay[..., 0] = fp * 0.95   # Red = FP
        err_overlay[..., 2] = fn * 0.95   # Blue = FN
        err_overlay[..., 3] = (tp + fp + fn) * alpha
        axes[4].imshow(err_overlay)
        axes[4].set_title(f"5. Error Map (TP/FP/FN)", color="#10B981", fontsize=11, weight="bold")
        axes[4].axis("off")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if has_gt:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; gap: 20px; font-size: 0.85rem; color: #94A3B8; margin-top: -8px; margin-bottom: 12px;">
                <span><span style="color: #10B981; font-weight: bold;">●</span> Green: True Positive (TP)</span>
                <span><span style="color: #F43F5E; font-weight: bold;">●</span> Red: False Positive (FP)</span>
                <span><span style="color: #38BDF8; font-weight: bold;">●</span> Blue: False Negative (FN)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

