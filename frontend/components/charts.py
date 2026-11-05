"""
charts.py
=========
Interactive Plotly charts for the medical AI frontend.
"""

from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def create_training_curves_plot(history: List[Dict]) -> go.Figure:
    """
    Generate an interactive Plotly figure displaying DANN training metrics:
    Segmentation Loss, Domain Loss, Total Loss, and Validation Dice over Epochs.
    """
    epochs = [h["epoch"] for h in history]
    seg_loss = [h["train_seg_loss"] for h in history]
    dom_loss = [h["train_domain_loss"] for h in history]
    total_loss = [h["train_total_loss"] for h in history]
    val_dice = [h["val_dice"] for h in history]
    lambda_p = [h["lambda_p"] for h in history]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Segmentation & Domain Losses",
            "Adult Validation Dice Score (Best: Epoch 6)",
            "GRL Adaptation Parameter (λ Schedule)",
            "Domain Classifier Accuracy"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # 1. Losses
    fig.add_trace(
        go.Scatter(x=epochs, y=seg_loss, name="Seg Loss", line=dict(color="#38BDF8", width=2.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=dom_loss, name="Domain Loss", line=dict(color="#F59E0B", width=2, dash="dot")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=total_loss, name="Total Loss", line=dict(color="#94A3B8", width=1.5)),
        row=1, col=1
    )

    # 2. Validation Dice
    fig.add_trace(
        go.Scatter(x=epochs, y=val_dice, name="Val Dice", line=dict(color="#10B981", width=3)),
        row=1, col=2
    )
    # Highlight Best Epoch (Epoch 6)
    fig.add_trace(
        go.Scatter(
            x=[6], y=[0.2652],
            mode="markers+text",
            name="Best Checkpoint (Ep 6)",
            marker=dict(color="#F43F5E", size=12, symbol="star"),
            text=["Best (0.2652)"],
            textposition="top center",
            textfont=dict(color="#F43F5E", size=11, family="Arial Black")
        ),
        row=1, col=2
    )

    # 3. GRL Lambda Schedule
    fig.add_trace(
        go.Scatter(x=epochs, y=lambda_p, name="λ Schedule", line=dict(color="#A855F7", width=2.5)),
        row=2, col=1
    )

    # 4. Domain Accuracy
    dom_acc = [h.get("train_domain_acc", 0.5) * 100 for h in history]
    fig.add_trace(
        go.Scatter(x=epochs, y=dom_acc, name="Domain Acc (%)", line=dict(color="#06B6D4", width=2.5)),
        row=2, col=2
    )
    fig.add_hline(y=50.0, line_dash="dash", line_color="#64748B", annotation_text="Ideal Confusion (50%)", row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#E2E8F0", family="Segoe UI"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=520,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )

    fig.update_xaxes(gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")

    return fig


def create_model_comparison_chart(unet_dice: float = 0.1451, dann_dice: float = 0.2652) -> go.Figure:
    """Generate comparative bar chart showing U-Net Zero-Shot vs DANN."""
    models = ["Baseline U-Net (Zero-Shot)", "DANN (Domain Adapted - Ep 6)"]
    dice_scores = [unet_dice, dann_dice]
    colors = ["#64748B", "#0284C7"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=models,
                y=dice_scores,
                text=[f"Dice: {s:.4f}" for s in dice_scores],
                textposition="auto",
                marker_color=colors,
                width=0.45,
            )
        ]
    )

    fig.update_layout(
        title="Model Performance Comparison (Neonatal Validation / Adaptation Benchmark)",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#E2E8F0"),
        height=380,
        yaxis=dict(range=[0, 0.35], gridcolor="#334155", title="Dice Coefficient"),
        xaxis=dict(gridcolor="#334155"),
        margin=dict(l=40, r=40, t=50, b=30),
    )
    return fig


def create_channel_sensitivity_chart() -> go.Figure:
    """Generate sensitivity chart of neonatal input channel mappings."""
    strategies = [
        "Primary Mapping<br>[ADC, Z-ADC, Z-ADC]",
        "Surrogate DWI<br>[-Z_ADC, ADC, 0]",
        "Zero-Padding<br>[0, ADC, 0]",
        "Triplicate ADC<br>[ADC, ADC, ADC]"
    ]
    dice_values = [0.1451, 0.4088, 0.0000, 0.0000]
    colors = ["#38BDF8", "#10B981", "#EF4444", "#F59E0B"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=strategies,
                y=dice_values,
                text=[f"{v:.4f}" for v in dice_values],
                textposition="auto",
                marker_color=colors,
                width=0.5,
            )
        ]
    )

    fig.update_layout(
        title="Input Channel Adaptation Sensitivity (Absent DWI & FLAIR Impact)",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(color="#E2E8F0"),
        height=380,
        yaxis=dict(range=[0, 0.5], gridcolor="#334155", title="Mean Dice Score"),
        xaxis=dict(gridcolor="#334155"),
        margin=dict(l=40, r=40, t=50, b=30),
    )
    return fig

