"""
plot_input_representations.py
=============================
Generates comparative bar charts and radar plots for all evaluated input representations.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("c:/ADHITHYAN/ischemic stroke lesions")
json_path = PROJECT_ROOT / "outputs/logs/controlled_input_representations_metrics.json"
out_dir = PROJECT_ROOT / "outputs/visualizations/scientific_evaluation"
out_dir.mkdir(parents=True, exist_ok=True)

with open(json_path) as f:
    data = json.load(f)

configs = [
    "config_A",
    "config_B",
    "config_C",
    "config_D1_dwi_aligned_zero",
    "config_D2_dwi_aligned_z",
    "config_D3_dwi_aligned_invz"
]

labels = [
    "A. [ADC, Z, Z]",
    "B. [ADC, -Z, -Z]",
    "C. [ADC, Z, -Z]",
    "D1. [-Z, ADC, 0]",
    "D2. [-Z, ADC, Z]",
    "D3. [-Z, ADC, -Z]"
]

base_dice = [data[c]["baseline"]["summary"]["dice"]["mean"] for c in configs]
dann_dice = [data[c]["dann"]["summary"]["dice"]["mean"] for c in configs]

base_prec = [data[c]["baseline"]["summary"]["precision"]["mean"] for c in configs]
dann_prec = [data[c]["dann"]["summary"]["precision"]["mean"] for c in configs]

base_rec = [data[c]["baseline"]["summary"]["recall"]["mean"] for c in configs]
dann_rec = [data[c]["dann"]["summary"]["recall"]["mean"] for c in configs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor="#0d1117")

x = np.arange(len(labels))
width = 0.35

for ax, metric_name, b_vals, d_vals, title in [
    (ax1, "Dice Score", base_dice, dann_dice, "Mean Dice Score Across Input Representations"),
    (ax2, "Precision", base_prec, dann_prec, "Mean Precision Across Input Representations")
]:
    ax.set_facecolor("#161b22")
    r1 = ax.bar(x - width/2, b_vals, width, label="Baseline Attention U-Net", color="#ff7b72", alpha=0.85, edgecolor="#f85149")
    r2 = ax.bar(x + width/2, d_vals, width, label="DANN Attention U-Net", color="#58a6ff", alpha=0.85, edgecolor="#1f6feb")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", color="white", fontsize=10, weight="bold")
    ax.set_ylabel(metric_name, color="white", fontsize=11, weight="bold")
    ax.set_title(title, color="white", fontsize=12, weight="bold", pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")
    ax.grid(axis="y", color="#30363d", linestyle="--", alpha=0.6)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=10)

    for r in r1:
        h = r.get_height()
        if h > 0.01:
            ax.text(r.get_x() + r.get_width()/2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", color="#ff7b72", fontsize=8, weight="bold")
    for r in r2:
        h = r.get_height()
        if h > 0.01:
            ax.text(r.get_x() + r.get_width()/2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", color="#58a6ff", fontsize=8, weight="bold")

plt.suptitle("Controlled Input Representation Benchmark (Neonatal BONBID-HIE Cohort)", color="white", fontsize=14, weight="bold", y=1.01)
plt.tight_layout()
out_file = out_dir / "input_representations_comparison.png"
plt.savefig(out_file, dpi=130, facecolor="#0d1117", bbox_inches="tight")
plt.close(fig)
print(f"Generated chart: {out_file}")

