"""Components package for frontend."""
from .header import render_header
from .cards import render_metric_card, render_info_card
from .visualizer import render_slice_viewer
from .charts import create_training_curves_plot, create_model_comparison_chart, create_channel_sensitivity_chart

