"""Page 1 — Dashboard: last run metrics, training curves, model status."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.sidebar import render_sidebar
from components.metrics_card import render_metrics_card
from utils.helpers import load_metrics_csv, ensure_src_on_path

render_sidebar()

st.title("📊 Dashboard")
st.caption("Overview of model status, last prediction, and training history.")

# ── Last prediction metrics ──────────────────────────────────────────────────
st.subheader("Last Prediction")
metrics = st.session_state.get("metrics", {})
if metrics:
    render_metrics_card(metrics, title="Evaluation vs ground truth")
elif st.session_state.get("change_map") is not None:
    import numpy as np
    cm = st.session_state["change_map"]
    changed_pct = 100.0 * cm.sum() / cm.size
    col1, col2, col3 = st.columns(3)
    col1.metric("Changed pixels", f"{cm.sum():,}")
    col2.metric("Changed area", f"{changed_pct:.2f}%")
    col3.metric("Image shape", f"{cm.shape[0]}×{cm.shape[1]}")
    st.info("No ground-truth label was provided, so evaluation metrics are unavailable. Supply a label on the Run Detection page.")
else:
    st.info("No prediction run yet. Go to **Run Detection** to get started.")

st.divider()

# ── Training curves ──────────────────────────────────────────────────────────
st.subheader("Training Curves")
ensure_src_on_path()
output_dir = Path(st.session_state.get("output_dir", "data/processed/artifacts"))

curves_png = output_dir / "training_curves.png"
metrics_csv = output_dir / "metrics.csv"

if curves_png.is_file():
    st.image(str(curves_png), caption="Training & validation curves", use_column_width=True)
else:
    st.info(f"No training curves found at `{curves_png}`. Train a model first.")

st.divider()

# ── Metrics CSV history ──────────────────────────────────────────────────────
st.subheader("Training History (CSV)")
if metrics_csv.is_file():
    try:
        df = load_metrics_csv(str(metrics_csv))
        st.dataframe(df, use_container_width=True)

        import plotly.express as px
        cols_to_plot = [c for c in df.columns if "f1" in c.lower() or "loss" in c.lower() or "jaccard" in c.lower()]
        if cols_to_plot:
            fig = px.line(df, y=cols_to_plot, title="Training metrics over epochs")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load metrics CSV: {e}")
else:
    st.info(f"No metrics CSV found at `{metrics_csv}`.")

st.divider()

# ── Model status ──────────────────────────────────────────────────────────────
st.subheader("Model Status")
model_path = st.session_state.get("model_path")
checkpoint = output_dir / "best_model.keras"

col1, col2 = st.columns(2)
with col1:
    if model_path and Path(model_path).is_file():
        st.success(f"✅ Model loaded: `{model_path}`")
    elif checkpoint.is_file():
        st.warning(f"⚠️ Default checkpoint found at `{checkpoint}` but not loaded. Set it in the sidebar.")
    else:
        st.error("❌ No model found. Train a model or provide a path in the sidebar.")

with col2:
    results_json = output_dir / "training_results.json"
    if results_json.is_file():
        import json
        with open(results_json) as f:
            results = json.load(f)
        st.json(results)
