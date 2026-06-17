"""Page 3 — Run Detection: configure and execute predict_pair()."""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.sidebar import render_sidebar
from utils.helpers import ensure_src_on_path, load_model_cached

render_sidebar()

st.title("🔍 Run Detection")
st.caption("Configure inference parameters and run the sliding-window change detector.")

# ── Pre-flight check ─────────────────────────────────────────────────────────
t1 = st.session_state.get("t1_path")
t2 = st.session_state.get("t2_path")
model_path = st.session_state.get("model_path")

if not t1 or not t2:
    st.warning("⚠️ T1 and T2 paths not set. Go to **Upload Images** first.")
    st.stop()

st.success(f"✅ T1: `{t1}`")
st.success(f"✅ T2: `{t2}`")

# ── Inference parameters ─────────────────────────────────────────────────────
st.subheader("Inference Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    patch_size = st.number_input("Patch size (px)", min_value=64, max_value=1024,
                                  value=512, step=64,
                                  help="Must match training patch size.")
with col2:
    overlap = st.number_input("Overlap (px)", min_value=0, max_value=256,
                               value=64, step=16,
                               help="Overlap between adjacent patches to reduce edge artefacts.")
with col3:
    batch_size = st.number_input("Batch size", min_value=1, max_value=16,
                                  value=4, step=1,
                                  help="Number of patches processed simultaneously.")

col4, col5 = st.columns(2)
with col4:
    threshold = st.slider("Change threshold", 0.0, 1.0, 0.5, 0.01,
                           help="Probability above which a pixel is classified as changed.")
with col5:
    normalize = st.selectbox("Normalisation", ["reflectance", "per_band"],
                              index=0 if st.session_state.get("normalize", "reflectance") == "reflectance" else 1)

# Output path
output_dir = Path(st.session_state.get("output_dir", "data/processed/artifacts"))
out_path = st.text_input(
    "Output path (.tif)",
    value=str(output_dir / "predictions" / "change_map.tif"),
    help="GeoTIFF output. PNG fallback if rasterio is unavailable.",
)

# Model path
mp_input = st.text_input(
    "Model path (.keras)",
    value=model_path or str(output_dir / "best_model.keras"),
    help="Path to trained best_model.keras",
)
if mp_input and Path(mp_input).is_file():
    st.session_state["model_path"] = mp_input
    model_path = mp_input
elif mp_input:
    st.error(f"Model not found: `{mp_input}`")

label_path = st.session_state.get("label_path")
if label_path:
    st.info(f"Ground-truth label: `{label_path}` → metrics will be computed after prediction.")

st.divider()

# ── Run ──────────────────────────────────────────────────────────────────────
if st.button("▶ Run Change Detection", type="primary", use_container_width=True):
    if not model_path or not Path(model_path).is_file():
        st.error("Model file not found. Check the path above.")
        st.stop()

    ensure_src_on_path()

    progress = st.progress(0, text="Loading model…")
    status = st.empty()

    try:
        # Step 1: load model
        model = load_model_cached(model_path)
        progress.progress(20, text="Model loaded. Loading image pair…")

        from preprocessing_data.oscd_loader import load_image_pair
        img1, img2 = load_image_pair(t1, t2, normalize=normalize)
        progress.progress(40, text=f"Images loaded ({img1.shape}). Running sliding-window inference…")

        # Step 2: sliding window
        from models.predict_model import predict_sliding_window, save_outputs
        import numpy as np

        prob_map = predict_sliding_window(
            model, img1, img2,
            patch_size=int(patch_size),
            overlap=int(overlap),
            batch_size=int(batch_size),
        )
        progress.progress(80, text="Inference done. Saving outputs…")

        change_map = (prob_map >= threshold).astype(np.uint8)
        changed_pct = 100.0 * change_map.sum() / change_map.size

        # Step 3: save
        ref_tif = None
        p = Path(t1)
        if p.is_dir():
            tifs = list(p.glob("*.tif"))
            ref_tif = str(tifs[0]) if tifs else None
        elif p.is_file():
            ref_tif = str(p)

        save_outputs(Path(out_path), prob_map, change_map, reference_tif=ref_tif)
        progress.progress(90, text="Computing metrics…")

        # Step 4: optional metrics
        metrics = {}
        if label_path and Path(label_path).is_file():
            from preprocessing_data.oscd_loader import load_label
            from models.metrics import compute_metrics
            label = load_label(label_path, target_shape=img1.shape[:2])
            metrics = compute_metrics(label.ravel(), (prob_map >= threshold).ravel())

        # Store in session
        st.session_state["prob_map"] = prob_map
        st.session_state["change_map"] = change_map
        st.session_state["metrics"] = metrics
        st.session_state["_out_path"] = out_path

        progress.progress(100, text="Done ✅")
        st.success(
            f"✅ Prediction complete! "
            f"Changed: {change_map.sum():,} px ({changed_pct:.2f}%). "
            f"Go to **Results** to view the change map."
        )

        # Quick summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Changed pixels", f"{change_map.sum():,}")
        c2.metric("Changed area", f"{changed_pct:.2f}%")
        c3.metric("Image size", f"{change_map.shape[0]}×{change_map.shape[1]}")

        if metrics:
            st.subheader("Quick metrics")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("F1", f"{metrics['f1']*100:.2f}%")
            mc2.metric("Precision", f"{metrics['precision']*100:.2f}%")
            mc3.metric("Recall", f"{metrics['recall']*100:.2f}%")

    except Exception as e:
        progress.empty()
        st.error(f"Prediction failed: {e}")
        import traceback
        st.code(traceback.format_exc())
