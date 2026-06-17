"""UCDNet Change Detection — Streamlit App Entry Point."""

import streamlit as st

st.set_page_config(
    page_title="UCDNet — Change Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ──────────────────────────────────────────────────
defaults = {
    "t1_path": None,  # Path to T1 image folder or file
    "t2_path": None,  # Path to T2 image folder or file
    "model_path": None,  # Path to best_model.keras
    "label_path": None,  # Optional ground-truth label
    "prob_map": None,  # np.ndarray from predict_pair
    "change_map": None,  # np.ndarray binary
    "metrics": {},  # dict from compute_metrics
    "train_history": None,  # Keras History object after training
    "settings": None,  # Settings dataclass instance
    "output_dir": "data/processed/artifacts",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ─────────────────────────────────────────────────────────────────
from components.sidebar import render_sidebar

render_sidebar()

# ── Landing page ─────────────────────────────────────────────────────────────
st.title("🛰️ UCDNet — Urban Change Detection")
st.markdown(
    """
    Detect man-made changes in **bi-temporal Sentinel-2 imagery** (13 spectral bands)
    using the **UCDNet** architecture ([Basavaraju et al., IEEE TGRS 2022](https://doi.org/10.1109/TGRS.2022.3161337))
    trained on the [OSCD dataset](https://ieee-datacomp.labri.fr/oscd/).
    """
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("**Step 1**\n\nUpload T1 & T2 images on the **Upload Images** page.")
with col2:
    st.info("**Step 2**\n\nRun inference on the **Run Detection** page.")
with col3:
    st.success("**Step 3**\n\nExplore change maps on the **Results** page.")
with col4:
    st.warning("**Optional**\n\nTrain a new model on the **Train Model** page.")

st.divider()

# Quick status panel
st.subheader("Session Status")
s = st.session_state
c1, c2, c3, c4 = st.columns(4)
c1.metric("T1 Image", "✅ Set" if s.t1_path else "❌ Not set")
c2.metric("T2 Image", "✅ Set" if s.t2_path else "❌ Not set")
c3.metric("Model", "✅ Loaded" if s.model_path else "❌ Not set")
c4.metric("Last prediction", "✅ Done" if s.change_map is not None else "—")
