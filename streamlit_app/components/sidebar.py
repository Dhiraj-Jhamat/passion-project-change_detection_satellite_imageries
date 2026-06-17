"""Sidebar: navigation info + model upload + global config."""

import tempfile
import streamlit as st
from pathlib import Path


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛰️ UCDNet")
        st.caption("Urban Change Detection · Sentinel-2 · OSCD")
        st.divider()

        st.markdown("### 🤖 Model")

        # Model file uploader
        model_file = st.file_uploader(
            "Upload model weights",
            type=["h5", "keras"],
            key="model_uploader",
            help="Upload your model.weights.h5 or best_model.keras file",
        )

        if model_file:
            # Save to temp dir
            tmp_model = Path(tempfile.gettempdir()) / f"ucdnet_{model_file.name}"
            tmp_model.write_bytes(model_file.read())
            st.session_state["model_path"] = str(tmp_model)
            st.success(f"✅ {model_file.name}")

        # Fallback: manual path input
        with st.expander("Or enter path manually"):
            model_input = st.text_input(
                "Model path",
                value=st.session_state.get("model_path") or "",
                placeholder="data/processed/artifacts/best_model.keras",
            )
            if model_input and Path(model_input).is_file():
                st.session_state["model_path"] = model_input
                st.success("Model found ✅")
            elif model_input:
                st.error("File not found")

        st.divider()

        # Output directory
        st.markdown("### 📁 Output")
        out_dir = st.text_input(
            "Output directory",
            value=st.session_state.get("output_dir", "data/processed/artifacts"),
        )
        st.session_state["output_dir"] = out_dir

        st.divider()
        st.markdown("### 📊 Session")

        s = st.session_state
        t1_path = s.get("t1_path")
        t2_path = s.get("t2_path")
        model_path = s.get("model_path")
        change_map = s.get("change_map")
        metrics = s.get("metrics")
        st.write(f"- T1: `{Path(t1_path).name if t1_path else '—'}`")
        st.write(f"- T2: `{Path(t2_path).name if t2_path else '—'}`")
        st.write(f"- Model: `{Path(model_path).name if model_path else '—'}`")
        st.write(f"- Prediction: {'done ✅' if change_map is not None else '—'}")
        if metrics:
            st.write(f"- F1: `{metrics.get('f1', 0)*100:.1f}%`")

        st.divider()
        if st.button("🗑️ Clear session", use_container_width=True):
            for key in ["t1_path", "t2_path", "prob_map", "change_map",
                        "metrics", "_img1_preview", "_img2_preview"]:
                st.session_state[key] = None if key != "metrics" else {}
            st.rerun()

        st.caption("UCDNet · IEEE TGRS 2022")
