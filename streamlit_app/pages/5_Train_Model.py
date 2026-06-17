"""Page 5 — Train Model: configure and launch UCDNet training."""

import subprocess
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.sidebar import render_sidebar
from utils.helpers import ensure_src_on_path

render_sidebar()

st.title("🏋️ Train Model")
st.caption("Configure hyperparameters and launch UCDNet training on the OSCD dataset.")

# ── Dataset ──────────────────────────────────────────────────────────────────
st.subheader("Dataset")
ensure_src_on_path()

data_root = st.text_input(
    "OSCD data root",
    placeholder="data/raw/onera-satellite-change-detection-dataset",
    help="Folder containing `images/` and `train_labels/` subdirectories.",
)
output_dir = st.text_input(
    "Output directory",
    value=st.session_state.get("output_dir", "data/processed/artifacts"),
    help="Where model checkpoints, metrics, and curves will be saved.",
)

if data_root and not Path(data_root).is_dir():
    st.error(f"Data root not found: `{data_root}`")

st.divider()

# ── Hyperparameters ───────────────────────────────────────────────────────────
st.subheader("Hyperparameters")

col1, col2, col3 = st.columns(3)
with col1:
    epochs = st.number_input("Epochs", 1, 300, 30, step=5)
    patch_size = st.number_input("Patch size", 64, 1024, 512, step=64)
with col2:
    batch_size = st.number_input("Batch size", 1, 16, 1)
    oversample = st.number_input("Oversample ratio", 1, 10, 3,
                                  help="How many times to repeat changed-class patches.")
with col3:
    use_augment = st.checkbox("Use augmentation", value=True)
    no_oversample = st.checkbox("Disable oversampling", value=False)

st.divider()

# ── City splits ───────────────────────────────────────────────────────────────
with st.expander("City splits (advanced)"):
    st.markdown("Default splits match the OSCD paper. Edit only if you know what you're doing.")

    try:
        from config import Settings
        default = Settings()
        train_cities = st.text_area("Train cities", value="\n".join(default.train_cities))
        val_cities = st.text_area("Val cities", value="\n".join(default.val_cities))
        test_cities = st.text_area("Test cities", value="\n".join(default.test_cities))
    except Exception:
        st.info("Could not load default city splits from config.py.")

st.divider()

# ── Launch ────────────────────────────────────────────────────────────────────
st.subheader("Launch Training")

mode = st.radio(
    "Execution mode",
    ["In-process (blocking)", "Subprocess (background)"],
    help="In-process: runs inside Streamlit — progress visible but UI is blocked. "
         "Subprocess: launches detached, check terminal for logs.",
)

if st.button("🚀 Start Training", type="primary", use_container_width=True):
    if data_root and not Path(data_root).is_dir():
        st.error("Data root directory not found.")
        st.stop()

    if mode == "In-process (blocking)":
        ensure_src_on_path()
        try:
            from config import load_settings
            overrides = {
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "patch_size": int(patch_size),
                "use_augmentation": use_augment,
            }
            if data_root:
                overrides["data_root"] = data_root
            if output_dir:
                overrides["output_dir"] = output_dir
            if no_oversample:
                overrides["oversample_ratio"] = 1
            else:
                overrides["oversample_ratio"] = int(oversample)

            settings = load_settings(**overrides)

            progress = st.progress(0, text="Importing training module…")
            from models.train_model import train

            progress.progress(10, text="Training started. This may take a long time…")
            st.info("Training is running. The UI will be unresponsive until it finishes. "
                    "Check the terminal for epoch-by-epoch output.")

            results = train(settings)

            progress.progress(100, text="Training complete ✅")
            st.success(f"✅ Training complete! Best model saved to `{results['checkpoint']}`")
            st.json(results)

            # Update session with new model path
            st.session_state["model_path"] = results["checkpoint"]

        except Exception as e:
            st.error(f"Training failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    else:
        # Build CLI command
        project_root = Path(__file__).resolve().parents[2]
        cmd = [
            sys.executable, str(project_root / "src" / "main.py"), "train",
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--patch-size", str(patch_size),
        ]
        if data_root:
            cmd += ["--data-root", data_root]
        if output_dir:
            cmd += ["--output-dir", output_dir]
        if not use_augment:
            cmd.append("--no-augment")
        if no_oversample:
            cmd.append("--no-oversample")

        st.code(" ".join(cmd), language="bash")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            st.info(f"Training launched (PID {proc.pid}). Monitor the terminal for progress.")
            st.session_state["_train_pid"] = proc.pid
        except Exception as e:
            st.error(f"Could not launch subprocess: {e}")

# ── Post-training quick view ──────────────────────────────────────────────────
st.divider()
st.subheader("Training Artifacts")
out = Path(output_dir)
curves = out / "training_curves.png"
if curves.is_file():
    st.image(str(curves), caption="Training curves", use_column_width=True)
else:
    st.info("Training curves will appear here after training completes.")
