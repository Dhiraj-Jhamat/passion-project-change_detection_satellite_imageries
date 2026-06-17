"""
src/models/predict_model.py — Evaluate trained UCDNet on the OSCD test set
============================================================================
Paper: UCDNet (Basavaraju et al., IEEE TGRS 2022)

Computes all paper metrics on the 30 held-out test patches:
  Accuracy, Precision, Recall, F1, Kappa (Ka), Jaccard Index (JI)

Also saves:
  - outputs/test_metrics.csv        — per-patch and overall metrics
  - outputs/predictions/<idx>/      — side-by-side PNG for each patch
      t1_rgb.png        T1 false-colour (B4-B3-B2)
      t2_rgb.png        T2 false-colour (B4-B3-B2)
      ground_truth.png  binary label
      prediction.png    model prediction
      overlay.png       TP/FP/FN coloured overlay on T2

Run (from project root):
    python -m src.models.predict_model
"""

import os
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import (
    IMAGES_ROOT,
    LABELS_ROOT,
    ALL_CITIES,
    N_TRAIN_PATCHES,
    N_VAL_PATCHES,
    N_TEST_PATCHES,
    INPUT_SHAPE,
    NUM_CLASSES,
    CHECKPOINT_PATH,
    PREDICTIONS_DIR,
)
from src.preprocessing_data.oscd_loader import get_split_paths
from src.models.ucdnet_architecture import build_ucdnet
from src.models.metrics import compute_metrics, average_metrics

TEST_METRICS_PATH = os.path.join(os.path.dirname(CHECKPOINT_PATH), "test_metrics.csv")


# ── Public API (used by streamlit_app & main.py) ──────────────────────────


def load_model(model_path):
    """Build UCDNet and load trained weights."""
    model = build_ucdnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.load_weights(model_path)
    return model


def predict_sliding_window(model, img1, img2, patch_size=512, overlap=64, batch_size=4):
    """
    Run sliding-window inference over a full-resolution image pair.

    Returns a 2D probability map (H, W) float32 in [0, 1].
    """
    H, W = img1.shape[:2]
    stride = patch_size - overlap

    # Collect patches
    patches_t1, patches_t2, positions = [], [], []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y, H - patch_size) if y + patch_size > H else y
            x1 = min(x, W - patch_size) if x + patch_size > W else x
            patches_t1.append(img1[y1:y1 + patch_size, x1:x1 + patch_size])
            patches_t2.append(img2[y1:y1 + patch_size, x1:x1 + patch_size])
            positions.append((y1, x1))

    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for i in range(0, len(patches_t1), batch_size):
        batch_t1 = np.stack(patches_t1[i:i + batch_size], axis=0)
        batch_t2 = np.stack(patches_t2[i:i + batch_size], axis=0)
        preds = model.predict({"T1": batch_t1, "T2": batch_t2}, verbose=0)
        for j, (y1, x1) in enumerate(positions[i:i + batch_size]):
            prob = preds[j][..., 1]
            prob_map[y1:y1 + patch_size, x1:x1 + patch_size] += prob
            count_map[y1:y1 + patch_size, x1:x1 + patch_size] += 1.0

    prob_map = np.divide(prob_map, count_map, where=count_map > 0)
    return prob_map


def save_outputs(out_path, prob_map, change_map, reference_tif=None):
    """Save probability map and binary change map to disk."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import rasterio
        from rasterio.profiles import DefaultGTiffProfile

        if reference_tif and Path(reference_tif).is_file():
            with rasterio.open(reference_tif) as src:
                profile = src.profile.copy()
        else:
            profile = DefaultGTiffProfile()
            profile.update(height=change_map.shape[0], width=change_map.shape[1], count=1)

        prob_path = out_path.with_suffix(".prob.tif")
        prof_float = dict(profile, dtype=rasterio.float32)
        with rasterio.open(prob_path, "w", **prof_float) as dst:
            dst.write(prob_map, 1)

        prof_uint8 = dict(profile, dtype=rasterio.uint8)
        with rasterio.open(out_path, "w", **prof_uint8) as dst:
            dst.write(change_map, 1)
    except Exception:
        png_path = out_path.with_suffix(".png")
        plt.imsave(str(png_path), change_map, cmap="RdYlBu_r", vmin=0, vmax=1)
        prob_png = png_path.with_suffix(".prob.png")
        plt.imsave(str(prob_png), prob_map, cmap="RdYlBu_r", vmin=0, vmax=1)


def predict_pair(
    model_path,
    t1_path,
    t2_path,
    out_path,
    label_path=None,
    patch_size=512,
    overlap=64,
    batch_size=4,
    threshold=0.5,
    normalize="reflectance",
):
    """Full inference pipeline: load model + images, run, save, return metrics."""
    from src.preprocessing_data.oscd_loader import load_image_pair, load_label

    model = load_model(model_path)
    img1, img2 = load_image_pair(t1_path, t2_path, normalize=normalize)
    prob_map = predict_sliding_window(model, img1, img2, patch_size, overlap, batch_size)
    change_map = (prob_map >= threshold).astype(np.uint8)
    save_outputs(Path(out_path), prob_map, change_map, reference_tif=t1_path)

    metrics = {}
    if label_path and Path(label_path).is_file():
        label = load_label(label_path, target_shape=img1.shape[:2])
        from .metrics import compute_metrics
        metrics = compute_metrics(label.ravel(), change_map.ravel())

    return metrics


# VISUALISATION


def to_rgb(patch_13band):
    """
    Extract false-colour RGB from a 13-band Sentinel-2 patch.
    Uses bands B4 (idx 3), B3 (idx 2), B2 (idx 1) — visible RGB.
    Stretches to [0, 1] with 2% clip for display.
    """
    rgb = patch_13band[..., [3, 2, 1]]  # (H, W, 3)
    lo = np.percentile(rgb, 2, axis=(0, 1))
    hi = np.percentile(rgb, 98, axis=(0, 1))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb


def save_prediction_visuals(idx, t1, t2, y_true_bin, y_pred_bin, out_dir):
    """
    Save five images for one test patch:
      t1_rgb.png, t2_rgb.png, ground_truth.png, prediction.png, overlay.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # RGB previews
    plt.imsave(os.path.join(out_dir, "t1_rgb.png"), to_rgb(t1))
    plt.imsave(os.path.join(out_dir, "t2_rgb.png"), to_rgb(t2))

    # Binary maps (white=changed, black=unchanged)
    plt.imsave(
        os.path.join(out_dir, "ground_truth.png"),
        y_true_bin,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    plt.imsave(
        os.path.join(out_dir, "prediction.png"), y_pred_bin, cmap="gray", vmin=0, vmax=1
    )

    # TP/FP/FN coloured overlay on T2 RGB
    rgb = to_rgb(t2)
    overlay = rgb.copy()

    TP_mask = (y_true_bin == 1) & (y_pred_bin == 1)  # green
    FP_mask = (y_true_bin == 0) & (y_pred_bin == 1)  # red
    FN_mask = (y_true_bin == 1) & (y_pred_bin == 0)  # blue

    overlay[TP_mask] = [0.0, 0.8, 0.0]
    overlay[FP_mask] = [0.9, 0.1, 0.1]
    overlay[FN_mask] = [0.1, 0.3, 0.9]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.axis("off")
    legend = [
        mpatches.Patch(color=(0.0, 0.8, 0.0), label="TP"),
        mpatches.Patch(color=(0.9, 0.1, 0.1), label="FP"),
        mpatches.Patch(color=(0.1, 0.3, 0.9), label="FN"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)
    ax.set_title(f"Patch {idx:03d} — TP/FP/FN overlay", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overlay.png"), dpi=120)
    plt.close()


def save_summary_figure(all_metrics, out_path):
    """Bar chart of all paper metrics averaged over test patches."""
    keys = ["accuracy", "precision", "recall", "f1", "kappa", "jaccard"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "Ka (Kappa)", "JI (Jaccard)"]
    values = [all_metrics[k] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color="steelblue", width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(
        "UCDNet — Test Set Metrics (paper: Acc=99.3%, F1=89.21%, Ka=88.85%, JI=80.53%)"
    )
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Summary figure saved → {out_path}")


# MAIN


def main():
    print("=" * 60)
    print("  UCDNet Testing")
    print("=" * 60)

    # Step 1 load the same patch split as training
    print("\n[1/4] Loading test patch paths ...")
    _, _, te_paths = get_split_paths(
        images_root=IMAGES_ROOT,
        labels_root=LABELS_ROOT,
        all_cities=ALL_CITIES,
        n_train=N_TRAIN_PATCHES,
        n_val=N_VAL_PATCHES,
        n_test=N_TEST_PATCHES,
    )
    print(f"  Test patches : {len(te_paths)}")

    # Step 2 load model
    print(f"\n[2/4] Loading model from {CHECKPOINT_PATH} ...")
    assert os.path.exists(
        CHECKPOINT_PATH
    ), f"Model not found at {CHECKPOINT_PATH}. Run train_model.py first."
    model = build_ucdnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.load_weights(CHECKPOINT_PATH)
    print("  Model loaded (weights only).")

    #  Step 3 run inference patch by patch
    print("\n[3/4] Running inference on test patches ...")
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    all_metrics = []
    rows = []  # for CSV

    for i, (t1_path, t2_path, y_path) in enumerate(te_paths):

        # Load raw arrays
        t1 = np.load(t1_path).astype(np.float32)  # (512,512,13)
        t2 = np.load(t2_path).astype(np.float32)
        y_oh = np.load(y_path).astype(np.float32)  # (512,512,2)

        # Predict
        t1_b = t1[np.newaxis]  # (1,512,512,13)
        t2_b = t2[np.newaxis]
        pred = model.predict({"T1": t1_b, "T2": t2_b}, verbose=0)  # (1,512,512,2)
        pred = pred[0]  # (512,512,2)

        # Binarise
        y_true_bin = np.argmax(y_oh, axis=-1).astype(np.int32)  # (512,512)
        y_pred_bin = np.argmax(pred, axis=-1).astype(np.int32)

        # Metrics
        m = compute_metrics(y_true_bin, y_pred_bin)
        all_metrics.append(m)

        print(
            f"  Patch {i+1:3d}/{len(te_paths)} | "
            f"Acc={m['accuracy']*100:.2f}%  "
            f"F1={m['f1']*100:.2f}%  "
            f"Ka={m['kappa']*100:.2f}%  "
            f"JI={m['jaccard']*100:.2f}%"
        )

        # Save visuals
        patch_dir = os.path.join(PREDICTIONS_DIR, f"patch_{i:03d}")
        save_prediction_visuals(i, t1, t2, y_true_bin, y_pred_bin, patch_dir)

        # CSV row
        rows.append(
            {
                "patch": i,
                "t1_path": t1_path,
                "t2_path": t2_path,
                **{k: f"{v:.6f}" for k, v in m.items()},
            }
        )

    # Step 4 aggregate and report
    print("\n[4/4] Aggregating results ...")
    avg = average_metrics(all_metrics)

    print("\n" + "=" * 60)
    print("  TEST SET RESULTS (averaged over 30 patches)")
    print("=" * 60)
    print(f"  Accuracy  : {avg['accuracy']  * 100:.2f}%   (paper: 99.30%)")
    print(f"  Precision : {avg['precision'] * 100:.2f}%   (paper: 92.53%)")
    print(f"  Recall    : {avg['recall']    * 100:.2f}%   (paper: 86.16%)")
    print(f"  F1        : {avg['f1']        * 100:.2f}%   (paper: 89.21%)")
    print(f"  Ka(Kappa) : {avg['kappa']     * 100:.2f}%   (paper: 88.85%)")
    print(f"  JI        : {avg['jaccard']   * 100:.2f}%   (paper: 80.53%)")
    print("=" * 60)

    # Save CSV
    fieldnames = [
        "patch",
        "t1_path",
        "t2_path",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "kappa",
        "jaccard",
    ]
    with open(TEST_METRICS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        # Final row = averages
        writer.writerow(
            {
                "patch": "AVERAGE",
                "t1_path": "",
                "t2_path": "",
                **{
                    k: f"{avg[k]:.6f}"
                    for k in [
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "kappa",
                        "jaccard",
                    ]
                },
            }
        )

    print(f"\n  Per-patch CSV → {TEST_METRICS_PATH}")

    # Save summary bar chart
    summary_path = os.path.join(os.path.dirname(CHECKPOINT_PATH), "test_summary.png")
    save_summary_figure(avg, summary_path)

    print(f"  Predictions  → {PREDICTIONS_DIR}/")
    print("\n✓ Testing complete!")


if __name__ == "__main__":
    main()
