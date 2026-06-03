"""Sliding-window inference on full-scene image pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from models.losses import make_loss
from models.metrics import compute_metrics, print_metrics
from models.ucdnet_architecture import BilinearResize, build_ucdnet
from preprocessing_data.oscd_loader import load_image_pair, load_label


def load_model(model_path: str | Path) -> tf.keras.Model:
    path = Path(model_path)
    if path.suffix == ".keras":
        return tf.keras.models.load_model(
            str(path),
            custom_objects={
                "BilinearResize": BilinearResize,
                "ucdnet_combined_loss": make_loss(),
            },
        )

    model = build_ucdnet(patch_size=None, num_bands=13, num_classes=2)
    model.load_weights(str(path))
    return model


def predict_sliding_window(
    model: tf.keras.Model,
    img1: np.ndarray,
    img2: np.ndarray,
    patch_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
) -> np.ndarray:
    h, w = img1.shape[:2]
    stride = max(1, patch_size - overlap)
    prob_acc = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    coords = [(y0, x0) for y0 in range(0, h, stride) for x0 in range(0, w, stride)]
    p1_batch, p2_batch, coord_batch = [], [], []

    def flush():
        nonlocal p1_batch, p2_batch, coord_batch
        if not p1_batch:
            return
        t1b = np.stack(p1_batch, axis=0)
        t2b = np.stack(p2_batch, axis=0)
        preds = model.predict({"T1": t1b, "T2": t2b}, verbose=0)
        for (y0, x0), pred in zip(coord_batch, preds):
            y2 = min(y0 + patch_size, h)
            x2 = min(x0 + patch_size, w)
            ph, pw = y2 - y0, x2 - x0
            prob_acc[y0:y2, x0:x2] += pred[:ph, :pw, 1]
            count[y0:y2, x0:x2] += 1.0
        p1_batch, p2_batch, coord_batch = [], [], []

    for y0, x0 in tqdm(coords, desc="Patches", unit="patch"):
        y2 = min(y0 + patch_size, h)
        x2 = min(x0 + patch_size, w)
        p1 = img1[y0:y2, x0:x2]
        p2 = img2[y0:y2, x0:x2]
        ph, pw = patch_size - p1.shape[0], patch_size - p1.shape[1]
        if ph > 0 or pw > 0:
            p1 = np.pad(p1, ((0, ph), (0, pw), (0, 0)), mode="constant")
            p2 = np.pad(p2, ((0, ph), (0, pw), (0, 0)), mode="constant")
        p1_batch.append(p1)
        p2_batch.append(p2)
        coord_batch.append((y0, x0))
        if len(p1_batch) == batch_size:
            flush()
    flush()

    count = np.where(count == 0, 1.0, count)
    return (prob_acc / count).astype(np.float32)


def save_outputs(
    out_path: Path,
    prob_map: np.ndarray,
    change_map: np.ndarray,
    reference_tif: str | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = out_path.with_suffix("")
    prob_path = Path(f"{base}_probability.tif")
    overlay_path = Path(f"{base}_overlay.png")

    saved_geo = False
    if reference_tif and Path(reference_tif).is_file():
        try:
            import rasterio

            with rasterio.open(reference_tif) as ref:
                profile = ref.profile.copy()
            h, w = change_map.shape
            profile.update(count=1, width=w, height=h)

            with rasterio.open(out_path, "w", **{**profile, "dtype": "uint8"}) as dst:
                dst.write(change_map.astype(np.uint8), 1)
            with rasterio.open(prob_path, "w", **{**profile, "dtype": "float32"}) as dst:
                dst.write(prob_map.astype(np.float32), 1)
            saved_geo = True
            print(f"  Change map (GeoTIFF): {out_path}")
            print(f"  Probability      : {prob_path}")
        except ImportError:
            pass

    if not saved_geo:
        from PIL import Image

        png_path = Path(f"{base}_change_map.png")
        Image.fromarray((change_map * 255).astype(np.uint8)).save(png_path)
        Image.fromarray((prob_map * 255).astype(np.uint8)).save(f"{base}_probability.png")
        print(f"  Change map (PNG): {png_path}")

    try:
        from PIL import Image

        grey = (prob_map * 200).clip(0, 200).astype(np.uint8)
        rgb = np.stack([grey, grey, grey], axis=2)
        mask = change_map.astype(bool)
        rgb[mask] = [230, 30, 30]
        Image.fromarray(rgb.astype(np.uint8), "RGB").save(overlay_path)
        print(f"  Overlay          : {overlay_path}")
    except Exception as exc:
        print(f"  [warn] overlay skipped: {exc}")


def predict_pair(
    model_path: str | Path,
    t1_path: str | Path,
    t2_path: str | Path,
    out_path: str | Path,
    *,
    label_path: str | Path | None = None,
    patch_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    threshold: float = 0.5,
    normalize: str = "reflectance",
) -> dict:
    print(f"  Model : {model_path}")
    print(f"  T1    : {t1_path}")
    print(f"  T2    : {t2_path}")

    model = load_model(model_path)
    img1, img2 = load_image_pair(t1_path, t2_path, normalize=normalize)
    print(f"  Shape : {img1.shape}")

    prob_map = predict_sliding_window(
        model, img1, img2,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
    )
    change_map = (prob_map >= threshold).astype(np.uint8)
    changed_pct = 100.0 * change_map.sum() / change_map.size
    print(f"  Changed pixels: {change_map.sum():,} ({changed_pct:.2f}%)")

    ref = None
    t1p = Path(t1_path)
    if t1p.is_dir():
        tifs = list(t1p.glob("*.tif"))
        ref = str(tifs[0]) if tifs else None
    elif t1p.is_file():
        ref = str(t1p)

    save_outputs(Path(out_path), prob_map, change_map, reference_tif=ref)

    metrics = {}
    if label_path:
        label = load_label(label_path, target_shape=img1.shape[:2])
        metrics = compute_metrics(label.ravel(), (prob_map >= threshold).ravel())
        print_metrics(metrics, title="Evaluation vs label")

    return {"prob_map": prob_map, "change_map": change_map, "metrics": metrics}
