"""
oscd_loader.py — OSCD Dataset Loader for UCDNet

Memory-efficient pipeline — patches are saved to disk as .npy files
and loaded one-at-a-time during training via tf.data.  This avoids
the 3+ GiB RAM spike from stacking all 287 patches at once.

Paper-correct data pipeline:
  1. Extract 512×512 patches (STRIDE=64) from all 24 cities.
  2. Save each patch as an individual .npy file under CACHE_DIR.
  3. Randomly assign patch indices → train (200) / val (57) / test (30).
  4. Return file-path lists; tf.data loads each file on demand.

Usage:
    from src.preprocessing_data.oscd_loader import get_split_paths, make_dataset
"""

import os
from pathlib import Path

import numpy as np
import rasterio
import tensorflow as tf
from PIL import Image as PILImage
from rasterio.enums import Resampling

try:
    _PIL_RESAMPLE = PILImage.Resampling.BILINEAR
except AttributeError:
    _PIL_RESAMPLE = PILImage.BILINEAR

#  Sentinel-2 band file names (13 bands)
BAND_FILES = [
    "B01.tif",
    "B02.tif",
    "B03.tif",
    "B04.tif",
    "B05.tif",
    "B06.tif",
    "B07.tif",
    "B08.tif",
    "B8A.tif",
    "B09.tif",
    "B10.tif",
    "B11.tif",
    "B12.tif",
]

#  Patch parameters
PATCH_SIZE = 512
STRIDE = 64  # step between patch starts (= overlap amount per paper)

#  Disk cache directory
# Patches are stored here on first run and reused on subsequent runs.
# Use OUTPUT_DIR env var parent so cache persists with other artifacts.
# Falls back to "patch_cache" (relative to CWD) when env is unset.
_OUTPUT_DIR = os.environ.get("UCDNET_OUTPUT_DIR", "")
CACHE_DIR = os.path.join(_OUTPUT_DIR, "patch_cache") if _OUTPUT_DIR else "patch_cache"


#  Low-level image helpers


def read_bands(city_dir, subdir="imgs_1_rect", target_h=None, target_w=None):
    """Read all 13 bands, resample to target size, normalise to [0,1]."""
    band_dir = os.path.join(city_dir, subdir)

    if target_h is None or target_w is None:
        with rasterio.open(os.path.join(band_dir, "B04.tif")) as src:
            target_h, target_w = src.height, src.width

    bands = []
    for bfile in BAND_FILES:
        with rasterio.open(os.path.join(band_dir, bfile)) as src:
            data = src.read(
                1, out_shape=(target_h, target_w), resampling=Resampling.bilinear
            ).astype(np.float32)
        bands.append(data)

    img = np.stack(bands, axis=-1)
    img = np.clip(img, 0, 10000) / 10000.0
    return img


def read_label(label_dir, city, target_h, target_w):
    """Read change-map, remap OSCD values (1=no-change, 2=change) → 0/1."""
    cm_path = os.path.join(label_dir, city, "cm", f"{city}-cm.tif")
    with rasterio.open(cm_path) as src:
        label = src.read(
            1, out_shape=(target_h, target_w), resampling=Resampling.nearest
        ).astype(np.int32)
    return np.where(label == 2, 1, 0)


def pad_to_patch(img1, img2, label):
    """Reflect-pad so spatial dims are at least PATCH_SIZE."""
    H, W, _ = img1.shape
    ph = max(0, PATCH_SIZE - H)
    pw = max(0, PATCH_SIZE - W)
    if ph > 0 or pw > 0:
        img1 = np.pad(img1, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        img2 = np.pad(img2, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        label = np.pad(label, ((0, ph), (0, pw)), mode="reflect")
    return img1, img2, label


def to_one_hot(label, num_classes=2):
    """(H, W) int → (H, W, C) one-hot float32."""
    return tf.keras.utils.to_categorical(label, num_classes=num_classes).astype(
        np.float32
    )


# Disk-cache helpers


def _patch_paths(idx, cache_dir=CACHE_DIR):
    """Return (t1_path, t2_path, y_path) for patch index idx."""
    base = os.path.join(cache_dir, f"{idx:05d}")
    return base + "_t1.npy", base + "_t2.npy", base + "_y.npy"


def build_patch_cache(images_root, labels_root, all_cities, cache_dir=CACHE_DIR):
    """
    Extract all patches from all cities and save to disk as .npy files.
    Skips extraction if cache already exists (checks for index file).

    Returns
    -------
    total : int   total number of patches saved
    """
    index_file = os.path.join(cache_dir, "total.txt")

    # Re-use existing cache
    if os.path.isfile(index_file):
        with open(index_file) as f:
            total = int(f.read().strip())
        print(f"  Patch cache found: {total} patches in '{cache_dir}'")
        return total

    os.makedirs(cache_dir, exist_ok=True)
    patch_idx = 0

    print(f"\nBuilding patch cache in '{cache_dir}' ...")
    for city in all_cities:
        print(f"  Extracting {city} ...", end=" ", flush=True)
        city_dir = os.path.join(images_root, city)

        img1 = read_bands(city_dir, "imgs_1_rect")
        img2 = read_bands(
            city_dir, "imgs_2_rect", target_h=img1.shape[0], target_w=img1.shape[1]
        )
        H, W, _ = img1.shape
        label = read_label(labels_root, city, H, W)

        img1, img2, label = pad_to_patch(img1, img2, label)
        H, W, _ = img1.shape
        count = 0

        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                t1 = img1[y : y + PATCH_SIZE, x : x + PATCH_SIZE]  # (512,512,13)
                t2 = img2[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                lbl = label[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                y_oh = to_one_hot(lbl)  # (512,512,2)

                p1, p2, py = _patch_paths(patch_idx, cache_dir)
                np.save(p1, t1)
                np.save(p2, t2)
                np.save(py, y_oh)

                patch_idx += 1
                count += 1

        print(f"done  ({count} patches)")

    # Write index so next run skips extraction
    with open(index_file, "w") as f:
        f.write(str(patch_idx))

    print(f"  Cache complete: {patch_idx} patches saved.\n")
    return patch_idx


# Main split function


def get_split_paths(
    images_root,
    labels_root,
    all_cities,
    n_train=200,
    n_val=57,
    n_test=30,
    seed=42,
    cache_dir=CACHE_DIR,
):
    """
    Paper-correct pipeline:
      1. Build (or reuse) patch cache on disk.
      2. Randomly assign patch indices → train / val / test.
      3. Return lists of file paths for each split.

    Returns
    -------
    tr_paths, vl_paths, te_paths : list of (t1_path, t2_path, y_path) tuples
    """
    total = build_patch_cache(images_root, labels_root, all_cities, cache_dir)

    assert total >= n_train + n_val + n_test, (
        f"Only {total} patches available; need "
        f"{n_train}+{n_val}+{n_test}={n_train+n_val+n_test}. "
        f"Consider reducing n_train/n_val/n_test."
    )

    np.random.seed(seed)
    idx = np.random.permutation(total)

    tr_idx = idx[:n_train]
    vl_idx = idx[n_train : n_train + n_val]
    te_idx = idx[n_train + n_val : n_train + n_val + n_test]

    def to_paths(indices):
        return [_patch_paths(i, cache_dir) for i in indices]

    print(
        f"  Split → train: {len(tr_idx)} | " f"val: {len(vl_idx)} | test: {len(te_idx)}"
    )

    return to_paths(tr_idx), to_paths(vl_idx), to_paths(te_idx)


# tf.data pipeline


# ── Inference I/O (used by streamlit_app) ────────────────────────────────

S2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B10", "B11", "B12",
]


def normalize_reflectance(img):
    return np.clip(img, 0, 10000).astype(np.float32) / 10000.0


def normalize_per_band(img):
    mn = img.min(axis=(0, 1), keepdims=True)
    mx = img.max(axis=(0, 1), keepdims=True)
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    return ((img - mn) / denom).astype(np.float32)


def _load_tif(path):
    try:
        with rasterio.open(path) as src:
            arr = src.read()
        return arr.transpose(1, 2, 0).astype(np.float32)
    except Exception:
        pass
    arr = np.array(PILImage.open(path)).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return arr


def _resize_to(img, h, w):
    out = []
    for c in range(img.shape[2]):
        ch = PILImage.fromarray(img[:, :, c]).resize((w, h), _PIL_RESAMPLE)
        out.append(np.array(ch, dtype=np.float32))
    return np.stack(out, axis=2)


def _load_from_path(path, num_bands=13):
    path = Path(path)
    if path.is_file():
        img = _load_tif(str(path))
        if img.shape[2] < num_bands:
            raise FileNotFoundError(f"{path} has {img.shape[2]} bands, need {num_bands}")
        return img[:, :, :num_bands]

    tifs = list(path.glob("*.tif")) + list(path.glob("*.TIF")) + list(path.glob("*.tiff"))
    found = {}
    for p in tifs:
        stem = p.stem.upper()
        for band in S2_BANDS:
            if stem == band.upper() or band.upper() in stem:
                found[band] = p
                break

    if len(found) >= num_bands:
        arrays = [_load_tif(str(found[b]))[:, :, 0] for b in S2_BANDS[:num_bands]]
        ref_h, ref_w = arrays[1].shape
        resized = []
        for arr in arrays:
            if arr.shape != (ref_h, ref_w):
                arr = np.array(
                    PILImage.fromarray(arr).resize((ref_w, ref_h), _PIL_RESAMPLE),
                    dtype=np.float32,
                )
            resized.append(arr)
        return np.stack(resized, axis=2).astype(np.float32)

    for p in tifs:
        try:
            img = _load_tif(str(p))
            if img.shape[2] >= num_bands:
                return img[:, :, :num_bands]
        except Exception:
            continue

    raise FileNotFoundError(
        f"Could not load {num_bands} bands from {path}. "
        f"Provide per-band TIFs (B01.tif … B12.tif) or one stacked GeoTIFF."
    )


def load_image_pair(t1_path, t2_path, num_bands=13, normalize="reflectance"):
    """Load a bi-temporal image pair for inference."""
    img1 = _load_from_path(t1_path, num_bands)
    img2 = _load_from_path(t2_path, num_bands)

    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = _resize_to(img2, h, w)

    if normalize == "per_band":
        img1 = normalize_per_band(img1)
        img2 = normalize_per_band(img2)
    else:
        img1 = normalize_reflectance(img1)
        img2 = normalize_reflectance(img2)

    return img1, img2


def load_label(path, target_shape=None):
    """Load a change-map label, remap OSCD values → 0/1."""
    path = Path(path)
    try:
        with rasterio.open(path) as src:
            lbl = src.read(1)
        lbl = np.where(lbl == 2, 1, np.where(lbl > 0, 1, 0)).astype(np.uint8)
    except Exception:
        lbl = (np.array(PILImage.open(path)) > 0).astype(np.uint8)

    if target_shape and lbl.shape != target_shape:
        h, w = target_shape
        lbl = np.array(PILImage.fromarray(lbl).resize((w, h), _PIL_RESAMPLE)) > 0
        lbl = lbl.astype(np.uint8)
    return lbl


def make_dataset(path_tuples, batch_size=1, shuffle=False):
    """
    Build a tf.data.Dataset that loads one patch at a time from disk.

    Parameters
    ----------
    path_tuples : list of (t1_path, t2_path, y_path)
    batch_size  : int
    shuffle     : bool

    Returns
    -------
    tf.data.Dataset yielding ({"T1": ..., "T2": ...}, Y)
    each of shape (batch, 512, 512, 13/2) float32
    """
    t1_paths = [p[0] for p in path_tuples]
    t2_paths = [p[1] for p in path_tuples]
    y_paths = [p[2] for p in path_tuples]

    def load_patch(t1_p, t2_p, y_p):
        t1 = tf.numpy_function(
            lambda p: np.load(p.decode()).astype(np.float32), [t1_p], tf.float32
        )
        t2 = tf.numpy_function(
            lambda p: np.load(p.decode()).astype(np.float32), [t2_p], tf.float32
        )
        y = tf.numpy_function(
            lambda p: np.load(p.decode()).astype(np.float32), [y_p], tf.float32
        )

        t1.set_shape([PATCH_SIZE, PATCH_SIZE, 13])
        t2.set_shape([PATCH_SIZE, PATCH_SIZE, 13])
        y.set_shape([PATCH_SIZE, PATCH_SIZE, 2])

        return {"T1": t1, "T2": t2}, y

    ds = tf.data.Dataset.from_tensor_slices((t1_paths, t2_paths, y_paths))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(t1_paths), reshuffle_each_iteration=True)

    ds = ds.map(load_patch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
