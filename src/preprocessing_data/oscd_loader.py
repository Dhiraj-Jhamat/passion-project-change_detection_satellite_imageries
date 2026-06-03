"""OSCD dataset loading and inference image I/O (importable module)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import tensorflow as tf
from rasterio.enums import Resampling

BAND_FILES = [
    "B01.tif", "B02.tif", "B03.tif", "B04.tif",
    "B05.tif", "B06.tif", "B07.tif", "B08.tif",
    "B8A.tif", "B09.tif", "B10.tif", "B11.tif", "B12.tif",
]

S2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B10", "B11", "B12",
]

try:
    from PIL import Image as PILImage

    _RESAMPLE = PILImage.Resampling.BILINEAR
except AttributeError:
    from PIL import Image as PILImage

    _RESAMPLE = PILImage.BILINEAR


def read_bands(city_dir, subdir: str = "imgs_1_rect", target_h=None, target_w=None):
    band_dir = city_dir / subdir if hasattr(city_dir, "__truediv__") else f"{city_dir}/{subdir}"

    if target_h is None or target_w is None:
        ref_path = f"{band_dir}/B04.tif"
        with rasterio.open(ref_path) as src:
            target_h, target_w = src.height, src.width

    bands = []
    for bfile in BAND_FILES:
        with rasterio.open(f"{band_dir}/{bfile}") as src:
            data = src.read(
                1,
                out_shape=(target_h, target_w),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
        bands.append(data)

    img = np.stack(bands, axis=-1)
    return np.clip(img, 0, 10000) / 10000.0


def read_label(label_dir, city: str, target_h: int, target_w: int):
    cm_path = f"{label_dir}/{city}/cm/{city}-cm.tif"
    with rasterio.open(cm_path) as src:
        label = src.read(
            1,
            out_shape=(target_h, target_w),
            resampling=Resampling.nearest,
        ).astype(np.int32)
    return np.where(label == 2, 1, 0)


def extract_patches(img1, img2, label, patch_size: int, stride: int, no_change_ratio: float = 1.0):
    h, w, _ = img1.shape
    changed, bg_pool = [], []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            t1_p = img1[y : y + patch_size, x : x + patch_size]
            t2_p = img2[y : y + patch_size, x : x + patch_size]
            lbl_p = label[y : y + patch_size, x : x + patch_size]
            if np.sum(lbl_p == 1) >= 5:
                changed.append((t1_p, t2_p, lbl_p))
            else:
                bg_pool.append((t1_p, t2_p, lbl_p))

    n_sample = max(1, int(len(changed) * no_change_ratio))
    if bg_pool:
        idx = np.random.choice(
            len(bg_pool),
            size=min(n_sample, len(bg_pool)),
            replace=False,
        )
        sampled = [bg_pool[i] for i in idx]
    else:
        sampled = []

    all_patches = changed + sampled
    t1_patches = [p[0] for p in all_patches]
    t2_patches = [p[1] for p in all_patches]
    lbl_patches = [p[2] for p in all_patches]
    return t1_patches, t2_patches, lbl_patches, len(changed)


def to_one_hot(label, num_classes: int = 2):
    return tf.keras.utils.to_categorical(label, num_classes=num_classes).astype(np.float32)


def load_oscd_dataset(
    images_root,
    labels_root,
    city_list: list[str],
    patch_size: int = 512,
    overlap: int = 64,
    no_change_ratio: float = 1.0,
):
    stride = patch_size - overlap
    t1_all, t2_all, y_all = [], [], []
    total_changed = total_patches = 0

    for city in city_list:
        print(f"  Loading {city} ...", end=" ", flush=True)
        city_dir = f"{images_root}/{city}"
        img1 = read_bands(city_dir, "imgs_1_rect")
        img2 = read_bands(city_dir, "imgs_2_rect", target_h=img1.shape[0], target_w=img1.shape[1])
        h, w, _ = img1.shape
        label = read_label(labels_root, city, h, w)

        t1p, t2p, lp, n_changed = extract_patches(
            img1, img2, label, patch_size, stride, no_change_ratio
        )
        for t1, t2, lbl in zip(t1p, t2p, lp):
            t1_all.append(t1)
            t2_all.append(t2)
            y_all.append(to_one_hot(lbl))

        total_changed += n_changed
        total_patches += len(t1p)
        print(f"{len(t1p)} patches ({n_changed} with change)")

    print(
        f"  Total: {total_patches} patches, {total_changed} changed "
        f"({100 * total_changed / max(total_patches, 1):.1f}%)"
    )
    return (
        np.array(t1_all, dtype=np.float32),
        np.array(t2_all, dtype=np.float32),
        np.array(y_all, dtype=np.float32),
    )


def _load_tif(path: str) -> np.ndarray:
    try:
        import rasterio

        with rasterio.open(path) as src:
            arr = src.read()
        return arr.transpose(1, 2, 0).astype(np.float32)
    except Exception:
        pass
    arr = np.array(PILImage.open(path)).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return arr


def normalize_reflectance(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 10000).astype(np.float32) / 10000.0


def normalize_per_band(img: np.ndarray) -> np.ndarray:
    mn = img.min(axis=(0, 1), keepdims=True)
    mx = img.max(axis=(0, 1), keepdims=True)
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    return ((img - mn) / denom).astype(np.float32)


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    out = []
    for c in range(img.shape[2]):
        ch = PILImage.fromarray(img[:, :, c]).resize((w, h), _RESAMPLE)
        out.append(np.array(ch, dtype=np.float32))
    return np.stack(out, axis=2)


def _load_from_path(path: str | Path, num_bands: int) -> np.ndarray:
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
        arrays = [_load_tif(str(found[b]))[:, :, 0] for b in S2_BANDS]
        ref_h, ref_w = arrays[1].shape
        resized = []
        for arr in arrays:
            if arr.shape != (ref_h, ref_w):
                arr = np.array(
                    PILImage.fromarray(arr).resize((ref_w, ref_h), _RESAMPLE),
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


def load_image_pair(
    t1_path: str | Path,
    t2_path: str | Path,
    num_bands: int = 13,
    normalize: str = "reflectance",
) -> tuple[np.ndarray, np.ndarray]:
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


def load_label(path: str | Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    path = Path(path)
    try:
        with rasterio.open(path) as src:
            lbl = src.read(1)
        lbl = np.where(lbl == 2, 1, np.where(lbl > 0, 1, 0)).astype(np.uint8)
    except Exception:
        lbl = (np.array(PILImage.open(path)) > 0).astype(np.uint8)

    if target_shape and lbl.shape != target_shape:
        h, w = target_shape
        lbl = np.array(PILImage.fromarray(lbl).resize((w, h), _RESAMPLE)) > 0
        lbl = lbl.astype(np.uint8)
    return lbl


