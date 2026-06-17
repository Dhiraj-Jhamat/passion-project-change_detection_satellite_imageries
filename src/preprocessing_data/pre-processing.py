"""
Load and pre-process OSCD bi-temporal Sentinel-2 imagery (assignment entry script).

Run from project root:
  python src/preprocessing_data/pre-processing.py --data-root data/raw/onera-satellite-change-detection-dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocessing_data.oscd_loader import load_oscd_dataset


def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate OSCD dataset layout")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--cities", nargs="*", default=["milano"])
    args = parser.parse_args()
    root = Path(args.data_root)
    images = root / "images"
    labels = root / "train_labels"
    if not images.is_dir():
        raise SystemExit(f"Missing images folder: {images}")
    load_oscd_dataset(images, labels, args.cities, patch_size=512, overlap=64)


if __name__ == "__main__":
    _main()
