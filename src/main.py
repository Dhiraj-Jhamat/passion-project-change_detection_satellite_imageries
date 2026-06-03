"""
Main entry point: train or predict with UCDNet.

From project root:
  python src/main.py train --epochs 30
  python src/main.py predict --t1 .../imgs_1_rect --t2 .../imgs_2_rect --model data/processed/artifacts/best_model.keras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_settings


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="OSCD dataset root (or set UCDNET_DATA_ROOT)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Artifacts directory (or set UCDNET_OUTPUT_DIR)",
    )


def _cmd_train(args: argparse.Namespace) -> None:
    overrides = {}
    if args.data_root:
        overrides["data_root"] = args.data_root
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.patch_size is not None:
        overrides["patch_size"] = args.patch_size
    if args.no_augment:
        overrides["use_augmentation"] = False
    if args.no_oversample:
        overrides["oversample_ratio"] = 1

    settings = load_settings(**overrides)

    from gpu import log_device_info
    from models.train_model import train

    log_device_info()

    train(settings)


def _cmd_predict(args: argparse.Namespace) -> None:
    from gpu import configure_gpu

    configure_gpu()
    settings = load_settings(
        data_root=args.data_root,
        output_dir=args.output_dir,
        patch_size=args.patch_size or 512,
        overlap=args.overlap or 64,
        inference_batch_size=args.batch_size or 4,
        threshold=args.threshold or 0.5,
    )

    model_path = args.model or str(settings.checkpoint_path)
    if not model_path:
        print("ERROR: provide --model or train a model first.", file=sys.stderr)
        sys.exit(1)

    from models.predict_model import predict_pair

    if not args.t1 or not args.t2:
        print("ERROR: --t1 and --t2 are required for predict.", file=sys.stderr)
        sys.exit(1)

    predict_pair(
        model_path=model_path,
        t1_path=args.t1,
        t2_path=args.t2,
        out_path=args.out,
        label_path=args.label,
        patch_size=settings.patch_size,
        overlap=settings.overlap,
        batch_size=settings.inference_batch_size,
        threshold=settings.threshold,
        normalize=args.normalize,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="change-detection",
        description="UCDNet — urban change detection from bi-temporal Sentinel-2 imagery",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train on OSCD labelled cities")
    _add_common(p_train)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--patch-size", type=int, default=None)
    p_train.add_argument("--no-augment", action="store_true")
    p_train.add_argument("--no-oversample", action="store_true")
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="Run inference on a T1/T2 image pair")
    _add_common(p_pred)
    p_pred.add_argument("--model", type=str, default=None, help="Path to best_model.keras")
    p_pred.add_argument("--t1", type=str, required=True)
    p_pred.add_argument("--t2", type=str, required=True)
    p_pred.add_argument(
        "--out",
        type=str,
        default="src/data/processed/artifacts/predictions/change_map.tif",
    )
    p_pred.add_argument("--label", type=str, default=None)
    p_pred.add_argument("--patch-size", type=int, default=None)
    p_pred.add_argument("--overlap", type=int, default=None)
    p_pred.add_argument("--batch-size", type=int, default=None)
    p_pred.add_argument("--threshold", type=float, default=None)
    p_pred.add_argument(
        "--normalize",
        choices=("reflectance", "per_band"),
        default="reflectance",
    )
    p_pred.set_defaults(func=_cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
