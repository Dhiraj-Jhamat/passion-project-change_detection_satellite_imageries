"""Runtime configuration (environment variables + defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    env = os.environ.get("UCDNET_PROJECT_ROOT") or os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    # src/config.py → project root is parent of src/
    return Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    project_root: Path = field(default_factory=_project_root)
    data_root: Path | None = None
    output_dir: Path | None = None

    patch_size: int = 512
    overlap: int = 64
    num_bands: int = 13
    num_classes: int = 2

    epochs: int = 30
    batch_size: int = 1
    learning_rate: float = 1e-4
    class_weights: tuple[float, float] = (0.1, 0.9)
    num_runs: int = 1
    seed: int = 42

    oversample_ratio: int = 3
    no_change_ratio: float = 1.0
    use_augmentation: bool = True

    inference_batch_size: int = 4
    threshold: float = 0.5

    train_cities: list[str] = field(
        default_factory=lambda: [
            "brasilia",
            "bercy",
            "bordeaux",
            "nantes",
            "paris",
            "rennes",
            "abudhabi",
            "cupertino",
            "pisa",
            "beirut",
            "lasvegas",
            "rio",
            "valencia",
            "aguasclaras",
            "saclay_e",
            "norcia",
        ]
    )
    val_cities: list[str] = field(
        default_factory=lambda: [
            "montpellier",
            "mumbai",
            "beihai",
            "hongkong",
            "chongqing",
        ]
    )
    test_cities: list[str] = field(default_factory=lambda: ["milano"])

    def __post_init__(self) -> None:
        if self.data_root is None:
            default_data = (
                self.project_root
                / "src"
                / "data"
                / "raw"
                / "onera-satellite-change-detection-dataset"
            )
            legacy = self.project_root / "onera-satellite-change-detection-dataset"
            if legacy.is_dir() and not default_data.is_dir():
                default_data = legacy
            self.data_root = Path(
                os.environ.get("UCDNET_DATA_ROOT", str(default_data))
            ).resolve()
        else:
            self.data_root = Path(self.data_root).resolve()

        if self.output_dir is None:
            self.output_dir = Path(
                os.environ.get(
                    "UCDNET_OUTPUT_DIR",
                    str(self.project_root / "src" / "data" / "processed" / "artifacts"),
                )
            ).resolve()
        else:
            self.output_dir = Path(self.output_dir).resolve()

    @property
    def images_root(self) -> Path:
        return self.data_root / "images"

    @property
    def labels_root(self) -> Path:
        return self.data_root / "train_labels"

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "best_model.keras"

    @property
    def metrics_csv(self) -> Path:
        return self.output_dir / "metrics.csv"

    @property
    def curves_path(self) -> Path:
        return self.output_dir / "training_curves.png"


def load_settings(**overrides) -> Settings:
    """Build settings from env + optional keyword overrides."""
    s = Settings()
    for key, value in overrides.items():
        if hasattr(s, key):
            setattr(s, key, value)
    s.__post_init__()
    s.output_dir.mkdir(parents=True, exist_ok=True)
    return s
