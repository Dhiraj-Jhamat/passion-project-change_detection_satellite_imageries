import os

# ── Env var keys (Docker convention; set in docker-compose.yml) ──
ENV_DATA_ROOT = "UCDNET_DATA_ROOT"
ENV_OUTPUT_DIR = "UCDNET_OUTPUT_DIR"

# ── Default paths (Docker-only; matched to docker-compose.yml mounts) ──
_DEFAULT_DATA_ROOT = "/data/oscd"
_DEFAULT_OUTPUT_DIR = "/app/data/processed/artifacts"

# ── Dataset paths ──
DATASET_ROOT = os.environ.get(ENV_DATA_ROOT, _DEFAULT_DATA_ROOT)
IMAGES_ROOT = os.path.join(DATASET_ROOT, "images")
LABELS_ROOT = os.path.join(DATASET_ROOT, "train_labels")

# ALL 24 CITIES
ALL_CITIES = [
    "aguasclaras",
    "bercy",
    "bordeaux",
    "nantes",
    "paris",
    "rennes",
    "saclay_e",
    "abudhabi",
    "cupertino",
    "pisa",
    "beihai",
    "hongkong",
    "beirut",
    "mumbai",
    "brasilia",
    "montpellier",
    "norcia",
    "rio",
    "saclay_w",
    "valencia",
    "dubai",
    "lasvegas",
    "milano",
    "chongqing",
]

# PATCH SPLIT COUNTS
N_TRAIN_PATCHES = 200
N_VAL_PATCHES = 57
N_TEST_PATCHES = 30

#  TRAINING HYPER-PARAMETERS
INPUT_SHAPE = (512, 512, 13)
NUM_CLASSES = 2
BATCH_SIZE = 1
EPOCHS = 60
LEARNING_RATE = 0.0001

# LOSS FUNCTION WEIGHTS
CLASS_WEIGHTS = (0.1, 0.9)

# OUTPUT PATHS (env-overridable)
OUTPUT_DIR = os.environ.get(ENV_OUTPUT_DIR, _DEFAULT_OUTPUT_DIR)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.keras")
CURVES_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.csv")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# ── Settings class (used by main.py & streamlit_app) ──────────────────────

class Settings:
    """Typed container for all configurable parameters.

    Created via :func:`load_settings` which reads env vars + CLI overrides.
    """

    def __init__(self, **kwargs):
        # Paths
        self.data_root = kwargs.pop("data_root", DATASET_ROOT)
        self.output_dir = kwargs.pop("output_dir", OUTPUT_DIR)

        # Training
        self.epochs = kwargs.pop("epochs", EPOCHS)
        self.batch_size = kwargs.pop("batch_size", BATCH_SIZE)
        self.patch_size = kwargs.pop("patch_size", INPUT_SHAPE[0])
        self.num_classes = kwargs.pop("num_classes", NUM_CLASSES)
        self.learning_rate = kwargs.pop("learning_rate", LEARNING_RATE)
        self.class_weights = kwargs.pop("class_weights", CLASS_WEIGHTS)
        self.input_shape = kwargs.pop("input_shape", INPUT_SHAPE)
        self.use_augmentation = kwargs.pop("use_augmentation", True)
        self.oversample_ratio = kwargs.pop("oversample_ratio", 3)

        # Inference
        self.overlap = kwargs.pop("overlap", 64)
        self.inference_batch_size = kwargs.pop("inference_batch_size", 4)
        self.threshold = kwargs.pop("threshold", 0.5)

        # City splits (default: use all cities for each set; user may narrow)
        self.train_cities = kwargs.pop("train_cities", list(ALL_CITIES))
        self.val_cities = kwargs.pop("val_cities", list(ALL_CITIES))
        self.test_cities = kwargs.pop("test_cities", list(ALL_CITIES))

        # Derived paths
        self.images_root = os.path.join(self.data_root, "images")
        self.labels_root = os.path.join(self.data_root, "train_labels")
        self.checkpoint_path = os.path.join(self.output_dir, "best_model.keras")
        self.curves_path = os.path.join(self.output_dir, "training_curves.png")
        self.metrics_path = os.path.join(self.output_dir, "metrics.csv")
        self.predictions_dir = os.path.join(self.output_dir, "predictions")

        # Remaining kwargs silently accepted for future compat
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)

    def __repr__(self):
        cls = type(self).__name__
        return (
            f"{cls}(data_root={self.data_root!r}, output_dir={self.output_dir!r}, "
            f"epochs={self.epochs}, batch_size={self.batch_size}, "
            f"patch_size={self.patch_size})"
        )


def load_settings(**overrides) -> Settings:
    """Build a :class:`Settings` from env vars + keyword overrides.

    Priority (highest wins):  ``**overrides`` > env vars > module defaults.

    Usage::

        settings = load_settings(data_root="/data/oscd", epochs=30)
        print(settings.checkpoint_path)
    """
    return Settings(**overrides)
