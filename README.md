[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23196209&assignment_repo_type=AssignmentRepo)

# Change Detection in Satellite Imageries

Implementation of **UCDNet** ([Basavaraju et al., IEEE TGRS 2022](https://doi.org/10.1109/TGRS.2022.3161337)) for detecting man-made changes in bi-temporal Sentinel-2 imagery (13 spectral bands), using the [OSCD](https://ieee-datacomp.labri.fr/oscd/) dataset.

**Python version:** 3.10–3.12 (3.11 recommended). See `requirements.txt`.

## Project organization

```
├── README.md
├── requirements.txt
├── notebooks/                    # Exploratory notebooks
├── reports/
│   ├── figures/                  # Generated figures for reporting
│   ├── final_project_report/     # Quarto final report
│   └── README.md
├── src/
│   ├── data/
│   │   ├── raw/                  # OSCD dataset (download separately)
│   │   └── processed/artifacts/  # Models, metrics, predictions
│   ├── main.py                   # CLI: train | predict
│   ├── config.py                 # Settings and city splits
│   ├── preprocessing_data/
│   │   └── pre-processing.py     # OSCD load / patch extraction
│   ├── feature_engineering/
│   │   └── build_features.py     # Augmentation & tf.data pipelines
│   ├── models/
│   │   ├── ucdnet_architecture.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       └── visualize.py
├── Dockerfile
└── docker-compose.yml
```

## Setup

### Option A — Conda + Poetry (recommended, from UCDNET)

```bash
conda env create -f environment.yml
conda activate change-detection
poetry lock    # first time only; commit poetry.lock for reproducibility
poetry install
```

`poetry.toml` installs into the active Conda env (no separate `.venv`). Registers CLI commands `ucdnet` and `change-detection`.

### Option B — pip + requirements.txt (course template)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Linux, install GDAL before `rasterio` if needed:

```bash
sudo apt-get install gdal-bin libgdal-dev
```

Optional GPU support after install:

```bash
pip install "tensorflow[and-cuda]>=2.15"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Copy environment template:

```bash
cp .env.example .env
```

## Dataset

Download OSCD and place under `src/data/raw/onera-satellite-change-detection-dataset/` (see `src/data/raw/README.md`), or set:

```bash
export UCDNET_DATA_ROOT="/path/to/onera-satellite-change-detection-dataset"
export UCDNET_OUTPUT_DIR="./src/data/processed/artifacts"
```

## Usage

**Train** (from project root):

```bash
python src/main.py train --epochs 30
# or, after poetry install:
ucdnet train --epochs 30
poetry run ucdnet train --epochs 30
```

**Predict** on a T1/T2 pair:

```bash
python src/main.py predict \
  --model src/data/processed/artifacts/best_model.keras \
  --t1  /path/to/city/imgs_1_rect \
  --t2  /path/to/city/imgs_2_rect \
  --out src/data/processed/artifacts/predictions/change_map.tif
```

Outputs: `best_model.keras`, `metrics.csv`, `training_curves.png`, `training_results.json`, and inference GeoTIFF/PNG maps.

## Docker

```bash
docker compose build
docker compose --profile gpu run --rm ucdnet-gpu train --epochs 30
```

## CLI reference

```text
python src/main.py train [--data-root PATH] [--output-dir PATH] [--epochs N] [--batch-size N]
                         [--patch-size N] [--no-augment] [--no-oversample]

python src/main.py predict --t1 PATH --t2 PATH [--model PATH] [--out PATH] [--label PATH]
                           [--patch-size N] [--overlap N] [--threshold T]
```

## Citation

```bibtex
@article{basavaraju2022ucdnet,
  title={UCDNet: A Deep Learning Model for Urban Change Detection From Bi-Temporal Multispectral Sentinel-2 Satellite Images},
  author={Basavaraju, K S and others},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  year={2022},
  doi={10.1109/TGRS.2022.3161337}
}
```

## Legacy reference

Earlier experiments from the UCDNET prototype live under `project/UCDNET/archive/legacy/` and are not used by this layout.
