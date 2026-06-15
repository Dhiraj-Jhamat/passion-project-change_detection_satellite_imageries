"""
train_model.py — Train UCDNet on the OSCD dataset

Memory-efficient: patches are loaded one-at-a-time from disk via tf.data.
No full dataset is held in RAM at any point.

Exact paper settings:
  - Adam lr=0.0001, 30 epochs, batch_size=1
  - Loss: WCCE + k_weight * Modified_Kappa  (Eq. 13–17)
  - Class weights: (0.1, 0.9)
  - Random 200/57/30 patch split from all 24 cities

Run from project root:
    python -m src.models.train_model
"""

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    IMAGES_ROOT, LABELS_ROOT,
    ALL_CITIES,
    N_TRAIN_PATCHES, N_VAL_PATCHES, N_TEST_PATCHES,
    INPUT_SHAPE, NUM_CLASSES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    CLASS_WEIGHTS,
    CHECKPOINT_PATH, CURVES_PATH, METRICS_PATH,
)
from src.preprocessing_data.oscd_loader import get_split_paths, make_dataset
from src.models.ucdnet_architecture import build_ucdnet
from src.models.losses import ucdnet_loss, k_warmup
from src.models.metrics import jaccard_index, f1_score


# CALLBACKS


def get_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(METRICS_PATH),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1,
        ),
    ]


class KappaWarmup(tf.keras.callbacks.Callback):
    """
    Linearly ramps the kappa loss weight from 0 → 1 over `warmup_epochs`.
    Prevents the kappa term from destabilising early training.
    """
    def __init__(self, warmup_epochs=10):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        w = min(1.0, (epoch + 1) / self.warmup_epochs)
        k_warmup.assign(w)
        print(f"  k_warmup = {w:.2f}")



# PLOT CURVES  (paper Fig. 6 + all tracked metrics)


def plot_curves(history):
    h = history.history

    # (train_key, val_key, y-axis / title label)
    candidates = [
        ("accuracy",      "val_accuracy",      "Accuracy"),
        ("loss",          "val_loss",           "Loss"),
        ("jaccard_index", "val_jaccard_index",  "Jaccard Index (IoU)"),
        ("f1_score",      "val_f1_score",       "F1 Score"),
        ("precision",     "val_precision",      "Precision"),
        ("recall",        "val_recall",         "Recall"),
    ]

    # Keep only metrics that were actually recorded
    metrics = [(tr, vl, title) for tr, vl, title in candidates if tr in h]

    n     = len(metrics)
    ncols = 3
    nrows = (n + ncols - 1) // ncols   # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for ax, (train_key, val_key, title) in zip(axes, metrics):
        ax.plot(h[train_key], label="Train", color="steelblue")
        if val_key in h:
            ax.plot(h[val_key], label="Val", color="coral")
        ax.set_title(f"Training and Validation {title}")
        ax.set_xlabel("No. of epochs")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide any unused subplot panels
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(CURVES_PATH, dpi=150)
    plt.close()
    print(f"  Curves saved → {CURVES_PATH}")



# MAIN


def main():
    print("=" * 60)
    print("  UCDNet Training  (paper-faithful, memory-efficient)")
    print("=" * 60)
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Class weights : {CLASS_WEIGHTS}")
    print(f"  Patch split   : {N_TRAIN_PATCHES} / {N_VAL_PATCHES} / {N_TEST_PATCHES}")

    # Step 1: build (or reuse) disk cache, get file-path splits 
    print("\n[1/4] Preparing patch file paths ...")
    tr_paths, vl_paths, te_paths = get_split_paths(
        images_root=IMAGES_ROOT,
        labels_root=LABELS_ROOT,
        all_cities=ALL_CITIES,
        n_train=N_TRAIN_PATCHES,
        n_val=N_VAL_PATCHES,
        n_test=N_TEST_PATCHES,
    )
    print(f"  Train: {len(tr_paths)} | Val: {len(vl_paths)} | Test: {len(te_paths)}")

    # Step 2 tf.data pipelines (load from disk on demand) 
    print("\n[2/4] Building tf.data pipelines ...")
    train_ds = make_dataset(tr_paths, batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = make_dataset(vl_paths, batch_size=BATCH_SIZE, shuffle=False)

    # Step 3: build and compile model
    print("\n[3/4] Building model ...")
    model = build_ucdnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=ucdnet_loss(class_weights=CLASS_WEIGHTS),
        metrics=[
            "accuracy",
            jaccard_index,
            f1_score,
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    # Step 4 train 
    print("\n[4/4] Training ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks() + [KappaWarmup(warmup_epochs=15)],
    )

    plot_curves(history)

    print("\n✓ Training complete!")
    print(f"  Best model  → {CHECKPOINT_PATH}")
    print(f"  Metrics CSV → {METRICS_PATH}")
    print(f"  Curves PNG  → {CURVES_PATH}")


if __name__ == "__main__":
    main()
    