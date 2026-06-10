"""Train UCDNet on the OSCD dataset."""

from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

from config import Settings
from feature_engineering.build_features import make_tf_dataset, oversample_changed_patches
from gpu import log_device_info
from models.losses import make_loss
from models.metrics import (
    changed_class_f1,
    changed_class_jaccard,
    compute_metrics,
    print_metrics,
)
from models.ucdnet_architecture import build_ucdnet
from preprocessing_data.oscd_loader import load_oscd_dataset
from visualization.visualize import plot_training_curves


def _evaluate_patch_dataset(model, ds) -> dict[str, float]:
    y_true_all, y_pred_all = [], []
    for inputs, lbl in ds:
        preds = model.predict(inputs, verbose=0)
        y_true_all.append(lbl[..., 1].numpy().ravel())
        y_pred_all.append(preds[..., 1].ravel())
    return compute_metrics(
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
    )


def train(settings: Settings) -> dict:
    log_device_info()

    if not settings.images_root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found at {settings.data_root}\n"
            "Set UCDNET_DATA_ROOT or pass --data-root to the OSCD folder."
        )

    tf.random.set_seed(settings.seed)
    np.random.seed(settings.seed)

    print("\n[1/4] Loading data")
    t1_tr, t2_tr, y_tr = load_oscd_dataset(
        settings.images_root,
        settings.labels_root,
        settings.train_cities,
        patch_size=settings.patch_size,
        overlap=settings.overlap,
        no_change_ratio=settings.no_change_ratio,
    )
    if settings.oversample_ratio > 1:
        t1_tr, t2_tr, y_tr = oversample_changed_patches(
            t1_tr, t2_tr, y_tr, settings.oversample_ratio
        )

    t1_vl, t2_vl, y_vl = load_oscd_dataset(
        settings.images_root,
        settings.labels_root,
        settings.val_cities,
        patch_size=settings.patch_size,
        overlap=settings.overlap,
        no_change_ratio=1.0,
    )

    train_ds = make_tf_dataset(
        t1_tr,
        t2_tr,
        y_tr,
        batch_size=settings.batch_size,
        augment=settings.use_augmentation,
    )
    val_ds = make_tf_dataset(
        t1_vl,
        t2_vl,
        y_vl,
        batch_size=settings.batch_size,
        augment=False,
        shuffle=False,
    )

    print("\n[2/4] Building model")
    model = build_ucdnet(
        patch_size=settings.patch_size,
        num_bands=settings.num_bands,
        num_classes=settings.num_classes,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings.learning_rate),
        loss=make_loss(class_weights=settings.class_weights),
        metrics=[changed_class_f1, changed_class_jaccard, "accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(settings.checkpoint_path),
            monitor="val_changed_class_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_changed_class_f1",
            mode="max",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_changed_class_f1",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(settings.metrics_csv)),
    ]

    print("\n[3/4] Training")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.epochs,
        callbacks=callbacks,
    )

    plot_training_curves(history, settings.curves_path)

    print("\n[4/4] Test-set evaluation (patch level)")
    test_metrics = {}
    if settings.test_cities:
        t1_te, t2_te, y_te = load_oscd_dataset(
            settings.images_root,
            settings.labels_root,
            settings.test_cities,
            patch_size=settings.patch_size,
            overlap=settings.overlap,
            no_change_ratio=1.0,
        )
        test_ds = make_tf_dataset(
            t1_te,
            t2_te,
            y_te,
            batch_size=settings.batch_size,
            augment=False,
            shuffle=False,
        )
        from models.predict_model import load_model

        best = load_model(settings.checkpoint_path)
        test_metrics = _evaluate_patch_dataset(best, test_ds)
        print_metrics(test_metrics, title="Test cities")

    results = {
        "checkpoint": str(settings.checkpoint_path),
        "epochs_trained": len(history.history.get("loss", [])),
        "test_metrics": test_metrics,
    }
    results_path = settings.output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Best model : {settings.checkpoint_path}")
    print(f"✓ Metrics CSV: {settings.metrics_csv}")
    print(f"✓ Curves PNG : {settings.curves_path}")
    return results
