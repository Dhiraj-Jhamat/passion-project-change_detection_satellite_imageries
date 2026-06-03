"""Evaluation metrics for the changed class."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Binary metrics from flat 0/1 arrays (changed class)."""
    y_true = y_true.astype(np.int32).ravel()
    y_pred = (y_pred.ravel() >= 0.5).astype(np.int32) if y_pred.dtype != np.int32 else y_pred.ravel()

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = tp + tn + fp + fn + 1e-9

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / total
    po = (tp + tn) / total
    pe = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / (total**2)
    kappa = (po - pe) / (1 - pe + 1e-9)
    jaccard = tp / (tp + fp + fn + 1e-9)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "kappa": float(kappa),
        "jaccard": float(jaccard),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def print_metrics(metrics: dict[str, float], title: str = "Metrics") -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for key in ("accuracy", "precision", "recall", "f1", "kappa", "jaccard"):
        print(f"  {key.capitalize():12s}: {metrics[key] * 100:6.2f} %")
    print(f"{'=' * 50}\n")


def changed_class_f1(y_true, y_pred, smooth=1e-6):
    """TensorFlow metric: F1 on the 'changed' class (channel 1)."""
    y_true_c = tf.cast(tf.argmax(y_true, axis=-1) == 1, tf.float32)
    y_pred_c = tf.cast(tf.argmax(y_pred, axis=-1) == 1, tf.float32)
    y_true_f = tf.reshape(y_true_c, [-1])
    y_pred_f = tf.reshape(y_pred_c, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return 2.0 * precision * recall / (precision + recall + smooth)


def changed_class_jaccard(y_true, y_pred, smooth=1e-6):
    """TensorFlow metric: IoU on the changed class."""
    y_true_c = tf.cast(tf.argmax(y_true, axis=-1) == 1, tf.float32)
    y_pred_c = tf.cast(tf.argmax(y_pred, axis=-1) == 1, tf.float32)
    y_true_f = tf.reshape(y_true_c, [-1])
    y_pred_f = tf.reshape(y_pred_c, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)
