"""
metrics.py — UCDNet Metrics


Two kinds of metrics are defined here:

  1. TF/Keras metrics (used during model.compile / model.fit)
       - jaccard_index   : IoU for the changed class
       - f1_score        : F1  for the changed class

  2. NumPy metrics (used during test-time evaluation, patch by patch)
       - compute_metrics  : returns dict of all paper metrics for one patch
       - average_metrics  : averages a list of per-patch metric dicts
"""

import numpy as np
import tensorflow as tf

SMOOTH = 1e-6


# 1. Keras (graph-mode) metrics 
# These are passed to model.compile(metrics=[...]) in train_model.py

def jaccard_index(y_true, y_pred, smooth=SMOOTH):
    """
    Jaccard Index (IoU) for the *changed* class (channel index 1).
    Works on one-hot y_true and softmax y_pred.
    """
    y_true_b = tf.cast(tf.argmax(y_true, axis=-1) == 1, tf.float32)
    y_pred_b = tf.cast(tf.argmax(y_pred, axis=-1) == 1, tf.float32)
    y_true_f = tf.reshape(y_true_b, [-1])
    y_pred_f = tf.reshape(y_pred_b, [-1])
    inter    = tf.reduce_sum(y_true_f * y_pred_f)
    union    = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)


def f1_score(y_true, y_pred, smooth=SMOOTH):
    """
    F1 score for the *changed* class (channel index 1).
    Works on one-hot y_true and softmax y_pred.
    """
    y_true_b = tf.cast(tf.argmax(y_true, axis=-1) == 1, tf.float32)
    y_pred_b = tf.cast(tf.argmax(y_pred, axis=-1) == 1, tf.float32)
    y_true_f = tf.reshape(y_true_b, [-1])
    y_pred_f = tf.reshape(y_pred_b, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    pr = (tp + smooth) / (tp + fp + smooth)
    re = (tp + smooth) / (tp + fn + smooth)
    return 2.0 * pr * re / (pr + re + smooth)


# 2. NumPy (test-time) metrics 
# These operate on binarised (H, W) arrays after argmax.

def compute_metrics(y_true_bin, y_pred_bin):
    """
    Compute all paper metrics for ONE patch.

    Parameters
    ----------
    y_true_bin : np.ndarray, shape (H, W), int  — ground truth  (changed=1)
    y_pred_bin : np.ndarray, shape (H, W), int  — prediction    (changed=1)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, kappa, jaccard
    """
    y_true_f = y_true_bin.ravel().astype(np.float32)
    y_pred_f = y_pred_bin.ravel().astype(np.float32)

    TP = np.sum(y_true_f * y_pred_f)
    TN = np.sum((1 - y_true_f) * (1 - y_pred_f))
    FP = np.sum((1 - y_true_f) * y_pred_f)
    FN = np.sum(y_true_f * (1 - y_pred_f))
    N  = TP + TN + FP + FN

    accuracy  = (TP + TN) / (N  + SMOOTH)
    precision = TP / (TP + FP + SMOOTH)
    recall    = TP / (TP + FN + SMOOTH)
    f1        = 2 * precision * recall / (precision + recall + SMOOTH)
    jaccard   = TP / (TP + FP + FN + SMOOTH)

    # Cohen's Kappa
    po    = (TP + TN) / (N + SMOOTH)
    pe    = (
        (TP + FP) * (TP + FN) + (TN + FN) * (TN + FP)
    ) / (N ** 2 + SMOOTH)
    kappa = (po - pe) / (1.0 - pe + SMOOTH)

    return dict(
        accuracy  = float(accuracy),
        precision = float(precision),
        recall    = float(recall),
        f1        = float(f1),
        kappa     = float(kappa),
        jaccard   = float(jaccard),
    )


def average_metrics(metric_list):
    """
    Average a list of per-patch metric dicts into a single dict.

    Parameters
    ----------
    metric_list : list of dicts returned by compute_metrics()

    Returns
    -------
    dict with the same keys, values averaged across all patches
    """
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}