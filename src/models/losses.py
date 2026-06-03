"""UCDNet combined loss: WCCE + modified Kappa (paper Eq. 13–17)."""

from __future__ import annotations

import tensorflow as tf


def wcce_loss(y_true, y_pred, class_weights=(0.1, 0.9)):
    weights = tf.constant(class_weights, dtype=tf.float32)
    y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    log_p = tf.math.log(y_pred_clipped)
    weighted = weights * y_true * log_p
    return -tf.reduce_mean(tf.reduce_sum(weighted, axis=-1))


def kappa_coefficient(y_true_flat, y_pred_flat, alpha=0.5, beta=0.5):
    y_true_f = tf.cast(y_true_flat, tf.float32)
    y_pred_f = tf.cast(y_pred_flat, tf.float32)

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    tn = tf.reduce_sum((1.0 - y_true_f) * (1.0 - y_pred_f))
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))

    a_tp = alpha * tp
    b_tn = beta * tn
    n = a_tp + b_tn + fp + fn + 1e-7

    p0 = (a_tp + b_tn) / n
    pe = ((a_tp + fp) * (a_tp + fn) + (b_tn + fn) * (b_tn + fp)) / (n * n)
    return (p0 - pe) / (1.0 - pe + 1e-7)


def combined_loss(y_true, y_pred, class_weights=(0.1, 0.9), alpha=0.5, beta=0.5):
    """Full UCDNet loss with dynamic kappa weighting."""
    l_wcce = wcce_loss(y_true, y_pred, class_weights=class_weights)

    y_true_ch = tf.cast(y_true[..., 1], tf.float32)
    y_pred_ch = tf.cast(y_pred[..., 1], tf.float32)
    y_true_f = tf.reshape(y_true_ch, [-1])
    y_pred_f = tf.reshape(y_pred_ch, [-1])

    ka = kappa_coefficient(y_true_f, y_pred_f, alpha=alpha, beta=beta)
    l_kappa_prime = 1.0 - ka
    l_mod_kappa = tf.math.log(tf.math.cosh(l_kappa_prime + 1e-7))
    k_weight = 1.0 + (l_kappa_prime / (ka + 1e-7))

    return l_wcce + k_weight * l_mod_kappa


def make_loss(class_weights=(0.1, 0.9)):
    """Factory for Keras compile (serialisable name)."""

    def loss(y_true, y_pred):
        return combined_loss(y_true, y_pred, class_weights=class_weights)

    loss.__name__ = "ucdnet_combined_loss"
    return loss
