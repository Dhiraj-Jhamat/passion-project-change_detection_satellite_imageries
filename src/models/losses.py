"""
losses.py — UCDNet Loss Function

Loss = WCCE + k_weight * L_modified_kappa   (Eq. 13–17)

Components:
  - WCCE              : Weighted Categorical Cross-Entropy  (Eq. 14)
  - L_modified_kappa  : Log-Cosh smoothed kappa loss       (Eq. 15–16)
  - k_weight          : Adaptive kappa weight               (Eq. 17)
  - k_warmup          : Trainable scalar that ramps k_weight
                        from 0→1 over the first N warmup epochs
                        (set via KappaWarmup callback in train_model.py)
"""

import tensorflow as tf

# Module-level constants 
K_WEIGHT_MAX = 2.0   # clip ceiling for k_weight (prevents instability)

# Warmup scalar: starts at 0, ramped to 1 by KappaWarmup callback.
# Kept here so both the loss closure and the callback reference the same Variable.
k_warmup = tf.Variable(0.0, trainable=False, dtype=tf.float32)


def ucdnet_loss(class_weights=(0.1, 0.9), alpha=0.3, beta=0.7):
    """
    Factory that returns the UCDNet combined loss function.

    Parameters
    ----------
    class_weights : tuple (w_unchanged, w_changed)
        Paper Eq. 14 — WCCE weights.  Default (0.1, 0.9).
    alpha : float
        Weight on TP in observed agreement (po).  Default 0.3.
    beta : float
        Weight on TN in observed agreement (po).  Default 0.7.

    Returns
    -------
    loss : callable  (y_true, y_pred) → scalar tensor
    """
    w = tf.constant(list(class_weights), dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred_c = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # ── Eq. 14: WCCE ──────────────────────────────────────────────────
        # L_wcce = -1/K * sum_{i,j,k} w_k * Y_ijk * log(p_ijk)
        K    = tf.cast(tf.shape(y_true)[-1], tf.float32)
        wcce = -tf.reduce_mean(
            tf.reduce_sum(y_true * tf.math.log(y_pred_c) * w, axis=-1)
        ) / K

        # ── Eq. 16: Soft kappa ────────────────────────────────────────────
        # changed class = channel index 1 (positive class)
        p = tf.reshape(y_pred_c[..., 1], [-1])   # predicted prob of change
        t = tf.reshape(y_true[..., 1],   [-1])   # ground-truth change mask

        # Raw (unweighted) counts — used for pe (expected agreement)
        TP_raw = tf.reduce_sum(t * p)
        TN_raw = tf.reduce_sum((1.0 - t) * (1.0 - p))
        FP_raw = tf.reduce_sum((1.0 - t) * p)
        FN_raw = tf.reduce_sum(t * (1.0 - p))
        N_raw  = TP_raw + TN_raw + FP_raw + FN_raw + 1e-7

        # Weighted counts — alpha on TP, beta on TN → po (observed agreement)
        TP_w = tf.reduce_sum(alpha * t * p)
        TN_w = tf.reduce_sum(beta  * (1.0 - t) * (1.0 - p))
        po   = (TP_w + TN_w) / N_raw

        # Expected agreement (unweighted)
        pe = (
            (TP_raw + FP_raw) * (TP_raw + FN_raw) +
            (TN_raw + FN_raw) * (TN_raw + FP_raw)
        ) / (N_raw * N_raw + 1e-7)

        # Cohen's kappa — clamped before it enters k_weight
        ka = (po - pe) / (1.0 - pe + 1e-7)
        ka = tf.clip_by_value(ka, -0.5, 1.0)

        # Log-Cosh smoothed kappa loss (Eq. 16)
        L_kappa          = 1.0 - ka
        L_modified_kappa = tf.math.log(tf.math.cosh(L_kappa + 1e-7))

        # Eq. 17: Adaptive kappa weight, ramped by warmup scalar
        k_weight = 1.0 + L_kappa / (ka + 0.1)          # +0.1 avoids near-zero ka
        k_weight = tf.clip_by_value(
            k_weight * k_warmup, 0.0, K_WEIGHT_MAX
        )

        return wcce + k_weight * L_modified_kappa

    loss.__name__ = "ucdnet_loss"
    return loss