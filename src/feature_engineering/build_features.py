"""Augmentation, oversampling, and TensorFlow dataset construction."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def oversample_changed_patches(t1, t2, y, oversample_ratio: int = 3):
    if oversample_ratio <= 1:
        return t1, t2, y

    has_change = np.any(y[..., 1] > 0, axis=(1, 2))
    idx_change = np.where(has_change)[0]
    idx_extra = np.tile(idx_change, oversample_ratio - 1)
    all_idx = np.concatenate([np.arange(len(t1)), idx_extra])
    np.random.shuffle(all_idx)

    n_change = len(idx_change) * oversample_ratio
    print(
        f"  Oversampling: {len(idx_change)} changed patches × {oversample_ratio} "
        f"→ {n_change}/{len(all_idx)} ({100 * n_change / len(all_idx):.1f}% changed)"
    )
    return t1[all_idx], t2[all_idx], y[all_idx]


def _aug_fn(t1, t2, lbl):
    stacked = tf.concat([t1, t2, lbl], axis=-1)
    stacked = tf.image.random_flip_left_right(stacked)
    stacked = tf.image.random_flip_up_down(stacked)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    stacked = tf.image.rot90(stacked, k=k)

    t1_out = stacked[:, :, :13]
    t2_out = stacked[:, :, 13:26]
    lbl_out = stacked[:, :, 26:]

    t1_out = tf.clip_by_value(t1_out + tf.random.uniform([], -0.07, 0.07), 0.0, 1.0)
    t2_out = tf.clip_by_value(t2_out + tf.random.uniform([], -0.07, 0.07), 0.0, 1.0)
    return t1_out, t2_out, lbl_out


def make_tf_dataset(t1, t2, y, batch_size: int = 1, augment: bool = False, shuffle: bool = True):
    def _map(t1_b, t2_b, lbl):
        if augment:
            t1_b, t2_b, lbl = _aug_fn(t1_b, t2_b, lbl)
        return {"T1": t1_b, "T2": t2_b}, lbl

    ds = tf.data.Dataset.from_tensor_slices((t1, t2, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(t1), 1), reshuffle_each_iteration=True)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
