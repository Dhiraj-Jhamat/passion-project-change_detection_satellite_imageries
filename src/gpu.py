"""GPU setup helpers."""

from __future__ import annotations

import tensorflow as tf


def configure_gpu(memory_growth: bool = True) -> list:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    return gpus


def log_device_info() -> None:
    gpus = configure_gpu()
    print(f"TensorFlow {tf.__version__}")
    print(f"GPUs: {len(gpus)}")
    for gpu in gpus:
        print(f"  {gpu}")
