"""Training curves and result plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(history, path: Path) -> None:
    keys = [
        ("loss", "Loss"),
        ("changed_class_f1", "F1 (changed)"),
        ("changed_class_jaccard", "Jaccard (changed)"),
    ]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    for ax, (key, title) in zip(axes, keys):
        if key in history.history:
            ax.plot(history.history[key], label="train")
            vk = f"val_{key}"
            if vk in history.history:
                ax.plot(history.history[vk], label="val")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
