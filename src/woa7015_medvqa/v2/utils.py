import math
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def display_examples(samples: list[dict[str, Any]], n=8, cols=4, figsize=(16, 8)):
    """
    samples: list of dataset samples (dicts)
    n: number of samples to show
    cols: number of columns in grid
    """
    n = min(n, len(samples))
    rows = math.ceil(n / cols)

    plt.figure(figsize=figsize)

    for i in range(n):
        sample = samples[i]
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]
        answer_type = sample["answer_type"]

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        ax.set_title(f"Q: {question}\nA: {answer}\nT: {answer_type}", fontsize=9)

    plt.tight_layout()
    plt.show()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def plot_history(history: dict[str, list[float]], title: str = "Training History"):
    """
    Plots training history with 2 subplots:
      (1) Loss curves
      (2) Metric curves (everything else)
    """
    if history is None or len(history) == 0:
        print("No history to plot.")
        return

    loss_keys = [k for k in history.keys() if "loss" in k.lower()]
    metric_keys = [k for k in history.keys() if k not in loss_keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14)

    # ---- Loss subplot ----
    ax0 = axes[0]
    if len(loss_keys) == 0:
        ax0.text(0.5, 0.5, "No loss keys found", ha="center", va="center")
    else:
        for k in loss_keys:
            ax0.plot(history[k], label=k)
        ax0.set_title("Loss")
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Loss")
        ax0.grid(True)
        ax0.legend()

    # ---- Metric subplot ----
    ax1 = axes[1]
    if len(metric_keys) == 0:
        ax1.text(0.5, 0.5, "No metric keys found", ha="center", va="center")
    else:
        for k in metric_keys:
            ax1.plot(history[k], label=k)
        ax1.set_title("Metrics")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Score")
        ax1.grid(True)
        ax1.legend()

    plt.tight_layout()
    plt.show()
