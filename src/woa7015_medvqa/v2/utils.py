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
