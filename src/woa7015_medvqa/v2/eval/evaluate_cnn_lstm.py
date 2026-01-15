from typing import Any, Dict, List

import torch
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader

from ..eval.metrics import compute_closed_metrics

console = Console()


@torch.no_grad()
def evaluate_baseline(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0

    pred_ids: List[int] = []
    gold_ids: List[int] = []

    for batch in loader:
        images = batch["images"].to(device)
        questions = batch["questions"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images, questions)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        bs = labels.size(0)
        total += bs
        total_loss += loss.item() * bs

        pred_ids.extend(preds.detach().cpu().tolist())
        gold_ids.extend(labels.detach().cpu().tolist())

    metrics = compute_closed_metrics(pred_ids, gold_ids, num_classes=num_classes)
    metrics["loss"] = total_loss / max(total, 1)

    console.print("[bold green]Baseline Test Evaluation[/bold green]")
    console.print(metrics)

    return metrics
