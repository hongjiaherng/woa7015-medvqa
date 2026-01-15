import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader

from ..eval.metrics import compute_closed_metrics

console = Console()


@dataclass
class BaselineTrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    device: str = "cuda"
    ckpt_dir: str = "./checkpoints/baseline"
    maximize_metric: bool = True
    best_metric: str = "val_closed_acc"  # or val_closed_macro_f1


def _save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, Any],
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
        },
        path,
    )


@torch.no_grad()
def evaluate_baseline_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> Dict[str, float]:
    """
    Computes classification metrics for baseline:
      - loss
      - closed_acc
      - closed_macro_f1
    """
    model.eval()
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

    closed_metrics = compute_closed_metrics(pred_ids, gold_ids, num_classes=num_classes)
    return {"loss": total_loss / max(total, 1), **closed_metrics}


def train_baseline(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: BaselineTrainConfig,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Returns a history dict:
      train_loss, val_loss
      train_closed_acc, val_closed_acc
      train_closed_macro_f1, val_closed_macro_f1
    """
    model = model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_closed_acc": [],
        "train_closed_macro_f1": [],
        "val_loss": [],
        "val_closed_acc": [],
        "val_closed_macro_f1": [],
    }

    best_score = None
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    console.print(
        f"[bold green]Training CNN–LSTM baseline ({cfg.epochs} epochs)[/bold green]"
    )

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        total = 0

        pred_ids: List[int] = []
        gold_ids: List[int] = []

        progress = Progress(
            TextColumn(f"[bold]Epoch {epoch + 1}/{cfg.epochs}[/bold]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            task = progress.add_task("train", total=len(train_loader))

            for batch in train_loader:
                images = batch["images"].to(cfg.device)
                questions = batch["questions"].to(cfg.device)
                labels = batch["labels"].to(cfg.device)

                logits = model(images, questions)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)

                bs = labels.size(0)
                total += bs
                running_loss += loss.item() * bs

                pred_ids.extend(preds.detach().cpu().tolist())
                gold_ids.extend(labels.detach().cpu().tolist())

                progress.update(task, advance=1)

        train_loss = running_loss / max(total, 1)
        train_metrics = compute_closed_metrics(
            pred_ids, gold_ids, num_classes=num_classes
        )

        val_metrics = evaluate_baseline_epoch(
            model, val_loader, cfg.device, num_classes=num_classes
        )

        history["train_loss"].append(train_loss)
        history["train_closed_acc"].append(train_metrics["closed_acc"])
        history["train_closed_macro_f1"].append(train_metrics["closed_macro_f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_closed_acc"].append(val_metrics["closed_acc"])
        history["val_closed_macro_f1"].append(val_metrics["closed_macro_f1"])

        console.print(
            f"[cyan]Epoch {epoch + 1}[/cyan] | "
            f"TrainLoss={train_loss:.4f} "
            f"TrainAcc={train_metrics['closed_acc']:.4f} "
            f"TrainMacroF1={train_metrics['closed_macro_f1']:.4f} | "
            f"ValLoss={val_metrics['loss']:.4f} "
            f"ValAcc={val_metrics['closed_acc']:.4f} "
            f"ValMacroF1={val_metrics['closed_macro_f1']:.4f}"
        )

        # Save last checkpoint
        _save_checkpoint(
            os.path.join(cfg.ckpt_dir, "last.pt"), model, optimizer, epoch, history
        )

        # Save best checkpoint
        current_score = history[cfg.best_metric][-1]
        improved = False
        if best_score is None:
            improved = True
        else:
            improved = (
                current_score > best_score
                if cfg.maximize_metric
                else current_score < best_score
            )

        if improved:
            best_score = current_score
            _save_checkpoint(
                os.path.join(cfg.ckpt_dir, "best.pt"), model, optimizer, epoch, history
            )
            console.print(
                f"[bold yellow]✅ Best updated: {cfg.best_metric}={best_score:.4f}[/bold yellow]"
            )

    console.print("[bold green]Done baseline training![/bold green]")
    return history
