import os
from dataclasses import dataclass

import numpy as np
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

from ..eval.metrics import compute_classification_metrics

console = Console()


@dataclass
class CNNLSTMTrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    device: str = "cuda"
    ckpt_dir: str = "./checkpoints/cnn_lstm"
    best_metric: str = "val_accuracy"  # ckpt selection metric
    maximize_metric: bool = True


def _save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: dict[str, list[float]],
) -> None:
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
def _eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0

    all_pred: list[int] = []
    all_gold: list[int] = []
    all_logits: list[np.ndarray] = []

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

        all_pred.extend(preds.detach().cpu().tolist())
        all_gold.extend(labels.detach().cpu().tolist())
        all_logits.append(logits.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0) if len(all_logits) else None

    metrics = compute_classification_metrics(
        pred_ids=all_pred,
        gold_ids=all_gold,
        logits=logits_np,
        num_classes=num_classes,
        topk=5,
    )
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def train_cnn_lstm(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CNNLSTMTrainConfig,
    num_classes: int,
) -> dict[str, list[float]]:
    model = model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_score: float | None = None
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    console.print(f"[bold green]Training CNN–LSTM ({cfg.epochs} epochs)[/bold green]")

    for epoch in range(cfg.epochs):
        model.train()

        running_loss = 0.0
        total = 0

        all_pred: list[int] = []
        all_gold: list[int] = []

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

                all_pred.extend(preds.detach().cpu().tolist())
                all_gold.extend(labels.detach().cpu().tolist())

                progress.update(task, advance=1)

        train_loss = running_loss / max(total, 1)
        train_acc = float(np.mean(np.asarray(all_pred) == np.asarray(all_gold)))

        val_metrics = _eval_epoch(
            model, val_loader, cfg.device, num_classes=num_classes
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        console.print(
            f"[cyan]Epoch {epoch + 1}[/cyan] | "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} | "
            f"ValLoss={val_metrics['loss']:.4f} ValAcc={val_metrics['accuracy']:.4f}"
        )

        _save_checkpoint(
            os.path.join(cfg.ckpt_dir, "last.pt"), model, optimizer, epoch, history
        )

        current_score = history[cfg.best_metric][-1]
        improved = best_score is None or (
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

    console.print("[bold green]Done CNN–LSTM training![/bold green]")
    return history
