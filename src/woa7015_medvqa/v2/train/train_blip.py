import os
from dataclasses import dataclass

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader

from ..eval.metrics import compute_text_metrics

console = Console()


@dataclass
class BLIPTrainConfig:
    epochs: int = 2
    lr: float = 1e-4
    device: str = "cuda"
    ckpt_dir: str = "./checkpoints/blip_lora"
    best_metric: str = "val_token_f1"  # ckpt selection metric
    maximize_metric: bool = True
    max_new_tokens: int = 10


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
def _generate_answers(
    model: torch.nn.Module,
    processor,
    loader: DataLoader,
    device: str,
    max_new_tokens: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns:
      preds: list[str]
      golds: list[str]
      answer_types: list[str]
    """
    model.eval()

    preds: list[str] = []
    golds: list[str] = []
    answer_types: list[str] = []

    for batch in loader:
        golds.extend(batch["gold_answers"])
        answer_types.extend(batch["answer_types"])

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
        )

        pred_text = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        preds.extend(pred_text)

    return preds, golds, answer_types


@torch.no_grad()
def evaluate_blip_val_epoch(
    model: torch.nn.Module,
    processor,
    loader: DataLoader,
    device: str,
    max_new_tokens: int,
) -> dict[str, float]:
    """
    Fast per-epoch validation metrics:
      - overall exact match
      - overall token_f1   (ckpt selection metric)
    """
    preds, golds, _ = _generate_answers(
        model, processor, loader, device, max_new_tokens=max_new_tokens
    )
    m = compute_text_metrics(preds, golds, bleu=False, rougeL=False, bertscore=False)
    return {
        "val_exact_match": m["exact_match"],
        "val_token_f1": m["token_f1"],
    }


def train_blip(
    model: torch.nn.Module,
    processor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: BLIPTrainConfig,
) -> dict[str, list[float]]:
    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_exact_match": [],
        "val_token_f1": [],
    }

    best_score: float | None = None
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    console.print(f"[bold green]Training BLIP+LoRA ({cfg.epochs} epochs)[/bold green]")

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        total = 0

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
                batch = {
                    k: v.to(cfg.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                loss = out.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = batch["input_ids"].size(0)
                running_loss += loss.item() * bs
                total += bs

                progress.update(task, advance=1)

        train_loss = running_loss / max(total, 1)

        val_metrics = evaluate_blip_val_epoch(
            model=model,
            processor=processor,
            loader=val_loader,
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens,
        )

        history["train_loss"].append(train_loss)
        history["val_exact_match"].append(val_metrics["val_exact_match"])
        history["val_token_f1"].append(val_metrics["val_token_f1"])

        console.print(
            f"[cyan]Epoch {epoch + 1}[/cyan] | "
            f"TrainLoss={train_loss:.4f} | "
            f"ValEM={val_metrics['val_exact_match']:.4f} "
            f"ValTokenF1={val_metrics['val_token_f1']:.4f}"
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

    console.print("[bold green]Done BLIP+LoRA training![/bold green]")
    return history
