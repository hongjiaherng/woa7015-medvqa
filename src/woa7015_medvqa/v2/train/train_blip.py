import os
from dataclasses import dataclass
from typing import Any, Dict, List

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

from ..eval.metrics import compute_open_metrics

console = Console()


@dataclass
class BlipTrainConfig:
    epochs: int = 2
    lr: float = 1e-4
    device: str = "cuda"
    ckpt_dir: str = "./checkpoints/blip_lora"
    maximize_metric: bool = True
    best_metric: str = "val_open_token_f1"  # or val_open_em
    max_new_tokens: int = 10


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
def generate_blip(
    model: torch.nn.Module,
    processor,
    loader: DataLoader,
    device: str,
    max_new_tokens: int = 10,
) -> Dict[str, List[str]]:
    model.eval()
    preds: List[str] = []
    golds: List[str] = []

    for batch in loader:
        golds.extend(batch["gold_answers"])

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

    return {"preds": preds, "golds": golds}


@torch.no_grad()
def evaluate_blip_epoch(
    model: torch.nn.Module,
    processor,
    loader: DataLoader,
    device: str,
    max_new_tokens: int = 10,
) -> Dict[str, float]:
    out = generate_blip(model, processor, loader, device, max_new_tokens=max_new_tokens)
    return compute_open_metrics(
        out["preds"], out["golds"], compute_bleu=False, compute_rougeL=False
    )


def train_blip(
    model: torch.nn.Module,
    processor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: BlipTrainConfig,
) -> Dict[str, Any]:
    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    history = {
        "train_loss": [],
        "val_open_em": [],
        "val_open_token_f1": [],
    }

    best_score = None
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

        val_metrics = evaluate_blip_epoch(
            model=model,
            processor=processor,
            loader=val_loader,
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens,
        )

        history["train_loss"].append(train_loss)
        history["val_open_em"].append(val_metrics["open_em"])
        history["val_open_token_f1"].append(val_metrics["open_token_f1"])

        console.print(
            f"[cyan]Epoch {epoch + 1}[/cyan] | "
            f"TrainLoss={train_loss:.4f} | "
            f"ValEM={val_metrics['open_em']:.4f} "
            f"ValTokenF1={val_metrics['open_token_f1']:.4f}"
        )

        # Save last
        _save_checkpoint(
            os.path.join(cfg.ckpt_dir, "last.pt"), model, optimizer, epoch, history
        )

        # Save best
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

    console.print("[bold green]Done BLIP+LoRA training![/bold green]")
    return history
