import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # Replaced rich with tqdm
from transformers import BlipProcessor

from ..eval.metrics import compute_text_metrics


@dataclass
class BLIPTrainConfig:
    epochs: int = 2
    lr: float = 1e-4
    device: str = "cuda"
    ckpt_dir: str = "./checkpoints/blip_lora"
    best_metric: str = "val_token_f1"  # ckpt selection metric
    maximize_metric: bool = True
    max_new_tokens: int = 20


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

    # Using tqdm for generation progress is optional but helpful
    for batch in tqdm(loader, desc="Evaluating"):
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
    Validation metrics:
      - val_loss
      - val_exact_match
      - val_token_f1
    """
    model.eval()

    # --- Part 1: Calculate Validation Loss ---
    running_loss = 0.0
    total = 0

    # We use a separate loop for loss because it requires a forward pass
    # with 'labels', whereas generation requires 'generate()' without labels.
    for batch in tqdm(loader, desc="Validating Loss", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Standard forward pass with labels -> returns loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )

        bs = input_ids.size(0)
        running_loss += outputs.loss.item() * bs
        total += bs

    val_loss = running_loss / max(total, 1)

    # --- Part 2: Calculate Generation Metrics (Existing logic) ---
    preds, golds, _ = _generate_answers(
        model, processor, loader, device, max_new_tokens=max_new_tokens
    )
    m = compute_text_metrics(preds, golds, bleu=False, rougeL=False, bertscore=False)

    return {
        "val_loss": val_loss,
        "val_exact_match": m["exact_match"],
        "val_token_f1": m["token_f1"],
    }


def train_blip(
    model: torch.nn.Module,
    processor: BlipProcessor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: BLIPTrainConfig,
) -> dict[str, list[float]]:
    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Added "val_loss" to history
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_exact_match": [],
        "val_token_f1": [],
    }

    best_score: float | None = None
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    print(f"Training BLIP+LoRA ({cfg.epochs} epochs)")

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            pixel_values = batch["pixel_values"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = input_ids.size(0)
            running_loss += loss.item() * bs
            total += bs

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / max(total, 1)

        # Run validation (now calculates loss + metrics)
        val_metrics = evaluate_blip_val_epoch(
            model=model,
            processor=processor,
            loader=val_loader,
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens,
        )

        # Append to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_exact_match"].append(val_metrics["val_exact_match"])
        history["val_token_f1"].append(val_metrics["val_token_f1"])

        # Updated print statement to include Val Loss
        print(
            f"Epoch {epoch + 1} | "
            f"TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_metrics['val_loss']:.4f} | "  # <--- Added
            f"ValEM={val_metrics['val_exact_match']:.4f} "
            f"ValTokenF1={val_metrics['val_token_f1']:.4f}"
        )

        _save_checkpoint(
            os.path.join(cfg.ckpt_dir, "last.pt"), model, optimizer, epoch, history
        )

        # Checkpointing logic remains the same (based on token_f1)
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
            print(f"-> Best updated: {cfg.best_metric}={best_score:.4f}")

    print("Done BLIP+LoRA training!")
    return history
