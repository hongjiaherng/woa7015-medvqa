from typing import Any, Dict, List

import torch
from rich.console import Console
from torch.utils.data import DataLoader

from ..eval.metrics import compute_open_metrics

console = Console()


@torch.no_grad()
def evaluate_blip(
    model: torch.nn.Module,
    processor,
    loader: DataLoader,
    device: str,
    max_new_tokens: int = 10,
    compute_bleu: bool = False,
    compute_rougeL: bool = False,
) -> Dict[str, Any]:
    model.eval()
    model.to(device)

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

    metrics = compute_open_metrics(
        preds,
        golds,
        compute_bleu=compute_bleu,
        compute_rougeL=compute_rougeL,
    )

    console.print("[bold green]BLIP Test Evaluation[/bold green]")
    console.print(metrics)

    return metrics
