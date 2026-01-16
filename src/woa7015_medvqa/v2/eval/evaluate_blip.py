import torch
from rich.console import Console
from torch.utils.data import DataLoader

from .metrics import compute_text_metrics, split_indices_by_answer_type

console = Console()


@torch.no_grad()
def evaluate_blip(
    model: torch.nn.Module,
    processor,
    loader: DataLoader,
    device: str,
    max_new_tokens: int = 10,
    bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, dict[str, float]]:
    """
    Returns:
      {
        "overall": {exact_match, token_f1},
        "open": {exact_match, token_f1, bleu, rougeL, bertscore_f1},
        "closed": {exact_match, token_f1},
      }
    """
    model.eval()
    model.to(device)

    preds: list[str] = []
    golds: list[str] = []
    types: list[str] = []

    for batch in loader:
        golds.extend(batch["gold_answers"])
        types.extend(batch["answer_types"])

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

    # Overall (fast)
    overall = compute_text_metrics(preds, golds)

    # OPEN subset (full)
    open_idx = split_indices_by_answer_type(types, "OPEN")
    open_preds = [preds[i] for i in open_idx]
    open_golds = [golds[i] for i in open_idx]

    open_metrics = compute_text_metrics(
        open_preds,
        open_golds,
        bleu=True,
        rougeL=True,
        bertscore=True,
        bertscore_model_type=bertscore_model_type,
    )

    # CLOSED subset (fast)
    closed_idx = split_indices_by_answer_type(types, "CLOSED")
    closed_preds = [preds[i] for i in closed_idx]
    closed_golds = [golds[i] for i in closed_idx]
    closed_metrics = compute_text_metrics(closed_preds, closed_golds)

    results = {
        "overall": overall,
        "open": open_metrics,
        "closed": closed_metrics,
    }

    console.print("[bold green]BLIP Test Evaluation[/bold green]")
    console.print(results)

    return results
