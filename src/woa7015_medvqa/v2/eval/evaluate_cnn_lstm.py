import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # Replaced rich with tqdm

from .metrics import compute_classification_metrics, split_indices_by_answer_type


@torch.no_grad()
def evaluate_cnn_lstm(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> dict[str, dict[str, float]]:
    """
    Returns:
      {
        "overall": {...},
        "open": {...},
        "closed": {...}
      }
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    all_logits = []
    all_pred: list[int] = []
    all_gold: list[int] = []
    all_types: list[str] = []

    total_loss = 0.0
    total = 0

    # Using tqdm for progress tracking
    pbar = tqdm(loader, desc="Evaluating CNN-LSTM")

    for batch in pbar:
        images = batch["images"].to(device)
        questions = batch["questions"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images, questions)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        bs = labels.size(0)
        total += bs
        total_loss += loss.item() * bs

        all_logits.append(logits.detach().cpu().numpy())
        all_pred.extend(preds.detach().cpu().tolist())
        all_gold.extend(labels.detach().cpu().tolist())
        all_types.extend(batch["answer_types"])

    logits_np = np.concatenate(all_logits, axis=0)

    # Overall metrics
    overall = compute_classification_metrics(
        pred_ids=all_pred,
        gold_ids=all_gold,
        logits=logits_np,
        num_classes=num_classes,
        topk=5,
    )
    overall["loss"] = total_loss / max(total, 1)

    # Split OPEN/CLOSED
    open_idx = split_indices_by_answer_type(all_types, "OPEN")
    closed_idx = split_indices_by_answer_type(all_types, "CLOSED")

    def subset_metrics(indices: list[int]) -> dict[str, float]:
        if len(indices) == 0:
            return {"accuracy": 0.0, "top5_accuracy": 0.0, "macro_f1": 0.0}

        pred_sub = [all_pred[i] for i in indices]
        gold_sub = [all_gold[i] for i in indices]
        logits_sub = logits_np[indices]

        m = compute_classification_metrics(
            pred_ids=pred_sub,
            gold_ids=gold_sub,
            logits=logits_sub,
            num_classes=num_classes,
            topk=5,
        )

        # rename key to consistent output
        m["top5_accuracy"] = m.pop("top5_accuracy", m.get("top5_accuracy", 0.0))
        return m

    open_m = subset_metrics(open_idx)
    closed_m = subset_metrics(closed_idx)

    results = {
        "overall": overall,
        "open": open_m,
        "closed": closed_m,
    }

    # Replaced console.print with standard print
    print("CNN-LSTM Test Evaluation Results:")
    # Using a simple loop to print clearly since we lost rich's pretty printing
    for category, metrics in results.items():
        print(f"[{category.upper()}]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return results
