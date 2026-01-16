import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipProcessor

# Assuming these are available in your .metrics module
from .metrics import compute_text_metrics, split_indices_by_answer_type


@torch.no_grad()
def evaluate_blip(
    model: torch.nn.Module,
    processor: BlipProcessor,
    loader: DataLoader,
    device: str,
    max_new_tokens: int = 20,
    bertscore_model_type: str = "distilbert-base-uncased",
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

    pbar = tqdm(loader, desc="Evaluating BLIP")

    for batch in pbar:
        # 1. Collect Gold Labels (and Types)
        golds.extend(batch["gold_answers"])
        types.extend(batch["answer_types"])

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        # 2. Generate
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
        )

        # 3. Decode & Normalize
        pred_text_batch = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Critical: Basic Normalization
        # This prevents "Yes." != "yes" errors
        pred_text_batch = [p.strip().lower() for p in pred_text_batch]

        preds.extend(pred_text_batch)

    # --- Metrics Computation ---

    # Helper to avoid crashes on empty lists
    def safe_compute(p_list, g_list, **kwargs):
        if not p_list:
            return {"exact_match": 0.0, "token_f1": 0.0}  # Return safe defaults
        return compute_text_metrics(p_list, g_list, **kwargs)

    print("Computing Overall Metrics...")
    overall = safe_compute(preds, golds)

    # OPEN subset (full metrics)
    print("Computing OPEN Metrics...")
    open_idx = split_indices_by_answer_type(types, "OPEN")
    open_preds = [preds[i] for i in open_idx]
    open_golds = [golds[i] for i in open_idx]

    # Only run expensive BERTScore if we actually have open questions
    if open_preds:
        print(f"  - Found {len(open_preds)} OPEN questions. Running BERTScore...")
        open_metrics = compute_text_metrics(
            open_preds,
            open_golds,
            bleu=True,
            rougeL=True,
            bertscore=True,
            bertscore_model_type=bertscore_model_type,
        )
    else:
        print("  - No OPEN questions found. Skipping detailed metrics.")
        open_metrics = {"exact_match": 0.0, "token_f1": 0.0}

    # CLOSED subset (fast metrics)
    print("Computing CLOSED Metrics...")
    closed_idx = split_indices_by_answer_type(types, "CLOSED")
    closed_preds = [preds[i] for i in closed_idx]
    closed_golds = [golds[i] for i in closed_idx]
    closed_metrics = safe_compute(closed_preds, closed_golds)

    results = {
        "overall": overall,
        "open": open_metrics,
        "closed": closed_metrics,
    }

    print("\nBLIP Test Evaluation Results:")
    for category, metrics in results.items():
        print(f"[{category.upper()}]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return results
