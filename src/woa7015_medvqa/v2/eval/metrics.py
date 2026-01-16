from collections import Counter

import evaluate
import numpy as np

from ..data.tokenizers import normalize_text


# ============================================================
# Open-ended metrics (text)
# ============================================================
def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()

    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0

    pred_cnt = Counter(pred_toks)
    gold_cnt = Counter(gold_toks)
    common = sum((pred_cnt & gold_cnt).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def compute_text_metrics(
    preds: list[str],
    golds: list[str],
    bleu: bool = False,
    rougeL: bool = False,
    bertscore: bool = False,
    bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, float]:
    """
    Returns:
      exact_match (normalized)
      token_f1 (SQuAD style)
      + optional BLEU / ROUGE-L / BERTScore
    """
    assert len(preds) == len(golds)

    ems = [exact_match(p, g) for p, g in zip(preds, golds)]
    f1s = [token_f1(p, g) for p, g in zip(preds, golds)]

    out: dict[str, float] = {
        "exact_match": float(np.mean(ems)),
        "token_f1": float(np.mean(f1s)),
    }

    if bleu or rougeL or bertscore:
        raise ImportError("Optional metrics require: pip install evaluate")

    if bleu:
        bleu_metric = evaluate.load("bleu")
        res = bleu_metric.compute(predictions=preds, references=[[g] for g in golds])
        out["bleu"] = float(res["bleu"])

    if rougeL:
        rouge_metric = evaluate.load("rouge")
        res = rouge_metric.compute(
            predictions=preds, references=golds, rouge_types=["rougeL"]
        )
        out["rougeL"] = float(res["rougeL"])

    if bertscore:
        bert_metric = evaluate.load("bertscore")
        res = bert_metric.compute(
            predictions=preds,
            references=golds,
            lang="en",
            model_type=bertscore_model_type,
        )
        out["bertscore_precision"] = float(np.mean(res["precision"]))
        out["bertscore_recall"] = float(np.mean(res["recall"]))
        out["bertscore_f1"] = float(np.mean(res["f1"]))

    return out


# ============================================================
# Closed-ended metrics (classification)
# ============================================================
def accuracy_from_ids(pred_ids: list[int], gold_ids: list[int]) -> float:
    pred = np.asarray(pred_ids)
    gold = np.asarray(gold_ids)
    return float(np.mean(pred == gold))


def topk_accuracy_from_logits(
    logits: np.ndarray, gold_ids: np.ndarray, k: int = 5
) -> float:
    """
    logits: (N, C)
    gold_ids: (N,)
    """
    topk = np.argsort(-logits, axis=1)[:, :k]
    correct = np.any(topk == gold_ids.reshape(-1, 1), axis=1)
    return float(np.mean(correct))


def macro_f1_from_ids(
    pred_ids: list[int], gold_ids: list[int], num_classes: int | None = None
) -> float:
    pred = np.asarray(pred_ids)
    gold = np.asarray(gold_ids)

    if num_classes is None:
        num_classes = int(max(pred.max(), gold.max()) + 1)

    f1s: list[float] = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (gold == c))
        fp = np.sum((pred == c) & (gold != c))
        fn = np.sum((pred != c) & (gold == c))

        if tp == 0 and fp == 0 and fn == 0:
            continue

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(float(f1))

    return float(np.mean(f1s)) if len(f1s) > 0 else 0.0


def compute_classification_metrics(
    pred_ids: list[int],
    gold_ids: list[int],
    logits: np.ndarray | None = None,
    num_classes: int | None = None,
    topk: int = 5,
) -> dict[str, float]:
    """
    Metrics:
      accuracy
      top5_accuracy (if logits provided)
      macro_f1
    """
    out = {
        "accuracy": accuracy_from_ids(pred_ids, gold_ids),
        "macro_f1": macro_f1_from_ids(pred_ids, gold_ids, num_classes=num_classes),
    }

    if logits is not None:
        gold_np = np.asarray(gold_ids)
        out[f"top{topk}_accuracy"] = topk_accuracy_from_logits(logits, gold_np, k=topk)

    return out


# ============================================================
# Split helpers for OPEN/CLOSED
# ============================================================
def filter_by_answer_type(items: list, answer_types: list[str], target: str) -> list:
    target = target.upper()
    return [x for x, t in zip(items, answer_types) if str(t).upper() == target]


def split_indices_by_answer_type(answer_types: list[str], target: str) -> list[int]:
    target = target.upper()
    return [i for i, t in enumerate(answer_types) if str(t).upper() == target]
