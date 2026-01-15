from collections import Counter

import evaluate
import numpy as np

from ..data.tokenizers import normalize_text

_BLEU = evaluate.load("bleu")
_ROUGE = evaluate.load("rouge")


# ============================================================
# Open-ended Metrics
# ============================================================
def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    """
    Token-level F1 similar to SQuAD:
    - tokenize by whitespace after normalization
    - compute overlap
    """
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


def compute_open_metrics(
    preds: list[str],
    golds: list[str],
    compute_bleu: bool = False,
    compute_rougeL: bool = False,
) -> dict[str, float]:
    """
    Computes open-ended metrics on raw text outputs.
    """
    assert len(preds) == len(golds)

    ems = [exact_match(p, g) for p, g in zip(preds, golds)]
    f1s = [token_f1(p, g) for p, g in zip(preds, golds)]

    out = {
        "open_em": float(np.mean(ems)),
        "open_token_f1": float(np.mean(f1s)),
    }

    # Optional BLEU / ROUGE-L
    if compute_bleu or compute_rougeL:
        raise ImportError(
            "Optional metrics require the `evaluate` library:\n"
            "pip install evaluate\n"
            "(Also ensure network access for first-time metric download.)"
        )

    if compute_bleu:
        # evaluate expects list of predictions + list of list references
        bleu = _BLEU.compute(predictions=preds, references=[[g] for g in golds])
        out["open_bleu"] = float(bleu["bleu"])

    if compute_rougeL:
        rouge = _ROUGE.compute(
            predictions=preds, references=golds, rouge_types=["rougeL"]
        )
        out["open_rougeL"] = float(rouge["rougeL"])

    return out


# ============================================================
# Closed-ended Metrics (classification)
# ============================================================
def _macro_f1_from_labels(
    pred_ids: list[int], gold_ids: list[int], num_classes: int | None = None
) -> float:
    """
    Computes macro-F1 for multiclass classification without sklearn dependency.
    """
    pred_ids = np.asarray(pred_ids)
    gold_ids = np.asarray(gold_ids)

    if num_classes is None:
        num_classes = int(max(pred_ids.max(), gold_ids.max()) + 1)

    f1s = []
    for c in range(num_classes):
        tp = np.sum((pred_ids == c) & (gold_ids == c))
        fp = np.sum((pred_ids == c) & (gold_ids != c))
        fn = np.sum((pred_ids != c) & (gold_ids == c))

        if tp == 0 and fp == 0 and fn == 0:
            # class not present at all -> skip
            continue

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)

    if len(f1s) == 0:
        return 0.0
    return float(np.mean(f1s))


def compute_closed_metrics(
    pred_ids: list[int], gold_ids: list[int], num_classes: int | None = None
) -> dict[str, float]:
    """
    Closed-set metrics for answer classification baselines.
    """
    assert len(pred_ids) == len(gold_ids)

    pred_ids = np.asarray(pred_ids)
    gold_ids = np.asarray(gold_ids)

    acc = float(np.mean(pred_ids == gold_ids))
    macro_f1 = _macro_f1_from_labels(
        pred_ids.tolist(), gold_ids.tolist(), num_classes=num_classes
    )

    return {
        "closed_acc": acc,
        "closed_macro_f1": macro_f1,
    }


# ============================================================
# Split Metrics by Answer Type (OPEN / CLOSED)
# ============================================================
def split_by_answer_type(
    preds: list,
    golds: list,
    answer_types: list[str],
    target_type: str,
) -> tuple[list, list]:
    """
    Returns subset of preds/golds for a given answer_type.
    """
    idx = [
        i for i, t in enumerate(answer_types) if str(t).upper() == target_type.upper()
    ]
    return [preds[i] for i in idx], [golds[i] for i in idx]


def compute_metrics_open_closed(
    open_preds: list[str],
    open_golds: list[str],
    closed_pred_ids: list[int],
    closed_gold_ids: list[int],
    num_closed_classes: int | None = None,
    compute_bleu: bool = False,
    compute_rougeL: bool = False,
) -> dict[str, float]:
    """
    Computes open metrics on open subset, and closed metrics on closed subset.
    """
    out = {}

    # Open subset (generative / text-based)
    if len(open_preds) > 0:
        out.update(
            compute_open_metrics(open_preds, open_golds, compute_bleu, compute_rougeL)
        )
    else:
        out.update({"open_em": 0.0, "open_token_f1": 0.0})

    # Closed subset (classification-based)
    if len(closed_pred_ids) > 0:
        out.update(
            compute_closed_metrics(closed_pred_ids, closed_gold_ids, num_closed_classes)
        )
    else:
        out.update({"closed_acc": 0.0, "closed_macro_f1": 0.0})

    return out
