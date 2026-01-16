import torch
from transformers import BlipProcessor

from typing import Any


def collate_fn_classify(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    questions = torch.stack([b["question"] for b in batch], dim=0)
    labels = torch.tensor([b["answer"] for b in batch], dtype=torch.long)
    answer_types = [b["answer_type"] for b in batch]

    return {
        "images": images,
        "questions": questions,
        "labels": labels,
        "answer_types": answer_types,
    }


def collate_fn_blip(batch: list[dict[str, Any]], processor: BlipProcessor):
    images = [b["image"] for b in batch]
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]
    answer_types = [b["answer_type"] for b in batch]

    inputs = processor(
        images=images,
        text=questions,
        padding=True,
        return_tensors="pt",
    )

    target_tokens = processor.tokenizer(
        answers,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    target_tokens[target_tokens == processor.tokenizer.pad_token_id] = -100

    inputs["labels"] = target_tokens["input_ids"]
    inputs["answer_types"] = answer_types
    inputs["gold_answers"] = answers

    return inputs
