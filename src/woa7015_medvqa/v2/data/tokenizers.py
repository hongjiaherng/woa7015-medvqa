import json
from collections import Counter
import re
import torch

UNK_ANS = "<unk>"


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def build_question_vocab(
    train_json_path: str, max_words: int = 500, english_only: bool = True
):
    with open(train_json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if english_only:
        samples = [x for x in samples if x.get("q_lang", "").lower() == "en"]

    counter = Counter()
    for x in samples:
        counter.update(normalize_text(x["question"]).split())

    vocab = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(max_words)]
    w2id = {w: i for i, w in enumerate(vocab)}
    return vocab, w2id


def make_question_encoder(w2id, max_len: int = 30):
    pad_id = w2id["<pad>"]
    unk_id = w2id["<unk>"]

    def encode(q: str):
        toks = normalize_text(q).split()[:max_len]
        ids = [w2id.get(t, unk_id) for t in toks]
        if len(ids) < max_len:
            ids += [pad_id] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    return encode


def build_answer_vocab(
    train_json_path: str, topk: int = 200, english_only: bool = True
):
    with open(train_json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if english_only:
        samples = [x for x in samples if x.get("q_lang", "").lower() == "en"]

    answers = [normalize_text(x["answer"]) for x in samples]
    counter = Counter(answers)

    vocab = [a for a, _ in counter.most_common(topk)] + [UNK_ANS]
    a2id = {a: i for i, a in enumerate(vocab)}
    id2a = {i: a for a, i in a2id.items()}
    return vocab, a2id, id2a


def make_answer_encoder(a2id):
    unk_id = a2id[UNK_ANS]

    def encode(ans: str):
        a = normalize_text(ans)
        return a2id.get(a, unk_id)

    return encode
