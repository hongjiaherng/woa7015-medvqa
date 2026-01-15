import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class SlakeDataset(Dataset):
    def __init__(
        self,
        json_path,
        img_root,
        q_tokenizer,
        a_vocab,
        transform=None,
        filter_lang="en",
    ):
        self.items = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.items = [item for item in self.items if item["q_lang"] == filter_lang]
        self.img_root = Path(img_root)
        self.q_tokenizer = q_tokenizer
        self.a_vocab = a_vocab
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(self.img_root / item["img_name"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        q_tokens = self.q_tokenizer(item["question"])
        a_idx = self.a_vocab.encode(item["answer"])
        return img, q_tokens, a_idx, idx
