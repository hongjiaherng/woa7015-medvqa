import json
from pathlib import Path
from typing import Any, Callable, Literal

from PIL import Image
from torch.utils.data import Dataset


class SLAKEDataset(Dataset):
    """
    SLAKE Dataset Loader (local json-based).

    Expects:
    root_dir/
      ├── imgs/
      │    ├── xmlab0/source.jpg
      │    ├── xmlab1/source.jpg
      │    └── ...
      ├── train.json
      ├── validation.json
      └── test.json

    Each record example:
    {
      "img_id": 102,
      "img_name": "xmlab102/source.jpg",
      "question": "...",
      "answer": "...",
      "q_lang": "en",
      "answer_type": "OPEN" / "CLOSED",
      ...
    }
    """

    def __init__(
        self,
        root_dir: Path,
        split: Literal["train", "validation", "test"] = "train",
        english_only: bool = True,
        image_transform: Callable | None = None,
        question_transform: Callable | None = None,
        answer_transform: Callable | None = None,
    ):
        super().__init__()

        assert split in ["train", "validation", "test"], (
            "split must be one of ['train', 'validation', 'test']"
        )

        self.root_dir = root_dir
        self.split = split
        self.english_only = english_only

        self.image_transform = image_transform
        self.question_transform = question_transform
        self.answer_transform = answer_transform

        ann_path = root_dir / f"{split}.json"
        assert ann_path.is_file(), f"Annotation file not found: {ann_path}"

        with open(ann_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        assert isinstance(samples, list), (
            f"Expected a list in {ann_path}, got {type(samples)}"
        )

        # Filter English only
        if self.english_only:
            samples = [x for x in samples if x.get("q_lang", "").lower() == "en"]

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_image_path(self, sample: dict[str, Any]) -> Path:
        """
        SLAKE provides img_name like: 'xmlab102/source.jpg'
        which is relative to imgs/ folder.
        """
        rel = sample["img_name"]  # required
        path = self.root_dir / "imgs" / rel

        assert path.is_file(), f"Image file not found: {path}"

        return path

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        # ---- Load image ----
        img_path = self._get_image_path(sample)
        image = Image.open(img_path).convert("RGB")

        # ---- Raw text ----
        question = sample["question"]
        answer = sample["answer"]
        answer_type = sample.get("answer_type")  # OPEN / CLOSED

        # ---- Optional transforms ----
        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.question_transform is not None:
            question = self.question_transform(question)

        if self.answer_transform is not None:
            answer = self.answer_transform(answer)

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
        }
