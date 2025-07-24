import os
import re
import torch
import random

from typing import Any, Dict
from torch.utils.data import Dataset


class NextWordPrediction(Dataset):
    def __init__(self, cfg: Dict[str, Any]):
        data_root = cfg.get("data_root")
        self.max_length = cfg.get("max_length")
        self.num_samples = cfg.get("num_samples")

        with open(os.path.join(data_root, "corpus.txt"), encoding="utf-8") as f:
            raw = f.read()
        text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", "", raw)

        unique_chars = sorted(set(text))

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.vocab = {c: idx + 2 for idx, c in enumerate(unique_chars)}
        self.vocab['<PAD>'] = self.pad_token_id
        self.vocab['<UNK>'] = self.unk_token_id
        self.vocab_size = len(self.vocab)
        cfg["vocab_size"] = self.vocab_size

        self.tokens = [self.vocab.get(c, self.unk_token_id) for c in text]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        start = random.randint(0, len(self.tokens) - self.max_length - 1)
        input_seq = self.tokens[start : start + self.max_length]
        
        label_seq = self.tokens[start + 1 : start + self.max_length + 1]
        mask = torch.tril(torch.ones((self.max_length, self.max_length), dtype=torch.long))
        return {
            "x": torch.tensor(input_seq, dtype=torch.long),
            "label": torch.tensor(label_seq, dtype=torch.long),
            "tag": torch.tensor(label_seq, dtype=torch.long),
            "mask": mask,
        }
