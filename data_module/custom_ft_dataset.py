from typing import List, Dict

import numpy as np

from data_module.chatlogs import Conversations
from data_module.parquet import ParquetTextDataset


class CustomFTDataset:
    def __init__(self, conv_path: str, parquet_path: str, tokenizer_path: str, chat_format: str, num_epochs: int|str=1, mix_ratio: int|str = 1) -> None:
        if isinstance(num_epochs, str): num_epochs = int(num_epochs)
        if isinstance(mix_ratio, str): mix_ratio = int(mix_ratio)
        self.conversations = Conversations(conv_path, tokenizer_path, chat_format)
        self.parquet_ds = ParquetTextDataset(parquet_path, tokenizer_path)
        self.epochs = num_epochs
        self.mix_ratio = mix_ratio
        self.parquet_idxs = np.random.choice(len(self.parquet_ds), len(self.conversations)*mix_ratio*num_epochs, replace=False)

    def __len__(self) -> int: return len(self.conversations) * (self.mix_ratio + 1) * self.epochs

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        if idx >= len(self): raise IndexError
        total_ratio = self.mix_ratio + 1
        if idx % total_ratio == 0 or self.mix_ratio == 0:
            return self.conversations[idx // total_ratio % len(self.conversations)]
        return self.parquet_ds[int(self.parquet_idxs[idx//total_ratio])]