from functools import partial
from typing import List, Dict, Optional

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

CROSS_ENTROPY_IGNORE_IDX = -100

class ParquetCustomDataReader(Dataset):
    def __init__(self, path: str) -> None:
        self.pq_file = pq.ParquetFile(path)
    def __len__(self) -> int: return self.pq_file.num_row_groups
    def __getitem__(self, idx) -> Dict[str, List[int]]: return self.pq_file.read_row_group(idx).to_pydict()

class DataModule(L.LightningDataModule):
    def __init__(self, train_ds_file: str, valid_ds_file: Optional[str], batch_size: int, max_seq_length: int, max_pad: bool = False,
                 pad_to_multiple_of: int = 1, pad_id: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_pad = max_pad
        self.train_ds = ParquetCustomDataReader(train_ds_file)
        self.valid_ds = ParquetCustomDataReader(valid_ds_file) if valid_ds_file else []
        if max_pad:
            self.collate_fn = partial(self.collate_max_pad_fn, max_seq_length, pad_id)
        elif pad_to_multiple_of > 1:
            self.collate_fn = partial(self.collate_batch_pad_multiple, max_seq_length, pad_id, pad_to_multiple_of)
        else:
            self.collate_fn = partial(self.collate_batch_pad_fn, max_seq_length, pad_id)

    @staticmethod
    def collate_batch_pad_fn(max_len: int, pad_id: int, batch: List[Dict[str, List[int]]]) -> dict:
        MAX_LEN = max_len
        max_len = max([len(x['input_ids']) for x in batch])

        input_ids = torch.stack([torch.tensor([0] * (max_len - len(x['input_ids'])) + x['input_ids']) for x in batch])
        labels = torch.stack([torch.tensor([CROSS_ENTROPY_IGNORE_IDX] * (max_len - len(x['labels'])) + x['labels']) for x in batch])
        attention_mask = (input_ids != torch.tensor(pad_id, dtype=input_ids.dtype)).long()

        if len(batch) == 1 and attention_mask.sum().item() == attention_mask.numel(): attention_mask = None

        out = {"input_ids": input_ids[:, :MAX_LEN], "labels": labels[:, :MAX_LEN]}
        if attention_mask is not None: out["attention_mask"] = attention_mask[:, :MAX_LEN]
        return out

    @staticmethod
    def collate_max_pad_fn(max_len: int, pad_id: int, batch: List[Dict[str, List[int]]]) -> dict:
        MAX_LEN = max_len

        input_ids = torch.stack([torch.tensor([0] * (MAX_LEN - len(x['input_ids'])) + x['input_ids']) for x in batch])
        labels = torch.stack([torch.tensor([CROSS_ENTROPY_IGNORE_IDX] * (MAX_LEN - len(x['labels'])) + x['labels']) for x in batch])
        attention_mask = (input_ids != torch.tensor(pad_id, dtype=input_ids.dtype)).long()

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    @staticmethod
    def collate_batch_pad_multiple(max_len: int, pad_id: int, multiple_of: int, batch: List[Dict[str, List[int]]]) -> dict:
        MAX_LEN = max_len
        max_len = max([len(x['input_ids']) for x in batch])
        max_len = ((max_len + multiple_of - 1) // multiple_of) * multiple_of

        input_ids = torch.stack([torch.tensor([0] * (max_len - len(x['input_ids'])) + x['input_ids']) for x in batch])
        labels = torch.stack([torch.tensor([CROSS_ENTROPY_IGNORE_IDX] * (max_len - len(x['labels'])) + x['labels']) for x in batch])
        attention_mask = (input_ids != torch.tensor(pad_id, dtype=input_ids.dtype)).long()

        if len(batch) == 1 and attention_mask.sum().item() == attention_mask.numel(): attention_mask = None

        out = {"input_ids": input_ids[:, :MAX_LEN], "labels": labels[:, :MAX_LEN]}
        if attention_mask is not None: out["attention_mask"] = attention_mask[:, :MAX_LEN]
        return out

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=4)