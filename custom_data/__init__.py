from functools import partial
from typing import Any, Iterator, List, Dict, Optional, Tuple, Union

import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import lightning as L
from torchtune.generation import get_causal_mask_from_padding_mask

CROSS_ENTROPY_IGNORE_IDX = -100
BatchType = List[Dict[str, List[Union[int, List[float]]]]]

def exists(x: Optional[Any]) -> bool: return x is not None

class ParquetCustomDataReader(Dataset):
    def __init__(self, path: str) -> None:
        self.pq_file = pq.ParquetFile(path)
    def __len__(self) -> int: return self.pq_file.num_row_groups
    def __getitem__(self, idx) -> Dict[str, List[int | float]]: return self.pq_file.read_row_group(idx).to_pydict()
    def __iter__(self) -> Iterator[Dict[str, List[int | float]]]:
        for i in range(len(self)): yield self[i]

def get_padded_logits(batch: BatchType, max_len: int) -> Optional[Tensor]:
    has_teacher_logits = batch[0].get('teacher_logits') is not None
    if not has_teacher_logits: return None
    teacher_logits = []
    for item in batch:
        logits = torch.tensor(item['teacher_logits'], dtype=torch.float32)[:max_len]
        input_size, vocab_size = logits.shape
        padding = torch.zeros((max_len - input_size, vocab_size), dtype=torch.float32)
        logits = torch.cat((padding, logits), dim=0)
        teacher_logits.append(logits)
    return torch.stack(teacher_logits, dim=0)

def get_padded_ids_and_labels(batch: BatchType, max_len: int) -> Tuple[Tensor, Tensor]:
    ids, labels = [], []
    for item in batch:
        _id, label = item['input_ids'][:max_len], item['labels'][:max_len]
        ids.append(torch.tensor([0] * (max_len - len(_id)) + _id))
        labels.append(torch.tensor([CROSS_ENTROPY_IGNORE_IDX] * (max_len - len(label)) + label))
    return torch.stack(ids, dim=0), torch.stack(labels, dim=0)

def get_causal_mask(input_ids: Tensor, pad_id: int, max_length) -> Tensor:
    padding_masks = input_ids != pad_id
    if not padding_masks.all():
        # padding_masks = torch.nn.functional.pad(padding_masks, (0, max_length), value=True)
        masks = get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_length)
    else:
        masks = torch.tril(torch.ones(max_length, max_length, dtype=torch.bool, device=input_ids.device)).unsqueeze(0)
    return masks[:, None, :, :]

class DataModule(L.LightningDataModule):
    def __init__(self, train_ds: Dataset, valid_ds: Optional[Dataset], batch_size: int, max_seq_length: int, max_pad: bool = False,
                 pad_to_multiple_of: int = 1, pad_id: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_pad = max_pad
        self.train_ds = train_ds
        self.valid_ds = valid_ds if exists(valid_ds) else []
        if max_pad:  # pad everything to max_length
            self.collate_fn = partial(self.collate_max_pad_fn, max_seq_length, pad_id)
        elif pad_to_multiple_of > 1:  # pad everything to min(max_length, max multiple length of item in batch)
            self.collate_fn = partial(self.collate_batch_pad_multiple, max_seq_length, pad_id, pad_to_multiple_of)
        else:  # pad everything to min(max_length, max item length in batch)
            self.collate_fn = partial(self.collate_batch_pad_fn, max_seq_length, pad_id)

    @staticmethod
    def collate_batch_pad_fn(max_len: int, pad_id: int, batch: BatchType) -> dict:
        max_len = min(max([len(x['input_ids']) for x in batch]), max_len)

        teacher_logits = get_padded_logits(batch, max_len)
        input_ids, labels = get_padded_ids_and_labels(batch, max_len)
        causal_mask = get_causal_mask(input_ids, pad_id, max_len)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": causal_mask, "teacher_logits": teacher_logits}

    @staticmethod
    def collate_max_pad_fn(max_len: int, pad_id: int, batch: BatchType) -> dict:
        teacher_logits = get_padded_logits(batch, max_len)
        input_ids, labels = get_padded_ids_and_labels(batch, max_len)
        causal_mask = get_causal_mask(input_ids, pad_id, max_len)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": causal_mask, "teacher_logits": teacher_logits}

    @staticmethod
    def collate_batch_pad_multiple(max_len: int, pad_id: int, multiple_of: int, batch: BatchType) -> dict:
        max_len = min(max([len(x['input_ids']) for x in batch]), max_len)
        max_len = ((max_len + multiple_of - 1) // multiple_of) * multiple_of
        teacher_logits = get_padded_logits(batch, max_len)
        input_ids, labels = get_padded_ids_and_labels(batch, max_len)
        causal_mask = get_causal_mask(input_ids, pad_id, max_len)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": causal_mask, "teacher_logits": teacher_logits}

    def train_dataloader(self):
        # damn these dataloader objects suck when num_workers > 0
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=0)
