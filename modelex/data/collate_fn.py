from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from modelex.utils.generation_utils import get_causal_mask_from_padding_mask, get_position_ids_from_padding_mask

CROSS_ENTROPY_IGNORE_IDX = -100
BatchType = List[Dict[str, List[int]]]

def get_mask_and_pos(input_ids: Tensor, pad_id: int, max_length: int) -> Tuple[Tensor, Tensor]:
    if input_ids.ndim == 3:  # hacks
        input_ids = torch.stack([x[0] for x in input_ids], dim=0)
    padding_masks: Tensor = input_ids != pad_id
    if not padding_masks.all():
        masks = get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_length)
        input_pos = get_position_ids_from_padding_mask(padding_masks)
    else:
        masks = torch.tril(torch.ones(max_length, max_length, dtype=torch.bool, device=input_ids.device)).unsqueeze(0)
        input_pos = torch.arange(0, max_length).unsqueeze(0)
    return masks.unsqueeze(1), input_pos

def get_padded_ids_and_labels(batch: BatchType, pad_id: int, max_len: int) -> Tuple[Tensor, Tensor]:
    ids, labels = [], []
    for item in batch:
        _id, label = item['input_ids'][:max_len], item['labels'][:max_len]
        if isinstance(_id, (np.ndarray, Tensor)):
            _id, label = _id.tolist(), label.tolist()
        ids.append(torch.tensor([pad_id] * (max_len - len(_id)) + _id))
        labels.append(torch.tensor([CROSS_ENTROPY_IGNORE_IDX] * (max_len - len(label)) + label))
    return torch.stack(ids, dim=0), torch.stack(labels, dim=0)

class CollateMaxPad:
    def __init__(self, max_len: int, pad_id: int):
        self.max_len = max_len
        self.pad_id = pad_id
    def __call__(self, batch: BatchType) -> dict:
        input_ids, labels = get_padded_ids_and_labels(batch, self.pad_id, self.max_len)
        causal_mask, input_pos = get_mask_and_pos(input_ids, self.pad_id, self.max_len)

        return {"input_ids": input_ids, "labels": labels, "mask": causal_mask, "input_pos": input_pos}

class CollateBatchMaxPad:
    def __init__(self, max_len: int, pad_id: int):
        self.max_len = max_len
        self.pad_id = pad_id
    def __call__(self, batch: BatchType) -> dict:
        max_len = min(max([len(x['input_ids']) for x in batch]), self.max_len)
        input_ids, labels = get_padded_ids_and_labels(batch, self.pad_id, max_len)
        if input_ids.size(0) > 1:
            causal_mask, input_pos = get_mask_and_pos(input_ids, self.pad_id, max_len)
            return {"input_ids": input_ids, "labels": labels, "mask": causal_mask, "input_pos": input_pos}
        else:
            return {"input_ids": input_ids, "labels": labels, "input_pos": torch.arange(0, max_len).unsqueeze(0)}

class CollateBatchPadMultiple:
    def __init__(self, max_len: int, pad_id: int, multiple_of: int):
        self.max_len = max_len
        self.pad_id = pad_id
        self.multiple_of = multiple_of
    def __call__(self, batch: BatchType) -> dict:
        max_len = min(max([len(x['input_ids']) for x in batch]), self.max_len)
        max_len = ((max_len + self.multiple_of - 1) // self.multiple_of) * self.multiple_of
        input_ids, labels = get_padded_ids_and_labels(batch, self.pad_id, max_len)
        causal_mask, input_pos = get_mask_and_pos(input_ids, self.pad_id, max_len)

        return {"input_ids": input_ids, "labels": labels, "mask": causal_mask, "input_pos": input_pos}

class CollateMaxPad2D:
    def __init__(self, max_len: int, pad_id: int):
        self.max_len = max_len
        self.pad_id = pad_id
    def __call__(self, batch: BatchType) -> dict:
        batch_size = len(batch)
        num_codebooks = len(batch[0]['input_ids'])
        padded_ids = torch.full((batch_size, num_codebooks, self.max_len), self.pad_id, dtype=torch.long)
        padded_labels = torch.full((batch_size, num_codebooks, self.max_len), CROSS_ENTROPY_IGNORE_IDX, dtype=torch.long)
        for b_idx, item in enumerate(batch):
            input_ids = item['input_ids']
            labels = item['labels']
            for seq_idx in range(num_codebooks):
                current_ids = input_ids[seq_idx][:self.max_len]
                current_labels = labels[seq_idx][:self.max_len]
                padded_ids[b_idx, seq_idx, :len(current_ids)] = current_ids
                padded_labels[b_idx, seq_idx, :len(current_labels)] = current_labels

        causal_mask, input_pos = get_mask_and_pos(padded_ids, self.pad_id, self.max_len)
        return {"input_ids": padded_ids, "labels": padded_labels, "mask": causal_mask, "input_pos": input_pos}