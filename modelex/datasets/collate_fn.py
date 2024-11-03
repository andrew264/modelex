from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torchtune.generation import get_causal_mask_from_padding_mask, get_position_ids_from_padding_mask

CROSS_ENTROPY_IGNORE_IDX = -100
BatchType = List[Dict[str, List[int]]]

def get_mask_and_pos(input_ids: Tensor, pad_id: int, max_length) -> Tuple[Tensor, Tensor]:
    padding_masks: Tensor = input_ids != pad_id
    if not padding_masks.all():
        masks = get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_length)
        input_pos = get_position_ids_from_padding_mask(padding_masks)
    else:
        masks = torch.tril(torch.ones(max_length, max_length, dtype=torch.bool, device=input_ids.device)).unsqueeze(0)
        input_pos = torch.arange(0, max_length).unsqueeze(0)
    return masks.unsqueeze(1), input_pos

def get_padded_ids_and_labels(batch: BatchType, max_len: int) -> Tuple[Tensor, Tensor]:
    ids, labels = [], []
    for item in batch:
        _id, label = item['input_ids'][:max_len], item['labels'][:max_len]
        ids.append(torch.tensor([0] * (max_len - len(_id)) + _id))
        labels.append(torch.tensor([CROSS_ENTROPY_IGNORE_IDX] * (max_len - len(label)) + label))
    return torch.stack(ids, dim=0), torch.stack(labels, dim=0)

class CollateMaxPad:
    def __init__(self, max_len: int, pad_id: int):
        self.max_len = max_len
        self.pad_id = pad_id
    def __call__(self, batch: BatchType) -> dict:
        input_ids, labels = get_padded_ids_and_labels(batch, self.max_len)
        causal_mask, input_pos = get_mask_and_pos(input_ids, self.pad_id, self.max_len)

        return {"input_ids": input_ids, "labels": labels, "mask": causal_mask, "input_pos": input_pos}

class CollateBatchPad:
    def __init__(self, max_len: int, pad_id: int):
        self.max_len = max_len
        self.pad_id = pad_id
    def __call__(self, batch: BatchType) -> dict:
        max_len = min(max([len(x['input_ids']) for x in batch]), self.max_len)
        input_ids, labels = get_padded_ids_and_labels(batch, max_len)
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
        input_ids, labels = get_padded_ids_and_labels(batch, max_len)
        causal_mask, input_pos = get_mask_and_pos(input_ids, self.pad_id, max_len)

        return {"input_ids": input_ids, "labels": labels, "mask": causal_mask, "input_pos": input_pos}