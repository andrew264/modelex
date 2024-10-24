from typing import Tuple

import torch
from torch import nn
from transformers import ROPE_INIT_FUNCTIONS

from models.config import ModelCfg

class RotaryEmbedding(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.rope_type = cfg.rope_type
        self.cache_len = cfg.max_seq_len
        self.original_max_len = cfg.max_seq_len
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(cfg, )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.cache_len:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len, **self.rope_kwargs)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.cache_len = seq_len
        if seq_len < self.original_max_len < self.cache_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.cache_len = self.original_max_len

    @torch.no_grad()
    def forward(self, position_ids: torch.LongTensor, dtype: torch.dtype = torch.bfloat16) -> Tuple[torch.Tensor, torch.Tensor]:
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=position_ids.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = position_ids.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos * self.attention_scaling  # yarn
        sin = sin * self.attention_scaling
        return cos.to(dtype=dtype), sin.to(dtype=dtype)