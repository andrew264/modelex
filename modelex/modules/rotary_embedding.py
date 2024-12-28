from typing import Tuple

import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.head_dim
        self.base = cfg.rope_base
        self.rope_init()

    def rope_init(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.LongTensor, dtype: torch.dtype = torch.bfloat16) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1: position_ids = position_ids.unsqueeze(0)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = position_ids.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)