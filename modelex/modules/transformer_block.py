from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from modelex.modules.attention import Attention
from modelex.modules.mlp import MLP

class Block(nn.Module):
    def __init__(self, cfg, layer_idx: int,) -> None:
        super().__init__()
        self.attn = Attention(cfg, layer_idx=layer_idx,)
        self.mlp = MLP(cfg, )
        self.sa_norm = nn.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp_norm = nn.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: Optional[int] = None, ):
        self.attn.setup_cache(batch_size=batch_size, dtype=dtype, max_seq_len=max_seq_len)
    def reset_cache(self): self.attn.reset_cache()
    def forward(self, x: Tensor, freqs: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        h = x + self.attn(self.sa_norm(x), freqs=freqs, attn_mask=attn_mask)
        return h + self.mlp(self.mlp_norm(h))