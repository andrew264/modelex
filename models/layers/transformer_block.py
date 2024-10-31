from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.config import ModelCfg, PeftCfg
from models.layers.attention import Attention
from models.layers.mlp import MLP

def exists(x: Optional[Any]) -> bool: return x is not None

class Block(nn.Module):
    def __init__(self, cfg: ModelCfg, layer_idx: int, peft_cfg: Optional[PeftCfg] = None,) -> None:
        super().__init__()
        self.attn = Attention(cfg, layer_idx=layer_idx, peft_cfg=peft_cfg)
        self.mlp = MLP(cfg, peft_cfg=peft_cfg,)
        self.sa_norm = nn.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp_norm = nn.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: Optional[int] = None, ):
        self.attn.setup_cache(batch_size=batch_size, dtype=dtype, max_seq_len=max_seq_len)
    def reset_cache(self): self.attn.reset_cache()
    def forward(self, x: Tensor, freqs: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        h = x + self.attn(self.sa_norm(x), freqs=freqs, attn_mask=attn_mask)
        return h + self.mlp(self.mlp_norm(h))