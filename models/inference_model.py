from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchtune.modules import TiedLinear

from models.config import ModelCfg
from models.layers.rotary_embedding import RotaryEmbedding
from models.layers.transformer_block import Block

def exists(x: Optional[Any]) -> bool: return x is not None

class LLM(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super(LLM, self).__init__()
        self.cfg = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token)
        self.layers = nn.ModuleList([Block(cfg, idx) for idx in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg)
        if not self.cfg.tie_word_embeddings:
            self.output = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        else:
            self.output = TiedLinear(self.tok_embeddings)
        self._cache_setup_complete = False
    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: Optional[int] = None, ):
        if self._cache_setup_complete: return
        for layer in self.layers:
            layer.setup_cache(batch_size=batch_size, dtype=dtype, max_seq_len=max_seq_len)
        self._cache_setup_complete = True
    def reset_cache(self):
        for layer in self.layers: layer.reset_cache()
    def caches_are_enabled(self): return self._cache_setup_complete
    @property
    def decoder_max_cache_seq_len(self): return self.cfg.max_seq_len
    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None, ) -> Tensor:
        x = self.tok_embeddings(x)
        if input_pos is None: input_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        freqs = self.rotary_emb(input_pos)

        for layer in self.layers:
            x = layer(x=x, freqs=freqs, attn_mask=mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits