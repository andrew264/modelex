from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modelex.models.llm.config import LLMConfig
from modelex.modules.linears import linear_factory
from modelex.utils import exists
from modelex.utils.kv_cache import KVCache

def rotate_half(x: Tensor) -> Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1) -> Tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, cfg: LLMConfig, layer_idx: int, ):
        super(Attention, self).__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_hidden_size = cfg.num_kv_heads * self.head_dim
        self.num_kv_groups = cfg.num_heads // cfg.num_kv_heads
        self.use_rope = layer_idx in cfg.rope_layers if cfg.rope_layers else True

        kwargs = {}
        if hasattr(cfg, 'peft') and cfg.peft and 'attn' in cfg.peft.layers:
            kwargs = {'peft_type': cfg.peft.type, 'rank': cfg.peft.rank, 'alpha': cfg.peft.alpha, 'dropout': cfg.peft.dropout, 'quantize_base': cfg.peft.quantize_base}

        self.q_proj = linear_factory(in_features=self.hidden_size, out_features=self.hidden_size, bias=cfg.attn_qkv_bias, **kwargs)
        self.k_proj = linear_factory(in_features=self.hidden_size, out_features=self.kv_hidden_size, bias=cfg.attn_qkv_bias, **kwargs)
        self.v_proj = linear_factory(in_features=self.hidden_size, out_features=self.kv_hidden_size, bias=cfg.attn_qkv_bias, **kwargs)
        self.o_proj = linear_factory(in_features=self.hidden_size, out_features=self.hidden_size, bias=cfg.attn_out_bias, **kwargs)

        self.cache_enabled = False
        self.kv_cache = None
        self.scaling = None
    def caches_are_setup(self) -> bool: return exists(self.kv_cache)
    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: int) -> None:
        if self.caches_are_setup(): print("Key value caches are already setup. You cannot call ``setup_caches()`` twice. Skipping.")
        else:
            self.kv_cache = KVCache(batch_size=batch_size, max_seq_len=max_seq_len, num_heads=self.num_kv_heads, head_dim=self.head_dim,
                                    dtype=dtype)
            self.cache_enabled = True
    def reset_cache(self):
        if not exists(self.kv_cache): raise RuntimeError("Key value caches are not setup. Call ``setup_caches()`` first.")
        self.kv_cache.reset()

    def forward(self, x: Tensor, freqs: Tuple[Tensor, Tensor], attn_mask: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.size()
        is_causal = attn_mask is None and seqlen > 1

        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, *freqs)

        if self.cache_enabled: k, v = self.kv_cache(k, v)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0., is_causal=is_causal, enable_gqa=self.num_kv_groups != 1, scale=self.scaling)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)
        return self.o_proj(attn)