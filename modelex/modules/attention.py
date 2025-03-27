from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
    def __init__(self, cfg, layer_idx: int, ):
        super(Attention, self).__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_hidden_size = cfg.num_kv_heads * self.head_dim
        self.num_kv_groups = cfg.num_heads // cfg.num_kv_heads
        qkv_out_dim = cfg.hidden_size + 2 * self.kv_hidden_size

        hidden = cfg.hidden_size
        kwargs = {}
        if hasattr(cfg, 'peft') and cfg.peft and 'attn' in cfg.peft.layers:
            kwargs = {'peft_type': cfg.peft.type, 'rank': cfg.peft.rank, 'alpha': cfg.peft.alpha, 'dropout': cfg.peft.dropout}

        self.qkv_proj = linear_factory(in_features=hidden, out_features=qkv_out_dim, bias=cfg.attn_qkv_bias, **kwargs)
        self.o_proj = linear_factory(in_features=hidden, out_features=hidden, bias=cfg.attn_out_bias, **kwargs)
        self._register_load_state_dict_pre_hook(self.fused_qkv_hook)

        self.cache_enabled = False
        self.kv_cache = None
        self.scaling = None

    @staticmethod
    def fused_qkv_hook(state_dict, prefix, *args, **kwargs):
        if prefix + 'q_proj.weight' in state_dict:
            q_weight = state_dict.pop(prefix + 'q_proj.weight')
            k_weight = state_dict.pop(prefix + 'k_proj.weight')
            v_weight = state_dict.pop(prefix + 'v_proj.weight')
            state_dict[prefix + 'qkv_proj.weight'] = torch.cat([q_weight, k_weight, v_weight])
        if prefix + 'q_proj.bias' in state_dict:
            q_bias = state_dict.pop(prefix + 'q_proj.bias')
            k_bias = state_dict.pop(prefix + 'k_proj.bias')
            v_bias = state_dict.pop(prefix + 'v_proj.bias')
            state_dict[prefix + 'qkv_proj.bias'] = torch.cat([q_bias, k_bias, v_bias])
    def caches_are_setup(self) -> bool: return exists(self.kv_cache)
    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: int) -> None:
        if self.caches_are_setup(): print("Key value caches are already setup. You cannot call ``setup_caches()`` twice. Skipping.")
        else:
            self.kv_cache = KVCache(batch_size=batch_size, max_seq_len=max_seq_len, num_heads=self.num_kv_heads, head_dim=self.head_dim,
                                    dtype=dtype, )
            self.cache_enabled = True
    def reset_cache(self):
        if not exists(self.kv_cache): raise RuntimeError("Key value caches are not setup. Call ``setup_caches()`` first.")
        self.kv_cache.reset()

    def forward(self, x: Tensor, freqs: Tuple[Tensor, Tensor], attn_mask: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.size()
        is_causal = attn_mask is None and seqlen > 1

        q, k, v = self.qkv_proj(x).split([self.hidden_size, self.kv_hidden_size, self.kv_hidden_size], dim=2)
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        q, k = apply_rotary_pos_emb(q, k, *freqs)

        if self.cache_enabled: k, v = self.kv_cache(k, v)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0., is_causal=is_causal, enable_gqa=self.num_kv_groups > 1, scale=self.scaling)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)
        return self.o_proj(attn)