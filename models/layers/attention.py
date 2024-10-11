from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Cache

from models.config import ModelCfg, PeftCfg

def rotate_half(x: Tensor) -> Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, pos_ids: Optional[Tensor] = None, unsqueeze_dim: int = 1) -> Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

try:
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    apply_rotary_pos_emb = liger_rotary_pos_emb
except ImportError:
    pass

def exists(x: Optional[Any]) -> bool: return x is not None

class Attention(nn.Module):
    def __init__(self, cfg: ModelCfg, layer_idx: int, peft_cfg: Optional[PeftCfg] = None):
        super(Attention, self).__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.kv_hidden_size = cfg.num_kv_heads * self.head_dim
        self.num_kv_groups = cfg.num_heads // cfg.num_kv_heads
        qkv_out_dim = cfg.hidden_size + 2 * self.kv_hidden_size

        if peft_cfg:
            if peft_cfg.type == 'dora': from torchtune.modules.peft import DoRALinear as Linear
            else: from torchtune.modules.peft import LoRALinear as Linear
            Linear = partial(Linear, rank=peft_cfg.rank, alpha=peft_cfg.alpha, dropout=peft_cfg.dropout, quantize_base=peft_cfg.quant_base)
            if 'qkv_proj' in peft_cfg.layers:
                self.qkv_proj = Linear(in_dim=cfg.hidden_size, out_dim=qkv_out_dim, use_bias=cfg.attn_qkv_bias)
            else:
                self.qkv_proj = nn.Linear(in_features=cfg.hidden_size, out_features=qkv_out_dim, bias=cfg.attn_qkv_bias)
            if 'o_proj' in peft_cfg.layers:
                self.o_proj = Linear(in_dim=cfg.hidden_size, out_dim=cfg.hidden_size, use_bias=cfg.attn_out_bias)
            else:
                self.o_proj = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.hidden_size, bias=cfg.attn_out_bias)
        else:
            self.qkv_proj = nn.Linear(in_features=cfg.hidden_size, out_features=qkv_out_dim, bias=cfg.attn_qkv_bias)
            self.o_proj = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.hidden_size, bias=cfg.attn_out_bias)
        self._register_load_state_dict_pre_hook(self.fused_qkv_hook)

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

    def forward(self, x: Tensor, freqs: Tensor, past_kv: Optional[Cache] = None, attn_mask: Optional[Tensor] = None, cache_position: Optional[Tensor] = None, ) -> Tensor:
        bsz, seqlen, _ = x.size()
        is_causal = attn_mask is None and seqlen > 1

        q, k, v = self.qkv_proj(x).split([self.hidden_size, self.kv_hidden_size, self.kv_hidden_size], dim=2)
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        q, k = apply_rotary_pos_emb(q, k, *freqs)

        if exists(cache_position) and exists(past_kv): k, v = past_kv.update(k, v, self.layer_idx, dict(cache_position=cache_position))

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        if exists(attn_mask): attn_mask = attn_mask[..., :k.shape[-2]]
        attn = F.scaled_dot_product_attention(q, k ,v, attn_mask=attn_mask, dropout_p=0., is_causal=is_causal).transpose(1, 2).view(bsz, seqlen, self.hidden_size)
        return self.o_proj(attn)
