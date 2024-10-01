from typing import Optional, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from models.config import ModelCfg, PeftCfg
from models.layers.attention import Attention
from models.layers.mlp import MLP
from models.layers.rotary_embedding import RotaryEmbedding

def get_rmsnorm():
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm as RMSNorm
        return RMSNorm
    except ImportError:
        return nn.RMSNorm

def exists(x: Optional[Any]) -> bool: return x is not None

class Block(nn.Module):
    def __init__(self, cfg: ModelCfg, layer_idx: int, peft_cfg: Optional[PeftCfg] = None) -> None:
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx=layer_idx, peft_cfg=peft_cfg)
        self.mlp = MLP(cfg, peft_cfg=peft_cfg)

        NORM_CLASS = get_rmsnorm()
        self.input_layernorm = NORM_CLASS(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = NORM_CLASS(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: Tensor, freqs: Tensor, past_kv: Optional[Cache] = None, attn_mask: Optional[Tensor] = None, cache_position: Optional[Tensor] = None, ) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), freqs=freqs, past_kv=past_kv, attn_mask=attn_mask, cache_position=cache_position)
        return h + self.mlp(self.post_attention_layernorm(h))
    
class Transformer(nn.Module):
    def __init__(self, cfg: ModelCfg, peft_cfg: Optional[None] = None, use_grad_checkpointing: bool = False) -> None:
        super(Transformer, self).__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token)
        self.layers = nn.ModuleList([Block(cfg, idx, peft_cfg) for idx in range(cfg.num_layers)])
        self.norm = get_rmsnorm()(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(dim=cfg.hidden_size//cfg.num_heads, base=cfg.rope_theta)

    def forward(self, x: Tensor, pos_ids: Optional[Tensor]=None, cache_position: Optional[Tensor]=None, attn_mask: Optional[Tensor]=None, past_kv: Optional[Cache]=None) -> Tensor:
        x = self.embed_tokens(x)
        if pos_ids is None: pos_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        causal_mask = _update_causal_mask(attn_mask, x, cache_position, past_kv)
        freqs = self.rotary_emb(pos_ids)

        for layer in self.layers:
            if self.training and self.use_grad_checkpointing: x = checkpoint(layer, x=x, freqs=freqs, past_kv=past_kv, attn_mask=causal_mask, cache_position=cache_position, use_reentrant=False)
            else: x = layer(x=x, freqs=freqs, past_kv=past_kv, attn_mask=causal_mask, cache_position=cache_position)
        return self.norm(x)
    
def _update_causal_mask(attn_mask: torch.Tensor, input_tensor: torch.Tensor, cache_position: torch.Tensor, past_kv: Cache,) -> Optional[Tensor]:
    past_seen_tokens = past_kv.get_seq_length() if exists(past_kv) else 0
    using_static_cache = isinstance(past_kv, StaticCache)

    if not exists(attn_mask): return attn_mask

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    seqlen = input_tensor.shape[1]
    if using_static_cache: target_length = past_kv.get_max_length()
    else: target_length = attn_mask.shape[-1] if isinstance(attn_mask, torch.Tensor) else past_seen_tokens + seqlen + 1

    causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(attn_mask,
        seqlen=seqlen, target_length=target_length, dtype=dtype, device=device, min_dtype=min_dtype, cache_position=cache_position, bsz=input_tensor.shape[0],)

    if exists(attn_mask) and attn_mask.device.type == "cuda": causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask
    
def _prepare_4d_causal_attention_mask_with_cache_position(
    attn_mask: torch.Tensor, seqlen: int, target_length: int, dtype: torch.dtype, device: torch.device, min_dtype: float, cache_position: torch.Tensor, bsz: int,) -> Tensor:
    if exists(attn_mask) and attn_mask.dim() == 4:
        causal_mask = attn_mask
    else:
        causal_mask = torch.full((seqlen, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if seqlen != 1: causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, -1, -1)
        if exists(attn_mask):
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attn_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attn_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

    return causal_mask
