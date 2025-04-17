import os
from typing import Optional, Union, Self

import torch
from torch import nn, Tensor

from modelex.models.granite.config import GraniteConfig
from modelex.models.base import BaseLLM
from modelex.models.registry import register_model
from modelex.modules.attention import Attention
from modelex.modules.mlp import MLP
from modelex.modules.rotary_embedding import RotaryEmbedding
from modelex.modules.linears import linear_factory
from modelex.modules.tied_linear import TiedLinear

try:
    from cut_cross_entropy import linear_cross_entropy as loss_fn
    LCS = True
except ImportError:
    from torch.nn.functional import cross_entropy as loss_fn
    LCS = False

class GraniteAttention(Attention):
    def __init__(self, cfg: GraniteConfig, layer_idx: int, ):
        super(GraniteAttention, self).__init__(cfg=cfg, layer_idx=layer_idx)
        self.scaling = cfg.attention_multiplier

class GraniteBlock(nn.Module):
    def __init__(self, cfg: GraniteConfig, layer_idx: int, ) -> None:
        super().__init__()
        self.attn = GraniteAttention(cfg, layer_idx=layer_idx, )
        self.mlp = MLP(cfg, )
        self.sa_norm = nn.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp_norm = nn.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.residual_multiplier = cfg.residual_multiplier
    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: Optional[int] = None):
        self.attn.setup_cache(batch_size=batch_size, dtype=dtype, max_seq_len=max_seq_len)
    def reset_cache(self): self.attn.reset_cache()
    def forward(self, x: Tensor, freqs: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        h = x + self.attn(self.sa_norm(x), freqs=freqs, attn_mask=attn_mask) * self.residual_multiplier
        return h + self.mlp(self.mlp_norm(h)) * self.residual_multiplier
    
@register_model("Granite")
class Granite(BaseLLM):
    def __init__(self, cfg: Union[GraniteConfig, dict, str], skip_peft: bool = False) -> None:
        if isinstance(cfg, dict): cfg = GraniteConfig(**cfg)
        elif isinstance(cfg, str): cfg = GraniteConfig.load_config(os.path.join(cfg, 'models.yaml'))
        super(Granite, self).__init__(cfg)
        if hasattr(cfg, 'peft') and skip_peft: cfg.peft = None
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.inference.pad_token)
        self.layers = nn.ModuleList([GraniteBlock(cfg, idx) for idx in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg)
        if not cfg.tie_word_embeddings:
            kwargs = {}
            if hasattr(cfg, 'peft') and cfg.peft and 'output' in cfg.peft.layers:
                kwargs = {'peft_type': cfg.peft.type, 'rank': cfg.peft.rank, 'alpha': cfg.peft.alpha, 'dropout': cfg.peft.dropout, 'quantize_base': cfg.peft.quantize_base}
            self.output = linear_factory(in_features=cfg.hidden_size, out_features=cfg.vocab_size, bias=False, **kwargs)
        else:
            self.output = TiedLinear(self.tok_embeddings)
        self.embedding_multiplier = cfg.embedding_multiplier
        self.logits_scaling = cfg.logits_scaling
        self._cache_setup_complete = False
        self.ignore_labels_cache = None
    @classmethod
    def from_config(cls, config: GraniteConfig, skip_peft: bool = False) -> Self:
        return cls(config, skip_peft)
    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: Optional[int] = None):
        if self._cache_setup_complete: return
        for layer in self.layers:
            layer.setup_cache(batch_size=batch_size, dtype=dtype, max_seq_len=max_seq_len)
        super().setup_cache(batch_size, dtype, max_seq_len)
    def reset_cache(self):
        for layer in self.layers: layer.reset_cache()
    def calc_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        return loss_fn(logits.contiguous().view(-1, self.cfg.vocab_size), labels.view(-1))
    def forward(self, input_ids: Tensor, input_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None, force_full_logits: bool = False, **kwargs) -> Tensor:
        x = self.tok_embeddings(input_ids) * self.embedding_multiplier
        if input_pos is None: input_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        freqs = self.rotary_emb(input_pos)
        for layer in self.layers:
            x = layer(x=x, freqs=freqs, attn_mask=mask)
        x = self.norm(x)
        if self.training or force_full_logits: logits = self.output(x)
        else: logits = self.output(x[:, -1:])
        return logits / self.logits_scaling
    def train_step(self, input_ids: Tensor, labels: Tensor, input_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None, **kwargs) -> dict:
        loss = None
        x = self.tok_embeddings(input_ids) * self.embedding_multiplier
        if input_pos is None: input_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        freqs = self.rotary_emb(input_pos)
        for layer in self.layers:
            x = layer(x=x, freqs=freqs, attn_mask=mask)
        x = self.norm(x)
        if LCS:
            loss = loss_fn(x, self.output.weight / self.logits_scaling, labels, shift=True)
            return dict(loss=loss)
        logits = self.output(x)
        if self.ignore_labels_cache is None:
            batch_size = labels.size(0)
            self.ignore_labels_cache = torch.full((batch_size, 1), -100, device=labels.device)
        labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[:labels.shape[0]])).contiguous()
        loss = self.calc_loss(logits, labels)
        return dict(logits=logits, loss=loss,)