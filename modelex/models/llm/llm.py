import os
from typing import Optional, Self, Union

import torch
import torch.nn as nn
from torch import Tensor

from modelex.models.base import BaseLLM
from modelex.models.llm.config import LLMConfig
from modelex.models.registry import register_model
from modelex.modules import Block, linear_factory, RotaryEmbedding, TiedLinear
from modelex.utils import exists

@register_model("LLM")
class LLM(BaseLLM):
    def __init__(self, cfg: Union[LLMConfig, dict, str], ) -> None:
        if isinstance(cfg, dict): cfg = LLMConfig(**cfg)
        elif isinstance(cfg, str): cfg = LLMConfig.load_config(os.path.join(cfg, 'models.yaml'))
        super(LLM, self).__init__(cfg)

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.inference.pad_token)
        self.layers = nn.ModuleList([Block(cfg, idx) for idx in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg)
        if not cfg.tie_word_embeddings:
            kwargs = {}
            if hasattr(cfg, 'peft') and cfg.peft and 'output' in cfg.peft.layers:
                kwargs = {'peft_type': cfg.peft.type, 'rank': cfg.peft.rank, 'alpha': cfg.peft.alpha, 'dropout': cfg.peft.dropout}
            self.output = linear_factory(in_features=cfg.hidden_size, out_features=cfg.vocab_size, bias=False, **kwargs)
        else:
            self.output = TiedLinear(self.tok_embeddings)
        self.loss_fn = nn.CrossEntropyLoss()
        self._cache_setup_complete = False
        self.ignore_labels_cache = None
    @classmethod
    def from_config(cls, config: LLMConfig) -> Self:
        return cls(config)
    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: Optional[int] = None, ):
        if self._cache_setup_complete: return
        for layer in self.layers:
            layer.setup_cache(batch_size=batch_size, dtype=dtype, max_seq_len=max_seq_len)
        super().setup_cache(batch_size, dtype, max_seq_len)
    def reset_cache(self):
        for layer in self.layers: layer.reset_cache()
    def calc_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self.loss_fn(logits.contiguous().view(-1, self.cfg.vocab_size), labels.view(-1))
    def forward(self, input_ids: Tensor, input_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None, labels: Optional[Tensor] = None, force_full_logits: bool = False, **kwargs) -> Tensor:
        x = self.tok_embeddings(input_ids)
        if input_pos is None: input_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        freqs = self.rotary_emb(input_pos)
        for layer in self.layers:
            x = layer(x=x, freqs=freqs, attn_mask=mask)
        x = self.norm(x)
        if self.training or exists(labels) or force_full_logits: logits = self.output(x)
        else: logits = self.output(x[:, -1:])
        return logits
    def train_step(self, input_ids: Tensor, input_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None, labels: Optional[Tensor] = None, force_full_logits: bool = False, **kwargs) -> dict:
        loss = None
        logits = self.forward(input_ids, input_pos, mask, labels, force_full_logits=force_full_logits)
        if exists(labels):
            if not exists(self.ignore_labels_cache):
                batch_size = labels.size(0)
                self.ignore_labels_cache = torch.full((batch_size, 1), -100, device=labels.device)
            labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[:labels.shape[0]])).contiguous()
            loss = self.calc_loss(logits, labels)
        return dict(logits=logits, loss=loss,)
