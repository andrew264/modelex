import os
from typing import Optional, Self, Union

import torch
import torch.nn as nn
from torch import Tensor

from modelex.models.base import BaseLLM
from modelex.models.llm.config import LLMTrainConfig
from modelex.models.registry import register_model
from modelex.modules import Block, linear_factory, RotaryEmbedding, TiedLinear

@register_model('LLMTrain')
class LLMTrain(BaseLLM):
    def __init__(self, cfg: Union[LLMTrainConfig, dict, str], ) -> None:
        if isinstance(cfg, dict): cfg = LLMTrainConfig(**cfg)
        elif isinstance(cfg, str): cfg = LLMTrainConfig.load_config(os.path.join(cfg, 'models.yaml'))
        super(LLMTrain, self).__init__(cfg)

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
        try:
            from cut_cross_entropy import linear_cross_entropy
            self.loss_fn = linear_cross_entropy
        except ImportError:
            raise ImportError('install apple/ml-cross-entropy for LLMTrain')
    @classmethod
    def from_config(cls, config: LLMTrainConfig) -> Self:
        return cls(config)
    def train_step(self, input_ids: Tensor, labels: Tensor, input_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None,
                **kwargs) -> dict:
        x = self.tok_embeddings(input_ids)
        if input_pos is None: input_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        freqs = self.rotary_emb(input_pos)
        for layer in self.layers:
            x = layer(x=x, freqs=freqs, attn_mask=mask)
        x = self.norm(x)
        loss = self.loss_fn(x, self.output.weight, labels, shift=True)
        return dict(loss=loss)