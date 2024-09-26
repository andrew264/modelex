from typing import Optional, Any

import torch
import torch.nn as nn
from torch import Tensor
from transformers import GenerationMixin, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ModuleUtilsMixin

from models.config import ModelCfg
from models.layers.transformer_block import Transformer


def exists(x: Optional[Any]) -> bool: return x is not None

class LLM(nn.Module, ModuleUtilsMixin, GenerationMixin):
    main_input_name = "inputs_embeds"
    _supports_cache_class = True

    def __init__(self, cfg: ModelCfg) -> None:
        super(LLM, self).__init__()
        self.cfg = cfg
        self.config = self.dummyconfig()

        self.model = Transformer(cfg=cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()

    def tie_weights(self):
        if self.cfg.tie_word_embeddings: self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, x: Tensor = None, pos_ids: Optional[Tensor] = None, cache_position: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None, past_kv: Optional[Cache] = None, **kwargs) -> CausalLMOutputWithPast:
        logits = self.lm_head(self.model(x=x, pos_ids=pos_ids, cache_position=cache_position, attn_mask=attn_mask, past_kv=past_kv))
        return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=past_kv, )

    def prepare_inputs_for_generation(self, input_ids: Tensor, past_key_values: Optional[Cache] = None, attention_mask=None, cache_position=None, use_cache=True, **kwargs,):
        past_length = 0
        x = input_ids
        attn_mask = attention_mask
        past_kv = past_key_values
        if exists(past_kv):
            past_length = cache_position[0] if cache_position is not None else past_kv.get_seq_length()
            max_cache_length = torch.tensor(past_kv.get_max_length(), device=x.device) if exists(past_kv.get_max_length()) else None
            
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            if exists(attn_mask) and attn_mask.shape[1] > x.shape[1]: x = x[:, -(attn_mask.shape[1] - past_length):]
            elif past_length < x.shape[1]: x = x[:, past_length:]
            if exists(max_cache_length) and exists(attn_mask) and cache_length + x.shape[1] > max_cache_length: attn_mask = attn_mask[:, -max_cache_length:]

        pos_ids = kwargs.get("pos_ids", None)
        if exists(attn_mask) and pos_ids is None:
            pos_ids = attn_mask.long().cumsum(-1) - 1
            pos_ids.masked_fill_(attn_mask == 0, 1)
            if past_kv: pos_ids = pos_ids[:, -x.shape[1]:]

        model_inputs = {"x": x.contiguous()}

        input_length = pos_ids.shape[-1] if exists(pos_ids) else x.shape[-1]
        if cache_position is None: cache_position = torch.arange(past_length, past_length + input_length, device=x.device)
        elif use_cache: cache_position = cache_position[-input_length:]

        model_inputs.update({"pos_ids": pos_ids, "cache_position": cache_position, "past_kv": past_kv, "attn_mask": attn_mask,})
        return model_inputs

    @classmethod
    def can_generate(cls) -> bool: return True

    def dummyconfig(self):  # HF compatability
        class Dummy:
            is_encoder_decoder = False
            use_cache = True
        return Dummy()