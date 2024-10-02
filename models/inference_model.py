from typing import Optional, Any

import torch
import torch.nn as nn
from torch import Tensor
from torchtune.modules import TiedLinear
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
        if not self.cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        else:
            self.lm_head = TiedLinear(self.model.embed_tokens)

    def forward(self, x: Tensor, pos_ids: Optional[Tensor] = None, cache_position: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None, past_kv: Optional[Cache] = None, **kwargs) -> CausalLMOutputWithPast:
        logits = self.lm_head(self.model(x=x, pos_ids=pos_ids, cache_position=cache_position, attn_mask=attn_mask, past_kv=past_kv))
        return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=past_kv, )

    def prepare_inputs_for_generation(self, input_ids: Tensor, past_key_values: Optional[Cache] = None, attention_mask=None, cache_position=None, use_cache=True, **kwargs,):
        past_length = 0
        if exists(past_key_values):
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = torch.tensor(past_key_values.get_max_length(), device=input_ids.device) if exists(past_key_values.get_max_length()) else None
            
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            if exists(attention_mask) and attention_mask.shape[1] > input_ids.shape[1]: input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]: input_ids = input_ids[:, past_length:]
            if exists(max_cache_length) and exists(attention_mask) and cache_length + input_ids.shape[1] > max_cache_length: attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if exists(attention_mask) and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values: position_ids = position_ids[:, -input_ids.shape[1]:]

        model_inputs = {"x": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if exists(position_ids) else input_ids.shape[-1]
        if cache_position is None: cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache: cache_position = cache_position[-input_length:]

        model_inputs.update({"pos_ids": position_ids, "cache_position": cache_position, "past_kv": past_key_values, "attn_mask": attention_mask,})
        return model_inputs

    @classmethod
    def can_generate(cls) -> bool: return True

    def dummyconfig(self):  # HF compatability
        class Dummy:
            is_encoder_decoder = False
            use_cache = True
        return Dummy()
