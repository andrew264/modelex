import contextlib
import os
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from modelex.modules import Block, RotaryEmbedding, TiedLinear2
from modelex.models.llm.config import ModelCfg, PeftCfg
from modelex.utils import exists

class LLM(nn.Module):
    def __init__(self, cfg: Union[ModelCfg, dict, str], peft_cfg: Optional[PeftCfg] = None) -> None:
        super(LLM, self).__init__()
        if isinstance(cfg, dict): cfg = ModelCfg(**cfg)
        elif isinstance(cfg, str): cfg = ModelCfg.from_yaml(os.path.join(cfg, 'models.yaml'))
        self.cfg = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token)
        self.layers = nn.ModuleList([Block(cfg, idx, peft_cfg) for idx in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg)

        if not cfg.tie_word_embeddings:
            if peft_cfg and 'output' in peft_cfg.layers:
                if peft_cfg.type == 'dora':
                    from torchtune.modules.peft import DoRALinear as Linear
                else:
                    from torchtune.modules.peft import LoRALinear as Linear
                Linear = partial(Linear, rank=peft_cfg.rank, alpha=peft_cfg.alpha, dropout=peft_cfg.dropout, quantize_base=peft_cfg.quant_base)
                self.output = Linear(in_dim=cfg.hidden_size, out_dim=cfg.vocab_size, use_bias=False)
            else:
                self.output = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.vocab_size, bias=False)
        else:
            self.output = TiedLinear2(self.tok_embeddings)

        self._cache_setup_complete = False
        self.num_output_chunks = 1
        self.offload_context = contextlib.nullcontext()
        self._embedding_device = None
    def _init_weights(self, module):
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if exists(module.bias): module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if exists(module.padding_idx): module.weight.data[module.padding_idx].zero_()
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
    @torch.compiler.disable
    def chunked_output(self, last_hidden_state: Tensor) -> list[Tensor]:
        return [self.output(chunk) for chunk in last_hidden_state.chunk(self.num_output_chunks, dim=1)]
    def set_output_chunks(self, chunks: int):
        self.num_output_chunks = chunks
    def set_offload_context(self, ctx):
        self.offload_context = ctx
    def offload_embeddings(self, offload: bool = True):
        if offload:
            self._embedding_device = torch.device('cpu')
            print(f'Offloading tok_embeddings to {self._embedding_device.type} | Size: {(self.tok_embeddings.weight.nbytes / (1024 ** 2)):.3f} MiB')
        else:
            self._embedding_device = self.output.weight.device
        self.tok_embeddings = self.tok_embeddings.to(device=self._embedding_device)
    def forward(self, input_ids: Tensor, input_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None, ) -> Union[Tensor, list[Tensor]]:
        device = input_ids.device
        if input_ids.device != self._embedding_device:
            input_ids = input_ids.to(device=self._embedding_device)
        x = self.tok_embeddings(input_ids).to(device=device)
        if input_pos is None: input_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        freqs = self.rotary_emb(input_pos)
        with self.offload_context:
            for layer in self.layers:
                x = layer(x=x, freqs=freqs, attn_mask=mask)
            x = self.norm(x)
        if self.num_output_chunks > 1:
            return self.chunked_output(x)
        else: return self.output(x)