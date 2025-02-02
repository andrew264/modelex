import contextlib
import os
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from modelex.models.llm.config import LLMConfig
from modelex.modules import Block, RotaryEmbedding, TiedLinear
from modelex.utils import exists

class LLM(nn.Module):
    def __init__(self, cfg: Union[LLMConfig, dict, str], ) -> None:
        super(LLM, self).__init__()
        if isinstance(cfg, dict): cfg = LLMConfig(**cfg)
        elif isinstance(cfg, str): cfg = LLMConfig.load_config(os.path.join(cfg, 'models.yaml'))
        self.cfg = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.inference.pad_token)
        self.layers = nn.ModuleList([Block(cfg, idx) for idx in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg)

        if not cfg.tie_word_embeddings:
            if cfg.peft and 'output' in cfg.peft.layers:
                if cfg.peft.type == 'dora':
                    from torchtune.modules.peft import DoRALinear as Linear
                else:
                    from torchtune.modules.peft import LoRALinear as Linear
                Linear = partial(Linear, rank=cfg.peft.rank, alpha=cfg.peft.alpha, dropout=cfg.peft.dropout)
                self.output = Linear(in_dim=cfg.hidden_size, out_dim=cfg.vocab_size, use_bias=False)
            else:
                self.output = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.vocab_size, bias=False)
        else:
            self.output = TiedLinear(self.tok_embeddings)

        self.loss_fn = nn.CrossEntropyLoss()
        self._cache_setup_complete = False
        self.num_output_chunks = 1
        self.offload_context = contextlib.nullcontext()
        self._embedding_device = None
        self.ignore_labels_cache = None
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
        try:
            from torchtune.modules.loss import CEWithChunkedOutputLoss, ForwardKLWithChunkedOutputLoss
            self.num_output_chunks = chunks
            self.loss_fn = CEWithChunkedOutputLoss(num_output_chunks=chunks)
            self.kd_loss_fn = ForwardKLWithChunkedOutputLoss(num_output_chunks=chunks)
        except ImportError as e:
            print("Please install torchtune to set output_chunks!")

    def set_offload_context(self, ctx):
        self.offload_context = ctx
    def offload_embeddings(self, offload: bool = True):
        if offload:
            self._embedding_device = torch.device('cpu')
            print(f'Offloading tok_embeddings to {self._embedding_device.type} | Size: {(self.tok_embeddings.weight.nbytes / (1024 ** 2)):.3f} MiB')
        else:
            self._embedding_device = self.output.weight.device
        self.tok_embeddings = self.tok_embeddings.to(device=self._embedding_device)
    def calc_loss(self, logits: Union[Tensor, list[Tensor]], labels: Tensor) -> Tensor:
        if isinstance(logits, Tensor):
            return self.loss_fn(logits.contiguous().view(-1, self.cfg.vocab_size), labels.view(-1))
        return self.loss_fn(logits, labels)
    def calc_kd_loss(self, teacher_logits: Tensor, logits: Union[list[Tensor], Tensor], labels: Tensor) -> Tensor:
        teacher_device = teacher_logits.device
        if isinstance(logits, list): logits = [chunk.to(device=teacher_device) for chunk in logits]
        else: logits = logits.to(device=teacher_device)
        labels = labels.to(device=teacher_device)
        try:
            from torchtune.modules.loss import ForwardKLLoss
        except ImportError as e:
            print("Please install torchtune to do knowledge distillation!")
            raise e
        kd_loss_fn = ForwardKLLoss()
        return kd_loss_fn(logits, teacher_logits, labels)
    def forward(self, input_ids: Tensor, input_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None, labels: Optional[Tensor] = None,
                teacher_logits: Optional[Tensor] = None, **kwargs) -> dict:
        device = input_ids.device
        loss, kd_loss = None, None
        if input_ids.device != self._embedding_device:
            input_ids = input_ids.to(device=self._embedding_device)
        x = self.tok_embeddings(input_ids).to(device=device)
        if input_pos is None: input_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        freqs = self.rotary_emb(input_pos)
        with self.offload_context:
            for layer in self.layers:
                x = layer(x=x, freqs=freqs, attn_mask=mask)
            x = self.norm(x)
        if self.output.weight.device != x.device: x = x.to(self.output.weight.device)
        if self.num_output_chunks > 1: logits = self.chunked_output(x)
        else: logits = self.output(x)
        if exists(labels):
            if not exists(self.ignore_labels_cache):
                batch_size = labels.size(0)
                self.ignore_labels_cache = torch.full((batch_size, 1), -100, device=labels.device)
            labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[:labels.shape[0]])).contiguous()
            loss = self.calc_loss(logits, labels)
        if exists(teacher_logits): kd_loss = self.calc_kd_loss(teacher_logits, logits, labels)
        return dict(logits=logits, loss=loss, kd_loss=kd_loss)