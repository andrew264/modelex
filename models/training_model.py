from functools import partial
from typing import Any, Optional
import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from models.config import ModelCfg, PeftCfg
from models.layers.transformer_block import Transformer

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    FUSED_CE = True
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None
    FUSED_CE = False

class LLMLit(L.LightningModule):
    def __init__(self, cfg: ModelCfg, peft_cfg: Optional[PeftCfg] = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.peft_cfg = peft_cfg
        self.save_hyperparameters()

        checkpointing = kwargs.get('enable_checkpointing', False)
        self.tie_word_embeddings = cfg.tie_word_embeddings

        self.model = Transformer(cfg=cfg, peft_cfg=peft_cfg, enable_checkpointing=checkpointing)
        if peft_cfg:
            if peft_cfg.type == 'dora': from torchtune.modules.peft import DoRALinear as Linear
            else: from torchtune.modules.peft import LoRALinear as Linear
            Linear = partial(Linear, rank=peft_cfg.rank, alpha=peft_cfg.alpha, dropout=peft_cfg.dropout)
            self.lm_head = Linear(in_dim=cfg.hidden_size, out_dim=cfg.vocab_size, use_bias=False)
        else:
            self.lm_head = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

        if FUSED_CE and LigerFusedLinearCrossEntropyLoss:
            self.loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        else: self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def _init_weights(self, module):
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()

    def tie_weights(self):
        if self.tie_word_embeddings: self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self,x: Tensor, labels: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        x = self.model(x=x, attn_mask=attn_mask)
        labels = labels[..., 1:].contiguous().view(-1)
        if FUSED_CE:
            x = x[..., :-1, :].view(-1, self.cfg.hidden_size)
            loss = self.loss(self.lm_head.weight, x, labels)
        else:
            x = self.lm_head(x)
            x = x[..., :-1, :].view(-1, self.cfg.hidden_size)
            loss = self.loss(x, labels)
        return loss

    def training_step(self, batch: dict):
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")

        loss = self(input_ids, labels, attention_mask)

        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train_ppl", torch.exp(loss), on_step=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch: dict):
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")

        loss = self(input_ids, labels, attention_mask)

        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train_ppl", torch.exp(loss), on_step=True, prog_bar=True, sync_dist=True)
        return loss