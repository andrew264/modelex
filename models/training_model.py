from functools import partial
from typing import Any, Optional
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchtune.modules import get_cosine_schedule_with_warmup

from models.config import ModelCfg, PeftCfg
from models.layers.transformer_block import Transformer

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    FUSED_CE = True
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None
    FUSED_CE = False

class TiedLinear:
    def __init__(self, tied_module: nn.Module):
        self.tied_module = tied_module
        if not hasattr(tied_module, "weight"): raise AttributeError("Provided module does not have attribute 'weight'. Please check your tied_module.")
    def __call__(self, x: Tensor) -> Tensor: return F.linear(x, self.tied_module.weight)
    @property
    def weight(self) -> Tensor: return self.tied_module.weight

class LLMLit(L.LightningModule):
    def __init__(self, cfg: ModelCfg, peft_cfg: Optional[PeftCfg] = None, *args: Any, **kwargs: Any) -> None:
        use_grad_checkpointing: bool = kwargs.pop('use_grad_checkpointing', False)
        lr: float = kwargs.pop('lr', 1e-4)
        use_scheduler: bool = kwargs.pop('use_scheduler', True)
        warmup: int = kwargs.pop('warmup', 100)
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.peft_cfg = peft_cfg
        self.save_hyperparameters()
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.warmup = warmup

        self.tie_word_embeddings = cfg.tie_word_embeddings

        self.model = Transformer(cfg=cfg, peft_cfg=peft_cfg, use_grad_checkpointing=use_grad_checkpointing)
        if not self.tie_word_embeddings:
            if peft_cfg and 'lm_head' in peft_cfg.layers:
                if peft_cfg.type == 'dora': from torchtune.modules.peft import DoRALinear as Linear
                else: from torchtune.modules.peft import LoRALinear as Linear
                Linear = partial(Linear, rank=peft_cfg.rank, alpha=peft_cfg.alpha, dropout=peft_cfg.dropout)
                self.lm_head = Linear(in_dim=cfg.hidden_size, out_dim=cfg.vocab_size, use_bias=False)
            else:
                self.lm_head = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.vocab_size, bias=False)
        else:
            self.lm_head = TiedLinear(self.model.embed_tokens)
        self.apply(self._init_weights)

        if FUSED_CE and LigerFusedLinearCrossEntropyLoss:
            self.loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        else: self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self._opt_class = None

    def _init_weights(self, module):
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()

    def forward(self, x: Tensor, labels: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
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
    
    def set_optimizer(self, opt) -> None: self._opt_class = opt
    
    def configure_optimizers(self):
        if self._opt_class is None: raise ValueError('Pls set opt wit .set_optimizer()')
        optimizer = self._opt_class(self.parameters(), lr=self.lr, weight_decay=0.1, betas=(0.9, 0.95),)
        if self.use_scheduler:
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup, num_training_steps=self.trainer.estimated_stepping_batches)
            lr_scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1,}
            return { "optimizer": optimizer, "lr_scheduler": lr_scheduler_config,}
        return optimizer