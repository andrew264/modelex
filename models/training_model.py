from functools import partial
from typing import Any, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torchtune.modules import get_cosine_schedule_with_warmup, TiedLinear

from models.config import ModelCfg, PeftCfg, TrainCfg
from models.layers.transformer_block import Transformer

class TiedLinear2(TiedLinear):
    @property
    def weight(self) -> Tensor: return self.tied_module.weight

class LLMLit(L.LightningModule):
    def __init__(self, cfg: ModelCfg, *args: Any, **kwargs: Any) -> None:
        peft_cfg: Optional[PeftCfg] = kwargs.pop('peft_cfg', None)
        train_cfg: TrainCfg = kwargs.pop('train_cfg')
        super().__init__(*args, **kwargs)
        if train_cfg.use_fused_ce and train_cfg.use_chunked_ce: raise ValueError('nuh uh; use either fused_ce or chunked_ce')
        self.cfg = cfg
        self.peft_cfg = peft_cfg
        self.train_cfg = train_cfg
        self.save_hyperparameters()
        self.use_fused_ce = train_cfg.use_fused_ce
        self.use_chunked_ce = train_cfg.use_chunked_ce
        self.num_output_chunks = train_cfg.num_output_chunks

        self.tie_word_embeddings = cfg.tie_word_embeddings

        self.model = Transformer(cfg=cfg, peft_cfg=peft_cfg, train_cfg=train_cfg)
        if not self.tie_word_embeddings:
            if peft_cfg and 'lm_head' in peft_cfg.layers:
                if peft_cfg.type == 'dora': from torchtune.modules.peft import DoRALinear as Linear
                else: from torchtune.modules.peft import LoRALinear as Linear
                Linear = partial(Linear, rank=peft_cfg.rank, alpha=peft_cfg.alpha, dropout=peft_cfg.dropout, quantize_base=peft_cfg.quant_base)
                self.lm_head = Linear(in_dim=cfg.hidden_size, out_dim=cfg.vocab_size, use_bias=False)
            else:
                self.lm_head = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.vocab_size, bias=False)
        else:
            self.lm_head = TiedLinear2(self.model.embed_tokens)
        # self.apply(self._init_weights)

        self._setup_loss_fn()
        self.ignore_labels_cache = torch.full((train_cfg.batch_size, 1), self.loss.ignore_index, device=self.device)

    def _init_weights(self, module):
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()

    def _setup_loss_fn(self):
        train_cfg = self.train_cfg
        if train_cfg.use_fused_ce:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            self.loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        elif train_cfg.use_chunked_ce:
            from torchtune.modules.loss import CEWithChunkedOutputLoss
            loss = CEWithChunkedOutputLoss(ignore_index=-100, num_output_chunks=train_cfg.num_output_chunks)
            loss.compute_cross_entropy = torch.compile(loss.compute_cross_entropy)
            self.loss = loss
        else: self.loss = nn.CrossEntropyLoss(ignore_index=-100)

        if train_cfg.use_kd:
            if train_cfg.use_chunked_ce:
                from torchtune.modules.loss import ForwardKLWithChunkedOutputLoss
                self.kld_loss = ForwardKLWithChunkedOutputLoss(num_output_chunks=train_cfg.num_output_chunks, ignore_index=-100)
            else:
                from torchtune.modules.loss import ForwardKLLoss
                self.kld_loss = ForwardKLLoss(ignore_index=-100)

    def _loss_fn(self, x: Tensor, labels: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        if self.use_fused_ce:
            x = x.contiguous().view(-1, self.cfg.hidden_size)
            loss = self.loss(self.lm_head.weight, x, labels.view(-1))
            logits = None
        elif self.use_chunked_ce:
            logits = [self.lm_head(chunk) for chunk in x.chunk(self.train_cfg.num_output_chunks, dim=1)]
            loss = self.loss(logits, labels)
        else:
            logits = self.lm_head(x)
            loss = self.loss(logits.contiguous().view(-1, self.cfg.vocab_size), labels.view(-1))
        return logits, loss

    def _kd_loss_fn(self, logits: Optional[Union[Tensor, List[Tensor]]], teacher_logits: Optional[Tensor], labels: Tensor) -> Tensor:
        if logits is None:
            raise ValueError('Student Logits are missing for Knowledge Distillation, Logits are not materialized for Fused CrossEntropy')
        if teacher_logits is None:
            raise ValueError('Teacher Logits not Found')
        if self.use_chunked_ce:
            chunk_size = self.train_cfg.num_output_chunks
            teacher_logits = [chunk for chunk in teacher_logits.chunk(chunk_size, dim=1)]
            if not isinstance(logits, list):
                logits = [chunk for chunk in logits.chunk(chunk_size)]
            loss = self.kld_loss(logits, teacher_logits, labels)
        else:
            loss = self.kld_loss(logits, teacher_logits, labels)
        return loss

    def forward(self, x: Tensor, labels: Tensor, attn_mask: Optional[Tensor] = None, teacher_logits: Optional[Tensor] = None, ) -> Tensor:
        x = self.model(x=x, attn_mask=attn_mask)
        labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])).contiguous()
        logits, class_loss = self._loss_fn(x, labels)
        if self.train_cfg.use_kd:
            kd_loss = self._kd_loss_fn(logits, teacher_logits, labels)
            loss = (1 - self.train_cfg.kll_loss_ratio) * class_loss + self.train_cfg.kll_loss_ratio * kd_loss
        else:
            loss = class_loss
        del teacher_logits
        return loss

    def training_step(self, batch: dict):
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask", None)
        teacher_logits = batch.get("teacher_logits", None)

        loss = self(input_ids, labels, attention_mask, teacher_logits)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_ppl", torch.exp(loss), prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict):
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask", None)
        teacher_logits = batch.get("teacher_logits", None)

        loss = self(input_ids, labels, attention_mask, teacher_logits)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_ppl", torch.exp(loss), prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.train_cfg.accelerator == 'cpu':
            from torch.optim.adamw import AdamW as Optimizer
        elif self.train_cfg.use_stage3:
            from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
            Optimizer = partial(DeepSpeedCPUAdam, adamw_mode=True, fp32_optimizer_states=True)
        else:
            from bitsandbytes.optim.adamw import PagedAdamW8bit as Optimizer

        optimizer = Optimizer(self.parameters(), lr=self.train_cfg.learning_rate, weight_decay=0.1, betas=(0.9, 0.95), )
        if self.train_cfg.use_scheduler:
            steps = self.trainer.estimated_stepping_batches
            warmup = int(self.train_cfg.warmup_ratio * steps)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=steps)
            lr_scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1, }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config, }
        return optimizer

class CustomCallback(L.Callback):
    def setup(self, trainer: L.Trainer, pl_module: LLMLit, stage: str) -> None:
        pl_module.ignore_labels_cache = pl_module.ignore_labels_cache.to(pl_module.device)