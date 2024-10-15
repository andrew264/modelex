from functools import partial
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.config import ModelCfg, PeftCfg, TrainCfg

class MLP(nn.Module):
    def __init__(self, cfg: ModelCfg, peft_cfg: Optional[PeftCfg], train_cfg: Optional[TrainCfg] = None) -> None:
        super(MLP, self).__init__()
        self.act_fn = F.silu
        if peft_cfg:
            if peft_cfg.type == 'dora': from torchtune.modules.peft import DoRALinear as Linear
            else: from torchtune.modules.peft import LoRALinear as Linear
            Linear = partial(Linear, rank=peft_cfg.rank, alpha=peft_cfg.alpha, dropout=peft_cfg.dropout, quantize_base=peft_cfg.quant_base)
            self.gate_proj = Linear(in_dim=cfg.hidden_size, out_dim=cfg.intermediate_size, use_bias=cfg.mlp_bias)
            self.up_proj = Linear(in_dim=cfg.hidden_size, out_dim=cfg.intermediate_size, use_bias=cfg.mlp_bias)
            self.down_proj = Linear(in_dim=cfg.intermediate_size, out_dim=cfg.hidden_size, use_bias=cfg.mlp_bias)
        else:
            self.gate_proj = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.intermediate_size, bias=cfg.mlp_bias)
            self.up_proj = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.intermediate_size, bias=cfg.mlp_bias)
            self.down_proj = nn.Linear(in_features=cfg.intermediate_size, out_features=cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x), inplace=True) * self.up_proj(x))