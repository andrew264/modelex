from functools import partial
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.config import ModelCfg, PeftCfg, TrainCfg

class MLP(nn.Module):
    def __init__(self, cfg: ModelCfg, peft_cfg: Optional[PeftCfg] = None, train_cfg: Optional[TrainCfg] = None) -> None:
        super(MLP, self).__init__()
        self.act_fn = F.silu
        hidden = cfg.hidden_size
        intermediate = cfg.intermediate_size
        bias = cfg.mlp_bias
        if peft_cfg:
            if peft_cfg.type == 'dora':
                from torchtune.modules.peft import DoRALinear as Linear
            else:
                from torchtune.modules.peft import LoRALinear as Linear
            Linear = partial(Linear, rank=peft_cfg.rank, alpha=peft_cfg.alpha, dropout=peft_cfg.dropout, quantize_base=peft_cfg.quant_base)

            self.w1 = Linear(in_dim=hidden, out_dim=intermediate, use_bias=bias)
            self.w3 = Linear(in_dim=hidden, out_dim=intermediate, use_bias=bias)
            self.w2 = Linear(in_dim=intermediate, out_dim=hidden, use_bias=bias)
        else:
            self.w1 = nn.Linear(hidden, intermediate, bias=bias)
            self.w3 = nn.Linear(hidden, intermediate, bias=bias)
            self.w2 = nn.Linear(intermediate, hidden, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act_fn(self.w1(x), inplace=True) * self.w3(x))