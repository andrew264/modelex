from functools import partial
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.config import ModelCfg, PeftCfg

try:
    from liger_kernel.transformers.functional import liger_swiglu
except ImportError:
    liger_swiglu = None


class MLP(nn.Module):
    def __init__(self, cfg: ModelCfg, peft_cfg: Optional[PeftCfg]) -> None:
        super(MLP, self).__init__()
        if peft_cfg:
            if peft_cfg.type == 'dora': from torchtune.modules.peft import DoRALinear as Linear
            else: from torchtune.modules.peft import LoRALinear as Linear
            Linear = partial(Linear, rank=peft_cfg.rank, alpha=peft_cfg.alpha, dropout=peft_cfg.dropout)
            self.gate_proj = Linear(in_dim=cfg.hidden_size, out_dim=cfg.intermediate_size, use_bias=cfg.mlp_bias)
            self.up_proj = Linear(in_dim=cfg.hidden_size, out_dim=cfg.intermediate_size, use_bias=cfg.mlp_bias)
            self.down_proj = Linear(in_dim=cfg.intermediate_size, out_dim=cfg.hidden_size, use_bias=cfg.mlp_bias)
        else:
            self.gate_proj = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.intermediate_size, bias=cfg.mlp_bias)
            self.up_proj = nn.Linear(in_features=cfg.hidden_size, out_features=cfg.intermediate_size, bias=cfg.mlp_bias)
            self.down_proj = nn.Linear(in_features=cfg.intermediate_size, out_features=cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        if liger_swiglu: return self.down_proj(liger_swiglu(self.gate_proj(x), self.up_proj(x)))
        return self.down_proj(F.silu(self.gate_proj(x), inplace=True) * self.up_proj(x))