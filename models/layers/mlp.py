import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.config import ModelCfg

try:
    from liger_kernel.transformers.functional import liger_swiglu
except ImportError:
    liger_swiglu = None


class MLP(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super(MLP, self).__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        if liger_swiglu: return self.down_proj(liger_swiglu(self.gate_proj(x), self.up_proj(x)))
        return self.down_proj(F.silu(self.gate_proj(x), inplace=True) * self.up_proj(x))