import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.config import ModelCfg

class MLP(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super(MLP, self).__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: Tensor) -> Tensor: return self.down_proj(F.silu(self.gate_proj(x), inplace=True) * self.up_proj(x))