import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .linears import linear_factory

class MLP(nn.Module):
    def __init__(self, cfg) -> None:
        super(MLP, self).__init__()
        self.act_fn = F.silu
        hidden = cfg.hidden_size
        intermediate = cfg.intermediate_size
        bias = cfg.mlp_bias
        kwargs = {}
        if hasattr(cfg, 'peft') and cfg.peft and 'mlp' in cfg.peft.layers:
            kwargs = {'peft_type': cfg.peft.type, 'rank': cfg.peft.rank, 'alpha': cfg.peft.alpha, 'dropout': cfg.peft.dropout}

        self.w1 = linear_factory(in_features=hidden, out_features=intermediate, bias=bias, **kwargs)
        self.w3 = linear_factory(in_features=hidden, out_features=intermediate, bias=bias, **kwargs)
        self.w2 = linear_factory(in_features=intermediate, out_features=hidden, bias=bias, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act_fn(self.w1(x), inplace=True) * self.w3(x))