import torch.nn.functional as F
from torch import nn, Tensor

class TiedLinear:
    def __init__(self, tied_module: nn.Module):
        self.tied_module = tied_module
        if not hasattr(tied_module, "weight"): raise AttributeError("Provided module does not have attribute 'weight'.")
    @property
    def weight(self) -> Tensor: return self.tied_module.weight
    def __call__(self, x: Tensor) -> Tensor: return F.linear(x, self.tied_module.weight)
    def to(self, *args, **kwargs): self.tied_module.to(*args, **kwargs)