from torch import Tensor
from torchtune.modules import TiedLinear

class TiedLinear2(TiedLinear):
    @property
    def weight(self) -> Tensor: return self.tied_module.weight
    def to(self, *args, **kwargs):
        self.tied_module.to(*args, **kwargs)