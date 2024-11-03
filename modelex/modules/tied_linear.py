from torch import Tensor
from torchtune.modules import TiedLinear

class TiedLinear2(TiedLinear):
    @property
    def weight(self) -> Tensor: return self.tied_module.weight