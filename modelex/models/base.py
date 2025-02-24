from abc import ABC, abstractmethod
from typing import Optional, Self

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch import Tensor

class BaseLLM(nn.Module, ABC):
    def __init__(self, cfg: BaseModel):
        super().__init__()
        self.cfg = cfg
        self._cache_setup_complete: bool = False
    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: Optional[int] = None, ):
        if self._cache_setup_complete: return
        self._cache_setup_complete = True
    def reset_cache(self):
        self._cache_setup_complete = True
    def caches_are_enabled(self): return self._cache_setup_complete
    @property
    def decoder_max_cache_seq_len(self): return self.cfg.max_seq_len
    def _init_weights(self, module):
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    def get_config(self) -> BaseModel:
        return self.cfg

    @classmethod
    @abstractmethod
    def from_config(cls, config: BaseModel) -> Self:
        ...

    def forward(self, **kwargs) -> Tensor:
        raise NotImplementedError(f'forward method is not implemented for {self.__class__.__name__}')

    def train_step(self, **kwargs) -> dict:
        raise NotImplementedError(f'train_step is not implemented for {self.__class__.__name__}')
