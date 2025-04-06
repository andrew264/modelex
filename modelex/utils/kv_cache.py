from typing import Tuple

import torch
from torch import nn, Tensor

class KVCache(nn.Module):
    def __init__(self, batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("cache_pos", torch.arange(0, cache_shape[2]), persistent=False)
        self.batch_size = batch_size

    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos -= self.size

    @property
    def size(self) -> int:
        return self.cache_pos[0].item()

    @torch.no_grad()
    def update(self, k_val: Tensor, v_val: Tensor) -> Tuple[Tensor, Tensor]:
        bsz, _, seq_len, _ = k_val.shape
        if bsz > self.k_cache.shape[0]:
            raise ValueError(f"The current cache has been setup with a batch size of {self.k_cache.shape[0]}"
                             f", but found new key tensors with batch size {k_val.shape[0]}!")

        assert (self.cache_pos[0] + seq_len) <= self.k_cache.shape[2], \
            (f"The current cache has been setup with a sequence length of {self.k_cache.shape[2]}"
             f", but the cache has reached a sequence length of {(self.size + seq_len)}!")

        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:, :, self.cache_pos[:seq_len]] = k_val
        v_out[:, :, self.cache_pos[:seq_len]] = v_val

        self.cache_pos.add_(seq_len)

        return k_out, v_out

    def forward(self, k_val: Tensor, v_val: Tensor) -> Tuple[Tensor, Tensor]: return self.update(k_val, v_val)