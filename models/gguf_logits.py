import os
from typing import Union

import llama_cpp
import torch
from torch import Tensor

from utils.diskcache import tensor_cache

class GGUFModelLogits:
    def __init__(self, name: str, n_ctx: int = 8192, device: str = 'cpu'):
        n_gpu_layers = 0
        if device.startswith('cuda') or device == 'gpu':
            n_gpu_layers = -1
        kwargs = dict(n_gpu_layers=n_gpu_layers, logits_all=True, n_ctx=n_ctx, offload_kqv=False, n_threads=os.cpu_count() - 1, verbose=False)
        if name.endswith('.gguf'):
            self.model = llama_cpp.Llama(model_path=name, **kwargs)
        else:
            self.model = llama_cpp.Llama.from_pretrained(repo_id=name, filename="*Q8*.gguf", **kwargs)

    @tensor_cache(50)
    def _get_logits(self, tokens: list[int]) -> Tensor:
        self.model.generate(tokens, top_k=-1, top_p=1, temp=0.).__next__()
        logits = torch.tensor(self.model.eval_logits, dtype=torch.float32)
        self.model.reset()
        return logits

    def __call__(self, tokens: Union[Tensor, list[Union[list[int], int]]]) -> Tensor:
        if isinstance(tokens, Tensor): tokens = tokens.tolist()
        if isinstance(tokens[0], int): tokens = [tokens]
        return torch.stack([self._get_logits(item) for item in tokens])