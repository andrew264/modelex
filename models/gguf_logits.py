import os

import torch
from torch import Tensor

import llama_cpp

class GGUFModelLogits:
    def __init__(self, name: str, n_ctx: int = 8192):
        kwargs = dict(logits_all=True, n_ctx=n_ctx, offload_kqv=False, n_threads=os.cpu_count() - 1, verbose=False)
        if name.endswith('.gguf'):
            self.model = llama_cpp.Llama(model_path=name, **kwargs)
        else:
            self.model = llama_cpp.Llama.from_pretrained(repo_id=name, filename="*Q8*.gguf", **kwargs)

    def _get_logits(self, tokens: Tensor) -> Tensor:
        self.model.generate(tokens.tolist(), top_k=-1, top_p=1, temp=0.).__next__()
        logits = torch.tensor(self.model.eval_logits, dtype=torch.float32)
        self.model.reset()
        return logits

    def __call__(self, tokens: Tensor) -> Tensor:
        if tokens.ndim == 1: tokens.unsqueeze_(0)
        return torch.stack([self._get_logits(item) for item in tokens])