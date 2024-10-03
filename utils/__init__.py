import os
import time
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file

def get_state_dict_from_safetensors(path: str | list[str], device: torch.device = torch.device('cpu')) -> Optional[dict]:
    state_dict = {}
    if isinstance(path, str): path = [path]
    if path:
        start = time.time()
        d = device.type if device.type == 'cpu' else device.index
        for p in path:
            with safe_open(p, framework="pt", device=d) as f:
                for k in f.keys(): state_dict[k] = f.get_tensor(k)
        if device.type != 'cpu': torch.cuda.synchronize(device)
        print(f"Loaded weights from {path} in {time.time() - start:.3f}s.")
    else: print("No weights found.")
    return state_dict if state_dict else None


def save_as_safetensors(state_dict: dict, path: str) -> None:
    start = time.time()
    safe_save_file(state_dict, path)
    print(f"Saved state_dict to {path} in {time.time() - start:.3f}s.")

def get_state_dict_from_deepspeed_ckpt(path: str) -> Optional[dict]:
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    return get_fp32_state_dict_from_zero_checkpoint(path)