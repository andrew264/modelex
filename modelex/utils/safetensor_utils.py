import time
from pathlib import Path
from typing import Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file

def get_state_dict_from_safetensors(path: str | list[str], device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.bfloat16) -> dict:
    state_dict = {}
    if isinstance(path, str): path = [path]
    if path:
        start = time.time()
        d = device.type if device.type == 'cpu' else 0
        for p in path:
            with safe_open(p, framework="pt", device=d) as f:
                for k in f.keys(): state_dict[k] = f.get_tensor(k).to(dtype=dtype)
        if device.type != 'cpu': torch.cuda.synchronize(device)
        print(f"Loaded weights from {path} in {time.time() - start:.3f}s.")
    else: print("No weights found.")
    return state_dict

def save_as_safetensors(state_dict: dict, path: Union[str, Path]) -> None:
    start = time.time()
    safe_save_file(state_dict, path)
    print(f"Saved state_dict to {path} in {time.time() - start:.3f}s.")