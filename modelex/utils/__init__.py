from typing import Any, Optional

from torch import nn

from .convert_hf_keys import convert_hf_state_dict, has_hf_keys
from .diskcache import tensor_cache
from .safetensor_utils import get_state_dict_from_safetensors, save_as_safetensors

def str2bool(v: str) -> bool: return v.lower() in ("yes", "true", "t", "1")

def exists(x: Optional[Any]) -> bool: return x is not None

def model_summary(model: nn.Module):
    def fmt(num: int) -> str:
        if num >= 1e9: return f"{num / 1e9:.2f}B"
        if num >= 1e6: return f"{num / 1e6:.2f}M"
        if num >= 1e3: return f"{num / 1e3:.2f}K"
        return str(num)

    total_params = active_params = 0
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        active_params += (param != 0).sum().item()
    inactive_params = total_params - active_params

    print(f"Total Parameters: {fmt(total_params)}")
    print(f"Active Parameters: {fmt(active_params)} ({(active_params / total_params) * 100:.2f}%)")
    print(f"Inactive Parameters: {fmt(inactive_params)} ({(inactive_params / total_params) * 100:.2f}%)")