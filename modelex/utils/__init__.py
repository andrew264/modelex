import contextlib
from typing import Any, Generator, Optional

import torch
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
        if param.requires_grad:
            active_params += num_params
    inactive_params = total_params - active_params

    print(f"Total Parameters: {fmt(total_params)}")
    print(f"Active Parameters: {fmt(active_params)} ({(active_params / total_params) * 100:.2f}%)")
    print(f"Inactive Parameters: {fmt(inactive_params)} ({(inactive_params / total_params) * 100:.2f}%)")

@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)

def get_torch_dtype(dtype: str) -> torch.dtype:
    """
    Converts a string representation of a data type to a torch.dtype.

    Args:
        dtype (str): The string representation of the data type.

    Returns:
        torch.dtype: The corresponding torch.dtype.
    """
    if dtype == 'float32':
        return torch.float32
    elif dtype == 'float16':
        return torch.float16
    elif dtype == 'bfloat16':
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid dtype: {dtype}")
