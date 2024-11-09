from typing import Any, Optional

from .convert_hf_keys import convert_hf_state_dict, has_hf_keys
from .diskcache import tensor_cache
from .safetensor_utils import get_state_dict_from_safetensors, save_as_safetensors

def str2bool(v: str) -> bool: return v.lower() in ("yes", "true", "t", "1")

def exists(x: Optional[Any]) -> bool: return x is not None