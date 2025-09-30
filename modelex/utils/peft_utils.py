from typing import Any, Dict, Optional, Set

import torch
from torch import nn, Tensor

def _get_lora_modules(state_dict: Dict[str, Any]) -> Set[str]:
    lora_keys = [k for k in state_dict.keys() if "lora" in k or "magnitude" in k]
    return set([k.replace(".lora_a.weight", "").replace(".lora_b.weight", "").replace(".magnitude", "") for k in lora_keys])

@torch.no_grad()
def get_merged_lora_ckpt(state_dict: Dict[str, Tensor], rank: int, alpha: float, ) -> Dict[str, Tensor]:
    lora_modules = _get_lora_modules(state_dict)
    for module in lora_modules:
        lora_a_weight: Tensor = state_dict[f"{module}.lora_a.weight"]
        lora_b_weight: Tensor = state_dict[f"{module}.lora_b.weight"]
        lora_magnitude: Optional[Tensor] = state_dict.get(f"{module}.magnitude", None)

        # If magnitude is present, calculate merged DoRA weight
        if lora_magnitude is not None:
            base_weight = state_dict[f"{module}.weight"].to(dtype=lora_a_weight.dtype)
            lora_weight = (alpha / rank) * lora_b_weight @ lora_a_weight
            merged_weight = base_weight + lora_weight
            weight_norm = torch.linalg.norm(base_weight + lora_weight, dim=1)
            mag_norm_scale = (lora_magnitude / weight_norm).view(-1, 1)
            merged_weight *= mag_norm_scale
            state_dict[f"{module}.weight"] = merged_weight
            del state_dict[f"{module}.magnitude"]
        else: # Otherwise it is just vanilla LoRA
            state_dict[f"{module}.weight"] = state_dict[f"{module}.weight"] + (alpha / rank) * lora_b_weight @ lora_a_weight
        del state_dict[f"{module}.lora_a.weight"]
        del state_dict[f"{module}.lora_b.weight"]

    return state_dict

def get_adapter_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    adapter_params = {}
    for k, v in model.named_modules():
        if hasattr(v, "adapter_params") and callable(v.adapter_params):
            current_adapter_params = v.adapter_params()
            for n, p in v.named_parameters(recurse=True):
                if n in current_adapter_params:
                    full_key = f"{k}.{n}" if k else n
                    adapter_params.update({full_key: p})
                    current_adapter_params.remove(n)
            assert current_adapter_params == [], f"Adapter params {current_adapter_params} not converted"
    return adapter_params

def set_trainable_params(model: nn.Module, adapter_params: Dict[str, Any]) -> None:
    for k, v in model.named_parameters(): v.requires_grad_(k in adapter_params)

def setup_model_for_peft(model: nn.Module, cfg) -> None:
    set_trainable_params(model, get_adapter_params(model))
    if cfg.peft.type == 'dora':  # This is for any adapters that need to be initialized after base weights have been loaded (e.g. DoRA).
        for m in model.modules():
            if hasattr(m, "initialize_dora_magnitude"): m.initialize_dora_magnitude()