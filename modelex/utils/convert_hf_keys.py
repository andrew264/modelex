import re
from typing import Dict

from torch import Tensor

HF_KEY_MAPPING: Dict[str, str] = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.o_proj.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}

def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    try:
        # Checks if there is a layer # in the key
        if any(k.isdigit() for k in key.split(".")):
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise KeyError(f'Error converting the state dict. Found unexpected key: "{key}". '
                       "Please make sure you're loading a checkpoint with the right format. ") from e

    return new_key

def convert_hf_state_dict(state_dict: Dict) -> Dict[str, Tensor]:
    converted_state_dict = {}
    for k, v in state_dict.items():
        new_k = get_mapped_key(k, HF_KEY_MAPPING)
        converted_state_dict[new_k] = v
    return converted_state_dict

def has_hf_keys(state_dict: Dict[str, Tensor]) -> bool:
    return any(key.startswith('model.') for key in state_dict)