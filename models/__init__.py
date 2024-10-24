HF_KEY_MAPPING = {
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

def convert_hf_state_dict(state_dict: dict):
    from torchtune.models.convert_weights import get_mapped_key
    converted_state_dict = {}
    for k, v in state_dict.items():
        new_k = get_mapped_key(k, HF_KEY_MAPPING)
        converted_state_dict[new_k] = v
    return converted_state_dict