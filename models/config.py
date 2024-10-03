from typing import Optional

import yaml

class Cfg:
    def __init__(self, **kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r', encoding='utf-8') as file: data = yaml.safe_load(file)
        return cls(**data)

    def to_yaml(self, path: str):
        with open(path, 'w', encoding='utf-8') as file: yaml.dump(self.__dict__, file)


class ModelCfg(Cfg):
    hidden_size: int = 2048
    num_layers: int = 16
    max_seq_len = 2048

    # Misc
    rms_norm_eps: float = 1e-05
    rope_theta: float = 500000.
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    # MLP
    intermediate_size: int = 8192
    mlp_bias: bool = False

    # Attention
    num_kv_heads: int = 8
    num_heads: int = 32
    attn_qkv_bias: bool = False
    attn_out_bias: bool = False

    # Tokenizer
    vocab_size: int = 128256
    pad_token: Optional[int] = None

class TrainCfg(Cfg):
    num_batch: int = 1
    num_epochs: int = 1
    enable_checkpointing: bool = True

class InferenceCfg(Cfg):
    num_beams: int = 2
    bos_token: int = 128000
    eos_tokens: list[int] = [128001, 128008, 128009]
    pad_token: int = 128004
    precision: str = 'bf16'
    chat_format: str = 'llama3'
    top_p: float = .95
    top_k: int = 12
    temperature: float = 1.

class PeftCfg(Cfg):
    type: str = 'lora'
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    layers: list[str] = ['qkv_proj', 'o_proj', 'mlp', 'lm_head']