from typing import Optional

import yaml

class Cfg:
    def __init__(self, **kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)

    @classmethod
    def from_yaml(cls, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as file: data = yaml.safe_load(file)
            print(f"Loaded {cls.__name__} from {path}")
            return cls(**data)
        except FileNotFoundError: return None

    def to_yaml(self, path: str):
        with open(path, 'w', encoding='utf-8') as file: yaml.dump(self.__dict__, file)
    def __str__(self): return '\n'.join(f'{k}: {v}' for k, v in self.__dict__.items())

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
    vocab_size: int = 32000
    pad_token: int = 0

class TrainCfg(Cfg):
    num_epochs: int = 1
    batch_size: int = 1
    num_accum_steps: int = 1
    precision: str = "bf16"
    use_grad_checkpointing: bool = False
    accelerator: str = "gpu"

    max_pad: bool = False
    pad_multiplier: int = 1

    use_fused_ce: bool = False
    use_chunked_ce: bool = False
    num_output_chunks: int = 1

    learning_rate: float = 5e-5
    use_scheduler: bool = True
    warmup_ratio: float = .1

    use_stage3: bool = False

    use_kd: bool = False
    kll_loss_ratio: float = .5

class InferenceCfg(Cfg):
    bos_token: int = 128000
    eos_tokens: list[int] = [128001, 128008, 128009]
    pad_token: int = 128004

    precision: str = 'bf16'
    chat_format: str = 'llama3'

    num_beams: int = 2
    top_p: float = .95
    top_k: int = 12
    temperature: float = 1.

class PeftCfg(Cfg):
    type: str = 'lora'
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    layers: list[str] = ['qkv_proj', 'o_proj', 'mlp', 'lm_head']
    quant_base: bool = False