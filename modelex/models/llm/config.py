import yaml

class Cfg:
    def __init__(self, **kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
    def checks(self): return
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r', encoding='utf-8') as file: data = yaml.safe_load(file)
        print(f"Loaded {cls.__name__} from {path}")
        obj = cls(**data)
        cls.checks(obj)
        return obj

    def to_yaml(self, path: str):
        with open(path, 'w', encoding='utf-8') as file: yaml.dump(self.__dict__, file)
    def __str__(self): return '\n'.join(f'{k}: {v}' for k, v in self.__dict__.items())

class ModelCfg(Cfg):
    hidden_size: int = 2048
    num_layers: int = 16
    max_seq_len = 4096

    # Misc
    rms_norm_eps: float = 1e-05
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    # rope
    rope_base: float = 10000.

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

    @property
    def head_dim(self): return self.hidden_size // self.num_heads
    @property
    def max_position_embeddings(self): return self.max_seq_len
    @property
    def num_attention_heads(self): return self.num_heads

class InferenceCfg(Cfg):
    bos_token: int = 128000
    eos_tokens: list[int] = [128001, 128008, 128009]
    pad_token: int = 128004

    precision: str = 'bf16'
    chat_format: str = 'llama3'

    top_k: int = 12
    temperature: float = 1.

class PeftCfg(Cfg):
    type: str = 'lora'
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    layers: list[str] = ['qkv_proj', 'o_proj', 'mlp', 'output']
    quant_base: bool = False