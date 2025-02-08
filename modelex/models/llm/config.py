from pathlib import Path
from typing import List, Literal, Optional, Self, Union

import yaml
from pydantic import BaseModel, Field

class InferenceConfig(BaseModel):
    bos_token: int = 1
    eos_tokens: List[int] = Field(default_factory=list)
    pad_token: int = 0

    precision: Literal['bf16', 'fp16', 'fp32'] = 'bf16'
    chat_format: Literal['llama3', 'gemma2', 'custom', 'chatml'] = 'chatml'

    top_k: int = Field(12, ge=1)
    temperature: float = Field(1.0, ge=0.0)

class PeftConfig(BaseModel):
    type: Literal['lora', 'dora'] = 'lora'
    rank: int = Field(8, ge=1)
    alpha: int = Field(16, ge=1)
    dropout: float = Field(0.05, ge=0.0, le=1.0)
    layers: List[str] = Field(['attn', 'mlp', 'output'])

class LLMConfig(BaseModel):
    type: str = "LLM"
    num_layers: int = 16
    max_seq_len: int = 4096
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_kv_heads: int = 8
    num_heads: int = 32
    tie_word_embeddings: bool = False

    # Misc configurations
    rms_norm_eps: float = 1e-05
    initializer_range: float = 0.02
    rope_base: float = 10000.0

    # bias
    attn_qkv_bias: bool = False
    attn_out_bias: bool = False
    mlp_bias: bool = False

    inference: Optional[InferenceConfig] = Field(default_factory=InferenceConfig)
    peft: Optional[PeftConfig] = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def max_position_embeddings(self) -> int:
        return self.max_seq_len

    @property
    def num_attention_heads(self) -> int:
        return self.num_heads

    def save_config(self, path: Union[str, Path]) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(), f)

    @classmethod
    def load_config(cls, path: Union[str, Path]) -> Self:
        with open(path, encoding='utf-8') as f:
            return cls(**yaml.safe_load(f))