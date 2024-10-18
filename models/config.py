import yaml

class Cfg:
    def __init__(self, **kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
    def checks(self): return
    @classmethod
    def from_yaml(cls, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as file: data = yaml.safe_load(file)
            print(f"Loaded {cls.__name__} from {path}")
            obj = cls(**data)
            cls.checks(obj)
            return obj
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
    accelerator: str = "gpu"

    offload_activations: bool = False
    use_grad_checkpointing: bool = False

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
    teacher_model: str = ""
    kll_loss_ratio: float = .5

    @property
    def is_online_kd(self) -> bool: return self.teacher_model and self.use_kd

    def checks(self):
        assert 0 <= self.kll_loss_ratio <= 1., "kll_loss_ratio must be between 0 and 1"
        assert 0 <= self.warmup_ratio <= 1., "warmup_ratio must be between 0 and 1"
        assert not(self.offload_activations and self.use_grad_checkpointing), "nuh uh; use either `offload_activations` or `use_grad_checkpointing`, not both at the same time"
        if self.accelerator == 'cpu':
            assert self.use_fused_ce is False, "can't do fused crossentropy in cpu"
            assert self.offload_activations is False, "can't offload activations when training with cpu"
            assert self.use_stage3 is False, "no stage3 for cpu training"
        if self.use_fused_ce:
            assert self.use_kd is False, "can't compute logits when using fused crossentropy, which is required for knowledge distillation"
            assert self.use_chunked_ce is False, "nuh uh; use either `use_fused_ce` or `use_chunked_ce`, not both at the same time"
        if self.use_chunked_ce:
            assert self.num_output_chunks > 1, "if you gonna use chunking you must also set the number of chunks"
            assert self.max_pad or self.pad_multiplier > 1, "uhm, make sure you pad sequence length properly before we chunk them into equal length"

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
    do_sample: bool = False

class PeftCfg(Cfg):
    type: str = 'lora'
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    layers: list[str] = ['qkv_proj', 'o_proj', 'mlp', 'lm_head']
    quant_base: bool = False