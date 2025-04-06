from modelex.models.llm.config import LLMConfig


class GraniteConfig(LLMConfig):
    residual_multiplier: float = 1.0
    attention_multiplier: float =1.0
    logits_scaling: float = 1.0
    embedding_multiplier: float = 1.0