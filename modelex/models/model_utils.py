from modelex.models.llm.config import LLMConfig
from modelex.models.llm.model import LLM

MODEL_MAPPING = {
    LLMConfig: LLM,
}

def instantiate_model(config):
    model_class = MODEL_MAPPING.get(type(config))
    if not model_class: raise ValueError(f"No model mapping found for config type: {type(config).__name__}")
    return model_class(config)