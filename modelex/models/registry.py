from pathlib import Path
from typing import Dict, Type
import yaml
import importlib

from modelex.models.base import BaseLLM

MODEL_REGISTRY: Dict[str, Type[BaseLLM]] = {}

def register_model(name: str):
    """Decorator to register model classes."""
    def decorator(cls: Type[BaseLLM]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model {name} already registered!")
        if not issubclass(cls, BaseLLM):
            raise ValueError(f"Class {cls.__name__} must inherit from BaseLLM")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def create_model(config_path: str | Path) -> BaseLLM:
    """
    Creates a model instance from a configuration file.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    model_type = config_data.get("type")
    if not model_type:
        raise ValueError("Config file must contain a 'type' field specifying the model type.")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_type}' not registered.  Available models: {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[model_type]
    module_path, class_name = model_cls.__module__, model_cls.__name__
    module = importlib.import_module(module_path)
    cfg_cls = getattr(module, f'{class_name}Config')
    cfg = cfg_cls.model_validate(config_data)

    return model_cls.from_config(cfg)