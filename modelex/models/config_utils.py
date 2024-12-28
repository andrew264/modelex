from typing import Dict, Type

import yaml
from pydantic import BaseModel

from modelex.models.llm.config import LLMConfig

CONFIG_MAPPING: Dict[str, Type[BaseModel]] = {
    "LLM": LLMConfig,
}

def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    config_type = config_data.get('type')
    if not config_type: raise ValueError("Config file must contain a 'type' field.")
    if config_type not in CONFIG_MAPPING.keys(): raise ValueError(f"Unknown config type: {config_type}")

    return CONFIG_MAPPING[config_type].load_config(config_path)