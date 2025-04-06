import logging
from pathlib import Path
from typing import Union

import torch

from modelex.models.registry import create_model
from modelex.utils import convert_hf_state_dict, get_state_dict_from_safetensors, has_hf_keys, set_default_dtype
from modelex.utils.peft_utils import setup_model_for_peft

logger = logging.getLogger(__name__)

def load_model(model_path: Union[str, Path], device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """
    Load model from configuration and weights.

    Args:
        model_path: Path to model directory
        device: Device to load model onto
        dtype: Data type for model weights



    Returns:
        Initialized model
    """
    path = Path(model_path)
    config_path = path / 'config.yaml'

    logger.info("Loading model configuration from %s", config_path)
    with device, set_default_dtype(dtype):
        model = create_model(config_path)

    # Find model weights files
    model_files = sorted(path.glob('model*.safetensors'))
    model_files = [p.absolute() for p in model_files]

    if not model_files:
        logger.warning("No model weights found, initializing with random weights")
        model.apply(model._init_weights)
        return model

    logger.info("Loading model weights from: %s", ", ".join(str(f) for f in model_files))
    model_sd = get_state_dict_from_safetensors(model_files)

    if not model_sd:
        logger.warning("Failed to load model weights, initializing with random weights")
        model.apply(model._init_weights)
        return model

    # Convert from HuggingFace format if needed
    if has_hf_keys(model_sd):
        logger.info("Converting state dict from HuggingFace format")
        model_sd = convert_hf_state_dict(model_sd)

    # Load weights
    missing, unexpected = model.load_state_dict(model_sd, strict=False)

    if missing:
        logger.warning("Missing keys in state dict: %s", missing[:10])
        if len(missing) > 10:
            logger.warning("... and %d more", len(missing) - 10)

    if unexpected:
        logger.warning("Unexpected keys in state dict: %s", unexpected[:10])
        if len(unexpected) > 10:
            logger.warning("... and %d more", len(unexpected) - 10)

    return model

def setup_peft_if_needed(model: torch.nn.Module) -> bool:
    """
    Set up Parameter-Efficient Fine-Tuning (PEFT) if configured.

    Args:
        model: The model to set up for PEFT

    Returns:
        True if PEFT was configured, False otherwise
    """
    cfg = model.get_config()

    if hasattr(cfg, 'peft') and cfg.peft:
        logger.info("Setting up model for Parameter-Efficient Fine-Tuning (PEFT)")
        setup_model_for_peft(model, cfg)
        return True

    return False