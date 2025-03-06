#!/usr/bin/env python
"""
Main training script for Large Language Models using the LLMTrainer.
Supports model loading, saving, and Parameter-Efficient Fine-Tuning (PEFT).
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Dict, Union

import torch

from modelex.models.registry import create_model
from modelex.training.trainer import LLMTrainer
from modelex.utils import (convert_hf_state_dict, get_state_dict_from_safetensors, has_hf_keys, save_as_safetensors)
from modelex.utils.peft_utils import get_adapter_params, setup_model_for_peft

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train or fine-tune a language model using LLMTrainer")
parser.add_argument("path", type=str, help="Path to the model directory containing config.yaml and trainer_config.yaml")
parser.add_argument("--resume", action="store_true", help="Resume training from existing optimizer state if available")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")


def remove_checkpoint_suffix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove activation checkpoint wrapper suffixes from model state dict keys.

    Args:
        state_dict: Model state dictionary

    Returns:
        Cleaned state dictionary
    """
    act_ckpt_wrapped_module = "._checkpoint_wrapped_module"
    return {k.replace(act_ckpt_wrapped_module, ''): v for k, v in state_dict.items()}

def load_model(model_path: Union[str, Path]) -> torch.nn.Module:
    """
    Load model from configuration and weights.

    Args:
        model_path: Path to model directory

    Returns:
        Initialized model
    """
    path = Path(model_path)
    config_path = path / 'config.yaml'

    logger.info("Loading model configuration from %s", config_path)
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

def save_model_weights(model: torch.nn.Module, output_path: Union[str, Path], is_peft: bool = False) -> None:
    """
    Save model weights to disk.

    Args:
        model: The model to save
        output_path: Directory where weights should be saved
        is_peft: Whether to save PEFT adapter weights only
    """
    path = Path(output_path)

    try:
        if is_peft:
            # Save only adapter parameters for PEFT
            logger.info("Saving PEFT adapter parameters")
            lora_params = remove_checkpoint_suffix(get_adapter_params(model))
            save_path = path / 'adapter.safetensors'
            save_as_safetensors(lora_params, save_path)
            logger.info("Saved adapter parameters to %s", save_path)
        else:
            # Save full model parameters
            logger.info("Saving full model parameters")
            model_params = remove_checkpoint_suffix(model.state_dict())
            save_path = path / 'model.safetensors'
            save_as_safetensors(model_params, save_path)
            logger.info("Saved model parameters to %s", save_path)
    except Exception as e:
        logger.error("Failed to save model weights: %s", str(e))
        raise

def save_optimizer_state(trainer: LLMTrainer, output_path: Union[str, Path]) -> None:
    """
    Save optimizer state to disk.

    Args:
        trainer: The LLMTrainer instance
        output_path: Directory where optimizer state should be saved
    """
    path = Path(output_path)
    opt_sd_file = path / 'optimizer.pt'

    try:
        torch.save(trainer.get_opt_state_dict(), opt_sd_file)
        logger.info("Saved optimizer state to %s", opt_sd_file)
    except Exception as e:
        logger.error("Failed to save optimizer state: %s", str(e))
        raise

def cleanup_memory() -> None:
    """Clean up memory to prevent leaks and OOM issues."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def main(args) -> None:
    """Main training function."""
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Set high precision for matmul operations
    torch.set_float32_matmul_precision('high')

    # Convert path to Path object for easier handling
    model_path = Path(args.path)

    if not model_path.exists():
        logger.error("Model path does not exist: %s", model_path)
        sys.exit(1)

    if not (model_path / 'config.yaml').exists():
        logger.error("Model configuration not found: %s", model_path / 'config.yaml')
        sys.exit(1)

    if not (model_path / 'trainer_config.yaml').exists():
        logger.error("Trainer configuration not found: %s", model_path / 'trainer_config.yaml')
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Starting training for model: %s", model_path)
    logger.info("=" * 80)

    try:
        # Load model
        model = load_model(model_path)

        # Setup PEFT if configured
        is_peft = setup_peft_if_needed(model)

        # Clean up memory before training
        cleanup_memory()

        # Initialize trainer
        trainer_config = model_path / 'trainer_config.yaml'
        logger.info("Initializing trainer with config: %s", trainer_config)
        trainer = LLMTrainer(model, trainer_config)

        # Resume from optimizer state if available and requested
        opt_sd_file = model_path / 'optimizer.pt'
        if args.resume and opt_sd_file.exists():
            logger.info("Resuming from optimizer state: %s", opt_sd_file)
            trainer.set_opt_state_dict(torch.load(opt_sd_file, weights_only=True))

        # Run training
        logger.info("Starting training")
        trainer.train()

        # Save model and optimizer state
        save_model_weights(model, model_path, is_peft=is_peft)
        save_optimizer_state(trainer, model_path)

        # Clean up resources
        trainer.close()
        logger.info("Training completed successfully")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        try:
            # Attempt to save current state
            save_model_weights(model, model_path, is_peft=is_peft)
            save_optimizer_state(trainer, model_path)
            trainer.close()
            logger.info("Saved current state before exit")
        except Exception as e:
            logger.error("Failed to save state on interrupt: %s", str(e))

    except Exception as e:
        logger.error("Training failed with error: %s", str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main(args=parser.parse_args())