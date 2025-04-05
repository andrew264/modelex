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

import torch

from modelex.training.trainer import LLMTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train or fine-tune a language model using LLMTrainer")
parser.add_argument("path", type=str, help="Path to the model directory containing config.yaml and trainer_config.yaml")
parser.add_argument("--resume", action="store_true", help="Resume training from existing optimizer state if available")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")


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

    trainer = None

    try:
        # Clean up memory before training
        cleanup_memory()

        # Initialize trainer
        logger.info("Initializing trainer with model path: %s", model_path)
        trainer = LLMTrainer(model_path)

        # Resume from optimizer state if available and requested
        opt_sd_file = model_path / 'optimizer.pt'
        if args.resume and opt_sd_file.exists():
            logger.info("Resuming from optimizer state: %s", opt_sd_file)
            trainer.set_opt_state_dict(torch.load(opt_sd_file, weights_only=True))

        # Run training
        logger.info("Starting training")
        trainer.train()

        # Save model and optimizer state
        trainer.save_model_weights()
        trainer.save_optimizer_state()

        # Clean up resources
        trainer.close()
        logger.info("Training completed successfully")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        try:
            # Attempt to save current state
            if trainer:
                trainer.save_model_weights()
                trainer.save_optimizer_state()
            trainer.close()
            logger.info("Saved current state before exit")
        except Exception as e:
            logger.error("Failed to save state on interrupt: %s", str(e))

    except Exception as e:
        logger.error("Training failed with error: %s", str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main(args=parser.parse_args())