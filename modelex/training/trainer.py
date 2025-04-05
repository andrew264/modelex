import enum
import importlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtune.training import get_memory_stats, set_activation_checkpointing
from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup
from torchtune.utils import batch_to_device
from tqdm import tqdm

from modelex.models.base import BaseLLM
from modelex.training.trainer_config import TrainerConfig
from modelex.utils import exists, get_torch_dtype, model_summary, save_as_safetensors
from modelex.utils.load import load_model, setup_peft_if_needed
from modelex.utils.peft_utils import get_adapter_params

logger = logging.getLogger(__name__)

class LogPrefix(enum.StrEnum):
    """Prefixes for logging categories."""
    TRAIN = 'train/'
    VALIDATION = 'valid/'

def get_instance(class_path: str) -> Any:
    """
    Dynamically import and return a class instance from a fully qualified path.

    Args:
        class_path: Fully qualified class path (e.g., 'module.submodule.ClassName')

    Returns:
        The class reference
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {class_path}: {e}")

class LLMTrainer:
    """
    Trainer for Large Language Models with support for mixed precision, activation checkpointing,
    and comprehensive logging.
    """

    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize the LLM trainer.

        Args:
            model_path: Path to the model directory containing config.yaml and trainer_config.yaml
        """
        self.model_path = model_path
        # Load configuration
        trainer_config = model_path / 'trainer_config.yaml'
        self.config = TrainerConfig.load_config(trainer_config)

        self.model: Optional[BaseLLM] = None
        self._is_peft: bool = False

        self.optimizer: Optional[Optimizer] = None
        self.lr_scheduler: Optional[LambdaLR] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.valid_dataloader: Optional[DataLoader] = None
        self.writer: Optional[SummaryWriter] = None
        self.device: Optional[torch.device] = None
        self.dtype: torch.dtype = torch.bfloat16

        # Tracking variables
        self.global_steps: Dict[LogPrefix, int] = {p: 0 for p in LogPrefix}
        self.items_per_epoch: int = 0
        self.log_dir: Optional[Path] = None

        # Set up all components
        self._setup()

    def _setup(self) -> None:
        """Initialize all trainer components."""
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logger()
        self._setup_misc()

    @property
    def is_peft(self) -> bool:
        return self._is_peft

    def _setup_model(self) -> None:
        """Set up the model including device placement and compilation if enabled."""
        # Configure device and move model
        self.device = torch.device(self.config.training.device)
        self.dtype = get_torch_dtype(self.config.training.dtype)
        self.model = load_model(self.model_path, self.device, self.dtype)
        self._is_peft = setup_peft_if_needed(self.model)
        self.model.to(device=self.device)

        # Compile model
        if self.config.training.compile:
            try:
                self.model.forward = torch.compile(self.model.forward, mode='max-autotune')
                logger.info("Model successfully compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Warning: Failed to compile model: {str(e)}")

    def _setup_optimizer(self) -> None:
        """Initialize the optimizer based on config."""
        opt_config = self.config.training.optimizer
        try:
            opt_class = opt_config.get_instance()
            # Optimize parameters that require gradients
            params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]

            if not params_to_optimize:
                raise ValueError("No parameters require gradients. Check model configuration.")

            self.optimizer = opt_class(params_to_optimize, **opt_config.params)
            logger.info(f"Using optimizer: {opt_class.__name__}")
        except Exception as e:
            raise RuntimeError(f"Failed to create optimizer: {str(e)}")

    def _setup_scheduler(self) -> None:
        """Initialize the learning rate scheduler if specified in config."""
        scheduler_config = self.config.training.scheduler
        if not exists(scheduler_config):
            logger.info("No learning rate scheduler configured")
            return

        try:
            scheduler_class = scheduler_config.get_instance()
            params = scheduler_config.params.copy()

            # Special handling for cosine scheduler
            if scheduler_class is get_cosine_schedule_with_warmup:
                num_warmup_steps = int(scheduler_config.warmup_ratio * self.total_steps)
                params.update({"num_warmup_steps": num_warmup_steps, "num_training_steps": self.total_steps})
                logger.info(f"Cosine scheduler with {num_warmup_steps} warmup steps of {self.total_steps} total steps")

            if self.optimizer is None:
                raise ValueError("Optimizer must be initialized before scheduler")

            self.lr_scheduler = scheduler_class(self.optimizer, **params)
            logger.info(f"Using learning rate scheduler: {scheduler_class.__name__}")
        except Exception as e:
            raise RuntimeError(f"Failed to create scheduler: {str(e)}")

    def _setup_data(self) -> None:
        """Set up training and validation data loaders."""
        data_config = self.config.data
        batch_size = self.config.training.batch_size

        # Common dataloader config
        dataloader_cfg = {"pin_memory": data_config.pin_memory, "batch_size": batch_size, "num_workers": data_config.num_workers, "drop_last": True}

        # Add collate function
        if exists(data_config.collate_fn):
            try:
                collate_fn_class = data_config.collate_fn.get_instance()
                collate_fn = collate_fn_class(**data_config.collate_fn.params)
                dataloader_cfg["collate_fn"] = collate_fn
            except Exception as e:
                raise RuntimeError(f"Failed to initialize collate function: {str(e)}")

        # Set up training dataset
        try:
            train_ds_class = data_config.train_dataset.get_instance()
            train_dataset = train_ds_class(**data_config.train_dataset.params)
            self.items_per_epoch = len(train_dataset)
            self.train_dataloader = DataLoader(train_dataset, **dataloader_cfg)
            logger.info(f"Training dataset: {len(train_dataset)} samples, {self.steps_per_epoch} steps per epoch")
        except Exception as e:
            raise RuntimeError(f"Failed to create training dataset: {str(e)}")

        # Set up validation dataset
        if exists(data_config.valid_dataset):
            try:
                valid_ds_class = data_config.valid_dataset.get_instance()
                valid_dataset = valid_ds_class(**data_config.valid_dataset.params)
                self.valid_dataloader = DataLoader(valid_dataset, **dataloader_cfg)
                logger.info(f"Validation dataset: {len(valid_dataset)} samples")
            except Exception as e:
                logger.warning(f"Warning: Failed to create validation dataset: {str(e)}")
                self.valid_dataloader = None

    def _setup_logger(self) -> None:
        """Set up TensorBoard logging."""
        if not exists(self.config.logging):
            logger.warning("Logging not configured")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = Path(self.config.logging.tensorboard_dir) / timestamp
            self.log_dir.mkdir(parents=True, exist_ok=True)

            config_path = self.log_dir / "trainer_config.yaml"
            self.config.save_config(config_path)
            logger.info(f"Saved configuration to {config_path}")

            self.writer = SummaryWriter(str(self.log_dir))
            logger.info(f"TensorBoard logs will be saved to {self.log_dir}")
        except Exception as e:
            logger.warning(f"Warning: Failed to set up logging: {str(e)}")
            self.writer = None

    def _setup_misc(self) -> None:
        """Set up miscellaneous training features like gradient checkpointing."""
        if self.config.training.checkpointing_layers:
            try:
                checkpointing_layers = {get_instance(layer_path) for layer_path in self.config.training.checkpointing_layers}

                set_activation_checkpointing(self.model, auto_wrap_policy=checkpointing_layers)
                logger.info(f"Activation checkpointing enabled for {len(checkpointing_layers)} layer types")
            except Exception as e:
                logger.warning(f"Warning: Failed to set up activation checkpointing: {str(e)}")

    def log(self, prefix: LogPrefix, **kwargs: Union[float, torch.Tensor, Dict[str, float]]) -> None:
        """
        Log metrics to TensorBoard.

        Args:
            prefix: Log category prefix
            **kwargs: Metrics to log (name-value pairs)
        """
        if not exists(self.config.logging) or self.writer is None:
            return

        step = self.global_steps[prefix]

        for name, value in kwargs.items():
            if isinstance(value, (float, int)):
                self.writer.add_scalar(f"{prefix}{name}", value, global_step=step)
            elif isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    self.writer.add_scalar(f"{prefix}{name}", value.item(), global_step=step)
                elif value.ndim in (3, 4):
                    dataformats = 'CHW' if value.ndim == 3 else 'NCHW'
                    self.writer.add_images(f"{prefix}{name}", value, global_step=step, dataformats=dataformats)
            elif isinstance(value, dict):
                for subname, subvalue in value.items():
                    self.writer.add_scalar(f"{prefix}{name}/{subname}", subvalue, global_step=step)

        # Flush logs
        if self.config.logging.save_frequency and (step + 1) % self.config.logging.save_frequency == 0:
            self.writer.flush()

    def get_opt_state_dict(self) -> Dict:
        """Get optimizer state dictionary for saving."""
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized")
        return self.optimizer.state_dict()

    def set_opt_state_dict(self, state_dict: Dict) -> None:
        """Load optimizer state from dictionary."""
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized")
        self.optimizer.load_state_dict(state_dict)

    @property
    def total_steps(self) -> int:
        """Calculate total training steps across all epochs."""
        return self.config.training.epochs * self.steps_per_epoch

    @property
    def steps_per_epoch(self) -> int:
        """Calculate number of optimization steps per epoch."""
        return self.items_per_epoch // (self.config.training.batch_size * self.config.training.gradient_accumulation_steps)

    def _loss_step(self, batch: dict, prefix: LogPrefix = LogPrefix.TRAIN) -> Tensor:
        """
        Perform a single forward pass and calculate loss.

        Args:
            batch: Input batch dictionary
            prefix: Logging prefix

        Returns:
            Loss tensor
        """
        batch_to_device(batch, self.device)

        input_ids = batch.get("input_ids")
        input_pos = batch.get("input_pos")
        mask = batch.get("mask", None)
        labels = batch.get("labels")

        output = self.model.train_step(input_ids=input_ids, input_pos=input_pos, mask=mask, labels=labels)

        loss = output.get('loss')

        self.log(prefix=prefix, crossentropy_loss=loss.item())

        return loss

    def _scheduler_step(self) -> None:
        """Update learning rate scheduler if configured."""
        if exists(self.lr_scheduler):
            self.lr_scheduler.step()

    def save_checkpoint(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save model and optimizer state.

        Args:
            path: Path to save checkpoint, defaults to log_dir/checkpoints/step_{step}.pt

        Returns:
            Path where checkpoint was saved
        """
        if path is None:
            if self.log_dir is None:
                raise ValueError("No log directory configured and no path provided")

            checkpoint_dir = self.log_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True, parents=True)

            global_step = self.global_steps[LogPrefix.TRAIN]
            path = checkpoint_dir / f"step_{global_step}.pt"
        else:
            path = Path(path)
            path.parent.mkdir(exist_ok=True, parents=True)

        checkpoint = {"model": self.model.state_dict(), "optimizer": self.get_opt_state_dict() if self.optimizer else None,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None, "global_steps": self.global_steps, "config": self.config
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        return path

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load model and optimizer state from checkpoint.

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])

        if checkpoint["optimizer"] and self.optimizer:
            self.set_opt_state_dict(checkpoint["optimizer"])

        if checkpoint["lr_scheduler"] and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if "global_steps" in checkpoint:
            self.global_steps = checkpoint["global_steps"]

        logger.info(f"Checkpoint loaded from {path}")

    def train(self) -> None:
        """
        Run the training loop for the specified number of epochs.
        """
        train_config = self.config.training
        model_summary(self.model)

        # Track metrics
        t0 = time.perf_counter()
        running_loss: Tensor = torch.tensor(0., device=self.device)
        num_tokens = 0

        # Determine maximum steps per epoch
        if exists(self.config.data.train_dataset.max_steps):
            max_steps_per_epoch = self.config.data.train_dataset.max_steps
        else:
            max_steps_per_epoch = self.steps_per_epoch

        log_prefix = LogPrefix.TRAIN

        try:
            for epoch_num in range(train_config.epochs):
                progress_bar = tqdm(total=max_steps_per_epoch, desc=f"Epoch {epoch_num + 1}/{train_config.epochs}")
                torch.cuda.empty_cache()

                self.model.train()

                for idx, batch in enumerate(self.train_dataloader):
                    # Stop if we've reached max steps for this epoch
                    if exists(max_steps_per_epoch) and (idx // train_config.gradient_accumulation_steps) >= max_steps_per_epoch:
                        break

                    # Count tokens that contribute to loss
                    current_num_tokens = (batch.get("labels") != -100).sum()
                    num_tokens += current_num_tokens

                    # Forward pass
                    loss = self._loss_step(batch, prefix=log_prefix)
                    running_loss += loss * current_num_tokens

                    if (idx + 1) % train_config.gradient_accumulation_steps == 0:
                        loss = running_loss / num_tokens

                        # Backward pass
                        loss.backward()

                        # Gradient clipping
                        grad_norm = None
                        if train_config.grad_clip.enabled:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=train_config.grad_clip.max_norm)

                        # Update parameters
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self._scheduler_step()

                        _loss = loss.item()
                        _perplexity = torch.exp(loss).item()

                        progress_bar.update(1)
                        progress_bar.set_description(f'Epoch: {epoch_num + 1}/{train_config.epochs} | '
                                                     f'Batch: {idx + 1}/{len(self.train_dataloader)} | '
                                                     f'Loss: {_loss:.3f} | '
                                                     f'PPL: {_perplexity:.3f}')

                        # Log metrics
                        if exists(self.config.logging) and self.global_steps[log_prefix] % self.config.logging.log_frequency == 0:
                            elapsed = time.perf_counter() - t0
                            tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0

                            log_dict = {'loss': _loss, 'perplexity': _perplexity,
                                        'lr': self.optimizer.param_groups[0]['lr'], 'tokens_per_sec': tokens_per_sec}

                            if self.device.type == 'cuda':
                                log_dict.update(get_memory_stats(device=self.device))

                            if train_config.grad_clip.enabled and exists(grad_norm):
                                log_dict['grad_norm'] = grad_norm

                            self.log(prefix=log_prefix, **log_dict)

                            if (exists(self.config.logging.checkpoint_frequency) and
                                    self.global_steps[log_prefix] % self.config.logging.checkpoint_frequency == 0):
                                self.save_checkpoint()

                        self.global_steps[log_prefix] += 1

                        running_loss = torch.tensor(0., device=self.device)
                        num_tokens = 0
                        t0 = time.perf_counter()

                # Clean up
                self.optimizer.zero_grad(set_to_none=True)

                # Run validation
                self._validation_pass()

                # Save end-of-epoch checkpoint
                if exists(self.config.logging) and self.config.logging.save_checkpoint_per_epoch:
                    self.save_checkpoint(path=self.log_dir / "checkpoints" / f"epoch_{epoch_num + 1}.pt" if self.log_dir else None)

            logger.info('Training Complete')

        except Exception as e:
            logger.error(f"Training interrupted: {str(e)}")
            # Save emergency checkpoint
            if exists(self.log_dir):
                self.save_checkpoint(self.log_dir / "checkpoints" / "emergency.pt")
            raise

    def _validation_pass(self) -> None:
        """Run validation on the validation dataset."""
        if not exists(self.valid_dataloader):
            return

        t0 = time.perf_counter()
        log_prefix = LogPrefix.VALIDATION

        if exists(self.config.data.valid_dataset.max_steps):
            max_steps = self.config.data.valid_dataset.max_steps
        else:
            max_steps = len(self.valid_dataloader)

        progress_bar = tqdm(total=max_steps, desc="Validation")

        total_loss = 0.0
        total_tokens = 0

        self.model.eval()

        try:
            for idx, batch in enumerate(self.valid_dataloader):
                if idx >= max_steps:
                    break

                current_num_tokens = (batch.get("labels") != -100).sum()
                total_tokens += current_num_tokens

                with torch.no_grad():
                    loss = self._loss_step(batch, prefix=log_prefix)

                total_loss += loss.item() * current_num_tokens

                _loss = loss.item()
                _perplexity = torch.exp(loss).item()

                progress_bar.update(1)
                progress_bar.set_description(f'Validation | Batch: {idx + 1}/{max_steps} | '
                                             f'Loss: {_loss:.3f} | '
                                             f'PPL: {_perplexity:.3f}')

                # Log
                if exists(self.config.logging) and self.global_steps[log_prefix] % self.config.logging.log_frequency == 0:
                    elapsed = time.perf_counter() - t0
                    tokens_per_sec = current_num_tokens / elapsed if elapsed > 0 else 0

                    log_dict = {'loss': _loss, 'perplexity': _perplexity, 'tokens_per_sec': tokens_per_sec
                    }

                    if self.device.type == 'cuda':
                        log_dict.update(get_memory_stats(device=self.device))

                    self.log(prefix=log_prefix, **log_dict)

                self.global_steps[log_prefix] += 1
                t0 = time.perf_counter()

                # Clear cache to prevent OOM
                torch.cuda.empty_cache()

            if total_tokens > 0 and exists(self.config.logging):
                avg_loss = total_loss / total_tokens
                avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

                self.log(prefix=log_prefix, avg_loss=avg_loss, avg_perplexity=avg_perplexity, total_tokens=total_tokens)

                logger.info(f"Validation complete - Avg Loss: {avg_loss:.4f}, Avg PPL: {avg_perplexity:.4f}")

        finally:
            self.model.train()

    @staticmethod
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

    def save_model_weights(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save model weights to disk.

        Args:
            output_path: Directory where weights should be saved
        """
        if output_path:
            path = Path(output_path)
        else:
            path = self.model_path

        try:
            if self._is_peft:
                # Save only adapter parameters for PEFT
                logger.info("Saving PEFT adapter parameters")
                lora_params = self.remove_checkpoint_suffix(get_adapter_params(self.model))
                save_path = path / 'adapter.safetensors'
                save_as_safetensors(lora_params, save_path)
                logger.info("Saved adapter parameters to %s", save_path)
            else:
                # Save full model parameters
                logger.info("Saving full model parameters")
                model_params = self.remove_checkpoint_suffix(self.model.state_dict())
                save_path = path / 'model.safetensors'
                save_as_safetensors(model_params, save_path)
                logger.info("Saved model parameters to %s", save_path)
        except Exception as e:
            logger.error("Failed to save model weights: %s", str(e))
            raise

    def save_optimizer_state(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save optimizer state to disk.

        Args:
            output_path: Directory where optimizer state should be saved
        """
        if output_path:
            path = Path(output_path)
        else:
            path = self.model_path
        opt_sd_file = path / 'optimizer.pt'

        try:
            torch.save(self.get_opt_state_dict(), opt_sd_file)
            logger.info("Saved optimizer state to %s", opt_sd_file)
        except Exception as e:
            logger.error("Failed to save optimizer state: %s", str(e))
            raise

    def close(self) -> None:
        """Clean up resources and save final logs."""
        logger.info('Closing Trainer...')

        if exists(self.writer):
            self.writer.flush()
            self.writer.close()
            logger.info(f"TensorBoard logs saved to {self.log_dir}")
