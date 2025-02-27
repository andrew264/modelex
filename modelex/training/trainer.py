import enum
import importlib
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
from modelex.utils import exists, model_summary

def get_instance(class_path) -> Any:
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class LogPrefix(enum.StrEnum):
    TRAIN = 'train/'
    VALIDATION = 'valid/'

class LLMTrainer:  # to new beginnings ig
    def __init__(self, model: BaseLLM, config: Union[TrainerConfig, str]):
        if isinstance(config, str):
            self.config = TrainerConfig.load_config(config)
        else:
            self.config = config
        self.model = model

        # dummy
        self.optimizer: Optional[Optimizer] = None
        self.lr_scheduler: Optional[LambdaLR] = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.device = None
        self.global_steps: Dict[LogPrefix, int] = {p: 0 for p in LogPrefix}
        self.items_per_epochs = 0

        self._setup()

    def _setup(self):
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logger()
        self._setup_misc()

    def _setup_model(self):
        self.device = torch.device(self.config.training.device)
        self.model.to(device=self.device, dtype=torch.bfloat16)
        if self.config.training.compile:
            self.model.forward = torch.compile(self.model.forward, mode='max-autotune')

    def _setup_optimizer(self):
        opt_config = self.config.training.optimizer
        opt_class = opt_config.get_instance()
        params_to_optimize = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optimizer = opt_class(params_to_optimize, **opt_config.params)

    def _setup_scheduler(self):
        scheduler_config = self.config.training.scheduler
        if not exists(scheduler_config): return
        scheduler_class = scheduler_config.get_instance()
        params = scheduler_config.params

        if scheduler_class is get_cosine_schedule_with_warmup:
            num_warmup_steps = scheduler_config.warmup_ratio * self.total_steps
            params |= dict(num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)

        assert exists(self.optimizer), 'optimizer first please'
        self.lr_scheduler = scheduler_class(self.optimizer, **params)

    def _setup_data(self):
        data = self.config.data
        batch_size = self.config.training.batch_size
        dataloader_cfg = dict(pin_memory=data.pin_memory, batch_size=batch_size, num_workers=data.num_workers, drop_last=True)

        if exists(data.collate_fn):
            collate_fn = data.collate_fn.get_instance()(**data.collate_fn.params)
            dataloader_cfg |= dict(collate_fn=collate_fn)

        train_ds_class = data.train_dataset.get_instance()
        train_dataset = train_ds_class(**data.train_dataset.params)
        self.items_per_epochs = len(train_dataset)
        self.train_dataloader = DataLoader(train_dataset, **dataloader_cfg)

        if exists(data.valid_dataset):
            valid_ds_class = data.valid_dataset.get_instance()
            valid_dataset = valid_ds_class(**data.valid_dataset.params)
            self.valid_dataloader = DataLoader(valid_dataset, **dataloader_cfg)

    def _setup_logger(self):
        if not exists(self.config.logging):
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(self.config.logging.tensorboard_dir) / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_config(log_dir / "trainer_config.yaml")  # dump config
        self.writer = SummaryWriter(str(log_dir))

    def _setup_misc(self):
        ### Gradient Checkpointing
        if self.config.training.checkpointing_layers:
            checkpointing_layers = {get_instance(l) for l in self.config.training.checkpointing_layers}
            set_activation_checkpointing(self.model, auto_wrap_policy=checkpointing_layers)

    def log(self, prefix: LogPrefix, **kwargs: Union[float, torch.Tensor, Dict[str, float]]):
        if not exists(self.config.logging):
            return
        step = self.global_steps[prefix]
        for name, value in kwargs.items():
            if isinstance(value, (float, int)):
                self.writer.add_scalar(f"{prefix}{name}", value, global_step=step)
            elif isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    self.writer.add_scalar(f"{prefix}{name}", value.item(), global_step=step)
                elif value.ndim == 3 or value.ndim == 4:
                    self.writer.add_images(f"{prefix}{name}", value, global_step=step, dataformats='CHW' if value.ndim == 3 else 'NCHW')
            elif isinstance(value, dict):
                for subname, subvalue in value.items():
                    self.writer.add_scalar(f"{prefix}{name}/{subname}", subvalue, global_step=step)
        if self.config.logging.save_frequency % (step + 1) == 0:
            self.writer.flush()

    def get_opt_state_dict(self):
        assert self.optimizer is not None
        return self.optimizer.state_dict()

    def set_opt_state_dict(self, state_dict):
        assert self.optimizer is not None
        self.optimizer.load_state_dict(state_dict)

    @property
    def total_steps(self):
        return self.config.training.epochs * self.steps_per_epoch
    @property
    def steps_per_epoch(self):
        return self.items_per_epochs // (self.config.training.batch_size * self.config.training.gradient_accumulation_steps)

    def _loss_step(self, batch: dict, prefix: LogPrefix = LogPrefix.TRAIN) -> Tensor:
        batch_to_device(batch, self.device)
        input_ids = batch.get("input_ids")
        input_pos = batch.get("input_pos")
        mask = batch.get("mask", None)
        labels = batch.get("labels")

        output = self.model.train_step(input_ids=input_ids, input_pos=input_pos, mask=mask, labels=labels)
        loss = output.get('loss')
        self.log(crossentropy_loss=loss.item(), prefix=prefix)

        return loss

    def _scheduler_step(self):
        if exists(self.lr_scheduler):
            self.lr_scheduler.step()

    def train(self):
        train_config = self.config.training
        model_summary(self.model)
        t0 = time.perf_counter()
        running_loss: Tensor = torch.tensor(0., device=self.device)
        num_tokens = 0
        if exists(self.config.data.train_dataset.max_steps):
            max_steps_per_epoch = self.config.data.train_dataset.max_steps
        else:
            max_steps_per_epoch = self.steps_per_epoch
        log_prefix = LogPrefix.TRAIN
        for epoch_num in range(train_config.epochs):
            progress_bar = tqdm(total=self.steps_per_epoch)
            torch.cuda.empty_cache()
            for idx, batch in enumerate(self.train_dataloader):
                if exists(max_steps_per_epoch) and (idx // train_config.gradient_accumulation_steps) == max_steps_per_epoch:
                    break
                current_num_tokens = (batch.get("labels") != -100).sum()
                num_tokens += current_num_tokens
                running_loss += self._loss_step(batch, prefix=log_prefix) * current_num_tokens
                if (idx + 1) % train_config.gradient_accumulation_steps == 0:
                    loss = running_loss / num_tokens
                    loss.backward()
                    grad_norm = None
                    if train_config.grad_clip.enabled:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=train_config.grad_clip.max_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._scheduler_step()
                    _loss = loss.item()
                    _perplexity = torch.exp(loss).item()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f'Epoch: {epoch_num} | Batch idx: {idx}/{max_steps_per_epoch} | Loss: {_loss:.3f} | Perplexity: {_perplexity:.3f}')

                    if exists(self.config.logging) and self.global_steps[log_prefix] % self.config.logging.log_frequency == 0:
                        time_per_step = time.perf_counter() - t0
                        log_dict = dict(loss=_loss, perplexity=_perplexity, lr=self.optimizer.param_groups[0]['lr'],
                                        tokens_per_sec=num_tokens / time_per_step)
                        if self.device.type == 'cuda':
                            log_dict.update(get_memory_stats(device=self.device))
                        if train_config.grad_clip.enabled and exists(grad_norm):
                            log_dict.update({'grad_norm': grad_norm})
                        self.log(prefix=log_prefix, **log_dict)

                    self.global_steps[log_prefix] += 1
                    running_loss = torch.tensor(0., device=self.device)
                    num_tokens = 0
                    t0 = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            self._validation_pass()
        print('Training Complete')

    def _validation_pass(self):
        if not exists(self.valid_dataloader): return
        t0 = time.perf_counter()
        progress_bar = tqdm(total=len(self.valid_dataloader))
        if exists(self.config.data.valid_dataset.max_steps):
            max_steps = self.config.data.valid_dataset.max_steps
        else:
            max_steps = len(self.valid_dataloader)
        log_prefix = LogPrefix.VALIDATION
        self.model.eval()
        for idx, batch in enumerate(self.valid_dataloader):
            if idx == max_steps:
                break
            current_num_tokens = (batch.get("labels") != -100).sum()
            with torch.no_grad():
                loss = self._loss_step(batch, prefix=log_prefix)
            _loss = loss.item()
            _perplexity = torch.exp(loss).item()
            progress_bar.update(1)
            progress_bar.set_description(f'Batch idx: {idx}/{max_steps} | Loss: {_loss:.3f} | Perplexity: {_perplexity:.3f}')

            if exists(self.config.logging) and self.global_steps[log_prefix] % self.config.logging.log_frequency == 0:
                time_per_step = time.perf_counter() - t0
                log_dict = dict(loss=_loss, perplexity=_perplexity, tokens_per_sec=current_num_tokens / time_per_step, prefix=LogPrefix.VALIDATION)
                if self.device.type == 'cuda':
                    log_dict.update(get_memory_stats(device=self.device))
                self.log(**log_dict)
            self.global_steps[log_prefix] += 1
            t0 = time.perf_counter()
            torch.cuda.empty_cache()  # something is really broken; i need this here to not OOM
        self.model.train()

    def close(self):
        print('Closing Trainer...')
        self.writer.flush()
        self.writer.close()