import argparse
import os

import torch
import lightning as L
from torchtune.modules.peft import get_adapter_params, set_trainable_params
from lightning.pytorch.strategies import DeepSpeedStrategy

from models.config import ModelCfg, PeftCfg
from models.training_model import LLMLit
from utils import get_state_dict_from_safetensors, safe_save_file
from custom_data import DataModule

torch.set_float32_matmul_precision('high')

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other): return self.start <= other <= self.end

def main(path: str, train_ds: str, valid_ds: str, device: str, bs: int, epochs: int, accum_steps: int, lr: float, use_scheduler: bool, warmup: int, use_grad_checkpointing: bool, val_interval: float, use_stage3: bool) -> None:
    print('='*75)
    print("Path: ", path)
    print("Training dataset: ", train_ds)
    print("Validation dataset: ", valid_ds)
    print("Device: ", device)
    print("Batch size: ", bs)
    print("Epochs: ", epochs)
    print("Accumulation steps: ", accum_steps)
    print("Learning Rate: ", lr)
    print("Using LR Scheduler: ", use_scheduler)
    print("Scheduler Warmup Steps: ", warmup)
    print("Using Gradient Checkpointing: ", use_grad_checkpointing)
    print("Validation Check Interval: ", val_interval)
    print("Using Stage 3 Offload: ", use_stage3)
    print('='*75)
    cfg = ModelCfg.from_yaml(os.path.join(path, 'model.yaml'))
    p_cfg = None
    if os.path.exists(os.path.join(path, 'peft.yaml')):
        p_cfg = PeftCfg.from_yaml(os.path.join(path, 'peft.yaml'))
        print('Peft Config Found')
    model_sd = get_state_dict_from_safetensors(os.path.join(path, 'model.safetensors'))
    model = LLMLit(cfg=cfg, peft_cfg=p_cfg, lr=lr, use_scheduler=use_scheduler, warmup=warmup, use_grad_checkpointing=use_grad_checkpointing)
    model.load_state_dict(model_sd, strict=False, assign=True)
    if p_cfg: set_trainable_params(model, get_adapter_params(model))
    datamod = DataModule(train_ds, valid_ds, batch_size=bs, max_seq_length=cfg.max_seq_len, pad_id=cfg.pad_token)
    if use_stage3:
        strategy = DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True, pin_memory=False)
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        model.set_optimizer(DeepSpeedCPUAdam)
    else:
        strategy = 'auto'
        from bitsandbytes.optim.adamw import AdamW8bit
        model.set_optimizer(AdamW8bit)

    trainer = L.Trainer(accelerator="gpu",
                        strategy=strategy,
                        precision="bf16-true",
                        max_epochs=epochs,
                        enable_checkpointing=False,
                        enable_progress_bar=True,
                        log_every_n_steps=accum_steps,
                        gradient_clip_val=1.0,
                        val_check_interval=val_interval,
                        accumulate_grad_batches=accum_steps)

    trainer.fit(model, train_dataloaders=datamod)
    if p_cfg:
        lora_params = get_adapter_params(model)
        safe_save_file(lora_params, os.path.join(path, 'adaptor.safetensors'))
    else:
        safe_save_file(model.state_dict(), os.path.join(path, 'model.safetensors'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate sequence")
    parser.add_argument("path", type=str, help="Path to the model (required)")
    parser.add_argument("train_data", type=str, help="Path to the parquet dataset (required)")
    parser.add_argument("validation_data", type=str, default=None, help="Path to the parquet dataset (optional)")
    parser.add_argument("--device", type=str, default="0", help="Device to run the model on (optional, defaults to 'gpu 0')")
    parser.add_argument("--bs", type=int, default=1, help="Batch size (Defaults to 1)")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs (Defaults to 1)")
    parser.add_argument("--accum-steps", type=int, default=8, help="Gradient Accumulation Steps (Defaults to 1)")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-4, help="Set Learning Rate")
    parser.add_argument("--use-scheduler", action=argparse.BooleanOptionalAction, default=False, help="Enable LR Scheduler")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--use-grad-checkpointing", action=argparse.BooleanOptionalAction, default=False, help="Enable Gradient Checkpointing")
    parser.add_argument("--validation-interval", type=float, default=1.0, choices=[Range(0.0, 1.0)], help="Validation Interval")
    parser.add_argument("--use-stage3", action=argparse.BooleanOptionalAction, default=False, help="Enable Stage 3 Offload")
    args = parser.parse_args()
    main(args.path, args.train_data, args.validation_data, args.device, args.bs, args.num_epochs, args.accum_steps, args.learning_rate, args.use_scheduler, args.warmup, args.use_grad_checkpointing, args.validation_interval, args.use_stage3)