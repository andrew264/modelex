import argparse
import glob
import importlib
import os
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy
from torchtune.modules.common_utils import _register_reparametrize_state_dict_hooks
from torchtune.modules.peft import get_adapter_params, set_trainable_params
from torchtune.training import cleanup_before_training, set_activation_checkpointing

from custom_data import DataModule, ParquetCustomDataReader
from models.config import ModelCfg, PeftCfg, TrainCfg
from models.layers.transformer_block import Block
from models.training_model import CustomCallback, LLMLit
from utils import get_state_dict_from_deepspeed_ckpt, get_state_dict_from_safetensors, safe_save_file

torch.set_float32_matmul_precision('high')

def setup_model_for_peft(model: torch.nn.Module, p_cfg: PeftCfg) -> None:
    set_trainable_params(model, get_adapter_params(model))
    if p_cfg.type == 'dora':
        for m in model.modules():
            if hasattr(m, "initialize_dora_magnitude"): m.initialize_dora_magnitude()  # i wish someone made documentation on this ffs

def create_instance_from_string(s: str, ):
    class_path, *args = s.split(':')
    if args: args = args[0].split(',')
    else: args = []
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(*args)

def main(path: str, train_ds: str, valid_ds: Optional[str], train_batches: Optional[int], validation_batches: Optional[int]) -> None:
    cfg = ModelCfg.from_yaml(os.path.join(path, 'model.yaml'))
    t_cfg = TrainCfg.from_yaml(os.path.join(path, 'train.yaml'))
    p_cfg = PeftCfg.from_yaml(os.path.join(path, 'peft.yaml'))
    print('=' * 75)
    print("Path: ", path)
    print("Training dataset: ", train_ds)
    print("Validation dataset: ", valid_ds)
    if train_batches:
        print("Training batches: ", train_batches)
    if validation_batches:
        print("Validation batches: ", validation_batches)
    print(t_cfg)
    if p_cfg: print(p_cfg)
    print('=' * 75)
    model_files = [os.path.abspath(p) for p in glob.glob(os.path.join(path, 'model*.safetensors'))]
    model_sd = get_state_dict_from_safetensors(model_files)

    model = LLMLit(cfg=cfg, peft_cfg=p_cfg, train_cfg=t_cfg, ).bfloat16()
    if not model_sd: model.apply(model._init_weights)

    if p_cfg and p_cfg.quant_base: _register_reparametrize_state_dict_hooks(model, dtype=model.model.embed_tokens.weight.dtype)
    _, unexpected = model.load_state_dict(model_sd, strict=False)
    print("Unexpected Keys: ", unexpected)

    if p_cfg: setup_model_for_peft(model, p_cfg)
    if t_cfg.use_grad_checkpointing: set_activation_checkpointing(model, auto_wrap_policy={Block})

    if os.path.exists(train_ds):
        train_ds = ParquetCustomDataReader(train_ds)
    else:
        train_ds = create_instance_from_string(train_ds)

    if os.path.exists(valid_ds):
        valid_ds = ParquetCustomDataReader(valid_ds)
    elif valid_ds is not None:
        valid_ds = create_instance_from_string(valid_ds)
    else:
        valid_ds = []
    datamod = DataModule(train_ds, valid_ds, batch_size=t_cfg.batch_size, max_seq_length=cfg.max_seq_len, max_pad=t_cfg.max_pad,
                         pad_to_multiple_of=t_cfg.pad_multiplier, pad_id=cfg.pad_token)

    if t_cfg.use_stage3:
        strategy = DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True, cpu_checkpointing=True, pin_memory=False)
    else: strategy = 'auto'

    ckpt_path = os.path.join(path, 'checkpoint')
    if not os.path.isdir(ckpt_path): os.mkdir(ckpt_path)
    ckpt_callback = ModelCheckpoint(ckpt_path, save_last=True, save_on_train_epoch_end=True)

    trainer = L.Trainer(accelerator=t_cfg.accelerator, strategy=strategy, precision=t_cfg.precision, max_epochs=t_cfg.num_epochs,
                        enable_progress_bar=True, log_every_n_steps=t_cfg.num_accum_steps, gradient_clip_val=1.0,
                        accumulate_grad_batches=t_cfg.num_accum_steps, callbacks=[ckpt_callback, CustomCallback()], limit_train_batches=train_batches,
                        limit_val_batches=validation_batches)
    cleanup_before_training()
    trainer.fit(model, train_dataloaders=datamod)

    if t_cfg.use_stage3:
        del model
        model = LLMLit(cfg=cfg, peft_cfg=p_cfg, ).cpu()
        model.load_state_dict(get_state_dict_from_deepspeed_ckpt(os.path.join(ckpt_path, 'last.ckpt')))
    if p_cfg:
        lora_params = get_adapter_params(model)
        safe_save_file(lora_params, os.path.join(path, 'adaptor.safetensors'))
    else:
        safe_save_file(model.state_dict(), os.path.join(path, 'model.safetensors'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument("path", type=str, help="Path to the model (required)")
    parser.add_argument("train_data", type=str, help="Path to the parquet dataset (required)")
    parser.add_argument("--validation", type=str, default="", help="Path to the parquet dataset (optional)")
    parser.add_argument("--train-batches", type=int, default=None, help="Num training batches per epoch (optional)")
    parser.add_argument("--validation-batches", type=int, default=None, help="Num validation batches per epoch (optional)")
    args = parser.parse_args()
    main(args.path, args.train_data, args.validation, args.train_batches, args.validation_batches)