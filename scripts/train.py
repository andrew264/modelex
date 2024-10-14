import argparse
import glob
import os

import torch
import lightning as L
from torchtune.modules.common_utils import _register_reparametrize_state_dict_hooks
from torchtune.modules.peft import get_adapter_params, set_trainable_params
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

from models.config import ModelCfg, PeftCfg, TrainCfg
from models.training_model import LLMLit, CustomCallback
from utils import get_state_dict_from_safetensors, safe_save_file, get_state_dict_from_deepspeed_ckpt
from custom_data import DataModule

torch.set_float32_matmul_precision('high')

def setup_model_for_peft(model: torch.nn.Module, p_cfg: PeftCfg)->None:
    set_trainable_params(model, get_adapter_params(model))
    if p_cfg.type == 'dora':
        for m in model.modules():
            if hasattr(m, "initialize_dora_magnitude"): m.initialize_dora_magnitude()  # i wish someone made documentation on this ffs

def main(path: str, train_ds: str, valid_ds: str,) -> None:
    cfg = ModelCfg.from_yaml(os.path.join(path, 'model.yaml'))
    t_cfg = TrainCfg.from_yaml(os.path.join(path, 'train.yaml'))
    p_cfg = PeftCfg.from_yaml(os.path.join(path, 'peft.yaml'))
    print('=' * 75)
    print("Path: ", path)
    print("Training dataset: ", train_ds)
    print("Validation dataset: ", valid_ds)
    print(t_cfg)
    if p_cfg: print(p_cfg)
    print('=' * 75)
    model_files = [os.path.abspath(p) for p in glob.glob(os.path.join(path, 'model*.safetensors'))]
    model_sd = get_state_dict_from_safetensors(model_files)
    model = LLMLit(cfg=cfg, peft_cfg=p_cfg, train_cfg=t_cfg,).bfloat16()
    if p_cfg and p_cfg.quant_base: _register_reparametrize_state_dict_hooks(model, dtype=model.model.embed_tokens.weight.dtype)
    _, unexpected = model.load_state_dict(model_sd, strict=False)
    print("Unexpected Keys: ", unexpected)
    if p_cfg: setup_model_for_peft(model, p_cfg)

    datamod = DataModule(train_ds, valid_ds, batch_size=t_cfg.batch_size, max_seq_length=cfg.max_seq_len, max_pad=t_cfg.max_pad,
                         pad_to_multiple_of=t_cfg.pad_multiplier, pad_id=cfg.pad_token)

    if t_cfg.use_stage3:
        strategy = DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True, cpu_checkpointing=True, pin_memory=False)
    else: strategy = 'auto'

    ckpt_path = os.path.join(path, 'checkpoint')
    if not os.path.isdir(ckpt_path): os.mkdir(ckpt_path)
    ckpt_callback = ModelCheckpoint(ckpt_path, save_last=True)

    trainer = L.Trainer(accelerator="gpu", strategy=strategy, precision=t_cfg.precision, max_epochs=t_cfg.num_epochs, enable_progress_bar=True,
                        log_every_n_steps=t_cfg.num_accum_steps, gradient_clip_val=1.0, accumulate_grad_batches=t_cfg.num_accum_steps,
                        callbacks=[ckpt_callback, CustomCallback()], )
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
    parser = argparse.ArgumentParser(description="generate sequence")
    parser.add_argument("path", type=str, help="Path to the model (required)")
    parser.add_argument("train_data", type=str, help="Path to the parquet dataset (required)")
    parser.add_argument("--validation", type=str, default="", help="Path to the parquet dataset (optional)")
    args = parser.parse_args()
    main(args.path, args.train_data, args.validation)