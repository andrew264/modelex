import argparse
import gc
import glob
import os

import torch

from modelex.models import instantiate_model, load_config
from modelex.training.trainer import Trainer
from modelex.utils import convert_hf_state_dict, get_state_dict_from_safetensors, has_hf_keys, save_as_safetensors
from modelex.utils.peft_utils import get_adapter_params, setup_model_for_peft

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description="train model")
parser.add_argument("path", type=str, help="Path to the model (required)")

def remove_checkpoint_suffix(state_dict: dict) -> dict:
    act_ckpt_wrapped_module = "._checkpoint_wrapped_module"
    return {k.replace(act_ckpt_wrapped_module, ''): v for k, v in state_dict.items()}

def main(args) -> None:
    path: str = args.path
    cfg = load_config(os.path.join(path, 'config.yaml'))
    print('=' * 75)
    print("Path: ", path)
    print('=' * 75)
    model_files = [os.path.abspath(p) for p in glob.glob(os.path.join(path, 'model*.safetensors'))]
    model_sd = get_state_dict_from_safetensors(model_files)

    model = instantiate_model(cfg, ).bfloat16()
    if not model_sd: model.apply(model._init_weights)

    if model_sd:
        if has_hf_keys(model_sd): model_sd = convert_hf_state_dict(model_sd)
        _, unexpected = model.load_state_dict(model_sd, strict=False)
        print("Unexpected Keys: ", unexpected)

    if hasattr(cfg, 'peft') and cfg.peft: setup_model_for_peft(model, cfg)

    trainer = Trainer(model, os.path.join(path, 'trainer_config.yaml'))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    opt_sd_file = os.path.join(path, 'optimizer.pt')
    if os.path.exists(opt_sd_file):
        print('Found Optimizer state_dict')
        trainer.set_opt_state_dict(torch.load(opt_sd_file, weights_only=True))
    trainer.train()

    if hasattr(cfg, 'peft') and cfg.peft:
        lora_params = remove_checkpoint_suffix(get_adapter_params(model))
        save_as_safetensors(lora_params, os.path.join(path, 'adaptor.safetensors'))
    else:
        save_as_safetensors(remove_checkpoint_suffix(model.state_dict()), os.path.join(path, 'model.safetensors'))
    torch.save(trainer.get_opt_state_dict(), opt_sd_file)
    print(f'Saved optimizer state_dict to {opt_sd_file}')

if __name__ == "__main__":
    main(args=parser.parse_args())