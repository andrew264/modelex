import argparse
import glob
import os

import torch
from torchtune.modules.common_utils import _register_reparametrize_state_dict_hooks
from torchtune.modules.peft import get_adapter_params, set_trainable_params
from torchtune.training import cleanup_before_training

from modelex.models.llm import ModelCfg, PeftCfg, LLM
from modelex.training import Trainer
from modelex.utils import get_state_dict_from_safetensors, save_as_safetensors

torch.set_float32_matmul_precision('high')

def setup_model_for_peft(model: torch.nn.Module, p_cfg: PeftCfg) -> None:
    set_trainable_params(model, get_adapter_params(model))
    if p_cfg.type == 'dora':
        for m in model.modules():
            if hasattr(m, "initialize_dora_magnitude"): m.initialize_dora_magnitude()  # i wish someone made documentation on this ffs

def remove_checkpoint_suffix(state_dict: dict) -> dict:
    act_ckpt_wrapped_module = "._checkpoint_wrapped_module"
    return {k.replace(act_ckpt_wrapped_module, ''): v for k, v in state_dict.items()}

def main(path: str, ) -> None:
    cfg = ModelCfg.from_yaml(os.path.join(path, 'model.yaml'))
    p_cfg = PeftCfg.from_yaml(os.path.join(path, 'peft.yaml'))
    print('=' * 75)
    print("Path: ", path)
    if p_cfg: print(p_cfg)
    print('=' * 75)
    model_files = [os.path.abspath(p) for p in glob.glob(os.path.join(path, 'model*.safetensors'))]
    model_sd = get_state_dict_from_safetensors(model_files)

    model = LLM(cfg=cfg, peft_cfg=p_cfg, ).bfloat16()
    if not model_sd: model.apply(model._init_weights)

    if p_cfg and p_cfg.quant_base: _register_reparametrize_state_dict_hooks(model, dtype=model.tok_embeddings.weight.dtype)
    _, unexpected = model.load_state_dict(model_sd, strict=False)
    print("Unexpected Keys: ", unexpected)

    if p_cfg: setup_model_for_peft(model, p_cfg)

    trainer = Trainer(model, os.path.join(path, 'trainer_config.yaml'))
    cleanup_before_training()
    trainer.train()

    if p_cfg:
        lora_params = remove_checkpoint_suffix(get_adapter_params(model))
        save_as_safetensors(lora_params, os.path.join(path, 'adaptor.safetensors'))
    else:
        save_as_safetensors(remove_checkpoint_suffix(model.state_dict()), os.path.join(path, 'model.safetensors'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument("path", type=str, help="Path to the model (required)")
    args = parser.parse_args()
    main(args.path)