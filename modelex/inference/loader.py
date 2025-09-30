import gc
import logging
from pathlib import Path
from typing import Tuple, Union

import torch
from tokenizers import Tokenizer

from modelex.models.base import BaseLLM
from modelex.models.llm.config import LLMConfig
from modelex.models.registry import create_model
from modelex.utils import (convert_hf_state_dict, get_state_dict_from_safetensors, has_hf_keys, set_default_dtype, )
from modelex.utils.peft_utils import get_merged_lora_ckpt

logger = logging.getLogger(__name__)

def get_quant_config(dtype: str):
    from torchao.quantization.quant_api import (Int4WeightOnlyConfig, Int8WeightOnlyConfig, Float8WeightOnlyConfig, )
    match dtype:
        case "int8": return Int8WeightOnlyConfig()
        case "float8": return Float8WeightOnlyConfig()
        case "int4": return Int4WeightOnlyConfig(use_hqq=True)
        case _: raise ValueError(f"Unknown quantization dtype: {dtype}")

def load_model_for_inference(path: Union[str, Path], device: torch.device) -> Tuple[BaseLLM, Tokenizer]:
    model_path = Path(path)
    config_path = model_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found at {config_path}")

    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    logger.info("Tokenizer loaded successfully.")

    with torch.device("cpu"), set_default_dtype(torch.bfloat16):
        model = create_model(config_path, skip_peft=True)

    cfg: LLMConfig = model.cfg.load_config(config_path)
    logger.info(f"Created model of type '{cfg.type}' on CPU.")

    model_files = list(model_path.glob("model*.safetensors"))
    if not model_files:
        raise FileNotFoundError(f"Model weight files not found in {model_path}")

    model_sd = get_state_dict_from_safetensors([str(p) for p in model_files], torch.device("cpu"))
    if has_hf_keys(model_sd):
        logger.info("Converting HuggingFace state dict keys to modelex format.")
        model_sd = convert_hf_state_dict(model_sd)

    adapter_path = model_path / "adapter.safetensors"
    peft_config = getattr(cfg, "peft", None)

    if peft_config and adapter_path.exists():
        logger.info(f"Found PEFT adapter weights at {adapter_path}.")
        adapter_sd = get_state_dict_from_safetensors(str(adapter_path), torch.device("cpu"))

        logger.info("Merging adapter weights with base model weights...")
        full_sd = model_sd | adapter_sd
        merged_sd = get_merged_lora_ckpt(full_sd, rank=peft_config.rank, alpha=peft_config.alpha)
        model.load_state_dict(merged_sd, strict=False, assign=True)
        del merged_sd, adapter_sd, full_sd
    else:
        logger.info("No PEFT adapters found or configured. Loading base weights.")
        model.load_state_dict(model_sd, strict=False, assign=True)

    del model_sd
    gc.collect()

    quant_dtype = getattr(cfg.inference, "quant_dtype", None)
    if quant_dtype:
        try:
            from torchao.quantization import quantize_
            quant_cfg = get_quant_config(quant_dtype)
            logger.info(f"Applying '{quant_dtype}' weight-only quantization...")
            quantize_(model, quant_cfg)
            logger.info("Quantization applied successfully.")
        except ImportError:
            logger.error("torchao is required for quantization. Please install it.")
            raise
        except Exception as e:
            logger.error(f"Failed to apply quantization: {e}")
            raise

    model = model.train(False).to(device=device)
    logger.info("Setting up KV cache for inference.")
    with device:
        model.setup_cache(batch_size=1, dtype=torch.bfloat16, max_seq_len=cfg.max_seq_len)
    logger.info("Model is fully loaded and ready for inference.")
    return model, tokenizer