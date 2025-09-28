import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import torch
from tokenizers import Tokenizer

from modelex.models.registry import create_model
from modelex.utils import convert_hf_state_dict, exists, get_state_dict_from_safetensors, has_hf_keys, set_default_dtype
from modelex.utils.generation_utils import generate, generate_stream
from modelex.utils.peft_utils import get_merged_lora_ckpt

logger = logging.getLogger(__name__)

def get_quant_config(dtype: str):
    from torchao.quantization.quant_api import Int8WeightOnlyConfig, Float8WeightOnlyConfig, Int4WeightOnlyConfig
    match dtype:
        case 'int8': return Int8WeightOnlyConfig()
        case 'float8': return Float8WeightOnlyConfig()
        case 'int4': return Int4WeightOnlyConfig(use_hqq=True)
        case _: raise ValueError(f'unknown quantization dtype: {dtype}')

class ModelGenerationHandler:
    """Handles loading and generation for ML models."""

    def __init__(self, path: Union[str, Path], device: Union[str, torch.device]):
        """
        Initialize the model generation handler.

        Args:
            path: Path to the model directory containing model files and config
            device: Device to load the model on (e.g., 'cuda', 'cpu')
        """
        self.path = Path(path)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cfg = None
        self.tokenizer: Optional[Tokenizer] = None
        self.model = None
        self._is_compiled = False

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None

    def load_tokenizer(self) -> None:
        """Load the tokenizer from the model path."""
        tokenizer_path = self.path / 'tokenizer.json'
        if tokenizer_path.exists():
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    def load_model(self, compiled: bool = False) -> None:
        """
        Load the model from the specified path.

        Args:
            compiled: Whether to compile the model using torch.compile

        Raises:
            FileNotFoundError: If model files are not found
        """
        # Load tokenizer
        if self.tokenizer is None:
            try:
                self.load_tokenizer()
            except FileNotFoundError:
                # Continue without tokenizer, will raise error if text input is used
                pass

        # Create model
        config_path = self.path / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with torch.device('cpu'), set_default_dtype(torch.bfloat16):
            model = create_model(config_path, skip_peft=True)  # always skip initializing peft stuff cuz we merge the adapter weights
        self.cfg = model.cfg.load_config(config_path)  # load cfg with peft parameters
        logger.info(f'{self.cfg.type} loaded')

        # Load model state dict
        model_files = list(self.path.glob('model*.safetensors'))
        if not model_files:
            raise FileNotFoundError(f"Model files not found in {self.path}")

        model_sd = self._get_state_dict(model_files)
        logger.info(f'found state_dict')

        # Handle PEFT adapters if present
        adapter_sd = {}
        adapter_path = self.path / 'adapter.safetensors'
        peft_config = getattr(self.cfg, 'peft', None)

        if peft_config and adapter_path.exists():
            logger.info(f'found adapter weights at {adapter_path}')
            adapter_sd = get_state_dict_from_safetensors(str(adapter_path), torch.device('cpu'))
        else:
            logger.info('no adapter weights found')
            self.cfg.peft = None

        # Merge adapter weights
        if adapter_sd and peft_config:
            logger.info(f'merging adapter weights with base weights')
            merged_sd = get_merged_lora_ckpt(model_sd | adapter_sd, rank=peft_config.rank, alpha=peft_config.alpha)
            model.load_state_dict(merged_sd, strict=False, assign=True)
            self.cfg.peft = peft_config
            del merged_sd
        else:  # Load state dict into model
            logger.info('loading the state_dict into model')
            model.load_state_dict(model_sd, strict=False, assign=True)

        # Clean up
        del model_sd, adapter_sd
        gc.collect()

        model.eval()
        logger.info('setting up kv cache')
        with self.device:
            model.setup_cache(batch_size=1, dtype=torch.bfloat16, max_seq_len=self.cfg.max_seq_len)

        if self.cfg.inference.quant_dtype is not None:
            from torchao.quantization import quantize_
            quant_cfg = get_quant_config(self.cfg.inference.quant_dtype)
            logger.info(f'applying {self.cfg.inference.quant_dtype} quantization to {self.cfg.type}')
            quantize_(model, quant_cfg, device=self.device)

        model = model.to(self.device)  # just to be sure

        for n, p in model.named_parameters():
            if p.device.type != self.device.type:
                logger.warning(f'layer: {n} | device {p.device} is not in the right device {self.device}')

        self.model = model

        # Compile
        if compiled:
            self._compile_model()

    @staticmethod
    def _get_state_dict(model_files: List[Path]) -> Dict[str, Any]:
        """
        Load state dict from safetensors files.

        Args:
            model_files: List of paths to model files

        Returns:
            Model state dict
        """
        model_sd = get_state_dict_from_safetensors([str(path) for path in model_files], torch.device('cpu'))

        # Convert HF format
        if has_hf_keys(model_sd):
            model_sd = convert_hf_state_dict(model_sd)

        return model_sd

    def _compile_model(self) -> None:
        """Compile the model using torch.compile for faster inference."""
        if self._is_compiled:
            return

        logger.info('Compiling model...')
        start = time.time()

        # Compile the forward function
        self.model.forward = torch.compile(self.model.forward, dynamic=True, mode="max-autotune-no-cudagraphs")

        # Warm-up runs for compilation
        for i in range(2):
            self.generate(list(range(10 * (i + 1))), max_new_tokens=5)

        self._is_compiled = True
        logger.info(f'Model compiled in {time.time() - start:.3f}s')

    def generate(self, prompt: Union[str, List[int]], max_new_tokens: Optional[int] = 512, return_tokens: bool = False,
                 skip_special_tokens: bool = True) -> Tuple[Union[str, List[int]], int, int, float]:
        """
        Generate text from a prompt.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Reset model cache
        self.model.reset_cache()
        gc.collect()
        torch.cuda.empty_cache()

        # Tokenize prompt
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer not loaded but string prompt provided")

            encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
            encoded_len = len(encoded.ids)
            tokens = torch.tensor([encoded.ids], device=self.device, dtype=torch.int64)
        else:
            tokens = torch.tensor([prompt], device=self.device, dtype=torch.int64)
            encoded_len = len(prompt)

        # Ensure prompt doesn't exceed max context length
        max_context_len = self.model.cfg.max_seq_len
        available_tokens = max_context_len - encoded_len

        if available_tokens < max_new_tokens:
            truncate_to = max_context_len - max_new_tokens
            if truncate_to > 0:
                tokens = tokens[:, -truncate_to:]
                encoded_len = tokens.size(1)
            else:
                max_new_tokens = available_tokens

        generation_config = self._prepare_generation_config(max_new_tokens)

        start = time.time()
        out = generate(self.model, tokens, **generation_config)
        out = out[0].tolist()

        total_tokens = len(out)
        new_tokens = out[encoded_len:]
        generation_time = time.time() - start

        if return_tokens:
            return new_tokens, len(new_tokens), total_tokens, generation_time

        # Decode text
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded but text output requested")

        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens)
        torch.cuda.empty_cache()
        return decoded, len(new_tokens), total_tokens, generation_time

    def generate_stream(self, prompt: Union[str, List[int]], max_new_tokens: Optional[int] = 512, return_tokens: bool = False,
                        skip_special_tokens: bool = True) -> Generator[Union[str, List[int]], None, None]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Reset model cache
        self.model.reset_cache()
        gc.collect()
        torch.cuda.empty_cache()

        # Tokenize prompt
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer not loaded but string prompt provided")

            encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
            encoded_len = len(encoded.ids)
            tokens = torch.tensor([encoded.ids], device=self.device, dtype=torch.int64)
        else:
            tokens = torch.tensor([prompt], device=self.device, dtype=torch.int64)
            encoded_len = len(prompt)

        # Ensure prompt doesn't exceed max context length
        max_context_len = self.model.cfg.max_seq_len
        available_tokens = max_context_len - encoded_len

        if available_tokens < max_new_tokens:
            truncate_to = max_context_len - max_new_tokens
            if truncate_to > 0:
                tokens = tokens[:, -truncate_to:]
                encoded_len = tokens.size(1)
            else:
                max_new_tokens = available_tokens

        generation_config = self._prepare_generation_config(max_new_tokens)

        for out in generate_stream(self.model, tokens, **generation_config):
            new_tokens = out[0].tolist()

            if return_tokens:
                yield new_tokens

            # Decode text
            if self.tokenizer is None:
                raise ValueError("Tokenizer not loaded but text output requested")

            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens)
            torch.cuda.empty_cache()
            yield decoded

    def _prepare_generation_config(self, max_new_tokens: int) -> Dict[str, Any]:
        """
        Prepare generation configuration parameters.

        Args:
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Dictionary of generation parameters
        """
        inference_cfg = getattr(self.cfg, 'inference', None)

        # Set up sampling parameters
        top_k = None
        if hasattr(inference_cfg, 'top_k') and inference_cfg.top_k is not None:
            if 1 <= inference_cfg.top_k <= self.cfg.vocab_size:
                top_k = inference_cfg.top_k

        top_p = None
        if hasattr(inference_cfg, 'top_p') and inference_cfg.top_p is not None:
            if 0 <= inference_cfg.top_p <= 1:
                top_p = inference_cfg.top_p

        return {'max_generated_tokens': max_new_tokens, 'pad_id': inference_cfg.pad_token if inference_cfg else None,
                'temperature': inference_cfg.temperature if inference_cfg else 1.0, 'top_k': top_k, 'top_p': top_p,
                'stop_tokens': inference_cfg.eos_tokens if inference_cfg else None,
                }