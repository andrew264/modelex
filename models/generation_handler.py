import gc
import glob
import os
import time
from typing import Any, Dict, Optional, List, Tuple, Union

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torchtune.modules.peft import get_merged_lora_ckpt
from transformers import GenerationConfig, LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper, StoppingCriteriaList
from transformers import LogitsWarper, StoppingCriteria, Cache

from models.config import ModelCfg, InferenceCfg, PeftCfg
from models.inference_model import LLM
from utils import get_state_dict_from_safetensors


def exists(x: Optional[Any]) -> bool: return x is not None

class TemperatureRangeLogitsWarper(LogitsWarper):
    def __init__(self, start: float, end: float, num_steps: int):
        super(TemperatureRangeLogitsWarper, self).__init__()
        if end < 0 or start < 0: raise ValueError("Temperature must be greater than 0.")
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self._step = (end - start) / num_steps
        self._current_step = 0

    def _get_temperature(self) -> float:
        if self._current_step >= self.num_steps: return self.end
        temp = self.start + self._current_step * self._step
        self._current_step += 1
        return temp

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor: return scores / self._get_temperature()

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops: Optional[List[Tensor]] = None, encounters=1):
        super().__init__()
        if stops is None: stops = []
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs) -> bool:
        stop_count = 0
        batch_size = input_ids.size(0)
        for batch in input_ids:
            for stop in self.stops:
                if torch.equal(stop, batch[-len(stop):]): stop_count += 1

        return stop_count >= batch_size * self.ENCOUNTERS


class StaticCache(Cache):
    def __init__(self, cfg: ModelCfg, compiled_mode: bool = False, dtype: torch.dtype = torch.bfloat16, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_compiled = compiled_mode
        self.dtype = dtype

        if device: self.device = device
        elif torch.cuda.is_available(): self.device = torch.device('cuda')
        else: self.device = torch.device('cpu')

        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        cache_shape = (batch_size, cfg.num_kv_heads, cfg.max_seq_len, cfg.hidden_size // cfg.num_heads)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device=self.device)
        for _ in range(cfg.num_layers):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            if self.is_compiled:
                torch._dynamo.mark_static_address(new_layer_key_cache)
                torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def get_seq_length(self, layer_idx: Optional[int] = -1) -> int | torch.Tensor: return (self.key_cache[layer_idx][0, :, 0].any(dim=-1)).sum()
    def get_max_length(self) -> Optional[int]: return self.cfg.max_seq_len

    def update(self, k: Tensor, v: Tensor, layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None,) -> Tuple[Tensor, Tensor]:
        bsz = k.shape[0]
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        if cache_position is None:
            k_out.copy_(k)
            v_out.copy_(v)
        else:
            k_out.index_copy_(2, cache_position, k)
            v_out.index_copy_(2, cache_position, v)

        if self.is_compiled: return k_out, v_out
        last_position = cache_position[-1] + 1
        return k_out[:bsz, :, :last_position], v_out[:bsz, :, :last_position]

    def reset(self):
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()


class ModelGenerationHandler:
    def __init__(self, path: str, device: Union[str, torch.device]):
        self.path = path
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cfg: Optional[ModelCfg] = None
        self.p_cfg: Optional[PeftCfg] = None
        self.infer_cfg: Optional[InferenceCfg] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.model: Optional[LLM] = None
        self.cache: Optional[Cache] = None
        self.stopping_criteria: Optional[StoppingCriteriaList] = None
        self.processor: Optional[LogitsProcessorList] = None

    @property
    def prompt_format(self,) -> str: return self.infer_cfg.chat_format if exists(self.infer_cfg) else ""

    def load_model(self, compiled: bool = False,):
        self.cfg = ModelCfg.from_yaml(os.path.join(self.path, 'model.yaml'))
        self.infer_cfg = InferenceCfg.from_yaml(os.path.join(self.path, 'inference.yaml'))
        tokenizer_path = os.path.join(self.path, 'tokenizer.json')
        self.tokenizer = None
        if os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(os.path.join(self.path, 'tokenizer.json'))
        p_cfg = None
        if os.path.exists(os.path.join(self.path, 'peft.yaml')):
            p_cfg = PeftCfg.from_yaml(os.path.join(self.path, 'peft.yaml'))
        self.p_cfg = p_cfg

        adaptor_sd = {}
        model_files = [os.path.abspath(path) for path in glob.glob(os.path.join(self.path, 'model*.safetensors'))]
        if model_files: model_sd = get_state_dict_from_safetensors(model_files, torch.device('cpu'))
        else: raise FileNotFoundError("Model file not found.")
        if self.p_cfg: adaptor_sd = get_state_dict_from_safetensors(os.path.join(self.path, 'adaptor.safetensors'), torch.device('cpu'))

        model = LLM(self.cfg).bfloat16()
        model.load_state_dict(model_sd, strict=False, assign=True)  # converts the keys to suit the model

        if adaptor_sd: model_sd = get_merged_lora_ckpt(model.state_dict() | adaptor_sd, rank=self.p_cfg.rank, alpha=self.p_cfg.alpha)
        model.load_state_dict(model_sd, strict=False, assign=True)
        del model_sd, adaptor_sd

        model.bos_token_id = self.infer_cfg.bos_token
        model.eval()
        model.to(dtype=torch.bfloat16)
        model.to(device=self.device)
        gc.collect()

        self.model = model

        self.cache = StaticCache(self.cfg, compiled_mode=compiled, batch_size=self.infer_cfg.num_beams, device=self.device)
        self.stopping_criteria = self._get_stop_criteria()
        self.set_processor()
        self.model.generation_config = self.get_gen_config(None)
        if compiled: self._compile_model()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device=self.device)


    def get_gen_config(self, max_new_tokens: Optional[int] = 512):
        cfg = self.infer_cfg
        return GenerationConfig(max_new_tokens=max_new_tokens, do_sample=cfg.do_sample, num_beams=cfg.num_beams, use_cache=True, pad_token_id=cfg.pad_token, bos_token_id=cfg.bos_token, eos_token_id=cfg.eos_tokens)

    def _get_stop_criteria(self, ):
        stopping_tokens: List[torch.Tensor] = [torch.tensor([eot], device=self.device) for eot in self.infer_cfg.eos_tokens]
        return StoppingCriteriaList([StoppingCriteriaSub(stops=stopping_tokens, encounters=1)])

    def set_processor(self):
        cfg = self.infer_cfg
        processors = []
        if cfg.temperature > 0.: processors.append(TemperatureRangeLogitsWarper(cfg.temperature, cfg.temperature/2, 24))
        if cfg.top_k > 0: processors.append(TopKLogitsWarper(top_k=cfg.top_k))
        if cfg.top_p < 1.: processors.append(TopPLogitsWarper(top_p=cfg.top_p))
        self.processor = LogitsProcessorList(processors)

    def _compile_model(self):
        print('Compiling...')
        start = time.time()
        self.model.forward = torch.compile(self.model.forward, fullgraph=True, mode="reduce-overhead")

        # Dry run for compilation
        inp = "Love is a beautiful and"
        for _ in range(2): self.generate(inp, max_new_tokens=10)  # Run twice cuz idl why; but this works? somehow?

        print(f'Compiled in {time.time() - start:.3f}s')

    def generate(self, prompt: Union[str, List[int]], max_new_tokens: Optional[int] = 512, return_tokens: bool = False) -> Tuple[Union[str, List[int]], int, int, float]:
        self.cache.reset()
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if isinstance(prompt, str):
            assert exists(self.tokenizer), "Tokenizer not found"
            encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
            encoded_len = len(encoded.ids)
            tokens = torch.tensor([encoded.ids], device=self.device)
            attention_mask = torch.tensor([encoded.attention_mask], device=self.device)
        else:
            tokens = torch.tensor([prompt], device=self.device)
            encoded_len = len(prompt)
            attention_mask = torch.tensor([[1]*len(prompt)], device=self.device)

        start = time.time()
        out = self.model.generate(input_ids=tokens,
                                  attention_mask=attention_mask,
                                  logits_processor=self.processor,
                                  past_key_values=self.cache,
                                  generation_config=self.get_gen_config(max_new_tokens=max_new_tokens),
                                  stopping_criteria=self.stopping_criteria)[0].tolist()

        total_tokens = len(out)
        out_tokens = out[encoded_len:]
        generation_time = time.time() - start

        if return_tokens: return out_tokens, len(out_tokens), total_tokens, generation_time

        decoded = self.tokenizer.decode(out_tokens, skip_special_tokens=True)
        return decoded, len(out_tokens), total_tokens, generation_time
