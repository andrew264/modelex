import gc
import glob
import os
import time
from typing import List, Optional, Tuple, Union

import torch
from tokenizers import Tokenizer

from modelex.models import instantiate_model, load_config
from modelex.utils import convert_hf_state_dict, exists, get_state_dict_from_safetensors, has_hf_keys
from modelex.utils.generation_utils import generate
from modelex.utils.peft_utils import get_merged_lora_ckpt

class ModelGenerationHandler:
    def __init__(self, path: str, device: Union[str, torch.device]):
        self.path = path
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cfg = None
        self.tokenizer: Optional[Tokenizer] = None
        self.model = None

    @property
    def prompt_format(self, ) -> str: return self.cfg.inference.chat_format if exists(self.cfg.inference) else ""

    def load_model(self, compiled: bool = False, ):
        self.cfg = load_config(os.path.join(self.path, 'config.yaml'))
        self.tokenizer = None
        if self.cfg.inference.tokenizer is not None:
            self.tokenizer = self.cfg.inference.tokenizer.get_instance()

        adaptor_sd = {}
        model_files = [os.path.abspath(path) for path in glob.glob(os.path.join(self.path, 'model*.safetensors'))]
        if model_files:
            model_sd = get_state_dict_from_safetensors(model_files, torch.device('cpu'))
            if has_hf_keys(model_sd): model_sd = convert_hf_state_dict(model_sd)
        else: raise FileNotFoundError(f"Model file not found in {model_files}.")
        if self.cfg.peft: adaptor_sd = get_state_dict_from_safetensors(os.path.join(self.path, 'adaptor.safetensors'), torch.device('cpu'))
        peft = self.cfg.peft
        self.cfg.peft = None

        model = instantiate_model(self.cfg)
        model.load_state_dict(model_sd, strict=False, assign=True)  # converts the keys to suit the models

        if adaptor_sd: model_sd = get_merged_lora_ckpt(model.state_dict() | adaptor_sd, rank=peft.rank, alpha=peft.alpha)
        model.load_state_dict(model_sd, strict=False, assign=True)
        del model_sd, adaptor_sd

        model.bos_token_id = self.cfg.inference.bos_token
        model.eval()
        model.to(dtype=torch.bfloat16)
        model.setup_cache(batch_size=1, dtype=torch.bfloat16, max_seq_len=self.cfg.max_seq_len)
        model.to(device=self.device)
        gc.collect()

        self.model = model
        if compiled: self._compile_model()

    def _compile_model(self):
        print('Compiling...')
        start = time.time()
        self.model.forward = torch.compile(self.model.forward, fullgraph=True, )

        # Dry run for compilation
        for _ in range(2): self.generate(list(range(10)), max_new_tokens=10)  # Run twice cuz idl why; but this works? somehow?

        print(f'Compiled in {time.time() - start:.3f}s')

    def generate(self, prompt: Union[str, List[int]], max_new_tokens: Optional[int] = 512, return_tokens: bool = False)\
            -> Tuple[Union[str, List[int]], int, int, float]:
        self.model.reset_cache()
        gc.collect()
        torch.cuda.empty_cache()
        if isinstance(prompt, str):
            assert exists(self.tokenizer), "Tokenizer not found"
            encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
            encoded_len = len(encoded.ids)
            tokens = torch.tensor([encoded.ids], device=self.device)
        else:
            tokens = torch.tensor([prompt], device=self.device)
            encoded_len = len(prompt)

        start = time.time()
        top_k = self.cfg.inference.top_k if 1 <= self.cfg.inference.top_k <= self.cfg.vocab_size else None
        if min(max_new_tokens, self.model.cfg.max_seq_len - encoded_len) < max_new_tokens:
            tokens = tokens[:, -(self.model.cfg.max_seq_len - max_new_tokens):]
            encoded_len = tokens.size(1)
        out, _ = generate(self.model, tokens, max_generated_tokens=max_new_tokens, pad_id=self.cfg.inference.pad_token,
                          temperature=self.cfg.inference.temperature, top_k=top_k, stop_tokens=self.cfg.inference.eos_tokens, )
        out = out[0].tolist()
        total_tokens = len(out)
        out_tokens = out[encoded_len:]
        generation_time = time.time() - start

        if return_tokens: return out_tokens, len(out_tokens), total_tokens, generation_time

        decoded = self.tokenizer.decode(out_tokens, skip_special_tokens=True)
        return decoded, len(out_tokens), total_tokens, generation_time