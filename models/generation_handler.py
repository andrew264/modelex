import gc
import glob
import os
import time
from typing import Any, List, Optional, Tuple, Union

import torch
from tokenizers import Tokenizer
from torchtune.generation import generate
from torchtune.modules.peft import get_merged_lora_ckpt

from models.config import InferenceCfg, ModelCfg, PeftCfg
from models.inference_model import LLM
from utils import get_state_dict_from_safetensors

def exists(x: Optional[Any]) -> bool: return x is not None

class ModelGenerationHandler:
    def __init__(self, path: str, device: Union[str, torch.device]):
        self.path = path
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cfg: Optional[ModelCfg] = None
        self.p_cfg: Optional[PeftCfg] = None
        self.infer_cfg: Optional[InferenceCfg] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.model: Optional[LLM] = None

    @property
    def prompt_format(self, ) -> str: return self.infer_cfg.chat_format if exists(self.infer_cfg) else ""

    def load_model(self, compiled: bool = False, ):
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
        model.setup_cache(batch_size=1, dtype=torch.bfloat16, max_seq_len=self.cfg.max_seq_len)
        model.to(device=self.device)
        gc.collect()

        self.model = model
        if compiled: self._compile_model()

    def _compile_model(self):
        print('Compiling...')
        start = time.time()
        self.model.forward = torch.compile(self.model.forward, fullgraph=True, mode="max-autotune")

        # Dry run for compilation
        for _ in range(2): self.generate(list(range(10)), max_new_tokens=10)  # Run twice cuz idl why; but this works? somehow?

        print(f'Compiled in {time.time() - start:.3f}s')

    def generate(self, prompt: Union[str, List[int]], max_new_tokens: Optional[int] = 512, return_tokens: bool = False) -> Tuple[
        Union[str, List[int]], int, int, float]:
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
        out, _ = generate(self.model, tokens, max_generated_tokens=max_new_tokens, pad_id=self.cfg.pad_token, temperature=self.infer_cfg.temperature,
                          top_k=self.infer_cfg.top_k, stop_tokens=self.infer_cfg.eos_tokens)
        out = out[0].tolist()
        total_tokens = len(out)
        out_tokens = out[encoded_len:]
        generation_time = time.time() - start

        if return_tokens: return out_tokens, len(out_tokens), total_tokens, generation_time

        decoded = self.tokenizer.decode(out_tokens, skip_special_tokens=True)
        return decoded, len(out_tokens), total_tokens, generation_time