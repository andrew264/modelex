import gc
import glob
import os
import time
from typing import List, Optional, Tuple, Union

import torch
from tokenizers import Tokenizer
from torchtune.generation import get_causal_mask_from_padding_mask, get_position_ids_from_padding_mask
from torchtune.generation._generation import sample, update_stop_tokens_tracker
from torchtune.modules.peft import get_merged_lora_ckpt

from modelex.models import instantiate_model, load_config
from modelex.utils import convert_hf_state_dict, exists, get_state_dict_from_safetensors, has_hf_keys

def generate_next_token(model, input_pos: torch.Tensor, x: torch.Tensor, q: torch.Tensor, *,
                        mask: Optional[torch.Tensor] = None, temperature: float = 1.0, top_k: Optional[int] = None,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = model(x, input_pos=input_pos, mask=mask)['logits']
    return sample(logits[:, -1].clone(), temperature=temperature, top_k=top_k, q=q), logits,

@torch.inference_mode()
def generate(model, prompt: torch.Tensor, *, max_generated_tokens: int, pad_id: int = 0, temperature: float = 1.0,
             top_k: Optional[int] = None, stop_tokens: Optional[List[int]] = None, rng: Optional[torch.Generator] = None,
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates tokens from a model conditioned on a prompt, and also returns logits for the generations.
    """
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    bsz, prompt_length = prompt.size()
    total_response_length = prompt_length + max_generated_tokens

    generated_tokens = prompt.clone()
    incremental_decoding = model.caches_are_enabled()

    max_seq_len = (total_response_length if not incremental_decoding else model.decoder_max_cache_seq_len)
    padding_masks = generated_tokens != pad_id

    if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(padding_masks, (0, max_generated_tokens), value=True)
        masks = get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_seq_len)
        input_pos = get_position_ids_from_padding_mask(padding_masks)
    else:
        masks = torch.tril(torch.ones(total_response_length, max_seq_len, dtype=torch.bool, device=prompt.device,)).unsqueeze(0)
        input_pos = torch.arange(0, total_response_length, device=generated_tokens.device).unsqueeze(0)

    if incremental_decoding: curr_masks = masks[:, :prompt_length]
    else: curr_masks = masks[:, :prompt_length, :prompt_length]

    q = torch.empty((bsz, model.tok_embeddings.num_embeddings), device=prompt.device).exponential_(1, generator=rng)
    tokens, generated_logits = generate_next_token(model, input_pos=input_pos[:, :prompt_length].squeeze(), mask=curr_masks, x=prompt,
                                                   temperature=temperature, top_k=top_k, q=q,)
    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
    curr_pos = prompt_length

    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
    stop_tokens = (torch.tensor(stop_tokens, device=prompt.device, dtype=tokens.dtype) if stop_tokens else None)
    stop_token_mask = torch.ones((bsz, prompt_length + 1), dtype=torch.int32, device=prompt.device)

    if stop_tokens is not None:
        stop_token_reached = update_stop_tokens_tracker(tokens, stop_tokens, stop_token_reached)
        if stop_token_reached.all().item(): return generated_tokens, generated_logits

    for _ in range(max_generated_tokens - 1):
        if stop_tokens is not None:
            stop_token_mask = torch.cat([stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1)
        if incremental_decoding:
            curr_input_pos = input_pos[:, curr_pos]
            curr_masks = masks[:, curr_pos, None, :]
        else:
            tokens = generated_tokens.clone()
            curr_input_pos = input_pos[:, : curr_pos + 1]
            curr_masks = masks[:, : curr_pos + 1, : curr_pos + 1]

        q = torch.empty((bsz, model.tok_embeddings.num_embeddings), device=prompt.device).exponential_(1, generator=rng)
        tokens, logits = generate_next_token(model, input_pos=curr_input_pos, x=tokens, mask=curr_masks, temperature=temperature, top_k=top_k, q=q,)
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        curr_pos += 1
        if incremental_decoding: generated_logits = torch.cat([generated_logits, logits], dim=1)
        else: generated_logits = logits

        if stop_tokens is not None:
            stop_token_reached = update_stop_tokens_tracker(tokens, stop_tokens, stop_token_reached)
            if stop_token_reached.all(): break
    if stop_tokens is not None:
        generated_tokens *= stop_token_mask
        generated_logits *= stop_token_mask[:, :-1, None]
    return generated_tokens, generated_logits

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
        tokenizer_path = os.path.join(self.path, 'tokenizer.json')
        self.tokenizer = None
        if os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(os.path.join(self.path, 'tokenizer.json'))

        adaptor_sd = {}
        model_files = [os.path.abspath(path) for path in glob.glob(os.path.join(self.path, 'model*.safetensors'))]
        if model_files:
            model_sd = get_state_dict_from_safetensors(model_files, torch.device('cpu'))
            if has_hf_keys(model_sd): model_sd = convert_hf_state_dict(model_sd)
        else: raise FileNotFoundError(f"Model file not found in {model_files}.")
        if self.cfg.peft: adaptor_sd = get_state_dict_from_safetensors(os.path.join(self.path, 'adaptor.safetensors'), torch.device('cpu'))
        peft = self.cfg.peft
        self.cfg.peft = None

        model = instantiate_model(self.cfg).bfloat16()
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

    def generate(self, prompt: Union[str, List[int]], max_new_tokens: Optional[int] = 512,
                 return_tokens: bool = False) -> Tuple[Union[str, List[int]], int, int, float]:
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
                          temperature=self.cfg.inference.temperature, top_k=top_k, stop_tokens=self.cfg.inference.eos_tokens,)
        out = out[0].tolist()
        total_tokens = len(out)
        out_tokens = out[encoded_len:]
        generation_time = time.time() - start

        if return_tokens: return out_tokens, len(out_tokens), total_tokens, generation_time

        decoded = self.tokenizer.decode(out_tokens, skip_special_tokens=True)
        return decoded, len(out_tokens), total_tokens, generation_time