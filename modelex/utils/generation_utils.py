from typing import List, Optional

import torch
from torch import Tensor

from modelex.models.base import BaseLLM

def get_causal_mask_from_padding_mask(padding_mask: Tensor, target_seq_len: Optional[int] = None) -> Tensor:
    bsz, seq_len = padding_mask.shape
    target_seq_len = seq_len if target_seq_len is None else target_seq_len
    if target_seq_len < seq_len: raise AssertionError("target_seq_len cannot be shorter than the sequence length of the padding mask.")

    mask = torch.tril(torch.ones(seq_len, target_seq_len, device=padding_mask.device, dtype=torch.bool), diagonal=0, ).repeat(bsz, 1, 1)
    mask.narrow(2, 0, seq_len).mul_(padding_mask[:, None, :].expand(-1, seq_len, -1))
    mask.diagonal(dim1=1, dim2=2).copy_(torch.tensor([True]))
    return mask

def get_position_ids_from_padding_mask(padding_mask: Tensor, ) -> Tensor:
    return ((padding_mask.cumsum(-1) - 1) * padding_mask).to(torch.int)

def update_stop_tokens_tracker(tokens: Tensor, stop_tokens: Tensor, stop_token_reached: Tensor) -> Tensor:
    """Updates which sequences have reached a stop token."""
    stop_token_reached_curr = torch.isin(tokens, stop_tokens).flatten()
    stop_token_reached |= stop_token_reached_curr
    return stop_token_reached

def multinomial_sample_one(probs: Tensor, q: Tensor) -> Tensor:
    """Samples from a multinomial distribution."""
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int64)

@torch.compile
def sample(logits: Tensor, *, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None, q: Optional[Tensor] = None, ) -> Tensor:
    # scale the logits based on temperature
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        # select the very last value from the top_k above as the pivot
        pivot = v.select(-1, -1).unsqueeze(-1)
        # set everything smaller than pivot value to inf since these
        # should be pruned
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        sorted_indices_to_remove[..., -1:] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    # change logits into probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # if q is None, we use the default softmax sampling trick
    if q is None:
        uniform_val = torch.rand_like(probs)
        epsilon = torch.finfo(uniform_val.dtype).eps / 2
        condition = uniform_val >= 1.0 - epsilon
        q = -torch.where(condition, -epsilon, torch.log(uniform_val))

    return multinomial_sample_one(probs, q)


def generate_next_token(model, input_pos: Tensor, x: Tensor, q: Tensor, *, mask: Optional[Tensor] = None, temperature: float = 1.0,
                        top_k: Optional[int] = None, top_p: Optional[float] = None,) -> Tensor:
    logits = model(input_ids=x, input_pos=input_pos, mask=mask)
    return sample(logits[:, -1].clone(), temperature=temperature, top_k=top_k, top_p=top_p, q=q)

@torch.no_grad()
def generate(model: BaseLLM, prompt: Tensor, *, max_generated_tokens: int, pad_id: int = 0, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
             stop_tokens: Optional[List[int]] = None, rng: Optional[torch.Generator] = None, ) -> Tensor:
    """
    Generates tokens from a model conditioned on a prompt, and also returns logits for the generations.
    """
    torch.cuda.empty_cache()
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
        masks = torch.tril(torch.ones(total_response_length, max_seq_len, dtype=torch.bool, device=prompt.device, )).unsqueeze(0)
        input_pos = torch.arange(0, total_response_length, device=generated_tokens.device).unsqueeze(0)

    if incremental_decoding: curr_masks = masks[:, :prompt_length]
    else: curr_masks = masks[:, :prompt_length, :prompt_length]

    q = torch.empty((bsz, model.tok_embeddings.num_embeddings), device=prompt.device).exponential_(1, generator=rng)
    tokens = generate_next_token(model, input_pos=input_pos[:, :prompt_length], mask=curr_masks, x=prompt,
                                                   temperature=temperature, top_k=top_k, top_p=top_p, q=q, )
    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
    curr_pos = prompt_length

    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
    stop_tokens = (torch.tensor(stop_tokens, device=prompt.device, dtype=tokens.dtype) if stop_tokens else None)
    stop_token_mask = torch.ones((bsz, prompt_length + 1), dtype=torch.int32, device=prompt.device)

    if stop_tokens is not None:
        stop_token_reached = update_stop_tokens_tracker(tokens, stop_tokens, stop_token_reached)
        if stop_token_reached.all().item(): return generated_tokens

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
        tokens = generate_next_token(model, input_pos=curr_input_pos.unsqueeze(0), x=tokens, mask=curr_masks, temperature=temperature, top_k=top_k, top_p=top_p, q=q, )
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        curr_pos += 1

        if stop_tokens is not None:
            stop_token_reached = update_stop_tokens_tracker(tokens, stop_tokens, stop_token_reached)
            if stop_token_reached.all(): break
    if stop_tokens is not None:
        generated_tokens *= stop_token_mask
    return generated_tokens