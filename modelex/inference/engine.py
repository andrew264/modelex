import asyncio
import logging
from pathlib import Path
from typing import Union

import torch
from tokenizers import Tokenizer

from modelex.inference.loader import load_model_for_inference
from modelex.inference.structs import (ErrorEvent, InferenceRequest, StopMaxTokensEvent, StopSequenceHitEvent, TokenEvent, )
from modelex.models.base import BaseLLM
from modelex.utils.generation_utils import sample

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self, model_path: Union[str, Path], device: str = "cuda", compile_model: bool = False, ):
        self.model_path = model_path
        self.device_str = device
        self.compile_model = compile_model
        self.model: BaseLLM
        self.tokenizer: Tokenizer
        self.device: torch.device
        self.request_queue = asyncio.Queue()
    async def load(self):
        logger.info("Loading model for inference...")
        self.device = torch.device(self.device_str)
        self.model, self.tokenizer = load_model_for_inference(self.model_path, self.device)
        if self.compile_model:
            logger.info("Compiling model with torch.compile...")
            try:
                self.model.forward = torch.compile(self.model.forward, dynamic=True, mode="max-autotune-no-cudagraphs")
                logger.info("Model compiled successfully.")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        logger.info("InferenceEngine is fully initialized and ready.")
    async def submit(self, request: InferenceRequest): await self.request_queue.put(request)
    async def run_loop(self):
        logger.info("InferenceEngine processing loop started.")
        while True:
            request = await self.request_queue.get()
            try:
                await self._process_request(request)
            except Exception as e:
                logger.error(f"Error processing request for session {request.session_id}: {e}", exc_info=True, )
                error_event = ErrorEvent(message=str(e))
                await request.results_queue.put(error_event)
            finally:
                self.request_queue.task_done()
    @torch.inference_mode()
    async def _process_request(self, request: InferenceRequest):
        self.model.reset_cache()
        cfg = request.generation_config
        results_queue = request.results_queue
        prompt_ids = torch.tensor([request.prompt_ids], device=self.device)
        prompt_len = prompt_ids.shape[1]
        decoder_max_cache_seq_len = self.model.decoder_max_cache_seq_len
        total_response_length = prompt_len + cfg.max_new_tokens
        masks = torch.tril(torch.ones(total_response_length, decoder_max_cache_seq_len, dtype=torch.bool, device=self.device)).unsqueeze(0)
        prompt_pos = torch.arange(0, prompt_len, device=self.device).unsqueeze(0)
        prompt_mask = masks[:, :prompt_len]
        with torch.no_grad():
            logits = self.model.forward(input_ids=prompt_ids, input_pos=prompt_pos, mask=prompt_mask)
        next_token_tensor = sample(logits[:, -1], temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p, )
        generated_token_ids = [next_token_tensor.item()]
        await results_queue.put(TokenEvent(token_id=generated_token_ids[0]))
        stop_sequences_ids = [self.tokenizer.encode(s).ids for s in cfg.stop_sequences]
        for i in range(1, cfg.max_new_tokens):
            current_token_input = next_token_tensor.view(1, 1)
            current_pos_val = prompt_len + i - 1
            current_pos = torch.tensor([current_pos_val], device=self.device).unsqueeze(0)
            current_mask = masks[:, current_pos_val, None, :]
            with torch.no_grad():
                logits = self.model.forward(input_ids=current_token_input, input_pos=current_pos, mask=current_mask)
            next_token_tensor = sample(logits[:, -1], temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p, )
            next_token_id = next_token_tensor.item()
            generated_token_ids.append(next_token_id)
            await results_queue.put(TokenEvent(token_id=next_token_id))
            stop_hit = False
            for j, stop_seq_ids in enumerate(stop_sequences_ids):
                if len(generated_token_ids) >= len(stop_seq_ids):
                    if generated_token_ids[-len(stop_seq_ids):] == stop_seq_ids:
                        await results_queue.put(StopSequenceHitEvent(sequence=cfg.stop_sequences[j]))
                        stop_hit = True
                        break
            if stop_hit: return
            await asyncio.sleep(0)
        await results_queue.put(StopMaxTokensEvent())