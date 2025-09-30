import argparse
import asyncio
import logging
import sys
import time
import uuid

import torch

from modelex.inference.conversation import DEFAULT_TOKENS
from modelex.inference.engine import InferenceEngine
from modelex.inference.structs import (ErrorEvent, GenerationConfig, InferenceRequest, StopMaxTokensEvent, StopSequenceHitEvent, TokenEvent, )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description="Generate sequences using the InferenceEngine")
parser.add_argument("path", type=str, help="Path to the model directory (required)")
parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (optional, defaults to 'cuda')")
parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the model (optional, defaults to False)")
parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--top-k", type=int, default=None, help="Top-K sampling")
parser.add_argument("--top-p", type=float, default=None, help="Top-P (nucleus) sampling")

async def main(args):
    engine = InferenceEngine(model_path=args.path, device=args.device, compile_model=args.compile, )
    await engine.load()

    engine_task = asyncio.create_task(engine.run_loop())
    logger.info("InferenceEngine is running. You can now enter prompts.")

    try:
        while True:
            try: prompt_text = input("Enter a prompt (or press Ctrl+C to exit): ")
            except KeyboardInterrupt:
                print("\nExiting...")
                break

            if not prompt_text: continue

            prompt_ids = engine.tokenizer.encode(DEFAULT_TOKENS["bos"] + prompt_text).ids
            results_queue = asyncio.Queue()
            generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                                 stop_sequences=[DEFAULT_TOKENS["eot"]])
            request = InferenceRequest(prompt_ids=prompt_ids, session_id=str(uuid.uuid4()), results_queue=results_queue,
                                       generation_config=generation_config)
            await engine.submit(request)

            start_time = time.perf_counter()
            generated_ids = []
            print("\nModel Output: ", end="", flush=True)
            while True:
                event = await results_queue.get()
                if isinstance(event, TokenEvent):
                    generated_ids.append(event.token_id)
                    token_text = engine.tokenizer.decode([event.token_id])
                    print(token_text, end="", flush=True)
                elif isinstance(event, (StopMaxTokensEvent, StopSequenceHitEvent, ErrorEvent)):
                    if isinstance(event, ErrorEvent): logger.error(f"\nAn error occurred during generation: {event.message}")
                    break
            generation_time = time.perf_counter() - start_time
            num_tokens = len(generated_ids)
            print()
            logger.info(f"Generated {num_tokens} tokens in {generation_time:.3f}s ({num_tokens / generation_time:.2f} tokens/s)")
            print("-" * 50)

    finally:
        engine_task.cancel()
        logger.info("InferenceEngine has been shut down.")

if __name__ == '__main__':
    parsed_args = parser.parse_args()
    try: asyncio.run(main(parsed_args))
    except KeyboardInterrupt: logger.info("Program interrupted by user.")