import argparse
import asyncio
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from modelex.inference.conversation import (ConversationFormatter, InferenceConversationData)
from modelex.inference.engine import InferenceEngine
from modelex.inference.parser import StreamingResponseParser
from modelex.inference.structs import (GenerationConfig, GenerationFinished, InferenceRequest)

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)], )
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Serve a modelex model")
parser.add_argument("path", type=str, help="Path to the model directory (required)")
parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (optional, defaults to 'cuda')", )
parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the model (optional, defaults to False)", )
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
parser.add_argument("--port", type=int, default=6969, help="Port to run the server on")

engine: Optional[InferenceEngine] = None
cli_args: Optional[argparse.Namespace] = None

class ChatRequest(BaseModel):
    messages: InferenceConversationData
    assistant_name: Optional[str] = "assistant"
    generation_config: Optional[GenerationConfig] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, cli_args
    args = cli_args
    engine = InferenceEngine(model_path=args.path, device=args.device, compile_model=args.compile)
    await engine.load()
    loop = asyncio.get_running_loop()
    engine_task = loop.create_task(engine.run_loop())
    logger.info("Server startup complete. Engine is running.")
    yield
    engine_task.cancel()
    logger.info("Server shutting down.")

app = FastAPI(lifespan=lifespan)

@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest):
    conversation = ConversationFormatter(assistant_name=body.assistant_name)
    conversation.add_msgs(body.messages)

    prompt_text = conversation.get_prompt_for_completion()
    prompt_ids = engine.tokenizer.encode(prompt_text).ids
    results_queue = asyncio.Queue()

    gen_config = body.generation_config or GenerationConfig()
    stop_sequences = set(gen_config.stop_sequences)
    stop_sequences.update(["<|eot_id|>", "</tool_call>", "</think>"])
    gen_config.stop_sequences = list(stop_sequences)


    inference_request = InferenceRequest(
        prompt_ids=prompt_ids,
        session_id=str(uuid.uuid4()),
        results_queue=results_queue,
        generation_config=gen_config,
    )
    await engine.submit(inference_request)

    async def stream_generator():
        parser = StreamingResponseParser(engine.tokenizer)
        try:
            async for event in parser.parse(results_queue):
                yield event.model_dump_json() + "\n"
                if isinstance(event, GenerationFinished):
                    break
        except Exception as e:
            logger.error(f"Error during stream generation: {e}", exc_info=True)
            error_event = GenerationFinished(reason="error", details=str(e))
            yield error_event.model_dump_json() + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

def main(args: argparse.Namespace):
    global cli_args
    cli_args = args
    uvicorn.run("modelex.scripts.serve:app", host=args.host, port=args.port, log_level="info", reload=False)

if __name__ == "__main__":
    parsed_args = parser.parse_args()
    main(parsed_args)