import argparse
from typing import List, Optional

import torch
from pydantic import BaseModel

from modelex.data.prompt_format import PromptFormatter
from modelex.generation import ModelGenerationHandler

parser = argparse.ArgumentParser(description="generate sequence")
parser.add_argument("path", type=str, help="Path to the models (required)")
parser.add_argument("--device", type=str, default="cuda", help="Device to run the models on (optional, defaults to 'cuda')", )

class ModelAPI:
    def __init__(self, path: str):
        self.device = None
        self.path = path
        self.model_handler: Optional[ModelGenerationHandler] = None

    def setup(self, device: str):
        self.device = device
        torch.set_float32_matmul_precision("high")
        self.model_handler = ModelGenerationHandler(self.path, self.device)
        self.model_handler.load_model(compiled=False)

    def __call__(self, request: dict) -> dict:
        assistant_name = request.get("assistant_name", "assistant")
        p = PromptFormatter(assistant_name=assistant_name, chat_format=self.model_handler.prompt_format, ).add_msgs(request["messages"])
        output_text, _, length, _ = self.model_handler.generate(p.get_prompt_for_completion(), max_new_tokens=1024)
        return {"response": output_text, "cur_length": length, "max_length": self.model_handler.cfg.max_seq_len, }

class PredictRequest(BaseModel):
    messages: List[dict]
    assistant_name: Optional[str] = "assistant"

api: Optional[ModelAPI] = None

def main(args):
    try:
        from fastapi import FastAPI, Request
        import uvicorn
    except ImportError as e:
        print("Please install fastapi and uvicorn to run serve!")
        raise e
    global api
    api = ModelAPI(args.path)
    api.setup(args.device)
    print("Model loaded!")

    app = FastAPI()
    @app.post("/predict")
    async def predict_route(request: Request):
        data = await request.json()
        predict_request = PredictRequest(**data)

        response = api(predict_request.model_dump())
        return response

    uvicorn.run(app, host="0.0.0.0", port=6969)

if __name__ == "__main__":
    main(args=parser.parse_args())