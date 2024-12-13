import argparse
from typing import List, Optional, Tuple

import litserve as ls
import torch

from modelex.datasets.prompt_format import Message, PromptFormatter
from modelex.generation import ModelGenerationHandler

class ModelAPI(ls.LitAPI):
    def __init__(self, path: str):
        super().__init__()
        self.device = None
        self.path = path
        self.model_handler: Optional[ModelGenerationHandler] = None
    def setup(self, device: str):
        self.device = device
        torch.set_float32_matmul_precision('high')
        self.model_handler = ModelGenerationHandler(self.path, self.device)
        self.model_handler.load_model(compiled=False)
    def decode_request(self, request, **kwargs) -> Tuple[List[Message], str]:
        assistant_name = request.get('assistant_name', 'assistant')
        return request['messages'], assistant_name
    def predict(self, request, **kwargs) -> Tuple[str, int]:
        messages, assistant_name = request
        p = PromptFormatter(assistant_name=assistant_name, chat_format=self.model_handler.prompt_format).add_msgs(messages)
        decoded, _, total_toks, _ = self.model_handler.generate(p.get_prompt_for_completion(), max_new_tokens=1024)
        return decoded, total_toks

    def encode_response(self, output, **kwargs):
        output_text, length = output
        return {'response': output_text, 'cur_length': length, 'max_length': self.model_handler.cfg.max_seq_len, }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate sequence")
    parser.add_argument("path", type=str, help="Path to the models (required)")
    parser.add_argument("--device", type=str, default="0", help="Device to run the models on (optional, defaults to 'gpu 0')")
    args = parser.parse_args()

    api = ModelAPI(args.path)
    server = ls.LitServer(api, devices=args.device)
    server.run(port=6969, generate_client_file=False)