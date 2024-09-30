import argparse
import datetime
import os
from typing import Tuple

import litserve as ls
import torch

from models.generation_handler import ModelGenerationHandler
from utils.prompt_format import Message, Prompt

class ModelAPI(ls.LitAPI):
    def __init__(self, path: str, assistant_name: str):
        super().__init__()
        self.device = None
        self.path = path
        self.model_handler = None
        self.sysprompt: str = ""
        self.assistant_name: str = assistant_name

    def setup(self, devices: str):
        self.device = devices
        torch.set_float32_matmul_precision('high')
        self.model_handler = ModelGenerationHandler(self.path, self.device)
        self.model_handler.load_model(compiled=False)
        with open(os.path.join(self.path, 'sysprompt.txt'), 'r', encoding='utf-8') as f: self.sysprompt = f.read().strip()

    def decode_request(self, request, **kwargs):
        msgs: list[Message] = request['msgs']
        top_p = request.get('top_p', 0.99)
        temp = request.get('temp', 1.7)
        return msgs, top_p, temp
    
    def predict(self, x, **kwargs) -> Tuple[str, int]:
        msgs, top_p, temp = x
        self.model_handler.set_processor(top_p, temp)
        dt = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        p = Prompt(assistant_name=self.assistant_name, sysprompt=self.sysprompt.format(datetime=dt))
        p.add_msgs(msgs)
        decoded, _, total_toks, _ = self.model_handler.generate(p.get_prompt_for_completion(), max_new_tokens=1024)
        return decoded, total_toks
    
    def encode_response(self, output, **kwargs):
        output_text, length = output
        return {'response': output_text, 'cur_length': length, 'max_length': self.model_handler.cfg.max_seq_len,}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate sequence")
    parser.add_argument("path", type=str, help="Path to the model (required)")
    parser.add_argument("--device", type=str, default="0", help="Device to run the model on (optional, defaults to 'gpu 0')")
    parser.add_argument("--botname", type=str, default="assistant", help="Username (optional, defaults to 'assistant')")
    args = parser.parse_args()

    api = ModelAPI(args.path, args.botname)
    server = ls.LitServer(api, devices=args.device)
    server.run(port=6969, generate_client_file=False)
