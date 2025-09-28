import argparse
import datetime
import logging
import os
import sys

import torch

from modelex.generation import ModelGenerationHandler
from modelex.utils.conversation_format import ConversationFormatter, TextContent, TextSegment

torch.set_float32_matmul_precision('high')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="generate sequence")
parser.add_argument("path", type=str, help="Path to the models (required)")
parser.add_argument("--device", type=str, default="cuda", help="Device to run the models on (optional, defaults to 'cuda')")
parser.add_argument("--name", type=str, default="user", help="Username (optional, defaults to 'user')")
parser.add_argument("--botname", type=str, default="assistant", help="Username (optional, defaults to 'assistant')")
parser.add_argument("--compile", action="store_true", help="Enable torch compile (optional, defaults to False)")

def multiline_input(name: str = 'User'):
    lines = []
    print(f'{name}: ', end="", flush=True)
    while True:
        try:
            line = input()
            if line == '': break
            lines.append(line)
        except KeyboardInterrupt:
            print()
            break
    return '\n'.join(lines)

def main(args):
    device = torch.device(args.device)
    model_handler = ModelGenerationHandler(args.path, device=device, )
    model_handler.load_model(compiled=args.compile)

    dt = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    with open(os.path.join(args.path, 'sysprompt.txt'), 'r', encoding='utf-8') as f: sysprompt = f.read().format(datetime=dt).strip()

    prompt = ConversationFormatter(assistant_name=args.botname)
    prompt.add_msg('system', [TextContent(segments=[TextSegment(text=sysprompt)])])

    while True:
        inp = multiline_input(args.name)
        if inp == '': break
        if inp.casefold() == 'reset':
            prompt.reset()
            prompt.add_msg('system', [TextContent(segments=[TextSegment(text=sysprompt)])])
            continue
        prompt.add_msg(args.name, [TextContent(segments=[TextSegment(text=inp)])])
        decoded = ""
        print(f"{args.botname}:", end="", flush=True)
        for text in model_handler.generate_stream(prompt.get_prompt_for_completion(), skip_special_tokens=False):
            print(text, end="", flush=True)
            decoded += text
        print()
        prompt.add_msg(args.botname, [TextContent(segments=[TextSegment(text=decoded)])])

if __name__ == '__main__':
    main(args=parser.parse_args())