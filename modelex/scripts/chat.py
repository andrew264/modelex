import argparse
import datetime
import os

import torch

from modelex.generation import ModelGenerationHandler
from modelex.utils.conversation_format import ConversationFormatter

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description="generate sequence")
parser.add_argument("path", type=str, help="Path to the models (required)")
parser.add_argument("--device", type=str, default="cuda", help="Device to run the models on (optional, defaults to 'cuda')")
parser.add_argument("--name", type=str, default="user", help="Username (optional, defaults to 'user')")
parser.add_argument("--botname", type=str, default="assistant", help="Username (optional, defaults to 'assistant')")

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
    model_handler.load_model()

    dt = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    with open(os.path.join(args.path, 'sysprompt.txt'), 'r', encoding='utf-8') as f: sysprompt = f.read().format(datetime=dt).strip()

    prompt = ConversationFormatter(args.botname, chat_format=model_handler.prompt_format).add_msg('system', [{"type": "text", "text": sysprompt}])

    while True:
        inp = multiline_input(args.name)
        if inp == '': break
        if inp.casefold() == 'reset':
            prompt.reset()
            continue
        prompt.add_msg(args.name, [{"type": "text", "text": inp}])
        decoded, num_tokens, _, generation_time = model_handler.generate(prompt.get_prompt_for_completion(), skip_special_tokens=False)
        if isinstance(decoded, str):
            prompt.add_msg(args.botname, [{"type": "text", "text": decoded}])

        print(f"{args.botname}: {decoded}")
        print(f"Generated {num_tokens} tokens in {generation_time:.3f}s ({num_tokens / generation_time:.3f} tokens/s)")

if __name__ == '__main__':
    main(args=parser.parse_args())