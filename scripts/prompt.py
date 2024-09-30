import datetime
import os
import torch
import argparse

from models.generation_handler import ModelGenerationHandler
from utils.prompt_format import Prompt

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate sequence")
    parser.add_argument("path", type=str, help="Path to the model (required)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (optional, defaults to 'cuda')")
    parser.add_argument("--name", type=str, default="user", help="Username (optional, defaults to 'user')")
    parser.add_argument("--botname", type=str, default="assistant", help="Username (optional, defaults to 'assistant')")

    args = parser.parse_args()
    device = torch.device(args.device)
    model_handler = ModelGenerationHandler(args.path, device=device,)
    model_handler.load_model()

    dt = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    with open(os.path.join(args.path, 'sysprompt.txt'), 'r', encoding='utf-8') as f: sysprompt = f.read().format(datetime=dt).strip()

    prompt = Prompt(args.botname, sysprompt)

    while True:
        inp = multiline_input(args.name)
        if inp == '': break
        if inp.casefold() == 'reset':
            prompt.reset()
            continue
        prompt.add_msg(args.name, inp)
        decoded, num_tokens, _, generation_time = model_handler.generate(prompt.get_prompt_for_completion())
        prompt.add_msg(args.botname, decoded)

        print(f"{args.botname}: {decoded}")
        print(f"Generated {num_tokens} tokens in {generation_time:.3f}s ({num_tokens / generation_time:.3f} tokens/s)")

