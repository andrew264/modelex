import argparse

import torch

from modelex.generation import ModelGenerationHandler

parser = argparse.ArgumentParser(description="generate sequence")
parser.add_argument("path", type=str, help="Path to the models (required)")
parser.add_argument("--device", type=str, default="cuda", help="Device to run the models on (optional, defaults to 'cuda')")

def main(args):
    device = torch.device(args.device)
    model_handler = ModelGenerationHandler(args.path, device=device, )
    model_handler.load_model()

    while True:
        prompt = input("Enter a prompt: ")
        if prompt == '': break
        decoded, num_tokens, _, generation_time = model_handler.generate(prompt)
        print(f"Model Output: {decoded}")
        print(f"Generated {num_tokens} tokens in {generation_time:.3f}s")

if __name__ == '__main__':
    main(args=parser.parse_args())