import argparse
import datetime
import json
import logging
import sys

try:
    import requests
except ImportError:
    print("The 'requests' package is required. Please install it with 'uv add requests'", file=sys.stderr)
    sys.exit(1)

from modelex.inference.conversation import InferenceItem, InferenceTextContent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Chat with a served Modelex model via its API endpoint.")
parser.add_argument("--host", type=str, default="localhost", help="Host of the inference server (default: localhost)")
parser.add_argument("--port", type=int, default=6969, help="Port of the inference server (default: 6969)")
parser.add_argument("--sysprompt", type=str, default="sysprompt.txt", help="Path to the system prompt file (optional, default: sysprompt.txt).")
parser.add_argument("--name", type=str, default="user", help="Your username (optional, default: 'user')")
parser.add_argument("--botname", type=str, default="assistant", help="The assistant's name (optional, default: 'assistant')")

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
    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    logger.info(f"Connecting to server at {api_url}")

    try:
        dt = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        with open(args.sysprompt, 'r', encoding='utf-8') as f:
            sysprompt_text = f.read().format(datetime=dt).strip()
        logger.info(f"Loaded system prompt from {args.sysprompt}")
    except FileNotFoundError:
        logger.warning(f"System prompt file not found at '{args.sysprompt}'. Starting without a system prompt.")
        sysprompt_text = ""

    messages: list[InferenceItem] = []
    if sysprompt_text: messages.append(InferenceItem(role='system', content=[InferenceTextContent(text=sysprompt_text)]))

    while True:
        user_input = multiline_input(args.name)
        if not user_input: break
        if user_input.casefold() == 'reset':
            messages.clear()
            if sysprompt_text:
                messages.append(InferenceItem(role='system', content=[InferenceTextContent(text=sysprompt_text)]))
            print("Chat history has been reset.")
            continue
        messages.append(InferenceItem(role=args.name, content=[InferenceTextContent(text=user_input)]))
        request_payload = {"messages": [msg.model_dump() for msg in messages], "assistant_name": args.botname}
        full_response_text = ""
        print(f"{args.botname}: ", end="", flush=True)
        try:
            with requests.post(api_url, json=request_payload, stream=True, timeout=300) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:  continue
                    try:
                        event = json.loads(line)
                        event_type = event.get("type")
                        if event_type == "text_chunk":
                            text_chunk = event.get("text", "")
                            print(text_chunk, end="", flush=True)
                            full_response_text += text_chunk
                        elif event_type == "finished":
                            break
                        elif event_type in ("parsing_error", "error") or event.get("reason") == "error":
                            error_details = event.get('details', event.get('error', 'Unknown error'))
                            logger.error(f"\nReceived error from server: {error_details}")
                            break
                    except json.JSONDecodeError:
                        logger.error(f"\nFailed to decode JSON from server response: {line}")

        except requests.exceptions.RequestException as e:
            logger.error(f"\nCould not connect to the server at {api_url}. Is it running?")
            logger.error(f"Details: {e}")
            messages.pop()
            continue

        print()
        if full_response_text: messages.append(InferenceItem(role=args.botname, content=[InferenceTextContent(text=full_response_text)]))

if __name__ == '__main__':
    main(args=parser.parse_args())