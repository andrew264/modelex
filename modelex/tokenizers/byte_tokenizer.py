import re

class ByteTokenizer:
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = {'<s>': 256, '</s>': 257, '<pad>': 258}
        if not all(isinstance(k, str) and isinstance(v, int) and v > 255 for k, v in special_tokens.items()):
            raise ValueError("Special tokens must be a dictionary of {str: int} where int >= 256.")
        self.special_tokens = special_tokens
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.token_regex = re.compile('|'.join(re.escape(token) for token in special_tokens.keys()))

    @property
    def vocab_size(self):
        return max(self.special_tokens.values(), default=255) + 1

    def encode(self, s: str, **kwargs) -> list[int]:
        encoded = []
        last_pos = 0
        for match in self.token_regex.finditer(s):
            if match.start() > last_pos:
                encoded.extend(s[last_pos:match.start()].encode())
            encoded.append(self.special_tokens[match.group()])
            last_pos = match.end()
        if last_pos < len(s):
            encoded.extend(s[last_pos:].encode())
        return encoded

    def decode(self, tokens: list[int], **kwargs) -> str:
        decoded = []
        byte_buffer = []
        for token in tokens:
            if token in self.inv_special_tokens:
                if byte_buffer:
                    decoded.append(bytes(byte_buffer).decode("utf-8", errors="backslashreplace"))
                    byte_buffer = []
                decoded.append(self.inv_special_tokens[token])
            elif 0 <= token < 256:
                byte_buffer.append(token)
            else:
                raise ValueError(f"Invalid token: {token}")
        if byte_buffer:
            decoded.append(bytes(byte_buffer).decode("utf-8", errors="backslashreplace"))
        return ''.join(decoded)
