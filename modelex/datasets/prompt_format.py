from enum import Enum
from typing import TypedDict

class Message(TypedDict):
    user: str
    message: str

class ChatFormat:
    BOT, EOT = '', ''
    SH, EH = '', ''

class Llama3Format(ChatFormat):
    BOT, EOT = '<|begin_of_text|>', '<|eot_id|>\n'
    SH, EH = '<|start_header_id|>', '<|end_header_id|>\n\n'

class Gemma2Format(ChatFormat):
    BOT, EOT = '<bos>', '<end_of_turn>'
    SH, EH = '<start_of_turn>', '\n'

class CustomFormat(ChatFormat):
    BOT, EOT = '', '</s>'
    SH, EH = '\n<|', '|>\n<s>'

class ChatFormatType(Enum):
    LLAMA3 = 'llama3'
    GEMMA2 = 'gemma2'
    CUSTOM = 'custom'

class Prompt(ChatFormat):
    def __init__(self, assistant_name: str, sysprompt: str, chat_format: str = "llama3") -> None:
        self.assistant_name = assistant_name
        self.sysprompt = sysprompt
        self.select_format(chat_format=chat_format)
        self.msgs: list[Message] = []
    def select_format(self, chat_format: str):
        match chat_format:
            case ChatFormatType.LLAMA3.value: self._apply_format(Llama3Format)
            case ChatFormatType.GEMMA2.value: self._apply_format(Gemma2Format)
            case ChatFormatType.CUSTOM.value: self._apply_format(CustomFormat)
            case _: raise ValueError(f"Unknown chat_format: {chat_format}")
    def _apply_format(self, fmt: ChatFormat): self.BOT, self.EOT, self.SH, self.EH = fmt.BOT, fmt.EOT, fmt.SH, fmt.EH
    def add_msg(self, user: str, msg: str) -> None: self.msgs.append(Message(user=user, message=msg))
    def add_msgs(self, msgs: list[Message]) -> None: self.msgs.extend(msgs)
    def reset(self) -> None: self.msgs = []
    def get_prompt_for_completion(self) -> str:
        out = f'{self.BOT}{self.SH}system{self.EH}{self.sysprompt}{self.EOT}'
        for m in self.msgs: out += f'{self.SH}{m["user"]}{self.EH}{m["message"]}{self.EOT}'
        out += f'{self.SH}{self.assistant_name}{self.EH}'
        return out