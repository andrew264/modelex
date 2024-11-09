from enum import StrEnum
from typing import Dict, Type, TypedDict

class Message(TypedDict):
    user: str
    message: str

class ChatFormat:
    BOT: str = ''  # Beginning of text token
    EOT: str = ''  # End of turn token
    SH: str = ''  # Start header token
    EH: str = ''  # End header token

class Llama3Format(ChatFormat):
    """Llama 3 specific chat format."""
    BOT: str = '<|begin_of_text|>'
    EOT: str = '<|eot_id|>\n'
    SH: str = '<|start_header_id|>'
    EH: str = '<|end_header_id|>\n\n'

class Gemma2Format(ChatFormat):
    """Gemma 2 specific chat format."""
    BOT: str = '<bos>'
    EOT: str = '<end_of_turn>'
    SH: str = '<start_of_turn>'
    EH: str = '\n'

class CustomFormat(ChatFormat):
    """Custom chat format."""
    BOT: str = ''
    EOT: str = '</s>'
    SH: str = '\n<|'
    EH: str = '|>\n<s>'

class ChatMLFormat(ChatFormat):
    """ChatML specific format."""
    BOT: str = ''
    EOT: str = '<|im_end|>\n'
    SH: str = '<|im_start|>'
    EH: str = '\n'

class ChatFormatType(StrEnum):
    LLAMA3 = 'llama3'
    GEMMA2 = 'gemma2'
    CUSTOM = 'custom'
    CHATML = 'chatml'

class ChatFormatFactory:
    _formats: Dict[ChatFormatType, Type[ChatFormat]] = {
        ChatFormatType.LLAMA3: Llama3Format,
        ChatFormatType.GEMMA2: Gemma2Format,
        ChatFormatType.CUSTOM: CustomFormat,
        ChatFormatType.CHATML: ChatMLFormat,
    }

    @classmethod
    def create(cls, format_type: ChatFormatType) -> ChatFormat:
        if format_type not in cls._formats: raise ValueError(f"Unsupported format type: {format_type}")
        return cls._formats[format_type]()

class Prompt(ChatFormat):
    def __init__(self, assistant_name: str, sysprompt: str, chat_format: str = "llama3") -> None:
        self.assistant_name = assistant_name
        self.sysprompt = sysprompt
        self._apply_format(chat_format)
        self.msgs: list[Message] = []
    def _apply_format(self, chat_format: str):
        fmt = ChatFormatFactory.create(ChatFormatType(chat_format))
        self.BOT, self.EOT = fmt.BOT, fmt.EOT
        self.SH, self.EH = fmt.SH, fmt.EH
    def __str__(self) -> str:
        out = f'{self.BOT}{self.SH}system{self.EH}{self.sysprompt}{self.EOT}'
        for m in self.msgs: out += f'{self.SH}{m["user"]}{self.EH}{m["message"]}{self.EOT}'
        return out
    def add_msg(self, user: str, msg: str) -> None: self.msgs.append(Message(user=user, message=msg))
    def add_msgs(self, msgs: list[Message]) -> None: self.msgs.extend(msgs)
    def reset(self) -> None: self.msgs = []
    def get_prompt_for_completion(self) -> str:
        out = str(self)
        out += f'{self.SH}{self.assistant_name}{self.EH}'
        return out