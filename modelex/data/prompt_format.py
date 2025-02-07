from enum import StrEnum
from typing import Dict, Self, Type, TypedDict

class Message(TypedDict):
    role: str
    content: str

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

DEFAULT_SYSTEM_PROMPT = "You are an intelligent and helpful assistant"

class PromptFormatter(ChatFormat):
    def __init__(self, assistant_name: str = "assistant", chat_format: str = "llama3") -> None:
        self.assistant_name = assistant_name
        self._apply_format(chat_format)
        self.msgs: list[Message] = []
    def _apply_format(self, chat_format: str):
        fmt = ChatFormatFactory.create(ChatFormatType(chat_format))
        self.BOT, self.EOT = fmt.BOT, fmt.EOT
        self.SH, self.EH = fmt.SH, fmt.EH
    def __str__(self) -> str:
        sysprompt = DEFAULT_SYSTEM_PROMPT
        msgs = self.msgs.copy()
        if msgs[0]['role'] == 'system':
            sysprompt = msgs.pop(0)['content']
        out = f'{self.BOT}{self.SH}system{self.EH}{sysprompt}{self.EOT}'
        for m in msgs: out += f'{self.SH}{m["role"]}{self.EH}{m["content"]}{self.EOT}'
        return out
    def add_msg(self, role: str, content: str) -> Self:
        self.msgs.append(Message(role=role, content=content))
        return self
    def add_msgs(self, msgs: list[Message]) -> Self:
        self.msgs.extend(msgs)
        return self
    def reset(self) -> Self:
        self.msgs: list[Message] = []
        return self
    def get_prompt_for_completion(self) -> str:
        return str(self) + f'{self.SH}{self.assistant_name}{self.EH}'