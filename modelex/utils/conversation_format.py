from enum import StrEnum
from typing import Dict, List, Literal, Self, Type, TypedDict, Union

class TextContent(TypedDict):
    """
    Represents a text content item in the JSON format.
    """
    type: Literal["text"]
    text: str

class ReasonContent(TypedDict):
    """
    Represents a reasoning content item in the JSON format.
    """
    type: Literal["reason"]
    text: str

ContentItem = Union[TextContent, ReasonContent]

class Item(TypedDict):
    """
    Represents a single item (e.g., a message)
    in the overall JSON structure.
    """
    role: str
    content: List[ContentItem]

Conversation = List[Item]

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

class GraniteFormat(ChatFormat):
    """Granite specific format."""
    BOT: str = ''
    EOT: str = '<|end_of_text|>\n'
    SH: str = '<|start_of_role|>'
    EH: str = '<|end_of_role|>'

class ChatFormatType(StrEnum):
    LLAMA3 = 'llama3'
    GEMMA2 = 'gemma2'
    CUSTOM = 'custom'
    CHATML = 'chatml'
    GRANITE = 'granite'

class ChatFormatFactory:
    _formats: Dict[ChatFormatType, Type[ChatFormat]] = {
        ChatFormatType.LLAMA3: Llama3Format,
        ChatFormatType.GEMMA2: Gemma2Format,
        ChatFormatType.CUSTOM: CustomFormat,
        ChatFormatType.CHATML: ChatMLFormat,
        ChatFormatType.GRANITE: GraniteFormat,
    }

    @classmethod
    def create(cls, format_type: ChatFormatType) -> ChatFormat:
        if format_type not in cls._formats: raise ValueError(f"Unsupported format type: {format_type}")
        return cls._formats[format_type]()

DEFAULT_SYSTEM_PROMPT = "You are an intelligent and helpful assistant"

class ConversationFormatter:
    def __init__(self, assistant_name: str = "assistant", chat_format: str = "llama3") -> None:
        self.assistant_name = assistant_name
        self._apply_format(chat_format)
        self.msgs: Conversation = []
    def _apply_format(self, chat_format: str):
        fmt = ChatFormatFactory.create(ChatFormatType(chat_format))
        self.BOT, self.EOT = fmt.BOT, fmt.EOT
        self.SH, self.EH = fmt.SH, fmt.EH
    def __str__(self) -> str:
        sysprompt = DEFAULT_SYSTEM_PROMPT
        msgs = self.msgs.copy()
        if msgs[0]['role'] == 'system':
            sysprompt = msgs.pop(0)['content'][0]['text']
        out = f'{self.BOT}{self.SH}system{self.EH}{sysprompt}{self.EOT}'
        for msg in msgs:
            out += f'{self.SH}{msg["role"]}{self.EH}'
            for row in msg['content']:
                out += row['text']
            out += self.EOT
        return out
    def add_msg(self, role: str, content: List[ContentItem]) -> Self:
        self.msgs.append({"role": role, "content": content})
        return self
    def add_msgs(self, msgs: Conversation) -> Self:
        self.msgs.extend(msgs)
        return self
    def reset(self) -> Self:
        self.msgs: Conversation = []
        return self
    def get_prompt_for_completion(self) -> str:
        return str(self) + f'{self.SH}{self.assistant_name}{self.EH}'