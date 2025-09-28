import json
from typing import Annotated, Any, Dict, List, Literal, Optional, Self, Union

from pydantic import BaseModel, Field

class TextSegment(BaseModel):
    text: str
    learnable: bool = True

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    segments: List[TextSegment]
    @property
    def full_text(self) -> str:
        return "".join(s.text for s in self.segments)

class ReasoningContent(BaseModel):
    type: Literal["reason"] = "reason"
    segments: List[TextSegment]
    @property
    def full_text(self) -> str:
        return "".join(s.text for s in self.segments)

class FunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Any]
    required: List[str] = Field(default_factory=list)

class FunctionSchema(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters

class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionSchema

class ToolsContent(BaseModel):
    type: Literal["tools"] = "tools"
    definitions: List[ToolDefinition]

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolCallContent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    calls: List[ToolCall]

class ToolResult(BaseModel):
    name: str
    content: Any

class ToolResultsContent(BaseModel):
    type: Literal["tool_response"] = "tool_response"
    results: List[ToolResult]

ContentItem = Annotated[Union[TextContent, ReasoningContent, ToolsContent, ToolCallContent, ToolResultsContent,], Field(discriminator="type"),]

class Item(BaseModel):
    role: str
    content: List[ContentItem]

ConversationData = List[Item]

DEFAULT_TOKENS = {"bos": "<|begin_of_text|>", "eot": "<|eot_id|>", "header_start": "<|start_header_id|>", "header_end": "<|end_header_id|>",
    "think_start": "<think>", "think_end": "</think>", "tools_start": "<tools>", "tools_end": "</tools>", "tool_call_start": "<tool_call>",
    "tool_call_end": "</tool_call>", "tool_response_start": "<tool_response>", "tool_response_end": "</tool_response>",
}

class ConversationFormatter:
    def __init__(self, assistant_name: str = "assistant", special_tokens: Optional[Dict[str, str]] = None, json_indent: Optional[int] = 0,
                 with_reason: bool = True, ) -> None:
        self.assistant_name = assistant_name
        self.json_indent = json_indent
        self.with_reason = with_reason
        self.tokens = DEFAULT_TOKENS.copy()
        if special_tokens: self.tokens.update(special_tokens)
        self.msgs: ConversationData = []
    def _serialize_tools(self, content: ToolsContent) -> str:
        try:
            definitions_as_dicts = [d.model_dump(exclude_none=True) for d in content.definitions]
            json_str = json.dumps(definitions_as_dicts, indent=self.json_indent)
            return f"{self.tokens['tools_start']}\n{json_str}\n{self.tokens['tools_end']}"
        except Exception:
            return f"{self.tokens['tools_start']}\n[]\n{self.tokens['tools_end']}"
    def _serialize_tool_calls(self, content: ToolCallContent) -> str:
        try:
            calls_as_dicts = [c.model_dump(exclude_none=True) for c in content.calls]
            call_strings = []
            for call_dict in calls_as_dicts:
                json_str = json.dumps(call_dict, indent=self.json_indent)
                call_strings.append(f"{self.tokens['tool_call_start']}\n{json_str}\n{self.tokens['tool_call_end']}")
            return "\n".join(call_strings)
        except Exception:
            return f"{self.tokens['tool_call_start']}\n{{}}\n{self.tokens['tool_call_end']}"
    def _serialize_tool_results(self, content: ToolResultsContent) -> str:
        try:
            results_as_dicts = [r.model_dump(exclude_none=True) for r in content.results]
            json_str = json.dumps(results_as_dicts, indent=self.json_indent)
            return f"{self.tokens['tool_response_start']}\n{json_str}\n{self.tokens['tool_response_end']}"
        except Exception:
            return f"{self.tokens['tool_response_start']}\n[]\n{self.tokens['tool_response_end']}"
    def __str__(self) -> str:
        parts = [self.tokens["bos"]]
        for item in self.msgs:
            header = f"{self.tokens['header_start']}{item.role}{self.tokens['header_end']}\n\n"
            turn_content_parts = []
            for content_part in item.content:
                if isinstance(content_part, TextContent): turn_content_parts.append(content_part.full_text)
                elif isinstance(content_part, ReasoningContent):
                    if self.with_reason:
                        think_block = f"{self.tokens['think_start']}{content_part.full_text}{self.tokens['think_end']}"
                        turn_content_parts.append(think_block)
                elif isinstance(content_part, ToolsContent): turn_content_parts.append(self._serialize_tools(content_part))
                elif isinstance(content_part, ToolCallContent): turn_content_parts.append(self._serialize_tool_calls(content_part))
                elif isinstance(content_part, ToolResultsContent): turn_content_parts.append(self._serialize_tool_results(content_part))
            full_turn_content = "".join(turn_content_parts)
            turn_string = f"{header}{full_turn_content}\n{self.tokens['eot']}"
            parts.append(turn_string)
        return "".join(parts)
    def add_msg(self, role: str, content: List[ContentItem]) -> Self:
        self.msgs.append(Item(role=role, content=content))
        return self
    def add_msgs(self, msgs: ConversationData) -> Self:
        self.msgs.extend(msgs)
        return self
    def reset(self) -> Self:
        self.msgs = []
        return self
    def get_prompt_for_completion(self): return str(self) + f"\n{self.tokens['header_start']}{self.assistant_name}{self.tokens['header_end']}\n"