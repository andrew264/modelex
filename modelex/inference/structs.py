import asyncio
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from modelex.inference.conversation import ToolCall

class GenerationConfig(BaseModel):
    max_new_tokens: int = 1024
    temperature: float = Field(1.0, ge=0.0)
    top_k: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stop_sequences: List[str] = Field(default_factory=list)

class InferenceRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: List[int]
    session_id: str
    generation_config: GenerationConfig
    results_queue: asyncio.Queue

class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    token_id: int

class StopSequenceHitEvent(BaseModel):
    type: Literal["stop_hit"] = "stop_hit"
    sequence: str

class StopMaxTokensEvent(BaseModel):
    type: Literal["stop_max"] = "stop_max"

class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str

InferenceEvent = Annotated[Union[TokenEvent, StopSequenceHitEvent, StopMaxTokensEvent, ErrorEvent,], Field(discriminator="type"),]

class TextChunk(BaseModel):
    type: Literal["text_chunk"] = "text_chunk"
    text: str

class ReasoningChunk(BaseModel):
    type: Literal["reasoning_chunk"] = "reasoning_chunk"
    text: str

class ToolCallCompleted(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall

class ParsingError(BaseModel):
    type: Literal["parsing_error"] = "parsing_error"
    raw_text: str
    error: str

class GenerationFinished(BaseModel):
    type: Literal["finished"] = "finished"
    reason: Literal["stop_sequence", "max_tokens", "error"]
    details: Optional[Any] = None

ParsedEvent = Annotated[Union[TextChunk, ReasoningChunk, ToolCallCompleted, ParsingError, GenerationFinished,], Field(discriminator="type"),]