import asyncio
import json
from enum import auto, Enum
from typing import AsyncGenerator

from pydantic import ValidationError
from tokenizers import Tokenizer

from modelex.inference.conversation import DEFAULT_TOKENS, ToolCall
from modelex.inference.structs import (ErrorEvent, GenerationFinished, InferenceEvent, ParsedEvent, ParsingError, ReasoningChunk, StopMaxTokensEvent,
                                       StopSequenceHitEvent, TextChunk, TokenEvent, ToolCallCompleted)

class ParserState(Enum):
    GENERATING_TEXT = auto()
    INSIDE_REASONING = auto()
    INSIDE_TOOL_CALL = auto()

class StreamingResponseParser:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.state = ParserState.GENERATING_TEXT
        self.buffer = ""
        self.tokens = {"think_start": DEFAULT_TOKENS["think_start"], "think_end": DEFAULT_TOKENS["think_end"],
            "tool_call_start": DEFAULT_TOKENS["tool_call_start"], "tool_call_end": DEFAULT_TOKENS["tool_call_end"], "eot": DEFAULT_TOKENS["eot"],
        }
    async def parse(self, event_stream: asyncio.Queue[InferenceEvent]) -> AsyncGenerator[ParsedEvent, None]:
        while True:
            event = await event_stream.get()
            if isinstance(event, TokenEvent):
                decoded_text = self.tokenizer.decode([event.token_id])
                if self.state == ParserState.GENERATING_TEXT:
                    if self.tokens["think_start"] in decoded_text:
                        parts = decoded_text.split(self.tokens["think_start"], 1)
                        if parts[0]: yield TextChunk(text=parts[0])
                        self.state = ParserState.INSIDE_REASONING
                        self.buffer = parts[1]
                    elif self.tokens["tool_call_start"] in decoded_text:
                        parts = decoded_text.split(self.tokens["tool_call_start"], 1)
                        if parts[0]: yield TextChunk(text=parts[0])
                        self.state = ParserState.INSIDE_TOOL_CALL
                        self.buffer = parts[1]
                    else: yield TextChunk(text=decoded_text)
                else: self.buffer += decoded_text
            elif isinstance(event, StopSequenceHitEvent):
                final_sequence = event.sequence
                if self.state == ParserState.INSIDE_REASONING and final_sequence == self.tokens["think_end"]:
                    yield ReasoningChunk(text=self.buffer)
                    self.buffer = ""
                    self.state = ParserState.GENERATING_TEXT
                elif self.state == ParserState.INSIDE_TOOL_CALL and final_sequence == self.tokens["tool_call_end"]:
                    try:
                        parsed_tool_call = ToolCall.model_validate_json(self.buffer)
                        yield ToolCallCompleted(tool_call=parsed_tool_call)
                    except (json.JSONDecodeError, ValidationError) as e:
                        yield ParsingError(raw_text=self.buffer, error=str(e))
                    finally:
                        yield GenerationFinished(reason="stop_sequence", details=final_sequence)
                        return
                else:
                    if self.buffer:
                        yield ParsingError(raw_text=self.buffer, error=f"Generation stopped by '{final_sequence}' before block was closed.")
                    yield GenerationFinished(reason="stop_sequence", details=final_sequence)
                    return

            elif isinstance(event, StopMaxTokensEvent):
                if self.buffer: yield ParsingError(raw_text=self.buffer, error="Max tokens reached before block was closed.")
                yield GenerationFinished(reason="max_tokens")
                return
            elif isinstance(event, ErrorEvent):
                yield GenerationFinished(reason="error", details=event.message)
                return