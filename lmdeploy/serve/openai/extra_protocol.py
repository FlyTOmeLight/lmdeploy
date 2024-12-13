import time
from typing import Any, Dict, List, Optional, Union, Literal

import shortuuid
from lmdeploy.serve.openai.protocol import (ResponseFormat, StreamOptions,
                                            Tool, ToolChoice, UsageInfo,
                                            ChatMessage, ChoiceLogprobs)
from pydantic import BaseModel, Field


class BatchChatCompletionRequest(BaseModel):
    """Chat completion request."""
    model: str
    # yapf: disable
    messages: Union[List, List[List[Dict[str, Any]]]] = Field(examples=[[[{'role': 'user', 'content': 'hi'}]]])  # noqa
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])
    tool_choice: Union[ToolChoice, Literal['auto', 'required','none']] = Field(default='auto', examples=['none'])  # noqa
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    n: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = Field(default=None, examples=[None])  # noqa
    max_tokens: Optional[int] = Field(default=None, examples=[None])
    stop: Optional[Union[str, List[str]]] = Field(default=None, examples=[None])  # noqa
    # yapf: enable
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = Field(default=None,
                                                    examples=[None])
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    response_format: Optional[ResponseFormat] = Field(default=None,
                                                      examples=[None])  # noqa
    # additional argument of lmdeploy
    repetition_penalty: Optional[float] = 1.0
    session_id: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True
    top_k: Optional[int] = 40
    seed: Optional[int] = None
    min_new_tokens: Optional[int] = Field(default=None, examples=[None])
    min_p: float = 0.0


class BatchChatCompletionResponseChoice(BaseModel):
    """Chat completion response choices."""
    index: int
    message: ChatMessage
    logprobs: Optional[ChoiceLogprobs] = None
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls']] = None
    usage: Optional[UsageInfo] = None


class BatchChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    object: str = 'batch.chat.completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[BatchChatCompletionResponseChoice]
    usage: UsageInfo
