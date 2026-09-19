"""What the official SDKs' stream calls return, built from their own types and accumulators.

The adapters rely on the SDKs to accumulate a stream's final object, so the tests feed them the
SDKs' real event types and processing instead of hand-made namespaces.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, AsyncIterator, Callable, Sequence
from typing import Any, cast

from anthropic import types as anthropic_types
from anthropic.lib.streaming._messages import accumulate_event, build_events
from anthropic.types.raw_message_delta_event import Delta
from openai.types.chat import ChatCompletionChunk

_accumulate_event = cast(Callable[..., Any], accumulate_event)
_ACCUMULATE_HAS_JSON_BUFS = "json_bufs" in inspect.signature(accumulate_event).parameters


class AnthropicStream:
    """What ``messages.stream()`` yields: the SDK's own processing of scripted raw events.

    The final message is the SDK's accumulated snapshot (cumulative usage deltas included).
    """

    def __init__(self, raw_events: Sequence[Any]) -> None:
        self._raw_events = raw_events
        self._snapshot: Any = None

    async def __aenter__(self) -> AnthropicStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        json_bufs: dict[int, bytes] = {}
        for raw in self._raw_events:
            kwargs = {"event": raw, "current_snapshot": self._snapshot}
            if _ACCUMULATE_HAS_JSON_BUFS:
                kwargs["json_bufs"] = json_bufs
            self._snapshot = _accumulate_event(**kwargs)
            for event in build_events(event=raw, message_snapshot=self._snapshot):
                yield event

    async def get_final_message(self) -> Any:
        return self._snapshot


def anthropic_message_start(
    *, model: str = "claude-sonnet-4-6", **usage: int
) -> anthropic_types.RawMessageStartEvent:
    counts = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    message = anthropic_types.Message(
        id="msg_test",
        type="message",
        role="assistant",
        model=model,
        content=[],
        stop_reason=None,
        usage=anthropic_types.Usage(**(counts | usage)),
    )
    return anthropic_types.RawMessageStartEvent(type="message_start", message=message)


def anthropic_block_start(index: int, kind: str, **fields: Any) -> Any:
    blocks = {
        "text": lambda: anthropic_types.TextBlock(type="text", text=""),
        "thinking": lambda: anthropic_types.ThinkingBlock(
            type="thinking", thinking="", signature=""
        ),
        "tool_use": lambda: anthropic_types.ToolUseBlock(
            type="tool_use", id=fields["id"], name=fields["name"], input={}
        ),
    }
    return anthropic_types.RawContentBlockStartEvent(
        type="content_block_start", index=index, content_block=blocks[kind]()
    )


def anthropic_text(index: int, text: str) -> Any:
    delta = anthropic_types.TextDelta(type="text_delta", text=text)
    return anthropic_types.RawContentBlockDeltaEvent(
        type="content_block_delta", index=index, delta=delta
    )


def anthropic_thinking(index: int, thinking: str) -> Any:
    delta = anthropic_types.ThinkingDelta(type="thinking_delta", thinking=thinking)
    return anthropic_types.RawContentBlockDeltaEvent(
        type="content_block_delta", index=index, delta=delta
    )


def anthropic_json(index: int, partial_json: str) -> Any:
    delta = anthropic_types.InputJSONDelta(type="input_json_delta", partial_json=partial_json)
    return anthropic_types.RawContentBlockDeltaEvent(
        type="content_block_delta", index=index, delta=delta
    )


def anthropic_block_stop(index: int) -> Any:
    return anthropic_types.RawContentBlockStopEvent(type="content_block_stop", index=index)


def anthropic_message_end(stop_reason: str = "end_turn", **usage: int) -> list[Any]:
    delta_usage = anthropic_types.MessageDeltaUsage(**({"output_tokens": 0} | usage))
    return [
        anthropic_types.RawMessageDeltaEvent(
            type="message_delta", delta=Delta(stop_reason=stop_reason), usage=delta_usage
        ),
        anthropic_types.RawMessageStopEvent(type="message_stop"),
    ]


class OpenAIStream:
    """What the ``openai`` SDK returns for ``stream=True`` (chat chunks or Responses events):
    an async context manager over the items, which may come from a raising async iterable."""

    def __init__(self, items: Sequence[Any] | AsyncIterable[Any]) -> None:
        self._items = items

    async def __aenter__(self) -> OpenAIStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        if isinstance(self._items, AsyncIterable):
            async for item in self._items:
                yield item
        else:
            for item in self._items:
                yield item


def openai_chunk(
    *,
    content: str | None = None,
    tool_calls: Sequence[dict[str, Any]] = (),
    finish_reason: str | None = None,
    model: str = "gpt-4o",
    reasoning: str | None = None,
    reasoning_field: str = "reasoning_content",
) -> ChatCompletionChunk:
    """One ``ChatCompletionChunk``; a tool-call dict takes ``index``, ``id``, ``name``,
    ``arguments`` (the SDK's shape: later fragments carry only the index and arguments)."""
    delta: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta[reasoning_field] = reasoning
    if tool_calls:
        delta["tool_calls"] = [_tool_call_delta(call) for call in tool_calls]
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    return _chunk(model, [choice])


def openai_usage_chunk(
    prompt_tokens: int, completion_tokens: int, *, model: str = "gpt-4o"
) -> ChatCompletionChunk:
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    return _chunk(model, [], usage=usage)


def _tool_call_delta(call: dict[str, Any]) -> dict[str, Any]:
    fragment: dict[str, Any] = {"index": call.get("index", 0)}
    if "id" in call:
        fragment["id"] = call["id"]
        fragment["type"] = "function"
    function = {key: call[key] for key in ("name", "arguments") if key in call}
    if function:
        fragment["function"] = function
    return fragment


def _chunk(
    model: str, choices: list[dict[str, Any]], usage: dict[str, int] | None = None
) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": choices,
            "usage": usage,
        }
    )
