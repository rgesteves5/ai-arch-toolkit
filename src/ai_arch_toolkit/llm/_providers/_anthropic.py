"""Provider for the Anthropic Messages API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ai_arch_toolkit.llm._async_http import async_post_json, async_stream_sse
from ai_arch_toolkit.llm._http import RetryConfig, post_json, stream_sse
from ai_arch_toolkit.llm._providers._base import BaseProvider
from ai_arch_toolkit.llm._types import (
    AudioPart,
    Content,
    ConversationItem,
    DocumentPart,
    ImagePart,
    JsonSchema,
    Response,
    StreamEvent,
    TextPart,
    ThinkingBlock,
    ThinkingConfig,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)

_BASE_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096


def _tool_to_anthropic(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def _content_to_anthropic(content: Content) -> str | list[dict[str, Any]]:
    """Convert Content to Anthropic wire format."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            if part.data:
                parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": part.data,
                        },
                    }
                )
            else:
                parts.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": part.url},
                    }
                )
        elif isinstance(part, DocumentPart):
            parts.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": part.data,
                    },
                }
            )
        elif isinstance(part, AudioPart):
            raise ValueError("Anthropic does not support audio content.")
    return parts


def _parse_response(raw: dict[str, Any]) -> Response:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thinking_blocks: list[ThinkingBlock] = []

    for block in raw.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                )
            )
        elif block.get("type") == "thinking":
            thinking_blocks.append(ThinkingBlock(text=block.get("thinking", "")))

    raw_usage = raw.get("usage", {})
    usage = Usage(
        input_tokens=raw_usage.get("input_tokens", 0),
        output_tokens=raw_usage.get("output_tokens", 0),
        total_tokens=raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0),
        cache_creation_tokens=raw_usage.get("cache_creation_input_tokens", 0),
        cache_read_tokens=raw_usage.get("cache_read_input_tokens", 0),
    )

    thinking = "\n\n".join(tb.text for tb in thinking_blocks)

    return Response(
        text="".join(text_parts).strip(),
        tool_calls=tuple(tool_calls),
        usage=usage,
        stop_reason=raw.get("stop_reason", ""),
        thinking=thinking,
        thinking_blocks=tuple(thinking_blocks),
        raw=raw,
    )


def _items_to_wire(items: list[ConversationItem]) -> list[dict[str, Any]]:
    """Convert conversation items to Anthropic wire format messages."""
    msgs: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def _flush_tool_results() -> None:
        if pending_tool_results:
            msgs.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for item in items:
        if isinstance(item, ToolResult):
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": item.tool_call_id,
                    "content": item.content,
                }
            )
        else:
            _flush_tool_results()
            if item.tool_calls:
                # Assistant message with tool_use blocks
                content_blocks: list[dict[str, Any]] = []
                if item.content:
                    wire_content = _content_to_anthropic(item.content)
                    if isinstance(wire_content, str):
                        content_blocks.append({"type": "text", "text": wire_content})
                    else:
                        content_blocks.extend(wire_content)
                for tc in item.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                msgs.append({"role": item.role, "content": content_blocks})
            else:
                msgs.append(
                    {
                        "role": item.role,
                        "content": _content_to_anthropic(item.content),
                    }
                )

    _flush_tool_results()
    return msgs


class AnthropicProvider(BaseProvider):
    """Anthropic Messages API provider."""

    def __init__(self, model: str, api_key: str, *, retry: RetryConfig | None = None) -> None:
        super().__init__(retry=retry)
        self._model = model
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        json_schema: JsonSchema | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        thinking: ThinkingConfig | None = kwargs.pop("thinking", None)
        cache_control: bool = kwargs.pop("cache_control", False)
        kwargs.pop("computer_use", False)  # consumed here; used in _request_headers

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _items_to_wire(messages),
            "max_tokens": kwargs.pop("max_tokens", _DEFAULT_MAX_TOKENS),
            **kwargs,
        }
        sys_text = system or ""
        if json_schema:
            schema_instruction = (
                f"\n\nYou must respond with valid JSON matching this schema:\n"
                f"```json\n{json.dumps(json_schema.schema, indent=2)}\n```"
            )
            sys_text = (sys_text + schema_instruction).strip()
        if sys_text:
            if cache_control:
                payload["system"] = [
                    {"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                payload["system"] = sys_text
        if tools:
            payload["tools"] = [_tool_to_anthropic(t) for t in tools]
        if thinking:
            if thinking.budget_tokens:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking.budget_tokens,
                }
            else:
                payload["thinking"] = {"type": "adaptive", "effort": thinking.effort}
        return payload

    def _request_headers(self, **kwargs: Any) -> dict[str, str]:
        """Return headers for a single request, adding beta headers as needed."""
        headers = dict(self._headers)
        if kwargs.get("cache_control"):
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        if kwargs.get("computer_use"):
            headers["anthropic-beta"] = "computer-use-2025-01-24"
        return headers

    def complete(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        json_schema: JsonSchema | None = None,
        **kwargs: Any,
    ) -> Response:
        timeout = kwargs.pop("timeout", 60)
        headers = self._request_headers(**kwargs)
        payload = self._build_payload(
            messages, system=system, tools=tools, json_schema=json_schema, **kwargs
        )
        raw = post_json(_BASE_URL, headers, payload, timeout=timeout, retry=self._retry)
        return _parse_response(raw)

    def stream(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        timeout = kwargs.pop("timeout", 120)
        headers = self._request_headers(**kwargs)
        payload = self._build_payload(messages, system=system, **kwargs)
        payload["stream"] = True
        for data in stream_sse(_BASE_URL, headers, payload, timeout=timeout, retry=self._retry):
            try:
                event = json.loads(data)
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text
            except json.JSONDecodeError:
                continue

    def stream_events(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        timeout = kwargs.pop("timeout", 120)
        headers = self._request_headers(**kwargs)
        payload = self._build_payload(messages, system=system, tools=tools, **kwargs)
        payload["stream"] = True

        # Track current content block for tool call accumulation
        current_block: dict[str, Any] | None = None
        tool_args_acc = ""

        for data in stream_sse(_BASE_URL, headers, payload, timeout=timeout, retry=self._retry):
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "content_block_start":
                block = event.get("content_block", {})
                current_block = block
                tool_args_acc = ""

            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                delta_type = delta.get("type", "")
                if delta_type == "text_delta":
                    if text := delta.get("text", ""):
                        yield StreamEvent(type="text", text=text)
                elif delta_type == "thinking_delta":
                    if thinking := delta.get("thinking", ""):
                        yield StreamEvent(type="thinking", thinking=thinking)
                elif delta_type == "input_json_delta":
                    tool_args_acc += delta.get("partial_json", "")

            elif event_type == "content_block_stop":
                if current_block and current_block.get("type") == "tool_use":
                    args: dict[str, object] = {}
                    if tool_args_acc:
                        try:
                            args = json.loads(tool_args_acc)
                        except json.JSONDecodeError:
                            args = {"_raw": tool_args_acc}
                    yield StreamEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=current_block.get("id", ""),
                            name=current_block.get("name", ""),
                            arguments=args,
                        ),
                    )
                current_block = None
                tool_args_acc = ""

            elif event_type == "message_delta":
                delta = event.get("delta", {})
                if raw_usage := event.get("usage"):
                    yield StreamEvent(
                        type="usage",
                        usage=Usage(
                            output_tokens=raw_usage.get("output_tokens", 0),
                        ),
                    )

            elif event_type == "message_stop":
                yield StreamEvent(type="done")

    async def acomplete(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        json_schema: JsonSchema | None = None,
        **kwargs: Any,
    ) -> Response:
        timeout = kwargs.pop("timeout", 60)
        headers = self._request_headers(**kwargs)
        payload = self._build_payload(
            messages, system=system, tools=tools, json_schema=json_schema, **kwargs
        )
        raw = await async_post_json(
            _BASE_URL, headers, payload, timeout=timeout, retry=self._retry
        )
        return _parse_response(raw)

    async def astream(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        timeout = kwargs.pop("timeout", 120)
        headers = self._request_headers(**kwargs)
        payload = self._build_payload(messages, system=system, **kwargs)
        payload["stream"] = True
        async for data in async_stream_sse(
            _BASE_URL, headers, payload, timeout=timeout, retry=self._retry
        ):
            try:
                event = json.loads(data)
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text
            except json.JSONDecodeError:
                continue

    async def astream_events(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        timeout = kwargs.pop("timeout", 120)
        headers = self._request_headers(**kwargs)
        payload = self._build_payload(messages, system=system, tools=tools, **kwargs)
        payload["stream"] = True

        current_block: dict[str, Any] | None = None
        tool_args_acc = ""

        async for data in async_stream_sse(
            _BASE_URL, headers, payload, timeout=timeout, retry=self._retry
        ):
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "content_block_start":
                block = event.get("content_block", {})
                current_block = block
                tool_args_acc = ""

            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                delta_type = delta.get("type", "")
                if delta_type == "text_delta":
                    if text := delta.get("text", ""):
                        yield StreamEvent(type="text", text=text)
                elif delta_type == "thinking_delta":
                    if thinking := delta.get("thinking", ""):
                        yield StreamEvent(type="thinking", thinking=thinking)
                elif delta_type == "input_json_delta":
                    tool_args_acc += delta.get("partial_json", "")

            elif event_type == "content_block_stop":
                if current_block and current_block.get("type") == "tool_use":
                    args: dict[str, object] = {}
                    if tool_args_acc:
                        try:
                            args = json.loads(tool_args_acc)
                        except json.JSONDecodeError:
                            args = {"_raw": tool_args_acc}
                    yield StreamEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=current_block.get("id", ""),
                            name=current_block.get("name", ""),
                            arguments=args,
                        ),
                    )
                current_block = None
                tool_args_acc = ""

            elif event_type == "message_delta":
                if raw_usage := event.get("usage"):
                    yield StreamEvent(
                        type="usage",
                        usage=Usage(
                            output_tokens=raw_usage.get("output_tokens", 0),
                        ),
                    )

            elif event_type == "message_stop":
                yield StreamEvent(type="done")
