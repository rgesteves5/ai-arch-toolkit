"""Provider for the OpenAI Responses API (/v1/responses)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ai_arch_toolkit.llm._async_http import async_post_json, async_stream_sse
from ai_arch_toolkit.llm._http import RetryConfig, post_json, stream_sse
from ai_arch_toolkit.llm._providers._base import BaseProvider
from ai_arch_toolkit.llm._types import (
    Content,
    ConversationItem,
    DocumentPart,
    ImagePart,
    JsonSchema,
    Response,
    ServerTool,
    StreamEvent,
    TextPart,
    ThinkingConfig,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)


def _content_to_responses(content: Content) -> str | list[dict[str, Any]]:
    """Convert Content to Responses API input format."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImagePart):
            url = f"data:{part.media_type};base64,{part.data}" if part.data else part.url
            parts.append({"type": "input_image", "image_url": url, "detail": part.detail})
        elif isinstance(part, DocumentPart):
            if part.data:
                parts.append(
                    {
                        "type": "input_file",
                        "file_data": f"data:{part.media_type};base64,{part.data}",
                    }
                )
            else:
                parts.append({"type": "input_file", "file_id": part.uri})
    return parts


def _items_to_input(
    items: list[ConversationItem],
) -> str | list[dict[str, Any]]:
    """Convert conversation items to Responses API input format."""
    # Single user string shortcut
    if len(items) == 1 and not isinstance(items[0], ToolResult):
        item = items[0]
        if not item.tool_calls and isinstance(item.content, str):
            return item.content

    result: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, ToolResult):
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": item.tool_call_id,
                    "output": item.content,
                }
            )
        elif item.tool_calls:
            # Emit function_call items for each tool call
            if item.content:
                result.append(
                    {
                        "type": "message",
                        "role": item.role,
                        "content": _content_to_responses(item.content),
                    }
                )
            for tc in item.tool_calls:
                result.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                )
        else:
            content = _content_to_responses(item.content)
            result.append({"type": "message", "role": item.role, "content": content})
    return result


def _parse_response(raw: dict[str, Any]) -> Response:
    """Parse a Responses API response."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for item in raw.get("output", []):
        item_type = item.get("type", "")
        if item_type == "message":
            for content_block in item.get("content", []):
                if content_block.get("type") == "output_text":
                    text_parts.append(content_block.get("text", ""))
        elif item_type == "function_call":
            raw_args = item.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": raw_args}
            tool_calls.append(
                ToolCall(
                    id=item.get("call_id", ""),
                    name=item.get("name", ""),
                    arguments=args,
                )
            )

    raw_usage = raw.get("usage", {})
    usage = Usage(
        input_tokens=raw_usage.get("input_tokens", 0),
        output_tokens=raw_usage.get("output_tokens", 0),
        total_tokens=raw_usage.get("total_tokens", 0),
    )

    return Response(
        text="".join(text_parts).strip(),
        tool_calls=tuple(tool_calls),
        usage=usage,
        stop_reason=raw.get("status", ""),
        raw=raw,
    )


class OpenAIResponsesProvider(BaseProvider):
    """OpenAI Responses API provider (/v1/responses)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        retry: RetryConfig | None = None,
        base_url: str = "https://api.openai.com",
    ) -> None:
        super().__init__(retry=retry)
        self._model = model
        self._url = f"{base_url}/v1/responses"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
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
        server_tools: list[ServerTool] | None = kwargs.pop("server_tools", None)
        previous_response_id: str | None = kwargs.pop("previous_response_id", None)

        input_data = _items_to_input(messages)
        payload: dict[str, Any] = {"model": self._model, "input": input_data}

        if system:
            payload["instructions"] = system

        all_tools: list[dict[str, Any]] = []
        if tools:
            for t in tools:
                all_tools.append(
                    {
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                )
        if server_tools:
            for st in server_tools:
                tool_def: dict[str, Any] = {"type": st.type, **st.config}
                all_tools.append(tool_def)
        if all_tools:
            payload["tools"] = all_tools

        if json_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema.name,
                    "schema": json_schema.schema,
                    "strict": json_schema.strict,
                }
            }

        if thinking:
            reasoning: dict[str, Any] = {"effort": thinking.effort}
            if thinking.budget_tokens:
                reasoning["budget_tokens"] = thinking.budget_tokens
            payload["reasoning"] = reasoning

        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        # Pass through remaining kwargs
        for k, v in kwargs.items():
            payload[k] = v

        return payload

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
        payload = self._build_payload(
            messages, system=system, tools=tools, json_schema=json_schema, **kwargs
        )
        raw = post_json(self._url, self._headers, payload, timeout=timeout, retry=self._retry)
        return _parse_response(raw)

    def stream(
        self,
        messages: list[ConversationItem],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        timeout = kwargs.pop("timeout", 120)
        payload = self._build_payload(messages, system=system, **kwargs)
        payload["stream"] = True
        for data in stream_sse(
            self._url, self._headers, payload, timeout=timeout, retry=self._retry
        ):
            try:
                event = json.loads(data)
                if event.get("type") == "response.output_text.delta" and (
                    text := event.get("delta", "")
                ):
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
        payload = self._build_payload(messages, system=system, tools=tools, **kwargs)
        payload["stream"] = True

        fn_acc: dict[str, dict[str, str]] = {}

        for data in stream_sse(
            self._url, self._headers, payload, timeout=timeout, retry=self._retry
        ):
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "response.output_text.delta":
                if text := event.get("delta", ""):
                    yield StreamEvent(type="text", text=text)

            elif event_type == "response.function_call_arguments.delta":
                call_id = event.get("call_id", "")
                if call_id not in fn_acc:
                    fn_acc[call_id] = {
                        "name": event.get("name", ""),
                        "arguments": "",
                    }
                fn_acc[call_id]["arguments"] += event.get("delta", "")

            elif event_type == "response.function_call_arguments.done":
                call_id = event.get("call_id", "")
                acc = fn_acc.pop(call_id, {"name": event.get("name", ""), "arguments": ""})
                raw_args = acc.get("arguments", event.get("arguments", "{}"))
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": raw_args}
                yield StreamEvent(
                    type="tool_call",
                    tool_call=ToolCall(id=call_id, name=acc.get("name", ""), arguments=args),
                )

            elif event_type == "response.completed":
                resp = event.get("response", {})
                raw_usage = resp.get("usage", {})
                if raw_usage:
                    yield StreamEvent(
                        type="usage",
                        usage=Usage(
                            input_tokens=raw_usage.get("input_tokens", 0),
                            output_tokens=raw_usage.get("output_tokens", 0),
                            total_tokens=raw_usage.get("total_tokens", 0),
                        ),
                    )
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
        payload = self._build_payload(
            messages, system=system, tools=tools, json_schema=json_schema, **kwargs
        )
        raw = await async_post_json(
            self._url, self._headers, payload, timeout=timeout, retry=self._retry
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
        payload = self._build_payload(messages, system=system, **kwargs)
        payload["stream"] = True
        async for data in async_stream_sse(
            self._url, self._headers, payload, timeout=timeout, retry=self._retry
        ):
            try:
                event = json.loads(data)
                if event.get("type") == "response.output_text.delta" and (
                    text := event.get("delta", "")
                ):
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
        payload = self._build_payload(messages, system=system, tools=tools, **kwargs)
        payload["stream"] = True

        fn_acc: dict[str, dict[str, str]] = {}

        async for data in async_stream_sse(
            self._url, self._headers, payload, timeout=timeout, retry=self._retry
        ):
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "response.output_text.delta":
                if text := event.get("delta", ""):
                    yield StreamEvent(type="text", text=text)

            elif event_type == "response.function_call_arguments.delta":
                call_id = event.get("call_id", "")
                if call_id not in fn_acc:
                    fn_acc[call_id] = {
                        "name": event.get("name", ""),
                        "arguments": "",
                    }
                fn_acc[call_id]["arguments"] += event.get("delta", "")

            elif event_type == "response.function_call_arguments.done":
                call_id = event.get("call_id", "")
                acc = fn_acc.pop(call_id, {"name": event.get("name", ""), "arguments": ""})
                raw_args = acc.get("arguments", event.get("arguments", "{}"))
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": raw_args}
                yield StreamEvent(
                    type="tool_call",
                    tool_call=ToolCall(id=call_id, name=acc.get("name", ""), arguments=args),
                )

            elif event_type == "response.completed":
                resp = event.get("response", {})
                raw_usage = resp.get("usage", {})
                if raw_usage:
                    yield StreamEvent(
                        type="usage",
                        usage=Usage(
                            input_tokens=raw_usage.get("input_tokens", 0),
                            output_tokens=raw_usage.get("output_tokens", 0),
                            total_tokens=raw_usage.get("total_tokens", 0),
                        ),
                    )
                yield StreamEvent(type="done")
