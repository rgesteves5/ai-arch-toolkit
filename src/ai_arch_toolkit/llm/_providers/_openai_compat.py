"""Provider for OpenAI-compatible APIs (OpenAI, xAI, Mistral, Groq)."""

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
    ThinkingConfig,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)

OPENAI_COMPAT_PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com",
        "path": "/v1/chat/completions",
        "env_key": "OPENAI_API_KEY",
    },
    "xai": {
        "base_url": "https://api.x.ai",
        "path": "/v1/chat/completions",
        "env_key": "XAI_API_KEY",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai",
        "path": "/v1/chat/completions",
        "env_key": "MISTRAL_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com",
        "path": "/openai/v1/chat/completions",
        "env_key": "GROQ_API_KEY",
    },
}


def _tool_to_openai(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _parse_tool_args(raw_args: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw_args}


def _content_to_openai(content: Content) -> str | list[dict[str, Any]]:
    """Convert Content to OpenAI wire format."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            url = f"data:{part.media_type};base64,{part.data}" if part.data else part.url
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": part.detail},
                }
            )
        elif isinstance(part, AudioPart):
            fmt = part.media_type.split("/")[-1] if part.media_type else "wav"
            parts.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": part.data, "format": fmt},
                }
            )
        elif isinstance(part, DocumentPart):
            raise ValueError("OpenAI Chat Completions does not support document/PDF content.")
    return parts


def _parse_response(raw: dict[str, Any]) -> Response:
    choices = raw.get("choices", [])
    if not choices:
        return Response(raw=raw)

    choice = choices[0]
    message = choice.get("message", {})
    text = message.get("content") or ""

    tool_calls: list[ToolCall] = []
    for tc in message.get("tool_calls", []):
        fn = tc.get("function", {})
        tool_calls.append(
            ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=_parse_tool_args(fn.get("arguments", "{}")),
            )
        )

    raw_usage = raw.get("usage", {})
    usage = Usage(
        input_tokens=raw_usage.get("prompt_tokens", 0),
        output_tokens=raw_usage.get("completion_tokens", 0),
        total_tokens=raw_usage.get("total_tokens", 0),
    )

    return Response(
        text=text,
        tool_calls=tuple(tool_calls),
        usage=usage,
        stop_reason=choice.get("finish_reason", ""),
        raw=raw,
    )


def _message_to_wire(item: ConversationItem) -> dict[str, Any]:
    """Convert a ConversationItem to the OpenAI wire format dict."""
    if isinstance(item, ToolResult):
        return {
            "role": "tool",
            "tool_call_id": item.tool_call_id,
            "content": item.content,
        }
    # Message with tool_calls (assistant turn)
    if item.tool_calls:
        msg: dict[str, Any] = {"role": item.role}
        if item.content:
            msg["content"] = _content_to_openai(item.content)
        else:
            msg["content"] = None
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in item.tool_calls
        ]
        return msg
    return {"role": item.role, "content": _content_to_openai(item.content)}


class OpenAICompatProvider(BaseProvider):
    """Handles OpenAI, xAI, Mistral, and Groq via the shared Chat Completions format."""

    def __init__(
        self,
        provider_name: str,
        model: str,
        api_key: str,
        *,
        retry: RetryConfig | None = None,
    ) -> None:
        super().__init__(retry=retry)
        cfg = OPENAI_COMPAT_PROVIDERS[provider_name]
        self._url = cfg["base_url"] + cfg["path"]
        self._model = model
        self._provider_name = provider_name
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
        # Pop known kwargs before spreading the rest
        thinking: ThinkingConfig | None = kwargs.pop("thinking", None)
        audio = kwargs.pop("audio", None)

        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(_message_to_wire(m) for m in messages)

        payload: dict[str, Any] = {"model": self._model, "messages": msgs, **kwargs}
        if tools:
            payload["tools"] = [_tool_to_openai(t) for t in tools]
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.name,
                    "schema": json_schema.schema,
                    "strict": json_schema.strict,
                },
            }
        if thinking:
            payload["reasoning_effort"] = thinking.effort
            if self._provider_name == "groq":
                payload["include_reasoning"] = True
        if audio:
            payload["modalities"] = ["text", "audio"]
            payload["audio"] = audio
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
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                if text := delta.get("content"):
                    yield text
            except (json.JSONDecodeError, KeyError, IndexError):
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
        payload["stream_options"] = {"include_usage": True}

        # Accumulate tool call deltas: index -> {id, name, arguments}
        tc_acc: dict[int, dict[str, str]] = {}

        for data in stream_sse(
            self._url, self._headers, payload, timeout=timeout, retry=self._retry
        ):
            if data.strip() == "[DONE]":
                yield StreamEvent(type="done")
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            # Usage chunk (final chunk with usage info)
            if raw_usage := chunk.get("usage"):
                yield StreamEvent(
                    type="usage",
                    usage=Usage(
                        input_tokens=raw_usage.get("prompt_tokens", 0),
                        output_tokens=raw_usage.get("completion_tokens", 0),
                        total_tokens=raw_usage.get("total_tokens", 0),
                    ),
                )
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})

            # Text delta
            if text := delta.get("content"):
                yield StreamEvent(type="text", text=text)

            # Tool call deltas
            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                if idx not in tc_acc:
                    tc_acc[idx] = {
                        "id": tc_delta.get("id", ""),
                        "name": tc_delta.get("function", {}).get("name", ""),
                        "arguments": "",
                    }
                else:
                    if tc_id := tc_delta.get("id"):
                        tc_acc[idx]["id"] = tc_id
                    if fn_name := tc_delta.get("function", {}).get("name"):
                        tc_acc[idx]["name"] = fn_name
                tc_acc[idx]["arguments"] += tc_delta.get("function", {}).get("arguments", "")

            # Emit completed tool calls on finish_reason
            if choice.get("finish_reason") == "tool_calls":
                for _idx in sorted(tc_acc):
                    acc = tc_acc[_idx]
                    yield StreamEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=acc["id"],
                            name=acc["name"],
                            arguments=_parse_tool_args(acc["arguments"]),
                        ),
                    )
                tc_acc.clear()

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
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                if text := delta.get("content"):
                    yield text
            except (json.JSONDecodeError, KeyError, IndexError):
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
        payload["stream_options"] = {"include_usage": True}

        tc_acc: dict[int, dict[str, str]] = {}

        async for data in async_stream_sse(
            self._url, self._headers, payload, timeout=timeout, retry=self._retry
        ):
            if data.strip() == "[DONE]":
                yield StreamEvent(type="done")
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            if raw_usage := chunk.get("usage"):
                yield StreamEvent(
                    type="usage",
                    usage=Usage(
                        input_tokens=raw_usage.get("prompt_tokens", 0),
                        output_tokens=raw_usage.get("completion_tokens", 0),
                        total_tokens=raw_usage.get("total_tokens", 0),
                    ),
                )
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})

            if text := delta.get("content"):
                yield StreamEvent(type="text", text=text)

            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                if idx not in tc_acc:
                    tc_acc[idx] = {
                        "id": tc_delta.get("id", ""),
                        "name": tc_delta.get("function", {}).get("name", ""),
                        "arguments": "",
                    }
                else:
                    if tc_id := tc_delta.get("id"):
                        tc_acc[idx]["id"] = tc_id
                    if fn_name := tc_delta.get("function", {}).get("name"):
                        tc_acc[idx]["name"] = fn_name
                tc_acc[idx]["arguments"] += tc_delta.get("function", {}).get("arguments", "")

            if choice.get("finish_reason") == "tool_calls":
                for _idx in sorted(tc_acc):
                    acc = tc_acc[_idx]
                    yield StreamEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=acc["id"],
                            name=acc["name"],
                            arguments=_parse_tool_args(acc["arguments"]),
                        ),
                    )
                tc_acc.clear()
