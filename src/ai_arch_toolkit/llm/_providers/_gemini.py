"""Provider for the Google Gemini generateContent API."""

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

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_CACHE_URL = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
_VALID_ROLES = {"user", "assistant"}


def _tool_to_gemini(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "functionDeclarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ]
        }
    ]


def _content_to_gemini_parts(content: Content) -> list[dict[str, Any]]:
    """Convert Content to Gemini parts format."""
    if isinstance(content, str):
        return [{"text": content}]
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"text": part.text})
        elif isinstance(part, ImagePart):
            if part.data:
                parts.append({"inlineData": {"mimeType": part.media_type, "data": part.data}})
            else:
                parts.append({"fileData": {"mimeType": part.media_type, "fileUri": part.url}})
        elif isinstance(part, DocumentPart):
            if part.uri:
                parts.append({"fileData": {"mimeType": part.media_type, "fileUri": part.uri}})
            else:
                parts.append({"inlineData": {"mimeType": part.media_type, "data": part.data}})
        elif isinstance(part, AudioPart):
            parts.append({"inlineData": {"mimeType": part.media_type, "data": part.data}})
    return parts


def _parse_response(raw: dict[str, Any]) -> Response:
    candidates = raw.get("candidates", [])
    if not candidates:
        return Response(raw=raw)

    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for part in parts:
        if "text" in part:
            text_parts.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append(
                ToolCall(
                    id="",  # Gemini doesn't provide a tool call ID
                    name=fc.get("name", ""),
                    arguments=fc.get("args", {}),
                )
            )

    raw_usage = raw.get("usageMetadata", {})
    usage = Usage(
        input_tokens=raw_usage.get("promptTokenCount", 0),
        output_tokens=raw_usage.get("candidatesTokenCount", 0),
        total_tokens=raw_usage.get("totalTokenCount", 0),
    )

    return Response(
        text="".join(text_parts).strip(),
        tool_calls=tuple(tool_calls),
        usage=usage,
        stop_reason=candidate.get("finishReason", ""),
        raw=raw,
    )


def _items_to_contents(items: list[ConversationItem]) -> list[dict[str, Any]]:
    """Convert conversation items to Gemini contents format."""
    contents: list[dict[str, Any]] = []
    pending_fn_responses: list[dict[str, Any]] = []

    def _flush_fn_responses() -> None:
        if pending_fn_responses:
            contents.append({"role": "user", "parts": list(pending_fn_responses)})
            pending_fn_responses.clear()

    for item in items:
        if isinstance(item, ToolResult):
            # Try to parse content as JSON for Gemini's functionResponse format
            try:
                response_data = json.loads(item.content)
            except (json.JSONDecodeError, TypeError):
                response_data = {"result": item.content}
            pending_fn_responses.append(
                {
                    "functionResponse": {
                        "name": item.name,
                        "response": response_data,
                    }
                }
            )
        else:
            _flush_fn_responses()
            if item.tool_calls:
                # Assistant turn with function calls
                parts: list[dict[str, Any]] = []
                if item.content:
                    parts.extend(_content_to_gemini_parts(item.content))
                for tc in item.tool_calls:
                    parts.append(
                        {
                            "functionCall": {
                                "name": tc.name,
                                "args": tc.arguments,
                            }
                        }
                    )
                contents.append({"role": "model", "parts": parts})
            else:
                if item.role not in _VALID_ROLES:
                    raise ValueError(
                        f"Gemini does not support role {item.role!r}. "
                        "Use the system= parameter for system prompts."
                    )
                role = "model" if item.role == "assistant" else item.role
                contents.append({"role": role, "parts": _content_to_gemini_parts(item.content)})

    _flush_fn_responses()
    return contents


class GeminiProvider(BaseProvider):
    """Google Gemini generateContent provider."""

    def __init__(self, model: str, api_key: str, *, retry: RetryConfig | None = None) -> None:
        super().__init__(retry=retry)
        self._model = model
        self._api_key = api_key
        self._url = f"{_BASE_URL}/{model}:generateContent"
        self._stream_url = f"{_BASE_URL}/{model}:streamGenerateContent?alt=sse"
        self._headers = {
            "x-goog-api-key": api_key,
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
        google_search = kwargs.pop("google_search", None)

        payload: dict[str, Any] = {"contents": _items_to_contents(messages)}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = _tool_to_gemini(tools)

        if google_search:
            tools_list = payload.get("tools", [])
            if isinstance(google_search, dict):
                threshold = google_search.get("threshold", 0.7)
            else:
                threshold = 0.7
            tools_list.append(
                {
                    "googleSearchRetrieval": {
                        "dynamicRetrievalConfig": {
                            "mode": "MODE_DYNAMIC",
                            "dynamicThreshold": threshold,
                        }
                    }
                }
            )
            payload["tools"] = tools_list

        # Map common kwargs to generationConfig
        gen_config: dict[str, Any] = {}
        if "temperature" in kwargs:
            gen_config["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            gen_config["maxOutputTokens"] = kwargs.pop("max_tokens")
        if "top_p" in kwargs:
            gen_config["topP"] = kwargs.pop("top_p")
        if json_schema:
            gen_config["responseMimeType"] = "application/json"
            gen_config["responseSchema"] = json_schema.schema
        if thinking:
            gen_config["thinkingConfig"] = {"thinkingBudget": thinking.budget_tokens or 8192}
        if gen_config:
            payload["generationConfig"] = gen_config

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
        for data in stream_sse(
            self._stream_url, self._headers, payload, timeout=timeout, retry=self._retry
        ):
            try:
                chunk = json.loads(data)
                for candidate in chunk.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "text" in part:
                            yield part["text"]
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
        for data in stream_sse(
            self._stream_url, self._headers, payload, timeout=timeout, retry=self._retry
        ):
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            for candidate in chunk.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        yield StreamEvent(type="text", text=part["text"])
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        yield StreamEvent(
                            type="tool_call",
                            tool_call=ToolCall(
                                id="",
                                name=fc.get("name", ""),
                                arguments=fc.get("args", {}),
                            ),
                        )

            if raw_usage := chunk.get("usageMetadata"):
                yield StreamEvent(
                    type="usage",
                    usage=Usage(
                        input_tokens=raw_usage.get("promptTokenCount", 0),
                        output_tokens=raw_usage.get("candidatesTokenCount", 0),
                        total_tokens=raw_usage.get("totalTokenCount", 0),
                    ),
                )

        yield StreamEvent(type="done")

    def create_cache(
        self,
        contents: list[ConversationItem],
        *,
        system: str | None = None,
        ttl: str = "300s",
    ) -> str:
        """Create cached content and return the cache name."""
        payload: dict[str, Any] = {
            "model": f"models/{self._model}",
            "contents": _items_to_contents(contents),
            "ttl": ttl,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        raw = post_json(_CACHE_URL, self._headers, payload, timeout=60, retry=self._retry)
        return raw.get("name", "")

    def complete_with_cache(
        self,
        cache_name: str,
        messages: list[ConversationItem],
        **kwargs: Any,
    ) -> Response:
        """Complete using cached content."""
        timeout = kwargs.pop("timeout", 60)
        payload: dict[str, Any] = {
            "contents": _items_to_contents(messages),
            "cachedContent": cache_name,
        }
        gen_config: dict[str, Any] = {}
        if "temperature" in kwargs:
            gen_config["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            gen_config["maxOutputTokens"] = kwargs.pop("max_tokens")
        if gen_config:
            payload["generationConfig"] = gen_config
        raw = post_json(self._url, self._headers, payload, timeout=timeout, retry=self._retry)
        return _parse_response(raw)

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
        async for data in async_stream_sse(
            self._stream_url, self._headers, payload, timeout=timeout, retry=self._retry
        ):
            try:
                chunk = json.loads(data)
                for candidate in chunk.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "text" in part:
                            yield part["text"]
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
        async for data in async_stream_sse(
            self._stream_url, self._headers, payload, timeout=timeout, retry=self._retry
        ):
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            for candidate in chunk.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        yield StreamEvent(type="text", text=part["text"])
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        yield StreamEvent(
                            type="tool_call",
                            tool_call=ToolCall(
                                id="",
                                name=fc.get("name", ""),
                                arguments=fc.get("args", {}),
                            ),
                        )

            if raw_usage := chunk.get("usageMetadata"):
                yield StreamEvent(
                    type="usage",
                    usage=Usage(
                        input_tokens=raw_usage.get("promptTokenCount", 0),
                        output_tokens=raw_usage.get("candidatesTokenCount", 0),
                        total_tokens=raw_usage.get("totalTokenCount", 0),
                    ),
                )

        yield StreamEvent(type="done")
