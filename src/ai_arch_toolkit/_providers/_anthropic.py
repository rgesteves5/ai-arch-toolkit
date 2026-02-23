"""Provider for the Anthropic Messages API."""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ai_arch_toolkit._http import RetryConfig, async_post_json, async_stream_sse
from ai_arch_toolkit._pricing import pricing
from ai_arch_toolkit._providers._base import BaseProvider
from ai_arch_toolkit._response import Response, ToolCall, Usage

_BASE_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096

# Parameters safe to forward to the Anthropic Messages API.
_ANTHROPIC_PARAMS = {"temperature", "top_p", "top_k", "stop_sequences", "max_tokens"}


class _StreamState:
    """Per-stream metadata accumulator (concurrent-safe: one per call)."""

    __slots__ = ("model", "raw", "stop_reason", "usage")

    def __init__(self) -> None:
        self.usage: Usage | None = None
        self.model: str = ""
        self.stop_reason: str = ""
        self.raw: dict[str, Any] | None = None


def _tool_to_anthropic(tool: dict[str, Any]) -> dict[str, Any]:
    """Map generic tool dict (with ``parameters``) to Anthropic wire format."""
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", tool.get("input_schema", {})),
    }


def _messages_to_wire(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract system messages and return remaining messages."""
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg.get("content", ""))
        elif msg.get("tool_use_id"):
            # tool_result — wrap in Anthropic format
            wire.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_use_id"],
                            "content": msg.get("content", ""),
                        }
                    ],
                }
            )
        else:
            wire.append({"role": msg["role"], "content": msg["content"]})
    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, wire


def _parse_response(raw: dict[str, Any], model: str) -> Response:
    """Convert Anthropic API response to our Response dataclass."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in raw.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    input=block.get("input", {}),
                )
            )

    raw_usage = raw.get("usage", {})
    input_tokens = raw_usage.get("input_tokens", 0)
    output_tokens = raw_usage.get("output_tokens", 0)
    cache_write = raw_usage.get("cache_creation_input_tokens", 0)
    cache_read = raw_usage.get("cache_read_input_tokens", 0)

    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write,
        cache_read_tokens=cache_read,
    )

    cost, cost_estimated = pricing.estimate_cost(
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write,
        cache_read_tokens=cache_read,
    )

    return Response(
        text="".join(text_parts).strip(),
        tool_calls=tuple(tool_calls),
        usage=usage,
        cost=cost,
        cost_estimated=cost_estimated,
        stop_reason=raw.get("stop_reason", ""),
        model=raw.get("model", model),
        raw=raw,
    )


def _build_payload(
    wire_messages: list[dict[str, Any]],
    *,
    model: str,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    unknown = set(kwargs) - _ANTHROPIC_PARAMS
    if unknown:
        warnings.warn(
            f"Unknown parameter(s) ignored for Anthropic: {sorted(unknown)}",
            stacklevel=4,
        )
    filtered = {k: v for k, v in kwargs.items() if k in _ANTHROPIC_PARAMS}
    payload: dict[str, Any] = {
        "model": model,
        "messages": wire_messages,
        "max_tokens": filtered.pop("max_tokens", _DEFAULT_MAX_TOKENS),
        **filtered,
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = [_tool_to_anthropic(t) for t in tools]
    return payload


def _parse_stream_usage(event: dict[str, Any]) -> Usage | None:
    """Extract usage from a message_delta or message_start event."""
    raw_usage = event.get("usage", {})
    if not raw_usage:
        return None
    return Usage(
        input_tokens=raw_usage.get("input_tokens", 0),
        output_tokens=raw_usage.get("output_tokens", 0),
        cache_write_tokens=raw_usage.get("cache_creation_input_tokens", 0),
        cache_read_tokens=raw_usage.get("cache_read_input_tokens", 0),
    )


class AnthropicProvider(BaseProvider):
    """Anthropic Messages API provider (async-only, reuses httpx client)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
        retry: RetryConfig | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url or _BASE_URL
        self._retry = retry
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(headers=self._headers)

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Response:
        timeout = kwargs.pop("timeout", 60)
        msg_system, wire = _messages_to_wire(messages)
        effective_system = system if system is not None else msg_system
        payload = _build_payload(
            wire, model=self._model, system=effective_system, tools=tools, **kwargs
        )
        raw = await async_post_json(
            self._base_url,
            payload=payload,
            timeout=timeout,
            retry=self._retry,
            client=self._client,
        )
        return _parse_response(raw, self._model)

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[str], _StreamState]:
        timeout = kwargs.pop("timeout", 120)
        msg_system, wire = _messages_to_wire(messages)
        effective_system = system if system is not None else msg_system
        payload = _build_payload(wire, model=self._model, system=effective_system, **kwargs)
        payload["stream"] = True

        state = _StreamState()
        state.model = self._model

        async def _generate() -> AsyncIterator[str]:
            async for data in async_stream_sse(
                self._base_url,
                payload=payload,
                timeout=timeout,
                retry=self._retry,
                client=self._client,
            ):
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text

                elif event_type == "message_start":
                    msg = event.get("message", {})
                    state.model = msg.get("model", self._model)
                    usage = _parse_stream_usage(msg)
                    if usage:
                        state.usage = usage

                elif event_type == "message_delta":
                    delta = event.get("delta", {})
                    state.stop_reason = delta.get("stop_reason", "")
                    usage = _parse_stream_usage(event)
                    if usage:
                        # Merge: message_start has input_tokens, message_delta has output_tokens.
                        # Use addition (not `or`) — values are complementary and 0 is valid.
                        prev = state.usage or Usage()
                        state.usage = Usage(
                            input_tokens=prev.input_tokens + usage.input_tokens,
                            output_tokens=prev.output_tokens + usage.output_tokens,
                            cache_write_tokens=prev.cache_write_tokens + usage.cache_write_tokens,
                            cache_read_tokens=prev.cache_read_tokens + usage.cache_read_tokens,
                        )

        return _generate(), state
