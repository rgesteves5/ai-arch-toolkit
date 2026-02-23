"""Provider for the OpenAI Chat Completions API."""

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

_BASE_URL = "https://api.openai.com/v1/chat/completions"
_DEFAULT_MAX_TOKENS = 4096

# Parameters safe to forward to the OpenAI Chat Completions API.
_OPENAI_PARAMS = {
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "response_format",
}


class _StreamState:
    """Per-stream metadata accumulator (concurrent-safe: one per call)."""

    __slots__ = ("model", "raw", "stop_reason", "usage")

    def __init__(self) -> None:
        self.usage: Usage | None = None
        self.model: str = ""
        self.stop_reason: str = ""
        self.raw: dict[str, Any] | None = None


def _tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    """Map generic tool dict to OpenAI wire format (function wrapper)."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", tool.get("input_schema", {})),
        },
    }


def _parse_tool_args(raw_args: str | dict[str, Any]) -> dict[str, Any]:
    """Parse tool call arguments (may be JSON string or dict)."""
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw_args}


def _messages_to_wire(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
) -> list[dict[str, Any]]:
    """Convert generic messages to OpenAI wire format.

    System stays as a regular message. tool_result messages use role="tool".
    If an explicit ``system`` is provided, system messages in the list are
    discarded (explicit overrides — same semantics as Anthropic provider).
    """
    wire: list[dict[str, Any]] = []
    if system is not None:
        wire.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            # Skip if explicit system was provided (explicit overrides)
            if system is None:
                wire.append({"role": "system", "content": msg.get("content", "")})
        elif msg.get("tool_use_id"):
            # tool_result — OpenAI uses role="tool" + tool_call_id
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": msg["tool_use_id"],
                    "content": str(msg.get("content", "")),
                }
            )
        elif role == "assistant" and msg.get("tool_calls"):
            # Assistant message with tool calls
            wire.append(
                {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("input", {})),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ],
                }
            )
        else:
            wire.append({"role": role, "content": msg.get("content", "")})
    return wire


def _parse_response(raw: dict[str, Any], model: str) -> Response:
    """Convert OpenAI API response to our Response dataclass."""
    choices = raw.get("choices", [])
    if not choices:
        return Response(raw=raw, model=model)

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
                input=_parse_tool_args(fn.get("arguments", "{}")),
            )
        )

    raw_usage = raw.get("usage", {})
    input_tokens = raw_usage.get("prompt_tokens", 0)
    output_tokens = raw_usage.get("completion_tokens", 0)

    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    cost, cost_estimated = pricing.estimate_cost(
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return Response(
        text=text.strip(),
        tool_calls=tuple(tool_calls),
        usage=usage,
        cost=cost,
        cost_estimated=cost_estimated,
        stop_reason=choice.get("finish_reason", ""),
        model=raw.get("model", model),
        raw=raw,
    )


def _build_payload(
    wire_messages: list[dict[str, Any]],
    *,
    model: str,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    unknown = set(kwargs) - _OPENAI_PARAMS
    if unknown:
        warnings.warn(
            f"Unknown parameter(s) ignored for OpenAI: {sorted(unknown)}",
            stacklevel=4,
        )
    filtered = {k: v for k, v in kwargs.items() if k in _OPENAI_PARAMS}
    payload: dict[str, Any] = {
        "model": model,
        "messages": wire_messages,
        **filtered,
    }
    if tools:
        payload["tools"] = [_tool_to_openai(t) for t in tools]
    return payload


class OpenAIProvider(BaseProvider):
    """OpenAI Chat Completions API provider (async-only, reuses httpx client)."""

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
            "Authorization": f"Bearer {api_key}",
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
        wire = _messages_to_wire(messages, system=system)
        payload = _build_payload(wire, model=self._model, tools=tools, **kwargs)
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
        wire = _messages_to_wire(messages, system=system)
        payload = _build_payload(wire, model=self._model, **kwargs)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

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
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Usage chunk (final chunk with usage info)
                if raw_usage := chunk.get("usage"):
                    state.usage = Usage(
                        input_tokens=raw_usage.get("prompt_tokens", 0),
                        output_tokens=raw_usage.get("completion_tokens", 0),
                    )
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})

                if text := delta.get("content"):
                    yield text

                if finish := choice.get("finish_reason"):
                    state.stop_reason = finish

                # Capture model from first chunk
                if model_name := chunk.get("model"):
                    state.model = model_name

        return _generate(), state
