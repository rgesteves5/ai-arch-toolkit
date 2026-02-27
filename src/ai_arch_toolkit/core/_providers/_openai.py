"""OpenAI provider — thin adapter over the ``openai`` SDK."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import AsyncIterator
from typing import Any

from ai_arch_toolkit.core._content import DocumentPart, ImagePart, _encode_b64, _is_url
from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._providers._base import (
    BaseProvider,
    StreamState,
    _parse_retry_after,
    parse_tool_args,
)
from ai_arch_toolkit.core._providers._imports import require_sdk
from ai_arch_toolkit.core._response import OutputSchema, Response, ToolCall, Usage

require_sdk("openai", "openai")
import openai  # noqa: E402

logger = logging.getLogger(__name__)

# Parameters safe to forward directly to the SDK.
_SDK_PARAMS = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "response_format",
    "parallel_tool_calls",
}


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def _content_to_sdk(content: Any) -> list[dict[str, Any]] | str:
    """Convert multimodal content to OpenAI content blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
        elif isinstance(part, ImagePart):
            if _is_url(part.source):
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": part.source},
                    }
                )
            else:
                b64 = _encode_b64(part.source)
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.media_type};base64,{b64}"},
                    }
                )
        elif isinstance(part, DocumentPart):
            b64 = _encode_b64(part.source)
            blocks.append(
                {
                    "type": "file",
                    "file": {
                        "filename": part.name or "document",
                        "file_data": f"data:{part.media_type};base64,{b64}",
                    },
                }
            )
        else:
            blocks.append({"type": "text", "text": str(part)})
    return blocks


def _tool_to_sdk(tool: dict[str, Any]) -> dict[str, Any]:
    """Map generic tool dict to OpenAI SDK format (function wrapper)."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", tool.get("parameters", {})),
        },
    }


def _messages_to_sdk(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
) -> list[dict[str, Any]]:
    """Convert generic messages to OpenAI SDK format.

    ``tool_use_id`` is the tool-result discriminator (role is ignored).
    System stays as a regular message role. tool results use role="tool".
    If ``system`` is provided, system messages in the list are discarded.
    """
    wire: list[dict[str, Any]] = []
    if system is not None:
        wire.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role", "user")
        if msg.get("tool_use_id"):
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": msg["tool_use_id"],
                    "content": str(msg.get("content", "")),
                }
            )
        elif role == "system":
            if system is None:
                wire.append({"role": "system", "content": msg.get("content", "")})
        elif role == "assistant" and msg.get("tool_calls"):
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
            wire.append({"role": role, "content": _content_to_sdk(msg.get("content", ""))})
    return wire


def _build_output_schema_format(
    output_schema: OutputSchema,
) -> dict[str, Any]:
    """Build OpenAI ``response_format`` for structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": output_schema.name,
            "schema": output_schema.schema,
            "strict": output_schema.strict,
        },
    }


def _extract_usage(sdk_usage: Any) -> Usage:
    """Convert SDK usage object to our Usage dataclass."""
    cache_read = 0
    details = getattr(sdk_usage, "prompt_tokens_details", None)
    if details:
        cache_read = getattr(details, "cached_tokens", 0) or 0
    return Usage(
        input_tokens=getattr(sdk_usage, "prompt_tokens", 0),
        output_tokens=getattr(sdk_usage, "completion_tokens", 0),
        cache_read_tokens=cache_read,
    )


def _parse_sdk_response(
    completion: Any,
    model: str,
    *,
    output_schema: OutputSchema | None = None,
) -> Response:
    """Convert ``openai.types.chat.ChatCompletion`` to our ``Response``."""
    choices = completion.choices or []
    if not choices:
        return Response(raw=completion, model=model)

    choice = choices[0]
    message = choice.message
    text = message.content or ""

    tool_calls: list[ToolCall] = []
    for tc in message.tool_calls or []:
        tool_calls.append(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=parse_tool_args(tc.function.arguments),
            )
        )

    parsed: Any = None
    if output_schema and text:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse structured output as JSON")

    usage = _extract_usage(completion.usage) if completion.usage else Usage()
    cost = _estimate_response_cost(model, usage)

    # Extract logprobs if present
    logprobs_data = getattr(choice, "logprobs", None)

    return Response(
        text=text.strip(),
        tool_calls=tuple(tool_calls),
        parsed=parsed,
        usage=usage,
        cost=cost,
        stop_reason=choice.finish_reason or "",
        model=completion.model or model,
        raw=completion,
        response_id=getattr(completion, "id", "") or "",
        logprobs=logprobs_data,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIProvider(BaseProvider):
    """OpenAI Chat Completions API provider via the official SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_sdk_kwargs(
        self,
        wire_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build kwargs dict for ``chat.completions.create()``."""
        # Extract our special params
        thinking = kwargs.pop("thinking", False)
        thinking_effort = kwargs.pop("thinking_effort", None)
        thinking_budget = kwargs.pop("thinking_budget", None)
        output_schema: OutputSchema | None = kwargs.pop("output_schema", None)
        tool_choice: str | None = kwargs.pop("tool_choice", None)
        json_mode: bool = kwargs.pop("json_mode", False)
        logprobs_flag: bool = kwargs.pop("logprobs", False)

        # Warn about unknown params
        unknown = set(kwargs) - _SDK_PARAMS
        if unknown:
            warnings.warn(
                f"Unknown parameter(s) ignored for OpenAI: {sorted(unknown)}. "
                f"Valid: {sorted(_SDK_PARAMS)}",
                stacklevel=4,
            )
        filtered = {k: v for k, v in kwargs.items() if k in _SDK_PARAMS}

        sdk_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": wire_messages,
            **filtered,
        }

        # Thinking → reasoning_effort (OpenAI naming)
        if thinking:
            sdk_kwargs["reasoning_effort"] = thinking_effort or "high"
        if thinking_budget:
            warnings.warn(
                "thinking_budget is not supported by OpenAI (only reasoning_effort string), "
                "ignoring",
                stacklevel=4,
            )

        # Structured output via native response_format
        if output_schema:
            sdk_kwargs["response_format"] = _build_output_schema_format(output_schema)

        if tools:
            fn_tools = [_tool_to_sdk(t) for t in tools if not t.get("_server_tool")]
            server_tools = [t for t in tools if t.get("_server_tool")]
            if fn_tools:
                sdk_kwargs["tools"] = fn_tools
            for st in server_tools:
                st_type = st["type"]
                if st_type == "web_search":
                    sdk_kwargs.setdefault("tools", []).append({"type": "web_search"})
                elif st_type == "code_execution":
                    sdk_kwargs.setdefault("tools", []).append({"type": "code_interpreter"})

        # tool_choice
        if tool_choice is not None:
            if tool_choice in ("auto", "required", "none"):
                sdk_kwargs["tool_choice"] = tool_choice
            else:
                sdk_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }

        # json_mode
        if json_mode:
            sdk_kwargs["response_format"] = {"type": "json_object"}

        # logprobs
        if logprobs_flag:
            sdk_kwargs["logprobs"] = True

        return sdk_kwargs

    # ------------------------------------------------------------------
    # complete
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Response:
        output_schema: OutputSchema | None = kwargs.get("output_schema")
        wire = _messages_to_sdk(messages, system=system)
        sdk_kwargs = self._build_sdk_kwargs(wire, tools=tools, **kwargs)

        try:
            completion = await self._client.chat.completions.create(**sdk_kwargs)
        except openai.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after") if exc.response else None
            raise RateLimitError(
                exc.response.status_code if exc.response else 429,
                str(exc.body),
                retry_after=_parse_retry_after(retry_after),
            ) from exc
        except openai.APIStatusError as exc:
            raise APIError(
                exc.response.status_code if exc.response else 500,
                str(exc.body),
            ) from exc

        return _parse_sdk_response(completion, self._model, output_schema=output_schema)

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[str], StreamState]:
        wire = _messages_to_sdk(messages, system=system)
        sdk_kwargs = self._build_sdk_kwargs(wire, tools=tools, **kwargs)

        state = StreamState()
        state.model = self._model

        async def _generate() -> AsyncIterator[str]:
            tc_acc: dict[int, dict[str, str]] = {}

            try:
                stream = await self._client.chat.completions.create(
                    **sdk_kwargs,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for chunk in stream:
                    # Usage chunk (final)
                    if chunk.usage:
                        state.usage = _extract_usage(chunk.usage)

                    choices = chunk.choices or []
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.delta

                    # Text content
                    if delta and delta.content:
                        yield delta.content

                    # Accumulate tool call deltas
                    if delta and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tc_acc:
                                tc_acc[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": (
                                        tc_delta.function.name
                                        if tc_delta.function and tc_delta.function.name
                                        else ""
                                    ),
                                    "arguments": "",
                                }
                            else:
                                if tc_delta.id:
                                    tc_acc[idx]["id"] = tc_delta.id
                                if tc_delta.function and tc_delta.function.name:
                                    tc_acc[idx]["name"] = tc_delta.function.name
                            if tc_delta.function and tc_delta.function.arguments:
                                tc_acc[idx]["arguments"] += tc_delta.function.arguments

                    finish = choice.finish_reason
                    if finish == "tool_calls" and tc_acc:
                        for _idx in sorted(tc_acc):
                            acc = tc_acc[_idx]
                            state.tool_calls.append(
                                ToolCall(
                                    id=acc["id"],
                                    name=acc["name"],
                                    input=parse_tool_args(acc["arguments"]),
                                )
                            )
                        tc_acc.clear()

                    if finish:
                        state.stop_reason = finish

                    if chunk.model:
                        state.model = chunk.model

                    state.raw = chunk

            except openai.RateLimitError as exc:
                retry_after = exc.response.headers.get("retry-after") if exc.response else None
                raise RateLimitError(
                    exc.response.status_code if exc.response else 429,
                    str(exc.body),
                    retry_after=_parse_retry_after(retry_after),
                ) from exc
            except openai.APIStatusError as exc:
                raise APIError(
                    exc.response.status_code if exc.response else 500,
                    str(exc.body),
                ) from exc

        return _generate(), state
