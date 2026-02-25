"""Anthropic provider — thin adapter over the ``anthropic`` SDK."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import AsyncIterator
from typing import Any

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._providers._base import (
    DEFAULT_THINKING_BUDGET,
    THINKING_EFFORT_BUDGETS,
    BaseProvider,
    StreamEvent,
    StreamState,
    _parse_retry_after,
    parse_tool_args,
)
from ai_arch_toolkit.core._providers._imports import require_sdk
from ai_arch_toolkit.core._response import OutputSchema, Response, ThinkingBlock, ToolCall, Usage

require_sdk("anthropic", "anthropic")
import anthropic  # noqa: E402

logger = logging.getLogger(__name__)

# Parameters safe to forward directly to the SDK.
_SDK_PARAMS = {"temperature", "top_p", "top_k", "stop_sequences", "max_tokens"}


# ---------------------------------------------------------------------------
# Adapter helpers — pure functions for message/tool/response conversion
# ---------------------------------------------------------------------------


def _tool_to_sdk(tool: dict[str, Any]) -> dict[str, Any]:
    """Map generic tool dict to Anthropic SDK format."""
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("input_schema", tool.get("parameters", {})),
    }


def _messages_to_sdk(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract system messages and convert the rest to SDK-compatible format.

    ``tool_use_id`` is treated as the tool-result discriminator (role is ignored).
    """
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("tool_use_id"):
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
        elif msg.get("role") == "system":
            system_parts.append(msg.get("content", ""))
        elif msg.get("role") == "assistant" and msg.get("tool_calls"):
            content_blocks: list[dict[str, Any]] = []
            text = msg.get("content", "")
            if text:
                content_blocks.append({"type": "text", "text": text})
            for tc in msg["tool_calls"]:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "input": tc.get("input", {}),
                    }
                )
            wire.append({"role": "assistant", "content": content_blocks})
        else:
            wire.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, wire


def _build_thinking_param(
    thinking: bool,
    thinking_effort: str | None,
    thinking_budget: int | None,
) -> dict[str, Any] | None:
    """Build the SDK ``thinking`` config dict, or None if disabled."""
    if not thinking:
        return None
    budget = thinking_budget or DEFAULT_THINKING_BUDGET
    cfg: dict[str, Any] = {"type": "enabled", "budget_tokens": budget}
    if thinking_effort:
        cfg["budget_tokens"] = THINKING_EFFORT_BUDGETS.get(
            thinking_effort, DEFAULT_THINKING_BUDGET
        )
    return cfg


def _build_output_config(output_schema: OutputSchema) -> dict[str, Any]:
    """Build native ``output_config`` for structured output (Anthropic JSON mode)."""
    return {
        "format": {
            "type": "json_schema",
            "schema": output_schema.schema,
        }
    }


def _extract_usage(sdk_usage: Any) -> Usage:
    """Convert SDK usage object to our Usage dataclass."""
    return Usage(
        input_tokens=getattr(sdk_usage, "input_tokens", 0),
        output_tokens=getattr(sdk_usage, "output_tokens", 0),
        cache_write_tokens=getattr(sdk_usage, "cache_creation_input_tokens", 0),
        cache_read_tokens=getattr(sdk_usage, "cache_read_input_tokens", 0),
    )


def _parse_sdk_response(
    message: Any,
    model: str,
    *,
    output_schema: OutputSchema | None = None,
) -> Response:
    """Convert an ``anthropic.types.Message`` to our ``Response``."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thinking_blocks: list[ThinkingBlock] = []
    parsed: Any = None

    for block in message.content:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        elif block_type == "thinking":
            thinking_blocks.append(ThinkingBlock(text=block.thinking))

    text = "".join(text_parts).strip()

    if output_schema and text:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse structured output as JSON")

    usage = _extract_usage(message.usage)
    cost, cost_estimated = _estimate_response_cost(model, usage)

    return Response(
        text=text,
        tool_calls=tuple(tool_calls),
        thinking=tuple(thinking_blocks),
        parsed=parsed,
        usage=usage,
        cost=cost,
        cost_estimated=cost_estimated,
        stop_reason=message.stop_reason or "",
        model=message.model or model,
        raw=message,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AnthropicProvider(BaseProvider):
    """Anthropic Messages API provider via the official SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._client = anthropic.AsyncAnthropic(
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
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build kwargs dict for ``messages.create()`` / ``messages.stream()``."""
        # Extract our special params before forwarding
        thinking = kwargs.pop("thinking", False)
        thinking_effort = kwargs.pop("thinking_effort", None)
        thinking_budget = kwargs.pop("thinking_budget", None)
        output_schema: OutputSchema | None = kwargs.pop("output_schema", None)

        # Warn about unknown params
        unknown = set(kwargs) - _SDK_PARAMS
        if unknown:
            warnings.warn(
                f"Unknown parameter(s) ignored for Anthropic: {sorted(unknown)}. "
                f"Valid: {sorted(_SDK_PARAMS)}",
                stacklevel=4,
            )
        filtered = {k: v for k, v in kwargs.items() if k in _SDK_PARAMS}

        sdk_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": wire_messages,
            **filtered,
        }

        if system:
            sdk_kwargs["system"] = system

        # Thinking
        thinking_cfg = _build_thinking_param(thinking, thinking_effort, thinking_budget)
        if thinking_cfg:
            sdk_kwargs["thinking"] = thinking_cfg
            # Anthropic requires temperature=1 when thinking is enabled
            sdk_kwargs.pop("temperature", None)

        # Structured output via native JSON mode
        if output_schema:
            sdk_kwargs["output_config"] = _build_output_config(output_schema)

        if tools:
            sdk_kwargs["tools"] = [_tool_to_sdk(t) for t in tools]

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
        msg_system, wire = _messages_to_sdk(messages)
        effective_system = system if system is not None else msg_system
        output_schema: OutputSchema | None = kwargs.get("output_schema")

        sdk_kwargs = self._build_sdk_kwargs(wire, system=effective_system, tools=tools, **kwargs)

        try:
            message = await self._client.messages.create(**sdk_kwargs)
        except anthropic.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after")
            raise RateLimitError(
                exc.response.status_code,
                str(exc.body),
                retry_after=_parse_retry_after(retry_after),
            ) from exc
        except anthropic.APIStatusError as exc:
            raise APIError(exc.response.status_code, str(exc.body)) from exc

        return _parse_sdk_response(message, self._model, output_schema=output_schema)

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
        msg_system, wire = _messages_to_sdk(messages)
        effective_system = system if system is not None else msg_system
        output_schema: OutputSchema | None = kwargs.get("output_schema")

        sdk_kwargs = self._build_sdk_kwargs(wire, system=effective_system, tools=tools, **kwargs)

        state = StreamState()
        state.model = self._model

        async def _generate() -> AsyncIterator[str]:
            current_block: Any = None
            tool_args_acc = ""

            try:
                async with self._client.messages.stream(**sdk_kwargs) as stream:
                    async for event in stream:
                        event_type = event.type

                        if event_type == "content_block_start":
                            current_block = event.content_block
                            tool_args_acc = ""

                        elif event_type == "content_block_delta":
                            delta = event.delta
                            delta_type = delta.type
                            if delta_type == "text_delta":
                                yield delta.text
                            elif delta_type == "input_json_delta":
                                tool_args_acc += delta.partial_json

                        elif event_type == "content_block_stop":
                            if current_block and getattr(current_block, "type", "") == "tool_use":
                                args = parse_tool_args(tool_args_acc) if tool_args_acc else {}
                                if (
                                    output_schema
                                    and getattr(current_block, "name", "") == output_schema.name
                                ):
                                    pass  # structured output handled via final message
                                else:
                                    state.tool_calls.append(
                                        ToolCall(
                                            id=getattr(current_block, "id", ""),
                                            name=getattr(current_block, "name", ""),
                                            input=args,
                                        )
                                    )
                            current_block = None
                            tool_args_acc = ""

                        elif event_type == "message_start":
                            state.model = getattr(event.message, "model", self._model)
                            if hasattr(event.message, "usage"):
                                state.usage = _extract_usage(event.message.usage)

                        elif event_type == "message_delta":
                            state.stop_reason = getattr(event.delta, "stop_reason", "") or ""
                            if hasattr(event, "usage") and event.usage:
                                delta_usage = _extract_usage(event.usage)
                                prev = state.usage or Usage()
                                state.usage = Usage(
                                    input_tokens=prev.input_tokens + delta_usage.input_tokens,
                                    output_tokens=(prev.output_tokens + delta_usage.output_tokens),
                                    cache_write_tokens=(
                                        prev.cache_write_tokens + delta_usage.cache_write_tokens
                                    ),
                                    cache_read_tokens=(
                                        prev.cache_read_tokens + delta_usage.cache_read_tokens
                                    ),
                                )

                    # After stream completes, extract thinking from final message
                    final = await stream.get_final_message()
                    state.raw = final
                    for block in final.content:
                        if getattr(block, "type", "") == "thinking":
                            state.thinking.append(ThinkingBlock(text=block.thinking))

            except anthropic.RateLimitError as exc:
                retry_after = exc.response.headers.get("retry-after")
                raise RateLimitError(
                    exc.response.status_code,
                    str(exc.body),
                    retry_after=_parse_retry_after(retry_after),
                ) from exc
            except anthropic.APIStatusError as exc:
                raise APIError(exc.response.status_code, str(exc.body)) from exc

        return _generate(), state

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[StreamEvent], StreamState]:
        msg_system, wire = _messages_to_sdk(messages)
        effective_system = system if system is not None else msg_system
        output_schema: OutputSchema | None = kwargs.get("output_schema")

        sdk_kwargs = self._build_sdk_kwargs(wire, system=effective_system, tools=tools, **kwargs)

        state = StreamState()
        state.model = self._model

        async def _generate() -> AsyncIterator[StreamEvent]:
            current_block: Any = None
            tool_args_acc = ""
            thinking_acc = ""

            try:
                async with self._client.messages.stream(**sdk_kwargs) as stream:
                    async for event in stream:
                        event_type = event.type

                        if event_type == "content_block_start":
                            current_block = event.content_block
                            tool_args_acc = ""
                            thinking_acc = ""

                        elif event_type == "content_block_delta":
                            delta = event.delta
                            delta_type = delta.type
                            if delta_type == "text_delta":
                                yield StreamEvent(kind="text", text=delta.text)
                            elif delta_type == "input_json_delta":
                                tool_args_acc += delta.partial_json
                            elif delta_type == "thinking_delta":
                                thinking_acc += getattr(delta, "thinking", "")

                        elif event_type == "content_block_stop":
                            if current_block and getattr(current_block, "type", "") == "tool_use":
                                args = parse_tool_args(tool_args_acc) if tool_args_acc else {}
                                if (
                                    output_schema
                                    and getattr(current_block, "name", "") == output_schema.name
                                ):
                                    pass  # structured output handled via final message
                                else:
                                    tc = ToolCall(
                                        id=getattr(current_block, "id", ""),
                                        name=getattr(current_block, "name", ""),
                                        input=args,
                                    )
                                    state.tool_calls.append(tc)
                                    yield StreamEvent(kind="tool_call", tool_call=tc)
                            elif (
                                current_block and getattr(current_block, "type", "") == "thinking"
                            ):
                                # Emit complete thinking block (buffered from deltas)
                                text = thinking_acc or getattr(current_block, "thinking", "")
                                if text:
                                    block = ThinkingBlock(text=text)
                                    state.thinking.append(block)
                                    yield StreamEvent(kind="thinking", thinking=block)
                            current_block = None
                            tool_args_acc = ""
                            thinking_acc = ""

                        elif event_type == "message_start":
                            state.model = getattr(event.message, "model", self._model)
                            if hasattr(event.message, "usage"):
                                state.usage = _extract_usage(event.message.usage)

                        elif event_type == "message_delta":
                            state.stop_reason = getattr(event.delta, "stop_reason", "") or ""
                            if hasattr(event, "usage") and event.usage:
                                delta_usage = _extract_usage(event.usage)
                                prev = state.usage or Usage()
                                state.usage = Usage(
                                    input_tokens=prev.input_tokens + delta_usage.input_tokens,
                                    output_tokens=(prev.output_tokens + delta_usage.output_tokens),
                                    cache_write_tokens=(
                                        prev.cache_write_tokens + delta_usage.cache_write_tokens
                                    ),
                                    cache_read_tokens=(
                                        prev.cache_read_tokens + delta_usage.cache_read_tokens
                                    ),
                                )

                    # After stream completes, get final message for raw state
                    final = await stream.get_final_message()
                    state.raw = final

            except anthropic.RateLimitError as exc:
                retry_after = exc.response.headers.get("retry-after")
                raise RateLimitError(
                    exc.response.status_code,
                    str(exc.body),
                    retry_after=_parse_retry_after(retry_after),
                ) from exc
            except anthropic.APIStatusError as exc:
                raise APIError(exc.response.status_code, str(exc.body)) from exc

        return _generate(), state
