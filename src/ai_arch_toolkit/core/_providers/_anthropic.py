"""Anthropic provider — thin adapter over the ``anthropic`` SDK."""

from __future__ import annotations

import json
import logging
import re
import warnings
from collections.abc import AsyncIterator
from typing import Any, cast

from ai_arch_toolkit.core._content import CachePart, DocumentPart, ImagePart, _encode_b64, _is_url
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
from ai_arch_toolkit.core._response import (
    Citation,
    OutputSchema,
    Response,
    ThinkingBlock,
    ToolCall,
    Usage,
)

require_sdk("anthropic", "anthropic")
import anthropic  # noqa: E402

logger = logging.getLogger(__name__)

# Parameters safe to forward directly to the SDK.
_SDK_PARAMS = {"temperature", "top_p", "top_k", "stop_sequences", "max_tokens"}

# Anthropic server tool type identifiers (versioned by Anthropic).
_SERVER_TOOL_TYPES: dict[str, str] = {
    "web_search": "web_search_20250305",
    "code_execution": "code_execution_20250522",
}

_TEMPERATURE_DEPRECATED_MODELS = {"claude-opus-4-7"}


# ---------------------------------------------------------------------------
# Adapter helpers — pure functions for message/tool/response conversion
# ---------------------------------------------------------------------------


def _content_to_sdk(content: Any) -> list[dict[str, Any]] | str:
    """Convert multimodal content to Anthropic content blocks.

    Returns a list of content blocks if multimodal, or a plain string.
    """
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
                        "type": "image",
                        "source": {"type": "url", "url": part.source},
                    }
                )
            else:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": _encode_b64(part.source),
                        },
                    }
                )
        elif isinstance(part, DocumentPart):
            block: dict[str, Any] = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": part.media_type,
                    "data": _encode_b64(part.source),
                },
            }
            if part.name:
                block["name"] = part.name
            blocks.append(block)
        elif isinstance(part, CachePart):
            blocks.append(
                {
                    "type": "text",
                    "text": part.content,
                    "cache_control": {"type": part.ttl},
                }
            )
        else:
            blocks.append({"type": "text", "text": str(part)})
    return blocks


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
            raw_content = msg.get("content", "")
            wire.append(
                {
                    "role": msg.get("role", "user"),
                    "content": _content_to_sdk(raw_content),
                }
            )
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


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json_text(text: str) -> str:
    """Return *text* with a wrapping Markdown code fence removed, if present."""
    stripped = text.strip()
    match = _JSON_FENCE_RE.search(stripped)
    return match.group(1).strip() if match else stripped


def _schema_prompt_instruction(output_schema: OutputSchema) -> str:
    """Build the system-prompt instruction for the ``"prompt"`` strategy.

    Asks the model for a raw JSON object matching the schema. Used for schemas
    that exceed Anthropic's native ``output_config`` complexity limit, where
    ``output_config`` would otherwise return a 400 "schema is too complex" error.
    """
    schema_text = json.dumps(output_schema.schema, indent=2)
    return (
        "IMPORTANT: Respond with ONLY a raw JSON object (no markdown code "
        "fences, no explanation, no text before or after). The JSON must match "
        "this schema:\n" + schema_text
    )


def _parse_structured_output(text: str, output_schema: OutputSchema) -> Any:
    """Parse structured-output *text*, coercing to the schema's Pydantic model.

    Tolerates Markdown-fenced JSON (the ``"prompt"`` strategy can produce it).
    When the schema carries a ``model_class``, the parsed data is validated into
    that model — at parity with the OpenAI, Gemini, and xAI adapters. Returns
    ``None`` if the text is not valid JSON, or the raw dict if validation fails.
    """
    try:
        data = json.loads(_extract_json_text(text))
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse structured output as JSON")
        return None
    if output_schema.model_class is None:
        return data
    try:
        return output_schema.model_class.model_validate(data)
    except Exception:
        logger.warning("Failed to validate structured output against schema")
        return data


def _uses_deprecated_temperature(model: str) -> bool:
    """Return True for Anthropic models that reject ``temperature``."""
    return model in _TEMPERATURE_DEPRECATED_MODELS


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
    citations: list[Citation] = []
    parsed: Any = None

    for block in message.content:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            text_parts.append(block.text)
            # Extract citations from text blocks
            for cite in getattr(block, "citations", None) or []:
                citations.append(
                    Citation(
                        text=getattr(cite, "cited_text", ""),
                        url=getattr(cite, "url", ""),
                        title=getattr(cite, "title", ""),
                        start_index=getattr(cite, "start_char_index", None),
                        end_index=getattr(cite, "end_char_index", None),
                    )
                )
        elif block_type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        elif block_type == "thinking":
            thinking_blocks.append(ThinkingBlock(text=block.thinking))

    text = "".join(text_parts).strip()

    if output_schema and text:
        parsed = _parse_structured_output(text, output_schema)

    usage = _extract_usage(message.usage)
    cost = _estimate_response_cost(model, usage)

    return Response(
        text=text,
        tool_calls=tuple(tool_calls),
        thinking=tuple(thinking_blocks),
        parsed=parsed,
        usage=usage,
        cost=cost,
        stop_reason=message.stop_reason or "",
        model=message.model or model,
        raw=message,
        response_id=getattr(message, "id", "") or "",
        citations=tuple(citations),
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
        timeout: float | None = None,
    ) -> None:
        self._model = model
        # Retry ownership belongs to LLM(RetryConfig(...)): hidden SDK retries
        # would be neither metered nor represented in Response.attempts.
        client_kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client = anthropic.AsyncAnthropic(**client_kwargs)

    async def close(self) -> None:
        await self._client.close()

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Count tokens using Anthropic's count_tokens API."""
        msg_system, wire = _messages_to_sdk(messages)
        effective_system = system if system is not None else msg_system
        sdk_kwargs: dict[str, Any] = {"model": self._model, "messages": wire}
        if effective_system:
            sdk_kwargs["system"] = effective_system
        if tools:
            fn_tools = [t for t in tools if not t.get("_server_tool")]
            if fn_tools:
                sdk_kwargs["tools"] = [_tool_to_sdk(t) for t in fn_tools]
        try:
            result = await self._client.messages.count_tokens(**sdk_kwargs)
            return result.input_tokens
        except anthropic.APIStatusError as exc:
            raise APIError(exc.response.status_code, str(exc.body)) from exc

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
        tool_choice: str | None = kwargs.pop("tool_choice", None)
        json_mode: bool = kwargs.pop("json_mode", False)
        structured_output_mode: str = kwargs.pop("structured_output_mode", "native")
        kwargs.pop("logprobs", None)  # Not supported by Anthropic

        if structured_output_mode not in ("native", "prompt"):
            raise ValueError(
                "structured_output_mode must be 'native' or 'prompt', "
                f"got {structured_output_mode!r}"
            )

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
        if _uses_deprecated_temperature(self._model):
            sdk_kwargs.pop("temperature", None)

        if system:
            sdk_kwargs["system"] = system

        # Thinking
        thinking_cfg = _build_thinking_param(thinking, thinking_effort, thinking_budget)
        if thinking_cfg:
            sdk_kwargs["thinking"] = thinking_cfg
            # Anthropic requires temperature=1 when thinking is enabled
            sdk_kwargs.pop("temperature", None)

        # Structured output: native ``output_config`` by default, or schema-in-
        # prompt for schemas that exceed Anthropic's native complexity limit.
        if output_schema:
            if structured_output_mode == "prompt":
                instruction = _schema_prompt_instruction(output_schema)
                base_system = sdk_kwargs.get("system", "") or ""
                sdk_kwargs["system"] = (
                    f"{base_system}\n\n{instruction}" if base_system else instruction
                )
            else:
                sdk_kwargs["output_config"] = _build_output_config(output_schema)

        if tools:
            fn_tools = [_tool_to_sdk(t) for t in tools if not t.get("_server_tool")]
            server_tools = [t for t in tools if t.get("_server_tool")]
            all_tools = fn_tools
            for st in server_tools:
                sdk_type = _SERVER_TOOL_TYPES.get(st["type"])
                if sdk_type:
                    all_tools.append({"type": sdk_type})
            if all_tools:
                sdk_kwargs["tools"] = all_tools

        # tool_choice
        if tool_choice is not None:
            if tool_choice == "auto":
                sdk_kwargs["tool_choice"] = {"type": "auto"}
            elif tool_choice == "required":
                sdk_kwargs["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                sdk_kwargs["tool_choice"] = {"type": "none"}
            else:
                sdk_kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        # json_mode — Anthropic has no native json_mode; append system instruction
        if json_mode:
            existing_system = sdk_kwargs.get("system", "")
            suffix = "Respond with valid JSON only."
            if existing_system:
                sdk_kwargs["system"] = f"{existing_system}\n\n{suffix}"
            else:
                sdk_kwargs["system"] = suffix

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

        logger.debug("complete start model=%s messages=%d", self._model, len(messages))
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

        resp = _parse_sdk_response(message, self._model, output_schema=output_schema)
        logger.debug(
            "complete done model=%s tokens_in=%d tokens_out=%d",
            self._model,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
        )
        return resp

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

        logger.debug("stream start model=%s", self._model)
        state = StreamState()
        state.model = self._model

        async def _generate() -> AsyncIterator[str]:
            current_block: Any = None
            tool_args_acc = ""

            try:
                async with self._client.messages.stream(**sdk_kwargs) as stream:
                    async for event in stream:
                        # The SDK stream yields a discriminated union keyed by ``event.type``.
                        # We narrow with ``cast`` once per branch so pyright can resolve the
                        # subsequent attribute access; the runtime check is the string match.
                        event_type = event.type

                        if event_type == "content_block_start":
                            ev = cast(anthropic.types.RawContentBlockStartEvent, event)
                            current_block = ev.content_block
                            tool_args_acc = ""

                        elif event_type == "content_block_delta":
                            ev = cast(anthropic.types.RawContentBlockDeltaEvent, event)
                            delta = ev.delta
                            delta_type = delta.type
                            if delta_type == "text_delta":
                                yield cast(anthropic.types.TextDelta, delta).text
                            elif delta_type == "input_json_delta":
                                tool_args_acc += cast(
                                    anthropic.types.InputJSONDelta, delta
                                ).partial_json

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
                            ev = cast(anthropic.types.RawMessageStartEvent, event)
                            state.model = getattr(ev.message, "model", self._model)
                            if hasattr(ev.message, "usage"):
                                state.usage = _extract_usage(ev.message.usage)

                        elif event_type == "message_delta":
                            ev = cast(anthropic.types.RawMessageDeltaEvent, event)
                            state.stop_reason = ev.delta.stop_reason or ""
                            if getattr(ev, "usage", None):
                                delta_usage = _extract_usage(ev.usage)
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
                            state.thinking.append(
                                ThinkingBlock(
                                    text=cast(anthropic.types.ThinkingBlock, block).thinking
                                )
                            )

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
                        # See comment in ``stream`` — same cast-per-branch pattern.
                        event_type = event.type

                        if event_type == "content_block_start":
                            ev = cast(anthropic.types.RawContentBlockStartEvent, event)
                            current_block = ev.content_block
                            tool_args_acc = ""
                            thinking_acc = ""

                        elif event_type == "content_block_delta":
                            ev = cast(anthropic.types.RawContentBlockDeltaEvent, event)
                            delta = ev.delta
                            delta_type = delta.type
                            if delta_type == "text_delta":
                                yield StreamEvent(
                                    kind="text", text=cast(anthropic.types.TextDelta, delta).text
                                )
                            elif delta_type == "input_json_delta":
                                tool_args_acc += cast(
                                    anthropic.types.InputJSONDelta, delta
                                ).partial_json
                            elif delta_type == "thinking_delta":
                                thinking_acc += cast(anthropic.types.ThinkingDelta, delta).thinking

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
                            ev = cast(anthropic.types.RawMessageStartEvent, event)
                            state.model = getattr(ev.message, "model", self._model)
                            if hasattr(ev.message, "usage"):
                                state.usage = _extract_usage(ev.message.usage)

                        elif event_type == "message_delta":
                            ev = cast(anthropic.types.RawMessageDeltaEvent, event)
                            state.stop_reason = ev.delta.stop_reason or ""
                            if getattr(ev, "usage", None):
                                delta_usage = _extract_usage(ev.usage)
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

    # ------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------

    async def batch_submit(
        self,
        requests: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Submit a batch via Anthropic's Message Batches API."""
        batch_requests = []
        for req in requests:
            custom_id = req.get("custom_id", "")
            messages = req.get("messages", [])
            req_system = req.get("system")
            tools = req.get("tools")
            req_kwargs = req.get("kwargs", {})

            _, wire = _messages_to_sdk(messages)
            params: dict[str, Any] = {
                "model": self._model,
                "messages": wire,
                "max_tokens": req_kwargs.get("max_tokens", 4096),
            }
            if req_system:
                params["system"] = req_system
            if tools:
                fn_tools = [t for t in tools if not t.get("_server_tool")]
                if fn_tools:
                    params["tools"] = [_tool_to_sdk(t) for t in fn_tools]

            batch_requests.append(
                {
                    "custom_id": custom_id,
                    "params": params,
                }
            )

        try:
            result = await self._client.messages.batches.create(requests=batch_requests)
            return result.id
        except anthropic.APIStatusError as exc:
            raise APIError(exc.response.status_code, str(exc.body)) from exc

    async def batch_status(self, batch_id: str) -> str:
        """Check batch status."""
        try:
            result = await self._client.messages.batches.retrieve(batch_id)
            return result.processing_status
        except anthropic.APIStatusError as exc:
            raise APIError(exc.response.status_code, str(exc.body)) from exc

    async def batch_results(self, batch_id: str) -> list[Any]:
        """Retrieve completed batch results."""
        from ai_arch_toolkit.core._batch import BatchResult

        results: list[BatchResult] = []
        try:
            async for entry in await self._client.messages.batches.results(batch_id):
                custom_id = getattr(entry, "custom_id", "")
                result_data = getattr(entry, "result", None)
                if result_data and getattr(result_data, "type", "") == "succeeded":
                    message = result_data.message
                    response = _parse_sdk_response(message, self._model)
                    results.append(BatchResult(custom_id=custom_id, response=response))
                else:
                    error_msg = str(getattr(result_data, "error", "unknown error"))
                    results.append(BatchResult(custom_id=custom_id, error=error_msg))
        except anthropic.APIStatusError as exc:
            raise APIError(exc.response.status_code, str(exc.body)) from exc
        return results
