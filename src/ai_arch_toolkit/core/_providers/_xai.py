"""xAI provider — thin adapter over the ``xai-sdk`` (gRPC-based)."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import AsyncIterator
from typing import Any

import grpc

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._providers._base import BaseProvider, StreamState, parse_tool_args
from ai_arch_toolkit.core._providers._imports import require_sdk
from ai_arch_toolkit.core._response import OutputSchema, Response, ThinkingBlock, ToolCall, Usage

require_sdk("xai_sdk", "xai")
import xai_sdk  # noqa: E402
from xai_sdk import chat as xai_chat  # noqa: E402

logger = logging.getLogger(__name__)

# Parameters safe to forward directly to the SDK.
_SDK_PARAMS = {
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "seed",
}


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def _messages_to_sdk(
    messages: list[dict[str, Any]],
) -> tuple[list[Any], str | None]:
    """Convert generic messages to xAI SDK Message objects.

    Returns (sdk_messages, system_text). System messages are extracted and
    returned separately since xAI passes system via ``create(messages=[...])``.
    ``tool_use_id`` is the tool-result discriminator (role is ignored).
    """
    sdk_msgs: list[Any] = []
    system_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")

        if msg.get("tool_use_id"):
            # Tool result
            content = msg.get("content", "")
            sdk_msgs.append(xai_chat.tool_result(str(content), tool_call_id=msg["tool_use_id"]))
        elif role == "system":
            system_parts.append(msg.get("content", ""))

        elif role == "assistant" and msg.get("tool_calls"):
            # Assistant with tool calls — build proto Message directly
            text = msg.get("content", "")
            content_parts = []
            if text:
                content_parts.append(xai_chat.chat_pb2.Content(text=text))
            pb_msg = xai_chat.chat_pb2.Message(
                content=content_parts,
                role=xai_chat.chat_pb2.MessageRole.ROLE_ASSISTANT,
            )
            for tc in msg["tool_calls"]:
                pb_tc = xai_chat.chat_pb2.ToolCall(
                    id=tc.get("id", ""),
                    type=xai_chat.chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                )
                pb_tc.function.CopyFrom(
                    xai_chat.chat_pb2.FunctionCall(
                        name=tc.get("name", ""),
                        arguments=json.dumps(tc.get("input", {})),
                    )
                )
                pb_msg.tool_calls.append(pb_tc)
            sdk_msgs.append(pb_msg)

        elif role == "assistant":
            sdk_msgs.append(xai_chat.assistant(msg.get("content", "")))

        else:
            sdk_msgs.append(xai_chat.user(msg.get("content", "")))

    system_text = "\n\n".join(system_parts) if system_parts else None
    return sdk_msgs, system_text


def _tool_to_sdk(tool: dict[str, Any]) -> Any:
    """Map generic tool dict to xAI SDK Tool."""
    schema = tool.get("input_schema", tool.get("parameters", {}))
    return xai_chat.tool(
        name=tool["name"],
        description=tool.get("description", ""),
        parameters=schema,
    )


def _build_response_format(output_schema: OutputSchema) -> Any:
    """Build xAI ``response_format`` for structured output."""
    rf = xai_chat.chat_pb2.ResponseFormat()
    rf.format_type = xai_chat.chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA
    rf.schema = json.dumps(output_schema.schema)
    return rf


def _extract_usage(sdk_usage: Any) -> Usage:
    """Convert SDK SamplingUsage to our Usage dataclass."""
    return Usage(
        input_tokens=getattr(sdk_usage, "input_tokens", 0)
        or getattr(sdk_usage, "prompt_tokens", 0),
        output_tokens=getattr(sdk_usage, "output_tokens", 0)
        or getattr(sdk_usage, "completion_tokens", 0),
        cache_read_tokens=getattr(sdk_usage, "cached_prompt_text_tokens", 0),
    )


def _parse_sdk_response(
    response: Any,
    model: str,
    *,
    output_schema: OutputSchema | None = None,
) -> Response:
    """Convert xAI SDK ``Response`` to our ``Response``.

    Note: ``response.content`` is a plain string here (SDK flattens it),
    unlike the proto ``Message.content`` which is a list of ``Content`` protos.
    """
    text = response.content or ""

    # Tool calls
    tool_calls: list[ToolCall] = []
    for tc in response.tool_calls or []:
        tool_calls.append(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=parse_tool_args(tc.function.arguments),
            )
        )

    # Thinking/reasoning
    thinking_blocks: list[ThinkingBlock] = []
    reasoning = response.reasoning_content
    if reasoning:
        thinking_blocks.append(ThinkingBlock(text=reasoning))

    # Structured output
    parsed: Any = None
    if output_schema and text:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse structured output as JSON")

    usage = _extract_usage(response.usage) if response.usage else Usage()
    cost, cost_estimated = _estimate_response_cost(model, usage)

    return Response(
        text=text.strip(),
        tool_calls=tuple(tool_calls),
        thinking=tuple(thinking_blocks),
        parsed=parsed,
        usage=usage,
        cost=cost,
        cost_estimated=cost_estimated,
        stop_reason=response.finish_reason or "",
        model=model,
        raw=response,
    )


def _grpc_code_to_http(code: grpc.StatusCode) -> int:
    """Map gRPC status code to approximate HTTP status code."""
    return {
        grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
        grpc.StatusCode.INVALID_ARGUMENT: 400,
        grpc.StatusCode.UNAUTHENTICATED: 401,
        grpc.StatusCode.PERMISSION_DENIED: 403,
        grpc.StatusCode.NOT_FOUND: 404,
        grpc.StatusCode.UNAVAILABLE: 503,
        grpc.StatusCode.INTERNAL: 500,
        grpc.StatusCode.DEADLINE_EXCEEDED: 504,
        grpc.StatusCode.CANCELLED: 499,
        grpc.StatusCode.UNKNOWN: 500,
        grpc.StatusCode.ABORTED: 409,
        grpc.StatusCode.DATA_LOSS: 500,
        grpc.StatusCode.OUT_OF_RANGE: 400,
        grpc.StatusCode.FAILED_PRECONDITION: 412,
        grpc.StatusCode.ALREADY_EXISTS: 409,
    }.get(code, 500)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class XAIProvider(BaseProvider):
    """xAI provider via the official ``xai-sdk`` (gRPC).

    Provides access to Grok models with native support for reasoning,
    server-side tools (web search, X search), and structured output.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
    ) -> None:
        self._model = model
        self._client = xai_sdk.AsyncClient(api_key=api_key)

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_create_kwargs(
        self,
        sdk_messages: list[Any],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build kwargs for ``client.chat.create()``."""
        # Extract special params
        thinking = kwargs.pop("thinking", False)
        thinking_effort = kwargs.pop("thinking_effort", None)
        thinking_budget = kwargs.pop("thinking_budget", None)
        if thinking_budget:
            warnings.warn(
                "thinking_budget is not supported by xAI (only reasoning_effort string), ignoring",
                stacklevel=4,
            )
        output_schema: OutputSchema | None = kwargs.pop("output_schema", None)

        # Warn about unknown params
        unknown = set(kwargs) - _SDK_PARAMS
        if unknown:
            warnings.warn(
                f"Unknown parameter(s) ignored for xAI: {sorted(unknown)}. "
                f"Valid: {sorted(_SDK_PARAMS)}",
                stacklevel=4,
            )
        filtered = {k: v for k, v in kwargs.items() if k in _SDK_PARAMS}

        # Build initial messages including system
        initial_messages = []
        if system:
            initial_messages.append(xai_chat.system(system))
        initial_messages.extend(sdk_messages)

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": initial_messages,
            **filtered,
        }

        # Reasoning — default to "high" when thinking=True but no effort specified
        if thinking:
            create_kwargs["reasoning_effort"] = thinking_effort or "high"
            incompatible = {"stop", "frequency_penalty", "presence_penalty"} & set(filtered)
            if incompatible:
                warnings.warn(
                    f"Parameters {sorted(incompatible)} are incompatible with "
                    f"xAI reasoning models",
                    stacklevel=4,
                )

        # Tools
        if tools:
            create_kwargs["tools"] = [_tool_to_sdk(t) for t in tools]

        # Structured output
        if output_schema:
            create_kwargs["response_format"] = _build_response_format(output_schema)

        return create_kwargs

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
        sdk_msgs, msg_system = _messages_to_sdk(messages)
        effective_system = system if system is not None else msg_system

        create_kwargs = self._build_create_kwargs(
            sdk_msgs, system=effective_system, tools=tools, **kwargs
        )

        try:
            chat = self._client.chat.create(**create_kwargs)
            response = await chat.sample()
        except grpc.aio.AioRpcError as exc:
            status_code = _grpc_code_to_http(exc.code())
            if exc.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise RateLimitError(status_code, exc.details() or str(exc)) from exc
            raise APIError(status_code, exc.details() or str(exc)) from exc

        return _parse_sdk_response(response, self._model, output_schema=output_schema)

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
        sdk_msgs, msg_system = _messages_to_sdk(messages)
        effective_system = system if system is not None else msg_system

        create_kwargs = self._build_create_kwargs(
            sdk_msgs, system=effective_system, tools=tools, **kwargs
        )

        state = StreamState()
        state.model = self._model

        async def _generate() -> AsyncIterator[str]:
            try:
                chat = self._client.chat.create(**create_kwargs)
                final_response = None
                async for response, chunk in chat.stream():
                    final_response = response

                    # Text content
                    if chunk.content:
                        yield chunk.content

                    # Reasoning/thinking content
                    if chunk.reasoning_content:
                        state.thinking.append(ThinkingBlock(text=chunk.reasoning_content))

                    # Tool calls — xAI delivers complete tool calls per chunk
                    # (unlike OpenAI's index-based delta accumulation)
                    for tc in chunk.tool_calls or []:
                        state.tool_calls.append(
                            ToolCall(
                                id=tc.id,
                                name=tc.function.name,
                                input=parse_tool_args(tc.function.arguments),
                            )
                        )

                # After stream completes, extract final state from response
                if final_response is not None:
                    state.raw = final_response
                    if final_response.usage:
                        state.usage = _extract_usage(final_response.usage)
                    state.stop_reason = final_response.finish_reason or ""

            except grpc.aio.AioRpcError as exc:
                status_code = _grpc_code_to_http(exc.code())
                if exc.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    raise RateLimitError(status_code, exc.details() or str(exc)) from exc
                raise APIError(status_code, exc.details() or str(exc)) from exc

        return _generate(), state
