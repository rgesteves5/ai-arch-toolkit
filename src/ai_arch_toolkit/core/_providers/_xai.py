"""xAI provider — Grok through the gRPC ``xai-sdk``, in the provider contract."""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Required, TypedDict, cast

from ai_arch_toolkit.core._content import CachePart, DocumentPart, ImagePart
from ai_arch_toolkit.core._exceptions import (
    APIError,
    ProviderError,
    ProviderTimeout,
    RateLimitError,
    RequestError,
    TransportError,
)
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._model_id import lookup
from ai_arch_toolkit.core._providers._base import (
    BaseProvider,
    Done,
    LoopAwareClientCache,
    Options,
    Prepared,
    mark_dispatched,
    merge_system_prompts,
    parse_options,
    parse_structured,
    parse_tool_args,
    refused_or_unread,
    system_content_text,
)
from ai_arch_toolkit.core._providers._imports import require_sdk
from ai_arch_toolkit.core._response import (
    OutputSchema,
    Response,
    StreamEvent,
    ThinkingBlock,
    ToolCall,
    Usage,
    _uncached_input_tokens,
)

with require_sdk("xai"):
    import grpc  # an xai-sdk dependency
    import xai_sdk
    from xai_sdk import chat as xai_chat
    from xai_sdk.aio.chat import Chat
    from xai_sdk.aio.chat import Client as ChatClient
    from xai_sdk.proto import chat_pb2, usage_pb2
    from xai_sdk.types.chat import AgentCount, ReasoningEffort, ToolMode

logger = logging.getLogger(__name__)

# SDK parameters forwarded as the caller gave them.
_FORWARDED = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "agent_count",
    }
)
# Refused with an error by xAI's reasoning models
# (https://docs.x.ai/developers/model-capabilities/text/reasoning).
_NOT_WHILE_REASONING = ("stop", "presence_penalty", "frequency_penalty")
_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})
# The multi-agent model runs 4 agents at low and medium effort, 16 at high and xhigh
# (https://docs.x.ai/developers/model-capabilities/text/multi-agent).
_AGENTS: dict[str, AgentCount] = {"low": 4, "medium": 4, "high": 16, "xhigh": 16}

# Transparent gRPC retries off: LLM(RetryConfig(...)) owns retries, so each physical attempt is
# metered and audited.
_CHANNEL_OPTIONS = [("grpc.enable_retries", 0)]

# The HTTP status of each gRPC code
# (https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto).
_HTTP_STATUS: dict[grpc.StatusCode, int] = {
    grpc.StatusCode.CANCELLED: 499,
    grpc.StatusCode.UNKNOWN: 500,
    grpc.StatusCode.INVALID_ARGUMENT: 400,
    grpc.StatusCode.NOT_FOUND: 404,
    grpc.StatusCode.ALREADY_EXISTS: 409,
    grpc.StatusCode.PERMISSION_DENIED: 403,
    grpc.StatusCode.UNAUTHENTICATED: 401,
    grpc.StatusCode.FAILED_PRECONDITION: 400,
    grpc.StatusCode.ABORTED: 409,
    grpc.StatusCode.OUT_OF_RANGE: 400,
    grpc.StatusCode.UNIMPLEMENTED: 501,
    grpc.StatusCode.INTERNAL: 500,
    grpc.StatusCode.DATA_LOSS: 500,
}


class _Create(TypedDict, total=False):
    """The ``chat.create`` arguments this adapter sends, in the SDK's own types."""

    model: Required[str]
    messages: Required[list[chat_pb2.Message]]
    max_tokens: int
    seed: int
    stop: list[str]
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    tools: list[chat_pb2.Tool]
    tool_choice: ToolMode | chat_pb2.ToolChoice
    response_format: chat_pb2.ResponseFormat
    reasoning_effort: ReasoningEffort
    agent_count: AgentCount


@dataclass(frozen=True, slots=True, kw_only=True)
class _Profile:
    """What one family of Grok models takes.

    ``efforts``: the ``reasoning_effort`` values the model documents (none: it takes no effort).
    ``reasons``: ``thinking`` is available, and ``stop`` and the penalties are refused.
    ``multi_agent``: the effort picks the number of agents; no ``max_tokens``, no client tools.
    """

    efforts: frozenset[str] = _EFFORTS
    reasons: bool = True
    multi_agent: bool = False


_CURRENT = _Profile()  # grok-4.7, grok-4.6, grok-4.5 (it serves xhigh as high), newer models
_GROK_4_3 = _Profile(efforts=_EFFORTS | {"none"})
_AUTO = _Profile(efforts=frozenset())  # reasons with no documented effort
_PLAIN = _Profile(efforts=frozenset(), reasons=False)
_MULTI_AGENT = _Profile(multi_agent=True)

# The families that differ from the current generation, a closed list: a model not listed (a new
# one) gets the current generation's rules. Efforts are those of each model's page
# (https://docs.x.ai/developers/models/grok-4.3 and the others); the ids retired on 2026-05-15
# are served by grok-4.3, the reasoning ones at low effort and the others at none, and
# grok-code-fast-1 by grok-build-0.1 (https://docs.x.ai/developers/migration/may-15-retirement).
_PROFILES: dict[str, _Profile] = {
    "grok-4.3": _GROK_4_3,
    "grok-4.20-multi-agent": _MULTI_AGENT,
    **dict.fromkeys(
        (
            "grok-4.20",
            "grok-4.20-reasoning",
            "grok-4.20-0309-reasoning",
            "grok-build-0.1",
            "grok-code-fast",
            "grok-code-fast-1",
        ),
        _AUTO,
    ),
    **dict.fromkeys(("grok-4.20-non-reasoning", "grok-4.20-0309-non-reasoning"), _PLAIN),
    **dict.fromkeys(("grok-4", "grok-4-fast-reasoning", "grok-4-1-fast-reasoning"), _GROK_4_3),
    **dict.fromkeys(
        ("grok-3", "grok-4-fast-non-reasoning", "grok-4-1-fast-non-reasoning"), _PLAIN
    ),
}


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def _user_text(content: Any) -> str:
    """A user turn's text: this adapter sends no images or documents to xAI."""
    if not isinstance(content, list):
        return content
    texts: list[str] = []
    for part in content:
        if isinstance(part, CachePart):
            texts.append(part.content)
        elif isinstance(part, ImagePart):
            warnings.warn("xAI does not support image input; image part dropped", stacklevel=4)
        elif isinstance(part, DocumentPart):
            warnings.warn(
                "xAI does not support document input; document part dropped", stacklevel=4
            )
        else:
            texts.append(part if isinstance(part, str) else str(part))
    return "\n".join(texts)


def _assistant(msg: dict[str, Any]) -> chat_pb2.Message:
    """An assistant turn with all its tool calls."""
    text = msg.get("content") or ""
    calls = [
        chat_pb2.ToolCall(
            id=call.get("id", ""),
            type=chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
            function=chat_pb2.FunctionCall(
                name=call.get("name", ""), arguments=json.dumps(call.get("input", {}))
            ),
        )
        for call in msg.get("tool_calls") or []
    ]
    return chat_pb2.Message(
        role=chat_pb2.MessageRole.ROLE_ASSISTANT,
        content=[chat_pb2.Content(text=text)] if text else [],
        tool_calls=calls,
    )


def _messages_to_sdk(messages: list[dict[str, Any]]) -> tuple[list[chat_pb2.Message], str | None]:
    """The SDK messages, and the text of the ``system()`` messages (sent as one system message).

    ``tool_use_id`` marks a tool result: each is its own message, with its call's id, in the order
    of the calls (https://docs.x.ai/docs/guides/function-calling).
    """
    sdk_messages: list[chat_pb2.Message] = []
    system: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        if msg.get("tool_use_id"):
            content = str(msg.get("content", ""))
            sdk_messages.append(xai_chat.tool_result(content, tool_call_id=msg["tool_use_id"]))
        elif role == "system":
            system.append(system_content_text(msg.get("content", "")))
        elif role == "assistant":
            sdk_messages.append(_assistant(msg))
        else:
            sdk_messages.append(xai_chat.user(_user_text(msg.get("content", ""))))
    return sdk_messages, "\n\n".join(system) if system else None


def _tool_to_sdk(tool: dict[str, Any]) -> chat_pb2.Tool:
    """A function tool."""
    return xai_chat.tool(
        name=tool["name"],
        description=tool.get("description", ""),
        parameters=tool.get("input_schema", tool.get("parameters", {})),
    )


def _tool_choice(choice: str) -> ToolMode | chat_pb2.ToolChoice:
    if choice == "auto" or choice == "required" or choice == "none":
        return choice
    return xai_chat.required_tool(choice)


def _build_response_format(output_schema: OutputSchema) -> chat_pb2.ResponseFormat:
    """Build xAI ``response_format`` for structured output."""
    return chat_pb2.ResponseFormat(
        format_type=chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
        schema=json.dumps(output_schema.schema),
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def _extract_usage(usage: usage_pb2.SamplingUsage) -> Usage:
    """The toolkit's usage; the gRPC API counts the answer and the reasoning apart, and both are
    billed as output."""
    cache_read = usage.cached_prompt_text_tokens
    return Usage(
        input_tokens=_uncached_input_tokens(usage.prompt_tokens, cache_read),
        output_tokens=usage.completion_tokens + usage.reasoning_tokens,
        cache_read_tokens=cache_read,
    )


def _provider_cost(response: xai_chat.Response) -> float | None:
    """The cost xAI reports for the request, when it reports a valid one."""
    cost = response.cost_usd
    if cost is not None and (cost < 0 or not math.isfinite(cost)):
        logger.warning("Ignoring invalid xAI provider-reported cost: %r", cost)
        return None
    return cost


def _parse_sdk_response(
    response: xai_chat.Response,
    model: str,
    *,
    output_schema: OutputSchema | None = None,
) -> Response:
    """The ``Response`` for the SDK's response (the base adds usage and cost)."""
    text = response.content
    reasoning = response.reasoning_content
    return Response(
        text=text.strip(),
        tool_calls=tuple(
            ToolCall(
                id=call.id, name=call.function.name, input=parse_tool_args(call.function.arguments)
            )
            for call in response.tool_calls
        ),
        thinking=(ThinkingBlock(text=reasoning),) if reasoning else (),
        parsed=parse_structured(text, output_schema) if output_schema and text else None,
        provider_cost=_provider_cost(response),
        stop_reason=response.finish_reason,
        model=model,
        raw=response,
        response_id=response.id,
    )


def _chunk_events(chunk: xai_chat.Chunk) -> list[StreamEvent]:
    """The text and reasoning deltas of one streamed chunk."""
    events: list[StreamEvent] = []
    if chunk.reasoning_content:
        events.append(
            StreamEvent(
                kind="thinking", thinking=ThinkingBlock(text=chunk.reasoning_content), partial=True
            )
        )
    if chunk.content:
        events.append(StreamEvent(kind="text", text=chunk.content))
    return events


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class XAIProvider(LoopAwareClientCache, BaseProvider[Prepared[Chat], xai_chat.Response]):
    """xAI provider via the official ``xai-sdk`` (gRPC)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        timeout: float | None = None,
    ) -> None:
        self._model = model
        self._install_client(
            lambda: xai_sdk.AsyncClient(
                api_key=api_key, channel_options=_CHANNEL_OPTIONS, timeout=timeout
            )
        )

    async def close(self) -> None:
        await self._client.close()

    def _profile(self) -> _Profile:
        found = lookup(self._model, _PROFILES)
        return found.value if found is not None else _CURRENT

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    def prepare(self, request: Request) -> Prepared[Chat]:
        options = parse_options(request.kwargs, _FORWARDED, "xAI")
        profile = self._profile()
        messages, message_system = _messages_to_sdk(request.messages)
        system = merge_system_prompts(request.system, message_system)
        params: _Create = {
            "model": self._model,
            "messages": [xai_chat.system(system), *messages] if system else messages,
        }
        params.update(cast("_Create", options.params))
        if options.thinking_budget:
            warnings.warn(
                "thinking_budget is not supported by xAI (only thinking_effort), ignoring",
                stacklevel=4,
            )
        if profile.multi_agent:
            self._agents(params, options)
        else:
            self._reasoning(params, options, profile)
        self._tools(params, request.tools, options, profile)
        if options.output_schema is not None:
            params["response_format"] = _build_response_format(options.output_schema)
        if options.json_mode:
            params["response_format"] = chat_pb2.ResponseFormat(
                format_type=chat_pb2.FormatType.FORMAT_TYPE_JSON_OBJECT
            )
        return Prepared(self._create(params), output_schema=options.output_schema)

    def _reasoning(self, params: _Create, options: Options, profile: _Profile) -> None:
        """``thinking_effort`` applies on its own, since these models reason unasked (as the
        Muse Spark, D13); ``thinking`` asks nothing more, except of a model that does not reason.
        """
        if options.thinking and not profile.reasons:
            raise RequestError(f"{self._model} does not reason: thinking is not available")
        if profile.reasons and (refused := [p for p in _NOT_WHILE_REASONING if p in params]):
            raise RequestError(
                f"{self._model} reasons, and xAI's reasoning models refuse {', '.join(refused)}"
            )
        effort = options.thinking_effort
        if effort is None:
            return
        if effort not in profile.efforts:
            takes = f"thinking_effort in {sorted(profile.efforts)}" if profile.efforts else None
            raise RequestError(
                f"{self._model} takes {takes or 'no thinking_effort'}, not {effort!r}"
            )
        params["reasoning_effort"] = cast("ReasoningEffort", effort)

    def _agents(self, params: _Create, options: Options) -> None:
        """The multi-agent model: the effort picks the number of agents, and ``max_tokens`` is
        not taken (the ``LLM`` always sends one)."""
        params.pop("max_tokens", None)
        effort = options.thinking_effort or ("low" if options.thinking else None)
        if effort is None:
            return
        if effort not in _AGENTS:
            raise RequestError(
                f"{self._model} takes thinking_effort in {sorted(_AGENTS)}, not {effort!r}"
            )
        params.setdefault("agent_count", _AGENTS[effort])

    def _tools(
        self,
        params: _Create,
        tools: list[dict[str, Any]] | None,
        options: Options,
        profile: _Profile,
    ) -> None:
        tools = tools or []
        # The SDK has server-side tools (xai_sdk.tools: web_search, x_search, code_execution…);
        # the toolkit sends them to xAI only once C05 adds them.
        if server := [tool["type"] for tool in tools if tool.get("_server_tool")]:
            raise RequestError(
                f"this adapter sends no xAI server tool yet ({', '.join(server)}): "
                "only function tools reach xAI"
            )
        if tools and profile.multi_agent:
            raise RequestError(f"{self._model} takes no client-side tools")
        if tools:
            params["tools"] = [_tool_to_sdk(tool) for tool in tools]
        if options.tool_choice is not None:
            params["tool_choice"] = _tool_choice(options.tool_choice)

    def _create(self, params: _Create) -> Chat:
        """The SDK's request: ``chat.create`` builds it locally, with no RPC."""
        chat: ChatClient = self._client.chat
        try:
            return chat.create(**params)
        except (TypeError, ValueError) as refused:  # the SDK's own validation
            raise RequestError(f"the xAI SDK refused the request: {refused}") from refused

    async def send(self, prepared: Prepared[Chat]) -> xai_chat.Response:
        mark_dispatched()  # every local step ran in prepare
        return await prepared.params.sample()

    async def open_stream(
        self, prepared: Prepared[Chat]
    ) -> AsyncIterator[StreamEvent | Done[xai_chat.Response]]:
        mark_dispatched()
        # The SDK pairs every chunk with the response accumulated so far.
        final: xai_chat.Response | None = None
        async for accumulated, chunk in prepared.params.stream():
            final = accumulated
            for event in _chunk_events(chunk):
                yield event
        if final is not None:
            yield Done(final)

    def assemble(self, final: xai_chat.Response, prepared: Prepared[Chat]) -> Response:
        return _parse_sdk_response(final, self._model, output_schema=prepared.output_schema)

    def usage(self, final: xai_chat.Response) -> Usage | None:
        # The stream accumulator copies every chunk's usage, so a stream without one ends with an
        # empty usage: only a usage with a field set was reported.
        return _extract_usage(final.usage) if final.usage.ListFields() else None

    def map_error(self, exc: Exception, *, sent: bool) -> ProviderError | None:
        """The one place that knows gRPC's errors.

        xAI bills requests that break its usage guidelines (https://docs.x.ai/developers/pricing)
        and documents nothing else about failed requests, so only ``RESOURCE_EXHAUSTED`` (a 429,
        D20) is unbilled. ``UNAVAILABLE`` also covers a connection that was never made: gRPC does
        not say whether the request left, so it stays indeterminate.
        """
        if not isinstance(exc, grpc.aio.AioRpcError):
            return refused_or_unread(exc, sent=sent)
        code = exc.code()
        details = exc.details() or code.name
        if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            return RateLimitError(429, details)
        if code == grpc.StatusCode.UNAVAILABLE:
            return TransportError(details)
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return ProviderTimeout(details)
        return APIError(_HTTP_STATUS.get(code, 500), details)
