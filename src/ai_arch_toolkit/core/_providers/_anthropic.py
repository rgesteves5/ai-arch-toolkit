"""Anthropic provider — the Messages API through ``anthropic``, in the provider contract."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, cast

from ai_arch_toolkit.core._content import CachePart, DocumentPart, ImagePart, _encode_b64, _is_url
from ai_arch_toolkit.core._exceptions import (
    APIError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._model_id import lookup
from ai_arch_toolkit.core._providers._base import (
    DEFAULT_THINKING_BUDGET,
    THINKING_EFFORT_BUDGETS,
    BaseProvider,
    Done,
    LoopAwareClientCache,
    Options,
    Prepared,
    StreamEvent,
    _parse_retry_after,
    mark_dispatched,
    merge_system_prompts,
    on_request,
    parse_options,
    parse_structured,
    parse_tool_args,
    refused_or_unread,
    system_content_text,
    transport_error,
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

with require_sdk("anthropic"):
    import anthropic
    import httpx2  # the anthropic SDK's transport
    from anthropic.resources.messages import AsyncMessages
    from anthropic.types import (
        CacheControlEphemeralParam,
        ContentBlock,
        ContentBlockParam,
        Message,
        MessageParam,
        OutputConfigParam,
        TextBlockParam,
        TextCitation,
        ToolChoiceParam,
        ToolParam,
        ToolResultBlockParam,
        ToolUnionParam,
        ToolUseBlockParam,
    )
    from anthropic.types import Usage as SDKUsage
    from anthropic.types.message_count_tokens_params import MessageCountTokensParams
    from anthropic.types.message_create_params import (
        MessageCreateParamsBase,
        MessageCreateParamsNonStreaming,
    )
    from anthropic.types.messages.batch_create_params import Request as BatchRequest

logger = logging.getLogger(__name__)


class Params(MessageCreateParamsBase, total=False):
    """``messages.create``/``messages.stream`` arguments: the SDK's own, and the body fields its
    signature lacks."""

    # anthropic 1.x removed temperature, top_p and top_k from its signatures (passing one is a
    # TypeError), but the API still takes them on the models that sample.
    extra_body: dict[str, object]


# SDK arguments and body fields forwarded as the caller gave them.
_FORWARDED = frozenset({"max_tokens", "stop_sequences", "temperature", "top_p", "top_k"})
_SAMPLING = ("temperature", "top_p", "top_k")
_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_MIN_BUDGET = 1024  # https://platform.claude.com/docs/en/build-with-claude/extended-thinking

# Failures before the request reached the server; any other transport failure may be billed.
_NOT_SENT = (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.PoolTimeout)
_TIMEOUTS = (httpx2.TimeoutException, anthropic.APITimeoutError)

# The error types and their statuses (https://platform.claude.com/docs/en/api/errors); an error
# event inside a stream carries only its type.
_ERROR_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "conflict_error": 409,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class _Profile:
    """How one family of Claude models is asked to think, per the documented tables
    (https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting and
    https://platform.claude.com/docs/en/build-with-claude/effort).

    ``thinking``: ``adaptive`` models take ``{type: "adaptive"}`` and an ``output_config.effort``
    in ``efforts``; ``extended`` models take a ``budget_tokens`` budget. ``sampling``: the model
    takes ``temperature``, ``top_p`` and ``top_k``. ``forced_tools``: it takes ``tool_choice``
    ``any`` and ``tool``.
    """

    thinking: Literal["adaptive", "extended"] = "adaptive"
    efforts: frozenset[str] = _EFFORTS
    sampling: bool = False
    forced_tools: bool = True


# The Claude 5 family (Opus 5, Sonnet 5, Fable 5, Mythos 5), Opus 4.8 and 4.7, and any newer
# model: adaptive thinking with every effort level, no sampling parameters.
_CURRENT = _Profile()
_EXTENDED = _Profile(
    thinking="extended", efforts=frozenset(THINKING_EFFORT_BUDGETS), sampling=True
)
# The families that differ from the current rules, a closed list.
_PROFILES: dict[str, _Profile] = {
    # Forced tool use returns a 400 (https://platform.claude.com/docs/en/api/errors).
    **dict.fromkeys(("claude-fable-5-1", "claude-mythos-5-1"), _Profile(forced_tools=False)),
    "claude-mythos-preview": _Profile(efforts=_EFFORTS - {"xhigh"}),
    **dict.fromkeys(
        ("claude-opus-4-6", "claude-sonnet-4-6"),
        _Profile(efforts=_EFFORTS - {"xhigh"}, sampling=True),
    ),
    **dict.fromkeys(
        (
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-opus-4-1",
            "claude-opus-4",
            "claude-opus-4-0",
            "claude-sonnet-4",
            "claude-sonnet-4-0",
        ),
        _EXTENDED,
    ),
}


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def _cache_control(part: CachePart) -> CacheControlEphemeralParam:
    if part.ttl == "5m" or part.ttl == "1h":
        return {"type": "ephemeral", "ttl": part.ttl}
    return {"type": "ephemeral"}


def _block(part: Any) -> ContentBlockParam:
    if isinstance(part, ImagePart) and isinstance(part.source, str) and _is_url(part.source):
        return {"type": "image", "source": {"type": "url", "url": part.source}}
    if isinstance(part, ImagePart):
        media = cast(
            "Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']", part.media_type
        )
        data = _encode_b64(part.source)
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
    if isinstance(part, DocumentPart):
        source = {
            "type": "base64",
            "media_type": part.media_type,
            "data": _encode_b64(part.source),
        }
        document: ContentBlockParam = {"type": "document", "source": cast("Any", source)}
        if part.name:
            document["title"] = part.name
        return document
    if isinstance(part, CachePart):
        return {"type": "text", "text": part.content, "cache_control": _cache_control(part)}
    return {"type": "text", "text": part if isinstance(part, str) else str(part)}


def _content_to_sdk(content: Any) -> list[ContentBlockParam] | str:
    """Convert multimodal content to Anthropic content blocks, or keep a plain string."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    return [_block(part) for part in content]


def _tool_to_sdk(tool: dict[str, Any]) -> ToolParam:
    """Map generic tool dict to Anthropic SDK format."""
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("input_schema", tool.get("parameters", {})),
    }


def _server_tool(tool: dict[str, Any]) -> ToolUnionParam:
    """A server tool with its name, at the version every current model takes; its config
    belongs to C05 (https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool,
    https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool)."""
    config = sorted(set(tool) - {"_server_tool", "type"})
    if config:
        raise RequestError(f"server tool {tool['type']!r}: config {config} is not supported yet")
    if tool["type"] == "web_search":
        return {"type": "web_search_20250305", "name": "web_search"}
    if tool["type"] == "code_execution":
        return {"type": "code_execution_20250825", "name": "code_execution"}
    raise RequestError(f"Anthropic has no server tool {tool['type']!r} in this adapter")


def _tool_choice(choice: str) -> ToolChoiceParam:
    if choice == "auto":
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    return {"type": "tool", "name": choice}


type SystemParam = str | list[TextBlockParam]


def _system_blocks(content: Any) -> list[TextBlockParam]:
    """Text blocks for one system message; a ``cache()`` part keeps its cache marker."""
    parts = content if isinstance(content, list | tuple) else [content]
    blocks: list[TextBlockParam] = []
    for part in parts:
        if isinstance(part, CachePart):
            if part.content:
                blocks.append(
                    {"type": "text", "text": part.content, "cache_control": _cache_control(part)}
                )
        elif text := system_content_text(part):
            blocks.append({"type": "text", "text": text})
    return blocks


def _as_system_blocks(system: SystemParam | None) -> list[TextBlockParam]:
    if not system:
        return []
    if isinstance(system, str):
        return [{"type": "text", "text": system}]
    return list(system)


def _merge_system(
    system: SystemParam | None, msg_system: SystemParam | None
) -> SystemParam | None:
    """``system=`` first, then the system messages; text blocks when either side has blocks.

    Blocks carry cache markers, so they are only used when needed; plain text stays a string.
    """
    if isinstance(system, list) or isinstance(msg_system, list):
        return [*_as_system_blocks(system), *_as_system_blocks(msg_system)] or None
    return merge_system_prompts(system, msg_system)


def _with_system_suffix(system: SystemParam | None, suffix: str) -> SystemParam:
    """Append an instruction to the system prompt, as a text block when it is blocks."""
    if isinstance(system, list):
        return [*system, {"type": "text", "text": suffix}]
    return f"{system}\n\n{suffix}" if system else suffix


def _text_of(content: list[ContentBlock]) -> str:
    return "".join(block.text for block in content if block.type == "text").strip()


def _replayable(msg: dict[str, Any]) -> list[ContentBlock] | None:
    """The content Claude sent for this turn, while the message still matches it.

    Thinking blocks, their signatures and server tool results must go back as received
    (https://platform.claude.com/docs/en/build-with-claude/thinking); a message whose text or
    tool calls were changed, or whose ``_raw`` came from another provider, is rebuilt instead.
    """
    raw = msg.get("_raw")
    if not isinstance(raw, Message) or _text_of(raw.content) != (msg.get("content") or "").strip():
        return None
    sent = [(b.id, b.name, b.input) for b in raw.content if b.type == "tool_use"]
    calls = [
        (c.get("id", ""), c.get("name", ""), c.get("input", {}))
        for c in msg.get("tool_calls") or []
    ]
    return list(raw.content) if sent == calls else None


def _tool_use(call: dict[str, Any]) -> ToolUseBlockParam:
    return {
        "type": "tool_use",
        "id": call.get("id", ""),
        "name": call.get("name", ""),
        "input": call.get("input", {}),
    }


def _assistant(msg: dict[str, Any]) -> MessageParam:
    """An assistant turn: as Claude sent it, else rebuilt from its text and calls."""
    if (replay := _replayable(msg)) is not None:
        return {"role": "assistant", "content": replay}
    calls = msg.get("tool_calls") or []
    if not calls:
        return {"role": "assistant", "content": _content_to_sdk(msg.get("content", ""))}
    text = msg.get("content") or ""
    blocks: list[ContentBlockParam] = [{"type": "text", "text": text}] if text else []
    blocks.extend(_tool_use(call) for call in calls)
    return {"role": "assistant", "content": blocks}


def _messages_to_sdk(
    messages: list[dict[str, Any]],
) -> tuple[SystemParam | None, list[MessageParam]]:
    """The system prompt, and the messages.

    ``tool_use_id`` marks a tool result: every result of a turn goes back in one user message, in
    call order (https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use).
    The system prompt is one string, or text blocks when a ``cache()`` part asks for a cache
    marker.
    """
    system_blocks: list[TextBlockParam] = []
    wire: list[MessageParam] = []
    results: list[ToolResultBlockParam] = []
    for msg in messages:
        if msg.get("tool_use_id"):
            content = msg.get("content", "")
            results.append(
                {"type": "tool_result", "tool_use_id": msg["tool_use_id"], "content": content}
            )
            continue
        if results:
            wire.append({"role": "user", "content": results})
            results = []
        role = msg.get("role", "user")
        if role == "system":
            system_blocks.extend(_system_blocks(msg.get("content", "")))
        elif role == "assistant":
            wire.append(_assistant(msg))
        elif role == "user":
            wire.append({"role": "user", "content": _content_to_sdk(msg.get("content", ""))})
        else:
            raise RequestError(f"the Messages API has no {role!r} role")
    if results:
        wire.append({"role": "user", "content": results})
    if not system_blocks:
        return None, wire
    if any("cache_control" in block for block in system_blocks):
        return system_blocks, wire
    return "\n\n".join(block["text"] for block in system_blocks), wire


def _build_output_config(output_schema: OutputSchema) -> OutputConfigParam:
    """Build native ``output_config`` for structured output (Anthropic JSON mode)."""
    return {"format": {"type": "json_schema", "schema": output_schema.schema}}


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


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def _extract_usage(usage: SDKUsage) -> Usage:
    """The toolkit's usage; the SDK types the cache counters as nullable."""
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
    )


def _citation(cite: TextCitation) -> Citation:
    if cite.type == "web_search_result_location":
        return Citation(text=cite.cited_text, url=cite.url, title=cite.title or "")
    if cite.type == "char_location":
        return Citation(
            text=cite.cited_text,
            url="",
            title="",
            start_index=cite.start_char_index,
            end_index=cite.end_char_index,
        )
    return Citation(text=cite.cited_text, url="", title="")


def _parse_sdk_response(
    message: Message,
    model: str,
    *,
    output_schema: OutputSchema | None = None,
) -> Response:
    """Convert an ``anthropic.types.Message`` to our ``Response`` (the base adds usage, cost).

    A thinking block whose text the model left out (``display: "omitted"``) adds no thinking.
    """
    text = _text_of(message.content)
    return Response(
        text=text,
        tool_calls=tuple(
            ToolCall(id=block.id, name=block.name, input=parse_tool_args(cast("Any", block.input)))
            for block in message.content
            if block.type == "tool_use"
        ),
        thinking=tuple(
            ThinkingBlock(text=block.thinking)
            for block in message.content
            if block.type == "thinking" and block.thinking
        ),
        parsed=parse_structured(text, output_schema) if output_schema and text else None,
        stop_reason=message.stop_reason or "",
        model=message.model or model,
        raw=message,
        response_id=message.id or "",
        citations=tuple(
            _citation(cite)
            for block in message.content
            if block.type == "text"
            for cite in block.citations or []
        ),
    )


def _stream_event(event: Any) -> StreamEvent | None:
    """A text delta, or a finished thinking block, from one SDK stream event."""
    if event.type == "content_block_delta" and event.delta.type == "text_delta":
        return StreamEvent(kind="text", text=event.delta.text)
    if (
        event.type == "content_block_stop"
        and event.content_block.type == "thinking"
        and event.content_block.thinking
    ):
        return StreamEvent(
            kind="thinking", thinking=ThinkingBlock(text=event.content_block.thinking)
        )
    return None


def _stream_error(body: object) -> ProviderError:
    """An error event inside a 200 stream, typed by its documented error type.

    Tokens may already be billed, so it stays indeterminate (a 429 never is, D20).
    """
    error = body.get("error") if isinstance(body, dict) else None
    kind = error.get("type") if isinstance(error, dict) else None
    status = _ERROR_STATUS.get(kind) if isinstance(kind, str) else None
    if status is None:
        return ResponseError(f"the stream reported an error: {body}")
    if status == 429:
        return RateLimitError(status, str(body))
    return APIError(status, str(body))


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AnthropicProvider(LoopAwareClientCache, BaseProvider[Prepared[Params], Message]):
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
        self._install_client(
            lambda: anthropic.AsyncAnthropic(**client_kwargs, http_client=_http())
        )

    async def close(self) -> None:
        await self._client.close()

    def _messages(self) -> AsyncMessages:
        return self._client.messages

    def _profile(self) -> _Profile:
        found = lookup(self._model, _PROFILES)
        return found.value if found is not None else _CURRENT

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Count tokens using Anthropic's count_tokens API."""
        msg_system, wire = _messages_to_sdk(messages)
        request: MessageCountTokensParams = {"model": self._model, "messages": wire}
        if effective_system := _merge_system(system, msg_system):
            request["system"] = effective_system
        if function_tools := [_tool_to_sdk(t) for t in tools or [] if not t.get("_server_tool")]:
            request["tools"] = list[ToolUnionParam](function_tools)
        with self._mapped():
            result = await self._messages().count_tokens(**request)
        return result.input_tokens

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    def prepare(self, request: Request) -> Prepared[Params]:
        options = parse_options(request.kwargs, _FORWARDED, "Anthropic")
        profile = self._profile()
        if options.structured_output_mode not in ("native", "prompt"):
            raise RequestError(
                "structured_output_mode must be 'native' or 'prompt', "
                f"got {options.structured_output_mode!r}"
            )
        max_tokens = options.params.get("max_tokens")
        if max_tokens is None:
            raise RequestError("the Messages API requires max_tokens")
        msg_system, wire = _messages_to_sdk(request.messages)
        params: Params = {"model": self._model, "messages": wire, "max_tokens": max_tokens}
        if "stop_sequences" in options.params:
            params["stop_sequences"] = options.params["stop_sequences"]
        thinking = self._thinking(params, options, profile)
        self._sampling(params, options, profile, thinking=thinking)
        self._output(params, options, profile)
        self._tools(params, request.tools, options, profile)
        system = _merge_system(request.system, msg_system)
        if options.output_schema is not None and options.structured_output_mode == "prompt":
            system = _with_system_suffix(system, _schema_prompt_instruction(options.output_schema))
        if options.json_mode:  # Anthropic has no json_mode of its own
            system = _with_system_suffix(system, "Respond with valid JSON only.")
        if system:
            params["system"] = system
        return Prepared(params, output_schema=options.output_schema)

    def _thinking(self, params: Params, options: Options, profile: _Profile) -> bool:
        """Adaptive models think on ``thinking=True``, with a summary to show (``display``
        defaults to ``"omitted"`` on the newer ones); ``thinking=False`` sends nothing, so a
        model that thinks by default keeps doing so. Older models take a budget."""
        if profile.thinking == "extended":
            return self._budget(params, options)
        if options.thinking_budget is not None:
            warnings.warn(
                f"thinking_budget is not taken by {self._model}, which thinks adaptively; "
                "thinking_effort sets its effort, ignoring",
                stacklevel=5,
            )
        if options.thinking:
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
        return options.thinking

    def _budget(self, params: Params, options: Options) -> bool:
        """An older model's thinking budget, added to ``max_tokens`` so it never takes the
        answer's room (D20). An effort or a budget turns thinking on by itself."""
        if not (options.thinking or options.thinking_effort or options.thinking_budget):
            return False
        budget = options.thinking_budget
        effort = options.thinking_effort
        if budget is None and effort is not None:
            if effort not in THINKING_EFFORT_BUDGETS:
                raise RequestError(
                    f"{self._model} takes thinking_effort in {sorted(THINKING_EFFORT_BUDGETS)}, "
                    f"not {effort!r}"
                )
            budget = THINKING_EFFORT_BUDGETS[effort]
        budget = DEFAULT_THINKING_BUDGET if budget is None else budget
        if budget < _MIN_BUDGET:
            raise RequestError(
                f"{self._model} takes a thinking_budget of at least {_MIN_BUDGET}, not {budget}"
            )
        params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        params["max_tokens"] = params["max_tokens"] + budget
        return True

    def _sampling(
        self, params: Params, options: Options, profile: _Profile, *, thinking: bool
    ) -> None:
        """``temperature``, ``top_p`` and ``top_k`` travel in the request body. A model without
        sampling refuses them; its ``temperature`` is dropped, since the ``LLM`` always sends one.
        Thinking needs the default temperature."""
        given = {name: options.params[name] for name in _SAMPLING if name in options.params}
        if not profile.sampling:
            if explicit := sorted(set(given) - {"temperature"}):
                raise RequestError(f"{self._model} takes no sampling parameters: {explicit}")
            return
        if thinking:
            given.pop("temperature", None)
        if given:
            params["extra_body"] = dict(given)

    def _output(self, params: Params, options: Options, profile: _Profile) -> None:
        """``output_config``: the structured output format and, on adaptive models, the effort
        (which applies with or without thinking)."""
        config: OutputConfigParam = {}
        if options.output_schema is not None and options.structured_output_mode == "native":
            config = _build_output_config(options.output_schema)
        effort = options.thinking_effort
        if profile.thinking == "adaptive" and effort is not None:
            if effort not in profile.efforts:
                raise RequestError(
                    f"{self._model} takes thinking_effort in {sorted(profile.efforts)}, "
                    f"not {effort!r}"
                )
            config["effort"] = cast("Literal['low', 'medium', 'high', 'xhigh', 'max']", effort)
        if config:
            params["output_config"] = config

    def _tools(
        self,
        params: Params,
        tools: list[dict[str, Any]] | None,
        options: Options,
        profile: _Profile,
    ) -> None:
        tools = tools or []
        if tools:
            functions: list[ToolUnionParam] = [
                _tool_to_sdk(tool) for tool in tools if not tool.get("_server_tool")
            ]
            params["tools"] = functions + [
                _server_tool(tool) for tool in tools if tool.get("_server_tool")
            ]
        choice = options.tool_choice
        if choice is None:
            return
        if not profile.forced_tools and choice not in ("auto", "none"):
            raise RequestError(
                f"{self._model} takes tool_choice 'auto' or 'none' only: forced tool use "
                "returns a 400 (https://platform.claude.com/docs/en/api/errors)"
            )
        params["tool_choice"] = _tool_choice(choice)

    async def send(self, prepared: Prepared[Params]) -> Message:
        return await self._messages().create(**prepared.params)

    async def open_stream(
        self, prepared: Prepared[Params]
    ) -> AsyncIterator[StreamEvent | Done[Message]]:
        # The SDK accumulates the final message, cumulative usage deltas included.
        async with self._messages().stream(**prepared.params) as stream:
            mark_dispatched()  # a response has begun: the request was sent
            async for event in stream:
                if (decoded := _stream_event(event)) is not None:
                    yield decoded
            yield Done(await stream.get_final_message())

    def assemble(self, final: Message, prepared: Prepared[Params]) -> Response:
        return _parse_sdk_response(final, self._model, output_schema=prepared.output_schema)

    def usage(self, final: Message) -> Usage | None:
        return _extract_usage(final.usage)

    def map_error(self, exc: Exception, *, sent: bool) -> ProviderError | None:
        """The one place that knows the SDK's errors.

        "Failed requests aren't charged", but a client that times out or disconnects mid-request
        is (https://support.claude.com/en/articles/8977456-how-do-i-pay-for-my-claude-api-usage):
        an error response is unbilled, a transport failure after sending is indeterminate. An
        error event inside a stream arrives with the stream's 200.
        """
        if isinstance(exc, anthropic.APIStatusError):
            status = exc.response.status_code
            if status == 200:
                return _stream_error(exc.body)
            if isinstance(exc, anthropic.RateLimitError):
                retry_after = _parse_retry_after(exc.response.headers.get("retry-after"))
                return RateLimitError(status, str(exc.body), retry_after=retry_after)
            return APIError(status, str(exc.body), delivery="unbilled")
        if isinstance(exc, anthropic.APIConnectionError):
            return transport_error(exc.__cause__ or exc, not_sent=_NOT_SENT, timeouts=_TIMEOUTS)
        if isinstance(exc, httpx2.TransportError):  # raised while reading a stream
            return transport_error(exc, not_sent=_NOT_SENT, timeouts=_TIMEOUTS)
        return refused_or_unread(exc, sent=sent)

    # ------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------

    async def batch_submit(
        self,
        requests: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Submit a batch via the Message Batches API; each body is what ``prepare`` builds."""
        batch = [
            BatchRequest(custom_id=req.get("custom_id", ""), params=self._batch_params(req))
            for req in requests
        ]
        with self._mapped():
            result = await self._messages().batches.create(requests=batch)
        return result.id

    def _batch_params(self, req: dict[str, Any]) -> MessageCreateParamsNonStreaming:
        """A batch line's body: the same request a call sends (4096 tokens by default), with
        the sampling fields in the body itself."""
        request = Request(
            messages=req.get("messages", []),
            system=req.get("system"),
            tools=req.get("tools"),
            model=self._model,
            kwargs={"max_tokens": 4096, **req.get("kwargs", {})},
        )
        params = dict(self.prepare(request).params)
        body = cast("dict[str, object]", params.pop("extra_body", {}))
        return cast("MessageCreateParamsNonStreaming", {**params, **body})

    async def batch_status(self, batch_id: str) -> str:
        """Check batch status."""
        with self._mapped():
            result = await self._messages().batches.retrieve(batch_id)
        return result.processing_status

    async def batch_results(self, batch_id: str) -> list[Any]:
        """Retrieve completed batch results."""
        from ai_arch_toolkit.core._batch import BatchResult

        results: list[BatchResult] = []
        with self._mapped():
            async for entry in await self._messages().batches.results(batch_id):
                result = entry.result
                if result.type == "succeeded":
                    answer = self._answer(result.message, Prepared(cast("Params", {})))
                    results.append(
                        BatchResult(custom_id=entry.custom_id, response=answer.response)
                    )
                elif result.type == "errored":
                    results.append(BatchResult(custom_id=entry.custom_id, error=str(result.error)))
                else:
                    results.append(BatchResult(custom_id=entry.custom_id, error=result.type))
        return results


def _http() -> Any:
    """The SDK's HTTP client, with the hook that marks a request as handed to the transport."""
    return anthropic.DefaultAsyncHttpxClient(event_hooks={"request": [on_request]})
