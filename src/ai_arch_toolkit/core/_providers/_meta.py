"""Meta provider — Muse Spark over the Meta Model API's Responses surface.

Meta ships no SDK of its own: its documentation points OpenAI-format clients at
``https://api.meta.ai/v1``. This adapter drives the official ``openai`` SDK's Responses API, the
surface Meta recommends for agents because it is the only OpenAI-compatible one that carries the
model's reasoning across tool turns (Chat Completions redacts it).

Requests are stateless (``store: false``). Each response carries its reasoning as encrypted items;
``Response.to_message()`` keeps the SDK response under ``_raw`` and the next request replays those
items, so a tool loop keeps its chain of thought without server-side conversation state.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import AsyncIterator
from typing import Any, cast

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
    BaseProvider,
    Done,
    LoopAwareClientCache,
    Options,
    Prepared,
    _parse_retry_after,
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
    StreamEvent,
    ThinkingBlock,
    ToolCall,
    Usage,
    _uncached_input_tokens,
)

with require_sdk("meta"):
    import httpx2  # the openai SDK's transport
    import openai
    from openai.resources.responses import AsyncResponses
    from openai.types.responses import (
        FunctionToolParam,
        ResponseFunctionToolCallParam,
        ResponseInputContentParam,
        ResponseInputImageParam,
        ResponseInputItemParam,
        ResponseInputMessageContentListParam,
        ResponseOutputItem,
        ResponseOutputMessageParam,
        ResponseTextConfigParam,
        ResponseUsage,
        ToolParam,
        WebSearchToolParam,
    )
    from openai.types.responses import Response as SDKResponse
    from openai.types.responses.input_token_count_params import InputTokenCountParams
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming,
        ResponseCreateParamsStreaming,
    )
    from openai.types.shared.reasoning_effort import ReasoningEffort
    from openai.types.shared_params import Reasoning

logger = logging.getLogger(__name__)

type Params = ResponseCreateParamsNonStreaming

DEFAULT_BASE_URL = "https://api.meta.ai/v1"

# Parameters forwarded to ``responses.create()`` as given (``max_tokens`` is renamed).
_FORWARDED = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "parallel_tool_calls",
        "prompt_cache_key",
        "safety_identifier",
    }
)

# Muse Spark's reasoning efforts (https://dev.meta.ai/docs/reasoning): "max" only on the standard
# tier of muse-spark-1.3, not on contributor-tier or older models; a newer model gets them all.
_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_PROFILES: dict[str, frozenset[str]] = dict.fromkeys(
    (
        "muse-spark-1.3-contributor",
        "muse-spark-1.2",
        "muse-spark-1.2-contributor",
        "muse-spark-1.1",
    ),
    _EFFORTS - {"max"},
)

# Meta's error codes and their HTTP statuses (https://dev.meta.ai/docs/error-handling). A failure
# inside a response or a stream carries only its code; "server_error" is the Responses API's code
# for the failure Meta answers with a 500 of type server_error. A 400 and a 500 both come with no
# code: an unknown or missing code gets no status.
_CODE_STATUS: dict[str, int] = {
    "invalid_api_key": 401,
    "billing_not_configured": 402,
    "model_not_found": 404,
    "file_not_found": 404,
    "payload_too_large": 413,
    "rate_limit_exceeded": 429,
    "server_error": 500,
    "server_shutting_down": 503,
    "service_overloaded": 503,
    "backend_unavailable": 503,
    "gateway_timeout": 504,
}

# Failures before the request reached the server; any other transport failure may be billed.
_NOT_SENT = (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.PoolTimeout)
_TIMEOUTS = (httpx2.TimeoutException, openai.APITimeoutError)

# Where the request leaves the openai SDK's types, and the live proof that Meta takes it (M01,
# blackboard/tasks/M01-meta-provider.md, probes of 2026-09-13). Each is built as a plain dict and
# cast once, where it is built:
# 1. an assistant turn rebuilt from its text and calls: the message item has no id or status, and
#    its text no annotations (ResponseOutputMessageParam and ResponseOutputTextParam require them);
#    the probes replayed turns with and without ids;
# 2. a function tool without strict (FunctionToolParam requires the key): every live tool loop;
# 3. an input image without detail (ResponseInputImageParam requires it): the image probe.


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def _media_url(source: str | bytes, media_type: str) -> str:
    """Return a URL or ``data:`` URL the API can read for an image or file."""
    if isinstance(source, str) and _is_url(source):
        return source
    return f"data:{media_type};base64,{_encode_b64(source)}"


def _part(part: Any) -> ResponseInputContentParam:
    if isinstance(part, CachePart):
        return {"type": "input_text", "text": part.content}  # caching is automatic
    if isinstance(part, ImagePart):
        image = {"type": "input_image", "image_url": _media_url(part.source, part.media_type)}
        return cast("ResponseInputImageParam", image)  # deviation 3: no detail
    if isinstance(part, DocumentPart):
        if isinstance(part.source, str) and _is_url(part.source):
            return {"type": "input_file", "file_url": part.source}
        data = _media_url(part.source, part.media_type)
        return {"type": "input_file", "filename": part.name or "document", "file_data": data}
    return {"type": "input_text", "text": part if isinstance(part, str) else str(part)}


def _content_to_sdk(content: Any) -> ResponseInputMessageContentListParam | str:
    """User content as Responses input parts, or a plain string."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    return [_part(part) for part in content]


def _text_of(content: Any) -> str:
    """The text of an assistant message's content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return system_content_text(content)


def _output_texts(output: list[ResponseOutputItem]) -> list[tuple[str, bool]]:
    """``(text, is_commentary)`` for each message item with text, in output order."""
    texts: list[tuple[str, bool]] = []
    for item in output:
        if item.type != "message":
            continue
        text = "".join(
            part.text if part.type == "output_text" else part.refusal for part in item.content
        )
        if text:
            texts.append((text, item.phase == "commentary"))
    return texts


def _joined_text(output: list[ResponseOutputItem]) -> str:
    return "\n\n".join(text for text, _ in _output_texts(output)).strip()


def _replayable_output(msg: dict[str, Any]) -> list[ResponseInputItemParam] | None:
    """The output items to replay for an assistant message, or ``None`` to rebuild it.

    Replays the SDK response kept under ``_raw`` when it still matches the message, so the
    encrypted reasoning travels with the turn. A message whose text or tool calls were changed
    after the response — or whose ``_raw`` came from another provider — is rebuilt from its
    fields instead, without reasoning that would no longer match.
    """
    raw = msg.get("_raw")
    if not isinstance(raw, SDKResponse):
        return None
    output = list(raw.output)
    if _joined_text(output) != _text_of(msg.get("content")).strip():
        return None
    raw_calls = [
        (item.call_id, item.name, parse_tool_args(item.arguments))
        for item in output
        if item.type == "function_call"
    ]
    msg_calls = [
        (tc.get("id", ""), tc.get("name", ""), dict(tc.get("input", {})))
        for tc in msg.get("tool_calls") or []
    ]
    if raw_calls != msg_calls:
        return None
    items = [
        cast("ResponseInputItemParam", item.model_dump(mode="json", exclude_none=True))
        for item in output
        # Without its encrypted content a reasoning item is only a reference, which a stateless
        # request cannot resolve (Meta answers 400 "reasoning item was not found").
        if item.type != "reasoning" or item.encrypted_content
    ]
    # A reasoning item must be followed by a message or function call before the next input.
    while items and items[-1].get("type") == "reasoning":
        items.pop()
    return items


def _function_call(call: dict[str, Any]) -> ResponseFunctionToolCallParam:
    return {
        "type": "function_call",
        "call_id": call.get("id", ""),
        "name": call.get("name", ""),
        "arguments": json.dumps(call.get("input", {})),
    }


def _assistant_items(msg: dict[str, Any]) -> list[ResponseInputItemParam]:
    """Responses input items for one assistant message."""
    replay = _replayable_output(msg)
    if replay is not None:
        return replay
    items: list[ResponseInputItemParam] = []
    text = _text_of(msg.get("content"))
    calls = msg.get("tool_calls") or []
    if text:
        message = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }
        if calls:
            message["phase"] = "commentary"  # text that leads into a tool call
        items.append(cast("ResponseOutputMessageParam", message))  # deviation 1: no id or status
    items.extend(_function_call(call) for call in calls)
    return items


def _input_items(messages: list[dict[str, Any]]) -> list[ResponseInputItemParam]:
    """Convert generic messages to Responses input items.

    ``tool_use_id`` marks a tool result. System messages keep their positions.
    """
    items: list[ResponseInputItemParam] = []
    for msg in messages:
        role = msg.get("role", "user")
        if msg.get("tool_use_id"):
            output = str(msg.get("content", ""))
            items.append(
                {"type": "function_call_output", "call_id": msg["tool_use_id"], "output": output}
            )
        elif role == "system":
            items.append(
                {"role": "system", "content": system_content_text(msg.get("content", ""))}
            )
        elif role == "assistant":
            items.extend(_assistant_items(msg))
        elif role == "user" or role == "developer":
            items.append({"role": role, "content": _content_to_sdk(msg.get("content", ""))})
        else:
            raise RequestError(f"the Responses API has no {role!r} role")
    return items


def _tool_to_sdk(tool: dict[str, Any]) -> FunctionToolParam:
    """A Responses function tool (flat, not nested)."""
    function = {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", tool.get("parameters", {})),
    }
    return cast("FunctionToolParam", function)  # deviation 2: no strict


def _hosted_tool(tool: dict[str, Any]) -> WebSearchToolParam:
    """The one server tool Meta runs; its config belongs to C05."""
    config = set(tool) - {"_server_tool", "type"}
    if tool["type"] != "web_search" or config:
        detail = f"config {sorted(config)}" if tool["type"] == "web_search" else "not run by Meta"
        raise RequestError(f"server tool {tool['type']!r}: {detail}")
    return {"type": "web_search"}


def _format(options: Options) -> ResponseTextConfigParam | None:
    if options.output_schema is not None:
        # Meta constrains decoding to the schema either way; strict only adds a server check
        # of OpenAI's strict subset, which a plain Pydantic schema fails.
        return {
            "format": {
                "type": "json_schema",
                "name": options.output_schema.name,
                "schema": options.output_schema.schema,
                "strict": False,
            }
        }
    if options.json_mode:
        return {"format": {"type": "json_object"}}
    return None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def _extract_usage(usage: ResponseUsage) -> Usage:
    """Convert Responses usage to our disjoint ``Usage`` (reasoning is inside output)."""
    cache_read = usage.input_tokens_details.cached_tokens or 0
    return Usage(
        input_tokens=_uncached_input_tokens(usage.input_tokens, cache_read),
        output_tokens=usage.output_tokens,
        cache_read_tokens=cache_read,
    )


def _stop_reason(response: SDKResponse) -> str:
    """``incomplete_details.reason`` when the response stopped early, else its status."""
    details = response.incomplete_details
    if details is not None and details.reason:
        return str(details.reason)
    refused = any(
        part.type == "refusal"
        for item in response.output
        if item.type == "message"
        for part in item.content
    )
    return "refusal" if refused else str(response.status or "")


def _thinking_blocks(output: list[ResponseOutputItem]) -> list[ThinkingBlock]:
    """One block per reasoning item that carries a summary (raw reasoning stays encrypted)."""
    blocks: list[ThinkingBlock] = []
    for item in output:
        if item.type == "reasoning":
            summary = "\n\n".join(s.text for s in item.summary if s.text)
            if summary:
                blocks.append(ThinkingBlock(text=summary))
    return blocks


def _citations(output: list[ResponseOutputItem]) -> list[Citation]:
    return [
        Citation(
            text=part.text[note.start_index : note.end_index],
            url=note.url or "",
            title=note.title or "",
            start_index=note.start_index,
            end_index=note.end_index,
        )
        for item in output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
        for note in part.annotations
        if note.type == "url_citation"
    ]


def _parse_sdk_response(
    response: SDKResponse,
    model: str,
    *,
    output_schema: OutputSchema | None = None,
) -> Response:
    """Convert an ``openai.types.responses.Response`` to our ``Response``."""
    output = list(response.output)
    texts = _output_texts(output)
    text = "\n\n".join(t for t, _ in texts).strip()

    parsed: Any = None
    if output_schema is not None:
        # Commentary that leads into a hosted tool is not the answer; parse the final message.
        finals = [t for t, commentary in texts if not commentary]
        candidate = (finals[-1] if finals else text).strip()
        if candidate:
            parsed = parse_structured(candidate, output_schema)

    return Response(
        text=text,
        tool_calls=tuple(
            ToolCall(id=item.call_id, name=item.name, input=parse_tool_args(item.arguments))
            for item in output
            if item.type == "function_call"
        ),
        thinking=tuple(_thinking_blocks(output)),
        parsed=parsed,
        stop_reason=_stop_reason(response),
        model=response.model or model,
        raw=response,
        response_id=response.id or "",
        citations=tuple(_citations(output)),
    )


def _reported(code: str | None, message: str, usage: Usage | None = None) -> ProviderError:
    """A failure Meta reported inside a response or a stream, typed by its code's status."""
    status = _CODE_STATUS.get(code) if code else None
    if status is None:
        return ResponseError(
            f"Meta reported a failure ({code or 'no code'}): {message}", usage=usage
        )
    if status == 429:
        return RateLimitError(status, message)
    return APIError(status, message, usage=usage)


def _failure(response: SDKResponse) -> ProviderError:
    """The error for a response whose status is ``failed``, with the usage it reported."""
    error = response.error
    usage = _extract_usage(response.usage) if response.usage is not None else None
    message = (error.message if error is not None else "") or "response failed"
    return _reported(error.code if error is not None else None, message, usage)


class _TextJoiner:
    """Separates the text of consecutive message items with a blank line, as ``complete`` joins
    them."""

    def __init__(self) -> None:
        self.item: str | None = None
        self.emitted = False

    def events(self, item_id: str, delta: str) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if item_id != self.item:
            if self.emitted:
                events.append(StreamEvent(kind="text", text="\n\n"))
            self.item = item_id
        if delta:
            self.emitted = True
            events.append(StreamEvent(kind="text", text=delta))
        return events


def _http() -> Any:
    """The SDK's HTTP client, with the hook that marks a request as handed to the transport."""
    return openai.DefaultAsyncHttpxClient(event_hooks={"request": [on_request]})


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MetaProvider(LoopAwareClientCache, BaseProvider[Prepared[Params], SDKResponse]):
    """Meta Model API provider (Muse Spark) via the ``openai`` SDK's Responses API."""

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
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url or DEFAULT_BASE_URL,
            "max_retries": 0,
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        def _new_client() -> openai.AsyncOpenAI:
            client = openai.AsyncOpenAI(**client_kwargs, http_client=_http())
            # The SDK reads OPENAI_ORG_ID / OPENAI_PROJECT_ID from the environment and sends them
            # as headers; they identify an OpenAI account and must not reach Meta.
            client.organization = None
            client.project = None
            return client

        self._install_client(_new_client)

    async def close(self) -> None:
        await self._client.close()

    def _responses(self) -> AsyncResponses:
        return self._client.responses

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    def prepare(self, request: Request) -> Prepared[Params]:
        options = parse_options(request.kwargs, _FORWARDED, "Meta")
        forwarded = dict(options.params)
        if "max_tokens" in forwarded:
            forwarded["max_output_tokens"] = forwarded.pop("max_tokens")
        params: Params = {
            "model": self._model,
            "input": _input_items(request.messages),
            # Stateless: no conversation is stored; reasoning comes back encrypted for replay.
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        params.update(cast("Params", forwarded))
        if request.system:
            params["instructions"] = request.system
        if reasoning := self._reasoning(options):
            params["reasoning"] = reasoning
        if tools := self._tools(request.tools, options):
            params["tools"] = tools
        if text := _format(options):
            params["text"] = text
        return Prepared(params, output_schema=options.output_schema)

    def _reasoning(self, options: Options) -> Reasoning:
        """Muse Spark always reasons (D13): the effort applies on its own, and thinking asks for
        a summary. Meta rejects logprobs from a reasoning model (https://dev.meta.ai/docs/reasoning).
        """
        if options.logprobs:
            raise RequestError(f"{self._model} reasons: Meta returns no logprobs")
        if options.thinking_budget:
            warnings.warn(
                "thinking_budget is not supported by Meta (Muse Spark takes thinking_effort), "
                "ignoring",
                stacklevel=5,
            )
        reasoning: Reasoning = {}
        effort = options.thinking_effort
        if effort is not None:
            found = lookup(self._model, _PROFILES)
            efforts = found.value if found is not None else _EFFORTS
            if effort not in efforts:
                raise RequestError(
                    f"{self._model} takes thinking_effort in {sorted(efforts)}, not {effort!r}"
                )
            reasoning["effort"] = cast("ReasoningEffort", effort)
        if options.thinking:
            reasoning["summary"] = "auto"
        return reasoning

    def _tools(self, tools: list[dict[str, Any]] | None, options: Options) -> list[ToolParam]:
        if options.tool_choice not in (None, "auto", "none"):
            raise RequestError(
                "Meta Model API only supports tool_choice='auto' ('none' is sent as no tools); "
                f"got {options.tool_choice!r}"
            )
        if not tools or options.tool_choice == "none":
            return []
        wire: list[ToolParam] = [
            _tool_to_sdk(tool) for tool in tools if not tool.get("_server_tool")
        ]
        return wire + [_hosted_tool(tool) for tool in tools if tool.get("_server_tool")]

    async def send(self, prepared: Prepared[Params]) -> SDKResponse:
        response = await self._responses().create(**prepared.params)
        if response.status == "failed":
            raise _failure(response)
        return response

    async def open_stream(
        self, prepared: Prepared[Params]
    ) -> AsyncIterator[StreamEvent | Done[SDKResponse]]:
        """Text streams with a blank line between message items; reasoning summaries stream as
        ``partial`` thinking. The terminal event carries the whole response — the source of the
        reasoning replayed on the next turn."""
        joiner = _TextJoiner()
        final: SDKResponse | None = None
        streaming: ResponseCreateParamsStreaming = {**prepared.params, "stream": True}
        stream = await self._responses().create(**streaming)
        async with stream:
            async for event in stream:
                if (
                    event.type == "response.output_text.delta"
                    or event.type == "response.refusal.delta"
                ):
                    for text_event in joiner.events(event.item_id, event.delta):
                        yield text_event
                elif event.type == "response.reasoning_summary_text.delta":
                    yield StreamEvent(
                        kind="thinking", thinking=ThinkingBlock(text=event.delta), partial=True
                    )
                elif event.type == "response.completed" or event.type == "response.incomplete":
                    final = event.response
                elif event.type == "response.failed":
                    raise _failure(event.response)
                elif event.type == "error":
                    raise _reported(event.code, event.message or "stream error")
        if final is not None:
            yield Done(final)

    def assemble(self, final: SDKResponse, prepared: Prepared[Params]) -> Response:
        return _parse_sdk_response(final, self._model, output_schema=prepared.output_schema)

    def usage(self, final: SDKResponse) -> Usage | None:
        return _extract_usage(final.usage) if final.usage is not None else None

    def map_error(self, exc: Exception, *, sent: bool) -> ProviderError | None:
        """The one place that knows the SDK's errors.

        Meta: "Failed requests may still incur charges depending on where processing occurred"
        (https://dev.meta.ai/docs/error-handling), so every failure after dispatch is
        indeterminate but a 429 (D20). An error payload inside a stream is typed by its code.
        """
        if isinstance(exc, openai.RateLimitError):
            retry_after = _parse_retry_after(exc.response.headers.get("retry-after"))
            return RateLimitError(exc.response.status_code, str(exc.body), retry_after=retry_after)
        if isinstance(exc, openai.APIStatusError):
            return APIError(exc.response.status_code, str(exc.body))
        if isinstance(exc, openai.APIConnectionError):
            return transport_error(exc.__cause__ or exc, not_sent=_NOT_SENT, timeouts=_TIMEOUTS)
        if isinstance(exc, httpx2.TransportError):  # raised while reading a stream
            return transport_error(exc, not_sent=_NOT_SENT, timeouts=_TIMEOUTS)
        if isinstance(exc, openai.APIError):
            body = exc.body if isinstance(exc.body, dict) else {}
            return _reported(body.get("code"), str(body.get("message") or exc.message))
        return refused_or_unread(exc, sent=sent)

    # ------------------------------------------------------------------
    # count_tokens
    # ------------------------------------------------------------------

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Count input tokens with Meta's ``POST /v1/responses/input_tokens``."""
        request: InputTokenCountParams = {"model": self._model, "input": _input_items(messages)}
        if system:
            request["instructions"] = system
        function_tools = [_tool_to_sdk(t) for t in tools or [] if not t.get("_server_tool")]
        if function_tools:
            request["tools"] = list[ToolParam](function_tools)
        with self._mapped():
            result = await self._responses().input_tokens.count(**request)
        return result.input_tokens or 0
