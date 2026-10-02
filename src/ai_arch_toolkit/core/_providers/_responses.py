"""The Responses API core: what the providers on the ``openai`` SDK's Responses surface share.

Meta drives its Model API through this surface (D11), and so does OpenAI's own host (D43). A
provider brings a :class:`ResponsesProfile`, the data that differs between them, and composes its
``prepare``, ``assemble`` and ``count_tokens`` from the functions here; :class:`ResponsesProvider`
does the I/O, reads the usage and types the errors.

Requests are stateless (``store: false``). Each response carries its reasoning as encrypted items;
``Response.to_message()`` keeps the SDK response under ``_raw`` and :func:`input_items` replays
those items on the next request, so a tool loop keeps its chain of thought without server-side
conversation state. They go back only to the provider and the model family that produced them:
"Persisted reasoning can be reused only within the same model family"
(https://developers.openai.com/api/docs/guides/reasoning).
"""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from ai_arch_toolkit.core._content import CachePart, DocumentPart, ImagePart, _encode_b64, _is_url
from ai_arch_toolkit.core._exceptions import (
    APIError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)
from ai_arch_toolkit.core._model_id import family, snapshot_base
from ai_arch_toolkit.core._providers._base import (
    BaseProvider,
    Done,
    LoopAwareClientCache,
    Options,
    Prepared,
    _parse_retry_after,
    on_request,
    parse_structured,
    parse_tool_args,
    refused_or_unread,
    strict_schema,
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

with require_sdk("openai"):  # the meta extra installs the same package
    import httpx2  # the openai SDK's transport
    import openai
    from openai.resources.responses import AsyncResponses
    from openai.types.responses import (
        FunctionToolParam,
        ResponseFunctionToolCallParam,
        ResponseIncludable,
        ResponseInputContentParam,
        ResponseInputImageParam,
        ResponseInputItemParam,
        ResponseInputMessageContentListParam,
        ResponseOutputItem,
        ResponseOutputMessageParam,
        ResponseTextConfigParam,
        ResponseUsage,
        ToolParam,
    )
    from openai.types.responses import Response as SDKResponse
    from openai.types.responses.input_token_count_params import InputTokenCountParams
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming,
        ResponseCreateParamsStreaming,
        ToolChoice,
    )
    from openai.types.responses.response_output_text import Logprob

type Params = ResponseCreateParamsNonStreaming

# Failures before the request reached the server; any other transport failure may be billed.
_NOT_SENT = (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.PoolTimeout)
_TIMEOUTS = (httpx2.TimeoutException, openai.APITimeoutError)

# Where the request leaves the openai SDK's types, and the proof that each provider takes it:
# Meta's live probes (M01, blackboard/tasks/M01-meta-provider.md, 2026-09-13), OpenAI's documents.
# Each is built as a plain dict and cast once, where it is built:
# 1. an assistant turn rebuilt from its text and calls: the message item has no id or status, and
#    its text no annotations (ResponseOutputMessageParam and ResponseOutputTextParam require them).
#    Meta: the probes replayed turns with and without ids. OpenAI: not yet proven live (O01
#    replayed its turns whole; the rebuilt shape is on O04's live check);
# 2. a function tool without strict, for a profile whose function_strict is None
#    (FunctionToolParam requires the key). Meta only: every live tool loop;
# 3. an input image without detail (ResponseInputImageParam requires it). Meta: the image probe.
#    OpenAI: "If you omit the parameter, it defaults to auto in both the Responses API and the
#    Chat Completions API" (https://developers.openai.com/api/docs/guides/images-vision).


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponsesProfile:
    """What differs between the providers that speak the Responses API.

    The rules of each model (its efforts, whether it reasons, its sampling) are not here: they
    come with code, and each adapter applies them in its own ``prepare``.

    Attributes:
        provider: The provider's name, in errors.
        families: Model families by id prefix, resolved by ``_model_id.family``. A turn's
            reasoning is replayed only within the family of the request (see :meth:`family`).
        include: What every request asks back besides the output.
        takes_tool_choice: The provider takes ``tool_choice`` (``auto``, ``none``, ``required``
            or a function's name). Without it only the model chooses: ``none`` is sent as no
            tools, and a forced choice is refused.
        function_strict: The ``strict`` flag every function tool carries; ``None`` leaves the
            key out (deviation 2).
        output_strict: The provider honours ``OutputSchema.strict``: a strict schema is sent
            strict, in OpenAI's strict subset (``strict_schema``). Without it every schema is sent
            non-strict.
        hosted_tools: The server tools the provider runs, by the toolkit's server-tool type, each
            with the Responses tool it is sent as. A server tool's config is refused (C05).
        codes: The HTTP status of each error code a failure inside a response or a stream
            carries; a code not listed gets none.
    """

    provider: str
    families: Mapping[str, str]
    include: tuple[ResponseIncludable, ...]
    takes_tool_choice: bool
    function_strict: bool | None
    output_strict: bool
    hosted_tools: Mapping[str, ToolParam]
    codes: Mapping[str, int]

    def family(self, model: str) -> str:
        """The family whose reasoning ``model`` can reuse: its family in this profile, else its
        own id without a snapshot suffix. Another provider's ids never fall into one of this
        profile's families, so its responses are never replayed here."""
        return family(model, self.families) or snapshot_base(model) or model


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


def _replayable_output(
    msg: dict[str, Any], profile: ResponsesProfile, family: str
) -> list[ResponseInputItemParam] | None:
    """The output items to replay for an assistant message, or ``None`` to rebuild it.

    Replays the SDK response kept under ``_raw`` when it came from the request's provider and
    model family and still matches the message, so the encrypted reasoning travels with the turn.
    A message whose ``_raw`` came from another provider or family, or whose text or tool calls
    were changed after the response, is rebuilt from its fields instead, without reasoning the
    model could not use.
    """
    raw = msg.get("_raw")
    if not isinstance(raw, SDKResponse) or profile.family(raw.model) != family:
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
        # The wire names: the SDK's async_ field is the request's "async".
        cast(
            "ResponseInputItemParam",
            item.model_dump(mode="json", exclude_none=True, by_alias=True),
        )
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


def _assistant_items(
    msg: dict[str, Any], profile: ResponsesProfile, family: str
) -> list[ResponseInputItemParam]:
    """Responses input items for one assistant message."""
    replay = _replayable_output(msg, profile, family)
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


def input_items(
    messages: list[dict[str, Any]], profile: ResponsesProfile, family: str
) -> list[ResponseInputItemParam]:
    """Convert generic messages to Responses input items for a request of model ``family``.

    ``tool_use_id`` marks a tool result. System messages keep their positions. An assistant
    turn's reasoning is replayed only when it came from ``family``
    (:meth:`ResponsesProfile.family`).
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
            items.extend(_assistant_items(msg, profile, family))
        elif role == "user" or role == "developer":
            items.append({"role": role, "content": _content_to_sdk(msg.get("content", ""))})
        else:
            raise RequestError(f"the Responses API has no {role!r} role")
    return items


def _function_tool(tool: dict[str, Any], profile: ResponsesProfile) -> FunctionToolParam:
    """A Responses function tool (flat, not nested)."""
    function: dict[str, Any] = {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", tool.get("parameters", {})),
    }
    if profile.function_strict is not None:
        function["strict"] = profile.function_strict
    return cast("FunctionToolParam", function)  # deviation 2 when strict is left out


def _hosted_tool(tool: dict[str, Any], profile: ResponsesProfile) -> ToolParam:
    """The Responses tool for a server tool the provider runs; its config belongs to C05."""
    kind = tool["type"]
    hosted = profile.hosted_tools.get(kind)
    if hosted is None:
        raise RequestError(f"server tool {kind!r}: not run by {profile.provider}")
    if config := set(tool) - {"_server_tool", "type"}:
        raise RequestError(f"server tool {kind!r}: config {sorted(config)}")
    return copy.deepcopy(hosted)


def _sends_tools(choice: str | None, profile: ResponsesProfile) -> bool:
    """Whether the request carries its tools, refusing a choice the provider does not take."""
    if profile.takes_tool_choice:
        return True
    if choice not in (None, "auto", "none"):
        raise RequestError(
            f"{profile.provider} only supports tool_choice='auto' ('none' is sent as no "
            f"tools); got {choice!r}"
        )
    return choice != "none"


def _tool_choice(choice: str) -> ToolChoice:
    if choice == "auto" or choice == "none" or choice == "required":
        return choice
    return {"type": "function", "name": choice}


def _tools(
    params: Params, tools: list[dict[str, Any]] | None, options: Options, profile: ResponsesProfile
) -> None:
    """The function tools, then the hosted ones, and the ``tool_choice`` where it is taken."""
    if not _sends_tools(options.tool_choice, profile) or not tools:
        return
    params["tools"] = [
        *(_function_tool(tool, profile) for tool in tools if not tool.get("_server_tool")),
        *(_hosted_tool(tool, profile) for tool in tools if tool.get("_server_tool")),
    ]
    if options.tool_choice is not None and profile.takes_tool_choice:
        params["tool_choice"] = _tool_choice(options.tool_choice)


def _text_format(options: Options, profile: ResponsesProfile) -> ResponseTextConfigParam | None:
    """The structured output's ``text.format``
    (https://developers.openai.com/api/docs/guides/structured-outputs).

    Strict adds a server check of OpenAI's strict subset, which a plain Pydantic schema fails
    unless it is normalized to it; a profile that does not honour ``OutputSchema.strict`` sends
    every schema non-strict (Meta constrains decoding to the schema either way, D13).
    """
    schema = options.output_schema
    if schema is not None:
        strict = profile.output_strict and schema.strict
        return {
            "format": {
                "type": "json_schema",
                "name": schema.name,
                "schema": strict_schema(schema.schema) if strict else schema.schema,
                "strict": strict,
            }
        }
    if options.json_mode:
        return {"format": {"type": "json_object"}}
    return None


def request_params(
    profile: ResponsesProfile,
    model: str,
    items: list[ResponseInputItemParam],
    *,
    system: str | None,
    tools: list[dict[str, Any]] | None,
    options: Options,
) -> Params:
    """The request every Responses provider sends for ``items``.

    Stateless, with the caller's SDK parameters, the instructions, the tools and the text format.
    The output limit is ``max_output_tokens`` here: ``max_tokens``, or Chat Completions'
    ``max_completion_tokens``, which wins when both are given. The model's own rules (reasoning,
    sampling) are the adapter's to add.
    """
    forwarded = dict(options.params)
    limit = forwarded.pop("max_completion_tokens", None) or forwarded.pop("max_tokens", None)
    forwarded.pop("max_tokens", None)
    if limit is not None:
        forwarded["max_output_tokens"] = limit
    # Stateless: no conversation is stored; the reasoning comes back encrypted for replay.
    params: Params = {"model": model, "input": items, "store": False}
    if profile.include:
        params["include"] = list(profile.include)
    params.update(cast("Params", forwarded))
    if system:
        params["instructions"] = system
    _tools(params, tools, options, profile)
    if text := _text_format(options, profile):
        params["text"] = text
    return params


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


def _logprobs(output: list[ResponseOutputItem]) -> tuple[Logprob, ...] | None:
    """The tokens' log probabilities of the output text, in order, when the request asked for
    them (``include: ["message.output_text.logprobs"]``); ``None`` otherwise."""
    found = tuple(
        logprob
        for item in output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
        for logprob in part.logprobs or ()
    )
    return found or None


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
        logprobs=_logprobs(output),
        citations=tuple(_citations(output)),
    )


def _reported(
    profile: ResponsesProfile, code: str | None, message: str, usage: Usage | None = None
) -> ProviderError:
    """A failure the provider reported inside a response or a stream, typed by its code."""
    status = profile.codes.get(code) if code else None
    if status is None:
        return ResponseError(
            f"{profile.provider} reported a failure ({code or 'no code'}): {message}", usage=usage
        )
    if status == 429:
        return RateLimitError(status, message)
    return APIError(status, message, usage=usage)


def _failure(profile: ResponsesProfile, response: SDKResponse) -> ProviderError:
    """The error for a response whose status is ``failed``, with the usage it reported."""
    error = response.error
    usage = _extract_usage(response.usage) if response.usage is not None else None
    message = (error.message if error is not None else "") or "response failed"
    return _reported(profile, error.code if error is not None else None, message, usage)


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


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def http_client() -> httpx2.AsyncClient:
    """The SDK's HTTP client, with the hook that marks a request as handed to the transport."""
    return openai.DefaultAsyncHttpxClient(event_hooks={"request": [on_request]})


class ResponsesProvider(LoopAwareClientCache, BaseProvider[Prepared[Params], SDKResponse]):
    """A provider on the Responses API: the I/O, the usage and the errors, by its profile.

    A subclass builds the SDK client and composes ``prepare``, ``assemble`` and ``count_tokens``
    from :func:`input_items`, :func:`request_params`, :func:`_parse_sdk_response` and
    :meth:`_count_input_tokens`; the request's model family and its model's rules are its own.
    """

    def __init__(
        self,
        model: str,
        profile: ResponsesProfile,
        client: Callable[[], openai.AsyncOpenAI],
    ) -> None:
        self._model = model
        self._profile = profile
        self._install_client(client)

    async def close(self) -> None:
        await self._client.close()

    def _responses(self) -> AsyncResponses:
        return self._client.responses

    async def send(self, prepared: Prepared[Params]) -> SDKResponse:
        response = await self._responses().create(**prepared.params)
        if response.status == "failed":
            raise _failure(self._profile, response)
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
                    raise _failure(self._profile, event.response)
                elif event.type == "error":
                    raise _reported(self._profile, event.code, event.message or "stream error")
        if final is not None:
            yield Done(final)

    def usage(self, final: SDKResponse) -> Usage | None:
        return _extract_usage(final.usage) if final.usage is not None else None

    def map_error(self, exc: Exception, *, sent: bool) -> ProviderError | None:
        """The one place that knows the SDK's errors.

        Every failure after dispatch is indeterminate but a 429 (D20). Meta: "Failed requests may
        still incur charges depending on where processing occurred"
        (https://dev.meta.ai/docs/error-handling). OpenAI documents no billing for error
        responses (https://developers.openai.com/api/docs/guides/error-codes), only that a 429 is
        not charged (https://platform.openai.com/docs/guides/flex-processing). An error payload
        inside a stream is typed by its code, through the profile.
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
            code, message = body.get("code"), str(body.get("message") or exc.message)
            return _reported(self._profile, code, message)
        return refused_or_unread(exc, sent=sent)

    async def _count_input_tokens(
        self,
        items: list[ResponseInputItemParam],
        system: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> int:
        """Count the input tokens of ``items`` with ``POST /v1/responses/input_tokens``; only the
        function tools are counted."""
        request: InputTokenCountParams = {"model": self._model, "input": items}
        if system:
            request["instructions"] = system
        function_tools = [
            _function_tool(tool, self._profile)
            for tool in tools or []
            if not tool.get("_server_tool")
        ]
        if function_tools:
            request["tools"] = list[ToolParam](function_tools)
        with self._mapped():
            result = await self._responses().input_tokens.count(**request)
        return result.input_tokens or 0
