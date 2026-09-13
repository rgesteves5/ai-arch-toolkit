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
from types import SimpleNamespace
from typing import Any

from ai_arch_toolkit.core._content import CachePart, DocumentPart, ImagePart, _encode_b64, _is_url
from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._providers._base import (
    BaseProvider,
    LoopAwareClientCache,
    StreamState,
    _parse_retry_after,
    parse_tool_args,
    system_content_text,
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

require_sdk("openai", "meta")
import openai  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.meta.ai/v1"

# Parameters safe to forward to ``responses.create()`` (``max_tokens`` is renamed).
_SDK_PARAMS = {
    "temperature",
    "top_p",
    "max_tokens",
    "parallel_tool_calls",
    "prompt_cache_key",
    "safety_identifier",
}

# HTTP statuses for error codes reported inside a response or stream, which carry no status line.
# Transient codes get retryable statuses; an unlisted code is a fault of the request (not retried).
_STREAM_ERROR_STATUS: dict[str | None, int] = {
    None: 500,
    "server_error": 500,
    "server_shutting_down": 503,
    "service_overloaded": 503,
    "backend_unavailable": 503,
    "gateway_timeout": 504,
    "rate_limit_exceeded": 429,
    "invalid_api_key": 401,
    "model_not_found": 404,
    "file_not_found": 404,
    "payload_too_large": 413,
}


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _media_url(source: str | bytes, media_type: str) -> str:
    """Return a URL or ``data:`` URL the API can read for an image or file."""
    if isinstance(source, str) and _is_url(source):
        return source
    return f"data:{media_type};base64,{_encode_b64(source)}"


def _content_to_sdk(content: Any) -> list[dict[str, Any]] | str:
    """Convert user content to Responses input parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "input_text", "text": part})
        elif isinstance(part, CachePart):
            parts.append({"type": "input_text", "text": part.content})  # caching is automatic
        elif isinstance(part, ImagePart):
            parts.append(
                {"type": "input_image", "image_url": _media_url(part.source, part.media_type)}
            )
        elif isinstance(part, DocumentPart):
            source = part.source
            if isinstance(source, str) and source.startswith(("https://", "http://")):
                parts.append({"type": "input_file", "file_url": source})
            else:
                parts.append(
                    {
                        "type": "input_file",
                        "filename": part.name or "document",
                        "file_data": _media_url(source, part.media_type),
                    }
                )
        else:
            parts.append({"type": "input_text", "text": str(part)})
    return parts


def _text_of(content: Any) -> str:
    """The text of an assistant message's content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return system_content_text(content)


def _output_texts(output: list[Any]) -> list[tuple[str, bool]]:
    """``(text, is_commentary)`` for each message item with text, in output order."""
    texts: list[tuple[str, bool]] = []
    for item in output:
        if getattr(item, "type", "") != "message":
            continue
        text = ""
        for part in getattr(item, "content", None) or []:
            kind = getattr(part, "type", "")
            if kind == "output_text":
                text += getattr(part, "text", "") or ""
            elif kind == "refusal":
                text += getattr(part, "refusal", "") or ""
        if text:
            texts.append((text, getattr(item, "phase", None) == "commentary"))
    return texts


def _joined_text(output: list[Any]) -> str:
    return "\n\n".join(text for text, _ in _output_texts(output)).strip()


def _replayable_output(msg: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The output items to replay for an assistant message, or ``None`` to rebuild it.

    Replays the SDK response kept under ``_raw`` when it still matches the message, so the
    encrypted reasoning travels with the turn. A message whose text or tool calls were changed
    after the response — or whose ``_raw`` came from another provider — is rebuilt from its
    fields instead, without reasoning that would no longer match.
    """
    raw = msg.get("_raw")
    output = getattr(raw, "output", None)
    if getattr(raw, "object", None) != "response" or not isinstance(output, list):
        return None
    if _joined_text(output) != _text_of(msg.get("content")).strip():
        return None
    raw_calls = [
        (item.call_id, item.name, parse_tool_args(item.arguments))
        for item in output
        if getattr(item, "type", "") == "function_call"
    ]
    msg_calls = [
        (tc.get("id", ""), tc.get("name", ""), dict(tc.get("input", {})))
        for tc in msg.get("tool_calls") or []
    ]
    if raw_calls != msg_calls:
        return None
    items = [
        item.model_dump(mode="json", exclude_none=True)
        for item in output
        # Without its encrypted content a reasoning item is only a reference, which a stateless
        # request cannot resolve (Meta answers 400 "reasoning item was not found").
        if getattr(item, "type", "") != "reasoning" or getattr(item, "encrypted_content", None)
    ]
    # A reasoning item must be followed by a message or function call before the next input.
    while items and items[-1].get("type") == "reasoning":
        items.pop()
    return items


def _assistant_items(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Responses input items for one assistant message."""
    replay = _replayable_output(msg)
    if replay is not None:
        return replay
    items: list[dict[str, Any]] = []
    text = _text_of(msg.get("content"))
    tool_calls = msg.get("tool_calls") or []
    if text:
        message: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }
        if tool_calls:
            message["phase"] = "commentary"  # text that leads into a tool call
        items.append(message)
    for tc in tool_calls:
        items.append(
            {
                "type": "function_call",
                "call_id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "arguments": json.dumps(tc.get("input", {})),
            }
        )
    return items


def _input_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert generic messages to Responses input items.

    ``tool_use_id`` marks a tool result. System messages keep their positions.
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        if msg.get("tool_use_id"):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg["tool_use_id"],
                    "output": str(msg.get("content", "")),
                }
            )
        elif role == "system":
            text = system_content_text(msg.get("content", ""))
            items.append({"role": "system", "content": text})
        elif role == "assistant":
            items.extend(_assistant_items(msg))
        else:
            items.append({"role": role, "content": _content_to_sdk(msg.get("content", ""))})
    return items


def _tool_to_sdk(tool: dict[str, Any]) -> dict[str, Any]:
    """Map a generic tool dict to a Responses function tool (flat, not nested)."""
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", tool.get("parameters", {})),
    }


def _hosted_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hosted: list[dict[str, Any]] = []
    for tool in tools:
        if tool["type"] == "web_search":
            hosted.append({"type": "web_search"})
        else:
            warnings.warn(
                f"Server tool {tool['type']!r} is not supported by Meta Model API, ignoring",
                stacklevel=5,
            )
    return hosted


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _extract_usage(sdk_usage: Any) -> Usage:
    """Convert Responses usage to our disjoint ``Usage`` (reasoning is inside output)."""
    if sdk_usage is None:
        return Usage()
    total_input = getattr(sdk_usage, "input_tokens", 0) or 0
    details = getattr(sdk_usage, "input_tokens_details", None)
    cache_read = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return Usage(
        input_tokens=_uncached_input_tokens(total_input, cache_read),
        output_tokens=getattr(sdk_usage, "output_tokens", 0) or 0,
        cache_read_tokens=cache_read,
    )


def _stop_reason(response: Any) -> str:
    """``incomplete_details.reason`` when the response stopped early, else its status."""
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None) if details is not None else None
    if reason:
        return str(reason)
    parts = (
        part
        for item in getattr(response, "output", None) or []
        if getattr(item, "type", "") == "message"
        for part in getattr(item, "content", None) or []
    )
    if any(getattr(part, "type", "") == "refusal" for part in parts):
        return "refusal"
    return str(getattr(response, "status", "") or "")


def _thinking_blocks(output: list[Any]) -> list[ThinkingBlock]:
    """One block per reasoning item that carries a summary (raw reasoning stays encrypted)."""
    blocks: list[ThinkingBlock] = []
    for item in output:
        if getattr(item, "type", "") != "reasoning":
            continue
        texts = [getattr(s, "text", "") for s in getattr(item, "summary", None) or []]
        summary = "\n\n".join(text for text in texts if text)
        if summary:
            blocks.append(ThinkingBlock(text=summary))
    return blocks


def _tool_call(item: Any) -> ToolCall:
    return ToolCall(id=item.call_id, name=item.name, input=parse_tool_args(item.arguments))


def _citations(output: list[Any]) -> list[Citation]:
    citations: list[Citation] = []
    for item in output:
        if getattr(item, "type", "") != "message":
            continue
        for part in getattr(item, "content", None) or []:
            text = getattr(part, "text", "") or ""
            for note in getattr(part, "annotations", None) or []:
                if getattr(note, "type", "") != "url_citation":
                    continue
                start = getattr(note, "start_index", None)
                end = getattr(note, "end_index", None)
                citations.append(
                    Citation(
                        text=text[start:end] if start is not None and end is not None else "",
                        url=getattr(note, "url", "") or "",
                        title=getattr(note, "title", "") or "",
                        start_index=start,
                        end_index=end,
                    )
                )
    return citations


def _parse_structured(text: str, output_schema: OutputSchema) -> Any:
    """Parse JSON output, validated into the schema's Pydantic model when it has one."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse structured output as JSON")
        return None
    if output_schema.model_class is None:
        return data
    try:
        return output_schema.model_class.model_validate(data)  # type: ignore[attr-defined]
    except Exception:
        logger.warning("Failed to validate structured output against schema")
        return data


def _parse_sdk_response(
    response: Any,
    model: str,
    *,
    output_schema: OutputSchema | None = None,
) -> Response:
    """Convert an ``openai.types.responses.Response`` to our ``Response``."""
    output = list(getattr(response, "output", None) or [])
    texts = _output_texts(output)
    text = "\n\n".join(t for t, _ in texts).strip()

    parsed: Any = None
    if output_schema is not None:
        # Commentary that leads into a hosted tool is not the answer; parse the final message.
        finals = [t for t, commentary in texts if not commentary]
        candidate = (finals[-1] if finals else text).strip()
        if candidate:
            parsed = _parse_structured(candidate, output_schema)

    usage = _extract_usage(getattr(response, "usage", None))
    return Response(
        text=text,
        tool_calls=tuple(
            _tool_call(item) for item in output if getattr(item, "type", "") == "function_call"
        ),
        thinking=tuple(_thinking_blocks(output)),
        parsed=parsed,
        usage=usage,
        cost=_estimate_response_cost(model, usage),
        stop_reason=_stop_reason(response),
        model=getattr(response, "model", "") or model,
        raw=response,
        response_id=getattr(response, "id", "") or "",
        citations=tuple(_citations(output)),
    )


def _api_error(exc: openai.APIStatusError) -> APIError:
    """Map an SDK status error to the toolkit's exception types."""
    response = exc.response
    status = response.status_code if response is not None else 500
    if isinstance(exc, openai.RateLimitError):
        retry_after = response.headers.get("retry-after") if response is not None else None
        return RateLimitError(status, str(exc.body), retry_after=_parse_retry_after(retry_after))
    return APIError(status, str(exc.body))


def _stream_error(code: str | None, message: str) -> APIError:
    """Map an error reported inside a response or stream (no HTTP status) to an exception."""
    status = _STREAM_ERROR_STATUS.get(code, 400)
    if status == 429:
        return RateLimitError(status, message)
    return APIError(status, message)


def _failure(response: Any) -> APIError:
    """The exception for a response whose status is ``failed``."""
    error = getattr(response, "error", None)
    if isinstance(error, dict):
        error = SimpleNamespace(**error)
    return _stream_error(
        getattr(error, "code", None), getattr(error, "message", "") or "response failed"
    )


def _sdk_error(exc: openai.APIError) -> APIError:
    """Map an SDK error raised for an error payload inside a stream (no HTTP status)."""
    body = exc.body if isinstance(exc.body, dict) else {}
    return _stream_error(body.get("code"), str(body.get("message") or exc.message))


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MetaProvider(LoopAwareClientCache, BaseProvider):
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
            import httpx

            client_kwargs["timeout"] = httpx.Timeout(timeout)

        def _new_client() -> openai.AsyncOpenAI:
            client = openai.AsyncOpenAI(**client_kwargs)
            # The SDK reads OPENAI_ORG_ID / OPENAI_PROJECT_ID from the environment and sends them
            # as headers; they identify an OpenAI account and must not reach Meta.
            client.organization = None
            client.project = None
            return client

        self._install_client(_new_client)

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build kwargs for ``responses.create()``."""
        thinking = kwargs.pop("thinking", False)
        thinking_effort = kwargs.pop("thinking_effort", None)
        thinking_budget = kwargs.pop("thinking_budget", None)
        output_schema: OutputSchema | None = kwargs.pop("output_schema", None)
        tool_choice: str | None = kwargs.pop("tool_choice", None)
        json_mode: bool = kwargs.pop("json_mode", False)
        kwargs.pop("logprobs", None)  # Muse Spark is a reasoning model: logprobs are rejected
        kwargs.pop("structured_output_mode", None)  # Anthropic-specific; ignored here

        unknown = set(kwargs) - _SDK_PARAMS
        if unknown:
            warnings.warn(
                f"Unknown parameter(s) ignored for Meta: {sorted(unknown)}. "
                f"Valid: {sorted(_SDK_PARAMS)}",
                stacklevel=4,
            )
        filtered = {k: v for k, v in kwargs.items() if k in _SDK_PARAMS}
        if "max_tokens" in filtered:
            filtered["max_output_tokens"] = filtered.pop("max_tokens")

        request: dict[str, Any] = {
            "model": self._model,
            "input": _input_items(messages),
            # Stateless: no conversation is stored; reasoning comes back encrypted for replay.
            "store": False,
            "include": ["reasoning.encrypted_content"],
            **filtered,
        }
        if system:
            request["instructions"] = system

        # Muse Spark always reasons: the effort applies on its own, thinking asks for a summary.
        reasoning: dict[str, Any] = {}
        if thinking_effort:
            reasoning["effort"] = thinking_effort
        if thinking:
            reasoning["summary"] = "auto"
        if reasoning:
            request["reasoning"] = reasoning
        if thinking_budget:
            warnings.warn(
                "thinking_budget is not supported by Meta (Muse Spark takes thinking_effort), "
                "ignoring",
                stacklevel=4,
            )

        if tool_choice not in (None, "auto", "none"):
            raise ValueError(
                "Meta Model API only supports tool_choice='auto' ('none' is sent as no tools); "
                f"got {tool_choice!r}"
            )
        if tools and tool_choice != "none":
            wire_tools = [_tool_to_sdk(t) for t in tools if not t.get("_server_tool")]
            wire_tools += _hosted_tools([t for t in tools if t.get("_server_tool")])
            if wire_tools:
                request["tools"] = wire_tools

        if output_schema is not None:
            # Meta constrains decoding to the schema either way; strict only adds a server check
            # of OpenAI's strict subset, which a plain Pydantic schema fails.
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": output_schema.name,
                    "schema": output_schema.schema,
                    "strict": False,
                }
            }
        elif json_mode:
            request["text"] = {"format": {"type": "json_object"}}

        return request

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
        request = self._build_request(messages, system=system, tools=tools, **kwargs)

        logger.debug("complete start model=%s messages=%d", self._model, len(messages))
        try:
            response = await self._client.responses.create(**request)
        except openai.APIStatusError as exc:
            raise _api_error(exc) from exc
        if getattr(response, "status", None) == "failed":
            raise _failure(response)

        resp = _parse_sdk_response(response, self._model, output_schema=output_schema)
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

    def _stream_core(
        self,
        request: dict[str, Any],
        state: StreamState,
    ) -> AsyncIterator[StreamEvent]:
        """Single SDK event loop shared by ``stream()`` and ``stream_events()``.

        Text deltas stream as they arrive, with a blank line between message items (as
        ``complete()`` joins them). Reasoning summaries stream as ``partial`` thinking events and
        ``state.thinking`` stays current, so an abandoned stream still finalizes with what came.
        The terminal event replaces the partial state with the full response, which becomes
        ``state.raw`` — the source of the reasoning replayed on the next turn.
        """

        async def _generate() -> AsyncIterator[StreamEvent]:
            text_item: str | None = None
            emitted_text = False
            summaries: dict[str, list[str]] = {}  # reasoning item id -> summary parts
            emitted_calls: set[str] = set()

            def _tool_event(item: Any) -> StreamEvent | None:
                if item.call_id in emitted_calls:
                    return None
                emitted_calls.add(item.call_id)
                tool_call = _tool_call(item)
                state.tool_calls.append(tool_call)
                return StreamEvent(kind="tool_call", tool_call=tool_call)

            try:
                stream = await self._client.responses.create(**request, stream=True)
                async for event in stream:
                    kind = getattr(event, "type", "")
                    if kind in ("response.output_text.delta", "response.refusal.delta"):
                        if event.item_id != text_item:
                            if emitted_text:
                                yield StreamEvent(kind="text", text="\n\n")
                            text_item = event.item_id
                        if event.delta:
                            emitted_text = True
                            yield StreamEvent(kind="text", text=event.delta)
                    elif kind == "response.reasoning_summary_text.delta":
                        parts = summaries.setdefault(event.item_id, [])
                        while len(parts) <= event.summary_index:
                            parts.append("")
                        parts[event.summary_index] += event.delta
                        state.thinking = [
                            ThinkingBlock(text="\n\n".join(p for p in ps if p))
                            for ps in summaries.values()
                        ]
                        yield StreamEvent(
                            kind="thinking", thinking=ThinkingBlock(text=event.delta), partial=True
                        )
                    elif kind == "response.output_item.done":
                        item = event.item
                        if getattr(item, "type", "") == "function_call" and (
                            tool_event := _tool_event(item)
                        ):
                            yield tool_event
                    elif kind in ("response.completed", "response.incomplete"):
                        response = event.response
                        output = list(getattr(response, "output", None) or [])
                        calls = [i for i in output if getattr(i, "type", "") == "function_call"]
                        for item in calls:
                            if tool_event := _tool_event(item):
                                yield tool_event
                        # Calls finish in any order; the turn keeps the response's order, which
                        # the replayed function calls and their tool results must share.
                        ids = {item.call_id for item in calls}
                        state.tool_calls = [_tool_call(item) for item in calls] + [
                            tc for tc in state.tool_calls if tc.id not in ids
                        ]
                        state.thinking = _thinking_blocks(output) or state.thinking
                        state.usage = _extract_usage(getattr(response, "usage", None))
                        state.stop_reason = _stop_reason(response)
                        state.model = getattr(response, "model", "") or state.model
                        state.raw = response
                    elif kind == "response.failed":
                        raise _failure(event.response)
                    elif kind == "error":
                        raise _stream_error(
                            getattr(event, "code", None),
                            getattr(event, "message", "") or "stream error",
                        )
            except openai.APIStatusError as exc:
                raise _api_error(exc) from exc
            except openai.APIConnectionError:
                raise  # transport failures propagate as they do from the OpenAI adapter
            except openai.APIError as exc:
                raise _sdk_error(exc) from exc

        return _generate()

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[str], StreamState]:
        request = self._build_request(messages, system=system, tools=tools, **kwargs)

        logger.debug("stream start model=%s", self._model)
        state = StreamState()
        state.model = self._model
        events = self._stream_core(request, state)

        async def _text_only() -> AsyncIterator[str]:
            async for event in events:
                if event.kind == "text" and event.text:
                    yield event.text

        return _text_only(), state

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[StreamEvent], StreamState]:
        request = self._build_request(messages, system=system, tools=tools, **kwargs)

        logger.debug("stream_events start model=%s", self._model)
        state = StreamState()
        state.model = self._model
        return self._stream_core(request, state), state

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
        request: dict[str, Any] = {"model": self._model, "input": _input_items(messages)}
        if system:
            request["instructions"] = system
        function_tools = [_tool_to_sdk(t) for t in tools or [] if not t.get("_server_tool")]
        if function_tools:
            request["tools"] = function_tools
        try:
            result = await self._client.responses.input_tokens.count(**request)
        except openai.APIStatusError as exc:
            raise _api_error(exc) from exc
        return result.input_tokens or 0
