"""OpenAI-compatible servers — Chat Completions through the ``openai`` SDK, in the contract.

A ``base_url`` on another host than OpenAI's own (Ollama, LM Studio, vLLM, a gateway) is served
here, with no OpenAI model rule: the request carries what the caller set (D43). OpenAI's own host
speaks the Responses API (``_openai.py``). Servers that also implement the Responses API stay on
Chat Completions until someone asks (D44).

OpenAI's Batch API (the JSONL file and the ``batches`` endpoints) lives here too: both OpenAI
adapters submit through it, and the official one reads with :func:`chat_batch_response` the
batches it submitted to Chat Completions before it moved to the Responses API.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import Any, Literal, cast, get_args

from ai_arch_toolkit.core._batch import BatchResult
from ai_arch_toolkit.core._content import (
    CachePart,
    DocumentPart,
    ImagePart,
    _encode_b64,
    _is_url,
)
from ai_arch_toolkit.core._exceptions import (
    APIError,
    ProviderError,
    RateLimitError,
    RequestError,
    ResponseError,
)
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._providers._base import (
    BaseProvider,
    Done,
    LoopAwareClientCache,
    Options,
    Prepared,
    _parse_retry_after,
    named_calls,
    on_request,
    parse_options,
    parse_structured,
    parse_tool_args,
    refused_or_unread,
    strict_schema,
    system_content_text,
    transport_error,
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

with require_sdk("openai"):
    import httpx2  # the openai SDK's transport
    import openai
    from openai.lib.streaming.chat import ChatCompletionStreamState
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionAssistantMessageParam,
        ChatCompletionChunk,
        ChatCompletionContentPartParam,
        ChatCompletionFunctionToolParam,
        ChatCompletionMessageFunctionToolCallParam,
        ChatCompletionMessageParam,
        ChatCompletionToolChoiceOptionParam,
    )
    from openai.types.chat.completion_create_params import (
        CompletionCreateParamsNonStreaming,
        CompletionCreateParamsStreaming,
        ResponseFormat,
    )
    from openai.types.shared.reasoning_effort import ReasoningEffort
    from openai.types.shared_params import ResponseFormatJSONSchema

logger = logging.getLogger(__name__)

type Params = CompletionCreateParamsNonStreaming
type BatchEndpoint = Literal["/v1/chat/completions", "/v1/responses"]

CHAT_COMPLETIONS: BatchEndpoint = "/v1/chat/completions"

# SDK parameters forwarded as the caller gave them (the output limit is sent as max_tokens).
_FORWARDED = frozenset(
    {
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
        "top_logprobs",
    }
)
_EFFORTS = frozenset(value for value in get_args(get_args(ReasoningEffort)[0]))

# Failures before the request reached the server; any other transport failure may be billed.
_NOT_SENT = (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.PoolTimeout)
_TIMEOUTS = (httpx2.TimeoutException, openai.APITimeoutError)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def _content_to_sdk(content: Any) -> list[ChatCompletionContentPartParam] | str:
    """User content as Chat Completions content parts, or a plain string."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    return [_part(part) for part in content]


def _part(part: Any) -> ChatCompletionContentPartParam:
    if isinstance(part, ImagePart):
        url = part.source if _is_url(part.source) else None
        if not isinstance(url, str):
            url = f"data:{part.media_type};base64,{_encode_b64(part.source)}"
        return {"type": "image_url", "image_url": {"url": url}}
    if isinstance(part, DocumentPart):
        data = f"data:{part.media_type};base64,{_encode_b64(part.source)}"
        return {"type": "file", "file": {"filename": part.name or "document", "file_data": data}}
    if isinstance(part, CachePart):
        return {"type": "text", "text": part.content}  # caching is automatic here
    return {"type": "text", "text": part if isinstance(part, str) else str(part)}


def _tool_calls(msg: dict[str, Any]) -> list[ChatCompletionMessageFunctionToolCallParam]:
    return [
        {
            "id": call.get("id", ""),
            "type": "function",
            "function": {
                "name": call.get("name", ""),
                "arguments": json.dumps(call.get("input", {})),
            },
        }
        for call in msg.get("tool_calls") or []
    ]


def _assistant(msg: dict[str, Any]) -> ChatCompletionAssistantMessageParam:
    content = msg.get("content")
    message: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": content if isinstance(content, str) or content is None else str(content),
    }
    if calls := _tool_calls(msg):
        message["tool_calls"] = calls
    return message


def _message(msg: dict[str, Any]) -> ChatCompletionMessageParam:
    """One neutral message on the wire; ``tool_use_id`` marks a tool result."""
    role = msg.get("role", "user")
    if msg.get("tool_use_id"):
        content = str(msg.get("content", ""))
        return {"role": "tool", "tool_call_id": msg["tool_use_id"], "content": content}
    if role in ("system", "developer"):
        text = system_content_text(msg.get("content", ""))
        return (
            {"role": "system", "content": text}
            if role == "system"
            else {
                "role": "developer",
                "content": text,
            }
        )
    if role == "assistant":
        return _assistant(msg)
    if role == "user":
        return {"role": "user", "content": _content_to_sdk(msg.get("content", ""))}
    raise RequestError(f"Chat Completions has no {role!r} role")


def _messages_to_sdk(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
) -> list[ChatCompletionMessageParam]:
    """Convert generic messages to Chat Completions messages.

    A non-empty ``system`` is sent as a leading system message; system messages in the list keep
    their positions (mid-conversation system messages are valid for Chat Completions and
    OpenAI-compatible servers). An assistant turn carries all its calls; each result is a
    ``tool`` message with its call's id.
    """
    head: list[ChatCompletionMessageParam] = (
        [{"role": "system", "content": system}] if system else []
    )
    return [*head, *(_message(msg) for msg in messages)]


def _tool_to_sdk(tool: dict[str, Any]) -> ChatCompletionFunctionToolParam:
    """A function tool; Chat Completions takes no server tools."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", tool.get("parameters", {})),
        },
    }


def _tool_choice(choice: str) -> ChatCompletionToolChoiceOptionParam:
    if choice == "auto" or choice == "required" or choice == "none":
        return choice
    return {"type": "function", "function": {"name": choice}}


def _build_output_schema_format(output_schema: OutputSchema) -> ResponseFormatJSONSchema:
    """Build the ``response_format`` for structured output (a strict schema in the strict
    subset)."""
    schema = output_schema.schema
    return {
        "type": "json_schema",
        "json_schema": {
            "name": output_schema.name,
            "schema": strict_schema(schema) if output_schema.strict else schema,
            "strict": output_schema.strict,
        },
    }


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def _extract_usage(sdk_usage: Any) -> Usage:
    """Convert SDK usage object to our Usage dataclass."""
    cache_read = 0
    details = getattr(sdk_usage, "prompt_tokens_details", None)
    if details:
        cache_read = getattr(details, "cached_tokens", 0) or 0
    total_input = getattr(sdk_usage, "prompt_tokens", 0) or 0
    completion = getattr(sdk_usage, "completion_tokens", 0) or 0
    total = getattr(sdk_usage, "total_tokens", 0) or 0
    # OpenAI includes reasoning in completion_tokens. Some compatible APIs expose it as a
    # separate generated-token component while still including it in total_tokens; the max keeps
    # standard OpenAI usage unchanged and avoids undercounting those compatible responses.
    output = max(completion, total - total_input)
    return Usage(
        input_tokens=_uncached_input_tokens(total_input, cache_read),
        output_tokens=output,
        cache_read_tokens=cache_read,
    )


def _reasoning_text(obj: Any) -> str:
    """Extract vendor reasoning text from a streaming delta or a message.

    Reads ``reasoning_content`` (DeepSeek, vLLM, LM Studio, SGLang) then
    ``reasoning`` (Ollama /v1, OpenRouter), per-field so a non-string value in
    one does not mask a valid string in the other. SDK pydantic models use
    ``extra="allow"``, so unknown wire fields surface as attributes.
    """
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _parse_sdk_response(
    completion: Any,
    model: str,
    *,
    output_schema: OutputSchema | None = None,
) -> Response:
    """Convert a ``ChatCompletion`` to our ``Response`` (the base adds usage and cost)."""
    choices = completion.choices or []
    if not choices:
        return Response(raw=completion, model=model)

    choice = choices[0]
    message = choice.message
    text = message.content or ""

    thinking: tuple[ThinkingBlock, ...] = ()
    if reasoning := _reasoning_text(message):
        thinking = (ThinkingBlock(text=reasoning),)

    tool_calls = tuple(
        ToolCall(id=tc.id, name=tc.function.name, input=parse_tool_args(tc.function.arguments))
        for tc in message.tool_calls or []
    )
    return Response(
        text=text.strip(),
        tool_calls=tool_calls,
        thinking=thinking,
        parsed=parse_structured(text, output_schema) if output_schema and text else None,
        stop_reason=choice.finish_reason or "",
        model=completion.model or model,
        raw=completion,
        response_id=getattr(completion, "id", "") or "",
        logprobs=getattr(choice, "logprobs", None),
    )


def _chunk_events(chunk: ChatCompletionChunk) -> Iterator[StreamEvent]:
    """The text and reasoning deltas of one chunk (reasoning comes from compatible servers)."""
    for choice in chunk.choices[:1]:
        delta = choice.delta
        if fragment := _reasoning_text(delta):
            yield StreamEvent(kind="thinking", thinking=ThinkingBlock(text=fragment), partial=True)
        if delta.content:
            yield StreamEvent(kind="text", text=delta.content)


# ---------------------------------------------------------------------------
# OpenAI's Batch API, for both OpenAI adapters
# ---------------------------------------------------------------------------


def batch_request(req: Mapping[str, Any], model: str) -> Request:
    """A batch line's request: the one a call makes, with 4096 output tokens by default."""
    return Request(
        messages=req.get("messages", []),
        system=req.get("system"),
        tools=req.get("tools"),
        model=model,
        kwargs={"max_tokens": 4096, **req.get("kwargs", {})},
    )


async def submit_batch(
    client: openai.AsyncOpenAI,
    endpoint: BatchEndpoint,
    bodies: Sequence[tuple[str, Mapping[str, object]]],
) -> str:
    """Upload one JSONL line per ``(custom_id, body)`` and create a batch on ``endpoint``."""
    lines = [
        json.dumps({"custom_id": custom_id, "method": "POST", "url": endpoint, "body": body})
        for custom_id, body in bodies
    ]
    file = await client.files.create(file=io.BytesIO("\n".join(lines).encode()), purpose="batch")
    batch = await client.batches.create(
        input_file_id=file.id, endpoint=endpoint, completion_window="24h"
    )
    return batch.id


async def batch_status(client: openai.AsyncOpenAI, batch_id: str) -> str:
    """The batch's status as the API reports it."""
    batch = await client.batches.retrieve(batch_id)
    return batch.status


async def batch_output(
    client: openai.AsyncOpenAI, batch_id: str
) -> tuple[str, list[dict[str, Any]]]:
    """The batch's endpoint and the entries of its output file (none before it has one)."""
    batch = await client.batches.retrieve(batch_id)
    if not batch.output_file_id:
        return batch.endpoint, []
    content = await client.files.content(batch.output_file_id)
    return batch.endpoint, [json.loads(line) for line in content.text.splitlines() if line.strip()]


def batch_results(
    entries: list[dict[str, Any]], parse: Callable[[dict[str, Any]], Response]
) -> list[BatchResult]:
    """One result per output entry: its error, or its body read by ``parse``."""
    results: list[BatchResult] = []
    for entry in entries:
        custom_id = entry.get("custom_id", "")
        body = (entry.get("response") or {}).get("body")
        if error := entry.get("error"):
            results.append(BatchResult(custom_id=custom_id, error=str(error)))
        elif body:
            results.append(BatchResult(custom_id=custom_id, response=parse(body)))
        else:
            results.append(BatchResult(custom_id=custom_id, error="empty response"))
    return results


def chat_batch_response(body: dict[str, Any], model: str) -> Response:
    """A Chat Completions body from a batch's output, with its usage and cost."""
    completion = ChatCompletion.model_construct(**body)
    usage = _extract_usage(completion.usage) if completion.usage else Usage()
    response = _parse_sdk_response(completion, model)
    cost = _estimate_response_cost(model, usage, is_batch=True)
    return dataclasses.replace(
        response, usage=usage, cost=cost, tool_calls=named_calls(response.tool_calls)
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider(
    LoopAwareClientCache, BaseProvider[Prepared[Params], ChatCompletion]
):
    """An OpenAI-compatible server's Chat Completions API via the official SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str,
        timeout: float | None = None,
    ) -> None:
        self._model = model
        # Retry ownership belongs to LLM(RetryConfig(...)): hidden SDK retries
        # would be neither metered nor represented in Response.attempts.
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": 0,
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        def _new_client() -> openai.AsyncOpenAI:
            client = openai.AsyncOpenAI(**client_kwargs, http_client=_http())
            # The SDK reads OPENAI_ORG_ID / OPENAI_PROJECT_ID from the environment and sends them
            # as headers; they identify an OpenAI account and must not reach another host (D48).
            client.organization = None
            client.project = None
            return client

        self._install_client(_new_client)

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    def prepare(self, request: Request) -> Prepared[Params]:
        options = parse_options(request.kwargs, _FORWARDED, "OpenAI")
        params: Params = {
            "model": self._model,
            "messages": _messages_to_sdk(request.messages, system=request.system),
        }
        _forward(params, options)
        self._reasoning(params, options)
        _tools(params, request.tools, options)
        _format(params, options)
        return Prepared(params, output_schema=options.output_schema)

    def _reasoning(self, params: Params, options: Options) -> None:
        """``reasoning_effort`` with ``thinking`` (the effort given, else ``"high"``), any the
        SDK names: what another server's model takes is its own."""
        if options.thinking_budget:
            warnings.warn(
                "thinking_budget is not supported by OpenAI (only reasoning_effort string), "
                "ignoring",
                stacklevel=5,
            )
        if not options.thinking:
            return
        effort = options.thinking_effort or "high"
        if effort not in _EFFORTS:
            raise RequestError(
                f"{self._model} takes thinking_effort in {sorted(_EFFORTS)}, not {effort!r}"
            )
        params["reasoning_effort"] = cast("ReasoningEffort", effort)

    async def send(self, prepared: Prepared[Params]) -> ChatCompletion:
        return await self._client.chat.completions.create(**prepared.params)

    async def open_stream(
        self, prepared: Prepared[Params]
    ) -> AsyncIterator[StreamEvent | Done[ChatCompletion]]:
        # The SDK's accumulator builds the final completion; its get_final_completion() would
        # raise on finish_reason="length", so the snapshot is read instead.
        snapshot = ChatCompletionStreamState()
        streaming: CompletionCreateParamsStreaming = {
            **prepared.params,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        stream = await self._client.chat.completions.create(**streaming)
        received = False
        async with stream:
            async for chunk in stream:
                received = True
                snapshot.handle_chunk(chunk)
                for event in _chunk_events(chunk):
                    yield event
        if received:
            yield Done(snapshot.current_completion_snapshot)

    def assemble(self, final: ChatCompletion, prepared: Prepared[Params]) -> Response:
        return _parse_sdk_response(final, self._model, output_schema=prepared.output_schema)

    def usage(self, final: ChatCompletion) -> Usage | None:
        return _extract_usage(final.usage) if final.usage else None

    def map_error(self, exc: Exception, *, sent: bool) -> ProviderError | None:
        """The one place that knows the SDK's errors.

        OpenAI documents no billing for error responses
        (https://developers.openai.com/api/docs/guides/error-codes), only that a 429 is not
        charged (https://platform.openai.com/docs/guides/flex-processing); every other failure
        after dispatch stays indeterminate.
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
        if isinstance(exc, openai.APIError):  # an error event inside the stream
            return ResponseError(f"the stream reported an error: {exc.message}")
        return refused_or_unread(exc, sent=sent)

    # ------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------

    async def batch_submit(self, requests: list[dict[str, Any]], **kwargs: Any) -> str:
        """Submit a batch via OpenAI's Batch API; each body is what ``prepare`` builds."""
        bodies = [
            (req.get("custom_id", ""), self.prepare(batch_request(req, self._model)).params)
            for req in requests
        ]
        with self._mapped():
            return await submit_batch(self._client, CHAT_COMPLETIONS, bodies)

    async def batch_status(self, batch_id: str) -> str:
        """Check batch status."""
        with self._mapped():
            return await batch_status(self._client, batch_id)

    async def batch_results(self, batch_id: str) -> list[Any]:
        """Retrieve completed batch results."""
        with self._mapped():
            _, entries = await batch_output(self._client, batch_id)
        return batch_results(entries, lambda body: chat_batch_response(body, self._model))


def _forward(params: Params, options: Options) -> None:
    """The caller's SDK parameters, with the output limit as ``max_tokens`` (the name other
    servers take; ``max_completion_tokens`` wins when both are given) and ``logprobs``."""
    forwarded = dict(options.params)
    limit = forwarded.pop("max_completion_tokens", None) or forwarded.pop("max_tokens", None)
    forwarded.pop("max_tokens", None)
    params.update(cast("Params", forwarded))
    if limit is not None:
        params["max_tokens"] = limit
    if options.logprobs:
        params["logprobs"] = True


def _tools(params: Params, tools: list[dict[str, Any]] | None, options: Options) -> None:
    """The function tools and the ``tool_choice``; a server tool is refused."""
    tools = tools or []
    if server := [tool["type"] for tool in tools if tool.get("_server_tool")]:
        raise RequestError(
            f"Chat Completions takes no server tool ({', '.join(server)}): "
            "only function tools reach an OpenAI-compatible server"
        )
    if tools:
        params["tools"] = [_tool_to_sdk(tool) for tool in tools]
    if options.tool_choice is not None:
        params["tool_choice"] = _tool_choice(options.tool_choice)


def _format(params: Params, options: Options) -> None:
    response_format: ResponseFormat | None = None
    if options.output_schema is not None:
        response_format = _build_output_schema_format(options.output_schema)
    if options.json_mode:
        response_format = {"type": "json_object"}
    if response_format is not None:
        params["response_format"] = response_format


def _http() -> Any:
    """The SDK's HTTP client, with the hook that marks a request as handed to the transport."""
    return openai.DefaultAsyncHttpxClient(event_hooks={"request": [on_request]})
