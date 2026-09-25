"""OpenAI provider — Chat Completions through the ``openai`` SDK, in the provider contract."""

from __future__ import annotations

import copy
import json
import logging
import warnings
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args
from urllib.parse import urlsplit

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
from ai_arch_toolkit.core._model_id import lookup
from ai_arch_toolkit.core._pricing import _estimate_response_cost
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

# SDK parameters forwarded as the caller gave them (max_tokens is placed by host).
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
_SAMPLING = ("temperature", "top_p", "top_logprobs")
_EFFORTS = frozenset(value for value in get_args(get_args(ReasoningEffort)[0]))
_OFFICIAL_HOST = "api.openai.com"

# Failures before the request reached the server; any other transport failure may be billed.
_NOT_SENT = (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.PoolTimeout)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Profile:
    """What one family of models takes through Chat Completions.

    ``thinking``: ``reasoning`` sends ``reasoning_effort`` and, with it, only the default
    ``temperature`` (the GPT-5 models refuse another, live probe of 2026-04-28 in
    ``scripts/model_probe_notes.md``); ``refused`` raises for a model that does not reason;
    ``passthrough`` applies no OpenAI rule (another server).

    ``tools_while_reasoning`` and ``sampling_while_reasoning``: the model takes tool calls, and
    ``temperature``, ``top_p`` and logprobs, at an effort other than ``"none"``. From GPT-5.4 on,
    Chat Completions takes tool calls only at ``"none"``
    (https://developers.openai.com/api/docs/guides/migrate-to-responses), and the GPT-6 models
    take sampling parameters only there (https://developers.openai.com/api/docs/guides/latest-model).
    ``default_effort``: the effort the model reasons at when the request sends none, where its
    page gives one; a model that takes no ``"none"`` always reasons.
    """

    thinking: Literal["reasoning", "refused", "passthrough"] = "reasoning"
    efforts: frozenset[str] = _EFFORTS
    tools_while_reasoning: bool = False
    sampling_while_reasoning: bool = True
    default_effort: str | None = None


_CURRENT = _Profile()  # GPT-5.4 and later
_EARLIER = _Profile(tools_while_reasoning=True)  # the reasoning models before GPT-5.4
_LEGACY = _Profile(thinking="refused")
_COMPATIBLE = _Profile(thinking="passthrough", tools_while_reasoning=True)
# No "none" (nor "minimal") effort, so Astra always reasons: no sampling, and tool calls only
# through the Responses API (https://developers.openai.com/api/docs/models/gpt-6-astra).
_ASTRA = _Profile(
    efforts=frozenset({"low", "medium", "high", "xhigh", "max"}), sampling_while_reasoning=False
)
# Sol and Luna take "none" and reason at "medium" when the request sends no effort
# (https://developers.openai.com/api/docs/models/gpt-6-sol and .../gpt-6-luna).
_SOL_LUNA = _Profile(
    efforts=frozenset({"none", "low", "medium", "high", "xhigh", "max"}),
    sampling_while_reasoning=False,
    default_effort="medium",
)

# The families that differ from the current generation, a closed list: a model not listed (a new
# one) gets the current generation's rules. https://developers.openai.com/api/docs/models
_PROFILES: dict[str, _Profile] = {
    "gpt-6-astra": _ASTRA,
    **dict.fromkeys(("gpt-6-sol", "gpt-6-luna"), _SOL_LUNA),
    **dict.fromkeys(
        (
            "gpt-5.3",
            "gpt-5.3-codex",
            "gpt-5.2",
            "gpt-5.2-pro",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-pro",
            "gpt-5-mini",
            "gpt-5-nano",
            "o4-mini-deep-research",
            "o3",
        ),
        _EARLIER,
    ),
    # The families before the reasoning generation:
    # https://developers.openai.com/api/docs/models/gpt-4o
    **dict.fromkeys(
        (
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
        ),
        _LEGACY,
    ),
}


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


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` in OpenAI's strict subset, as the SDK's ``parse()`` helpers do.

    Strict mode rejects an object without ``additionalProperties: false`` or with properties
    missing from ``required``, which is what ``model_json_schema()`` produces for any model with
    a default. Every object is closed and lists all its properties as required (a field with a
    default is then always sent, and validates), ``default: null`` is dropped (the field stays
    nullable), and a ``$ref`` with sibling keys is inlined.
    """
    root = copy.deepcopy(schema)
    return _ensure_strict(root, root, frozenset())


def _ensure_strict(
    node: dict[str, Any], root: dict[str, Any], inlining: frozenset[str]
) -> dict[str, Any]:
    for key in ("$defs", "definitions"):
        definitions = node.get(key)
        if isinstance(definitions, dict):
            for name, definition in definitions.items():
                if isinstance(definition, dict):
                    definitions[name] = _ensure_strict(definition, root, inlining)

    if node.get("type") == "object" and "additionalProperties" not in node:
        node["additionalProperties"] = False

    properties = node.get("properties")
    if isinstance(properties, dict):
        node["required"] = list(properties)
        for name, prop in properties.items():
            if isinstance(prop, dict):
                properties[name] = _ensure_strict(prop, root, inlining)

    items = node.get("items")
    if isinstance(items, dict):
        node["items"] = _ensure_strict(items, root, inlining)

    for key in ("anyOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            node[key] = [
                _ensure_strict(v, root, inlining) if isinstance(v, dict) else v for v in variants
            ]
    all_of = node.get("allOf")
    if isinstance(all_of, list) and len(all_of) == 1 and isinstance(all_of[0], dict):
        node.pop("allOf")
        node.update(all_of[0])

    if "default" in node and node["default"] is None:
        node.pop("default")

    ref = node.get("$ref")
    # A ref already being inlined is a cycle (a model that holds itself): leave it a reference.
    if isinstance(ref, str) and len(node) > 1 and ref.startswith("#/") and ref not in inlining:
        resolved: Any = root
        for part in ref[2:].split("/"):
            resolved = resolved.get(part) if isinstance(resolved, dict) else None
        if isinstance(resolved, dict):
            # A copy, so the definition never ends up containing itself; the node's keys win.
            inlined = {**copy.deepcopy(resolved), **node}
            inlined.pop("$ref")
            return _ensure_strict(inlined, root, inlining | {ref})
    return node


def _build_output_schema_format(output_schema: OutputSchema) -> ResponseFormatJSONSchema:
    """Build OpenAI ``response_format`` for structured output."""
    schema = output_schema.schema
    return {
        "type": "json_schema",
        "json_schema": {
            "name": output_schema.name,
            "schema": _strict_schema(schema) if output_schema.strict else schema,
            "strict": output_schema.strict,
        },
    }


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
    """Extract vendor reasoning text from a streaming delta, message, or dict.

    Reads ``reasoning_content`` (DeepSeek, vLLM, LM Studio, SGLang) then
    ``reasoning`` (Ollama /v1, OpenRouter), per-field so a non-string value in
    one does not mask a valid string in the other. SDK pydantic models use
    ``extra="allow"``, so unknown wire fields surface as attributes; batch
    bodies arrive as plain dicts.
    """
    for attr in ("reasoning_content", "reasoning"):
        value = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
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
# Provider
# ---------------------------------------------------------------------------


class OpenAIProvider(LoopAwareClientCache, BaseProvider[Prepared[Params], ChatCompletion]):
    """OpenAI Chat Completions API provider via the official SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._model = model
        self._official = base_url is None or urlsplit(base_url).hostname == _OFFICIAL_HOST
        # Retry ownership belongs to LLM(RetryConfig(...)): hidden SDK retries
        # would be neither metered nor represented in Response.attempts.
        client_kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._install_client(lambda: openai.AsyncOpenAI(**client_kwargs, http_client=_http()))

    async def close(self) -> None:
        await self._client.close()

    def _profile(self) -> _Profile:
        if not self._official:
            return _COMPATIBLE
        found = lookup(self._model, _PROFILES)
        return found.value if found is not None else _CURRENT

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    def prepare(self, request: Request) -> Prepared[Params]:
        options = parse_options(request.kwargs, _FORWARDED, "OpenAI")
        profile = self._profile()
        params: Params = {
            "model": self._model,
            "messages": _messages_to_sdk(request.messages, system=request.system),
        }
        self._forward(params, options)
        self._reasoning(params, options, profile)
        self._tools(params, request.tools, options, profile)
        self._sampling(params, profile)  # after the tools, which may set the effort
        self._format(params, options)
        return Prepared(params, output_schema=options.output_schema)

    def _forward(self, params: Params, options: Options) -> None:
        """The caller's SDK parameters; the output limit's name depends on the host.

        ``max_tokens`` is deprecated in favour of ``max_completion_tokens`` and not accepted by
        o-series models (https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create);
        other servers keep ``max_tokens``.
        """
        forwarded = dict(options.params)
        limit = forwarded.pop("max_completion_tokens", None) or forwarded.pop("max_tokens", None)
        forwarded.pop("max_tokens", None)
        params.update(cast("Params", forwarded))
        if limit is not None:
            params["max_completion_tokens" if self._official else "max_tokens"] = limit
        if options.logprobs:
            params["logprobs"] = True

    def _reasoning(self, params: Params, options: Options, profile: _Profile) -> None:
        if options.thinking_budget:
            warnings.warn(
                "thinking_budget is not supported by OpenAI (only reasoning_effort string), "
                "ignoring",
                stacklevel=5,
            )
        if not options.thinking:
            return
        if profile.thinking == "refused":
            raise RequestError(f"{self._model} does not reason: thinking is not available")
        effort = options.thinking_effort or "high"
        if effort not in profile.efforts:
            raise RequestError(
                f"{self._model} takes thinking_effort in {sorted(profile.efforts)}, not {effort!r}"
            )
        params["reasoning_effort"] = cast("ReasoningEffort", effort)
        default_only = profile.thinking == "reasoning" and effort != "none"
        if default_only and params.get("temperature") not in (None, 1, 1.0):
            params.pop("temperature")

    @staticmethod
    def _reasons(params: Params, profile: _Profile) -> bool:
        """The request runs at an effort other than ``"none"``: the one it sends, else the
        model's default."""
        effort = params.get("reasoning_effort") or profile.default_effort
        return "none" not in profile.efforts if effort is None else effort != "none"

    def _sampling(self, params: Params, profile: _Profile) -> None:
        """A model without sampling while it reasons loses it; the ``LLM`` always sends a
        ``temperature``, so it is dropped rather than refused."""
        if profile.sampling_while_reasoning or not self._reasons(params, profile):
            return
        for name in _SAMPLING:
            params.pop(name, None)
        params.pop("logprobs", None)

    def _tools(
        self,
        params: Params,
        tools: list[dict[str, Any]] | None,
        options: Options,
        profile: _Profile,
    ) -> None:
        tools = tools or []
        if server := [tool["type"] for tool in tools if tool.get("_server_tool")]:
            raise RequestError(
                f"Chat Completions takes no server tool ({', '.join(server)}): "
                "only function tools reach OpenAI through this adapter"
            )
        if tools or options.tool_choice not in (None, "none"):
            self._tool_effort(params, options, profile)
        if tools:
            params["tools"] = [_tool_to_sdk(tool) for tool in tools]
        takes_tools = profile.tools_while_reasoning or "none" in profile.efforts
        if options.tool_choice is not None and takes_tools:
            params["tool_choice"] = _tool_choice(options.tool_choice)

    def _tool_effort(self, params: Params, options: Options, profile: _Profile) -> None:
        """Tool calls at the effort the model takes them. A request that asks for no thinking is
        sent at ``"none"`` when the model would reason by default (GPT-6 Sol and Luna); tool
        calls while it reasons need the Responses API."""
        if profile.tools_while_reasoning or not self._reasons(params, profile):
            return
        if "none" not in profile.efforts:
            raise RequestError(
                f"{self._model} always reasons, so its tool calling requires the Responses API, "
                "which this provider does not use"
            )
        if not options.thinking:
            params["reasoning_effort"] = "none"
            return
        raise RequestError(
            f"{self._model} takes tool calls through Chat Completions only at thinking_effort "
            "'none': tool calling while it reasons requires the Responses API, which this "
            "provider does not use"
        )

    def _format(self, params: Params, options: Options) -> None:
        response_format: ResponseFormat | None = None
        if options.output_schema is not None:
            response_format = _build_output_schema_format(options.output_schema)
        if options.json_mode:
            response_format = {"type": "json_object"}
        if response_format is not None:
            params["response_format"] = response_format

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

    async def batch_submit(
        self,
        requests: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Submit a batch via OpenAI's Batch API; each body is what ``prepare`` builds."""
        import io

        lines = [
            json.dumps(
                {
                    "custom_id": req.get("custom_id", ""),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": self._batch_body(req),
                }
            )
            for req in requests
        ]
        content = "\n".join(lines)
        with self._mapped():
            file = await self._client.files.create(
                file=io.BytesIO(content.encode()), purpose="batch"
            )
            batch = await self._client.batches.create(
                input_file_id=file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
        return batch.id

    def _batch_body(self, req: dict[str, Any]) -> Params:
        """A batch line's body: the same typed request a call sends (4096 tokens by default)."""
        request = Request(
            messages=req.get("messages", []),
            system=req.get("system"),
            tools=req.get("tools"),
            model=self._model,
            kwargs={"max_tokens": 4096, **req.get("kwargs", {})},
        )
        return self.prepare(request).params

    async def batch_status(self, batch_id: str) -> str:
        """Check batch status."""
        with self._mapped():
            batch = await self._client.batches.retrieve(batch_id)
        return batch.status

    async def batch_results(self, batch_id: str) -> list[Any]:
        """Retrieve completed batch results."""
        from ai_arch_toolkit.core._batch import BatchResult

        with self._mapped():
            batch = await self._client.batches.retrieve(batch_id)
            if not batch.output_file_id:
                return []
            file_response = await self._client.files.content(batch.output_file_id)
        raw_text = file_response.text

        results: list[BatchResult] = []
        for line in raw_text.strip().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id", "")
            resp_body = entry.get("response", {}).get("body")
            error = entry.get("error")
            if error:
                results.append(BatchResult(custom_id=custom_id, error=str(error)))
            elif resp_body:
                response = self._parse_batch_response(resp_body)
                results.append(BatchResult(custom_id=custom_id, response=response))
            else:
                results.append(BatchResult(custom_id=custom_id, error="empty response"))
        return results

    def _parse_batch_response(self, body: dict[str, Any]) -> Response:
        """Build a Response from a raw batch response body dict."""
        choices = body.get("choices", [])
        if not choices:
            return Response(raw=body, model=self._model)

        choice = choices[0]
        message = choice.get("message", {})
        text = (message.get("content") or "").strip()

        thinking: tuple[ThinkingBlock, ...] = ()
        if reasoning := _reasoning_text(message):
            thinking = (ThinkingBlock(text=reasoning),)

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    input=parse_tool_args(fn.get("arguments", "{}")),
                )
            )

        raw_usage = body.get("usage", {})
        prompt_details = raw_usage.get("prompt_tokens_details") or {}
        cache_read = prompt_details.get("cached_tokens", 0) or 0
        total_input = raw_usage.get("prompt_tokens", 0) or 0
        completion = raw_usage.get("completion_tokens", 0) or 0
        total = raw_usage.get("total_tokens", 0) or 0
        usage = Usage(
            input_tokens=_uncached_input_tokens(total_input, cache_read),
            output_tokens=max(completion, total - total_input),
            cache_read_tokens=cache_read,
        )
        cost = _estimate_response_cost(self._model, usage)

        return Response(
            text=text,
            tool_calls=tuple(tool_calls),
            thinking=thinking,
            usage=usage,
            cost=cost,
            stop_reason=choice.get("finish_reason", ""),
            model=body.get("model", self._model),
            raw=body,
        )


def _http() -> Any:
    """The SDK's HTTP client, with the hook that marks a request as handed to the transport."""
    return openai.DefaultAsyncHttpxClient(event_hooks={"request": [on_request]})


_TIMEOUTS = (httpx2.TimeoutException, openai.APITimeoutError)
