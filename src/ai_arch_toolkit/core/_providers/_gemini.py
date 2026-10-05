"""Gemini provider — ``generateContent`` through ``google-genai``, in the provider contract."""

from __future__ import annotations

import base64
import json
import logging
import warnings
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from ai_arch_toolkit.core._content import CachePart, DocumentPart, ImagePart
from ai_arch_toolkit.core._exceptions import APIError, ProviderError, RateLimitError, RequestError
from ai_arch_toolkit.core._images import ImageRequest
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._model_id import lookup
from ai_arch_toolkit.core._providers import OWN_BASE_URLS
from ai_arch_toolkit.core._providers._base import (
    DEFAULT_THINKING_BUDGET,
    THINKING_EFFORT_BUDGETS,
    BaseProvider,
    CallPieces,
    Done,
    LoopAwareClientCache,
    Options,
    Prepared,
    _parse_retry_after,
    image_media_type,
    image_prompt,
    merge_system_prompts,
    on_request,
    parse_options,
    parse_structured,
    refused_or_unread,
    system_content_text,
    transport_error,
)
from ai_arch_toolkit.core._providers._imports import require_sdk
from ai_arch_toolkit.core._response import (
    Citation,
    GeneratedImage,
    OutputSchema,
    Response,
    StreamEvent,
    ThinkingBlock,
    ToolCall,
    Usage,
    _uncached_input_tokens,
)

with require_sdk("gemini"):
    import httpx  # a google-genai dependency
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types
    from google.genai.models import AsyncModels

logger = logging.getLogger(__name__)

# SDK config fields forwarded as the caller gave them (max_tokens becomes max_output_tokens).
_FORWARDED = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_output_tokens",
        "stop_sequences",
        "seed",
        "presence_penalty",
        "frequency_penalty",
    }
)
_MODES = {
    "auto": types.FunctionCallingConfigMode.AUTO,
    "required": types.FunctionCallingConfigMode.ANY,
    "none": types.FunctionCallingConfigMode.NONE,
}
_SERVER_TOOLS = {
    "web_search": types.Tool(google_search=types.GoogleSearch()),
    "code_execution": types.Tool(code_execution=types.ToolCodeExecution()),
}

# Failures before the request reached the server; any other transport failure may be billed.
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
# Billing: "If your request fails with a 400 or 500 error, you won't be charged for the tokens
# used" (https://ai.google.dev/gemini-api/docs/billing); a 429 never is (D20).
_UNBILLED_STATUSES = frozenset({400, 500})


class _Generate(TypedDict):
    """The ``generate_content`` arguments, in the SDK's own types."""

    model: str
    contents: list[types.ContentUnion]
    config: types.GenerateContentConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class _Profile:
    """How one family of Gemini models is told to think.

    ``levels``: the ``thinking_level`` values of a Gemini 3 model. ``budget``: the
    ``thinking_budget`` range of a Gemini 2.5 model (which takes no level), and whether it can
    turn thinking off with 0.
    """

    levels: frozenset[str] = frozenset({"low", "medium", "high"})
    budget: tuple[int, int] | None = None
    can_disable: bool = False


# Thinking controls per model (https://ai.google.dev/gemini-api/docs/generate-content/thinking);
# the Gemini 3 levels also in https://ai.google.dev/gemini-api/docs/thinking.
_CURRENT = _Profile()  # gemini-3.8-flash, gemini-3.7-flash, gemini-3.1-pro, and newer models
_MINIMAL = _Profile(levels=frozenset({"minimal", "low", "medium", "high"}))
_PROFILES: dict[str, _Profile] = {
    **dict.fromkeys(
        (
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash",
            "gemini-3-flash-preview",
        ),
        _MINIMAL,
    ),
    **dict.fromkeys(
        ("gemini-3-pro", "gemini-3-pro-preview"), _Profile(levels=frozenset({"low", "high"}))
    ),
    "gemini-2.5-pro": _Profile(budget=(128, 32768)),
    "gemini-2.5-flash": _Profile(budget=(0, 24576), can_disable=True),
    "gemini-2.5-flash-lite": _Profile(budget=(512, 24576), can_disable=True),
}


# The image models ("Nano Banana"), on generate_content with image output
# (https://ai.google.dev/gemini-api/docs/generate-content/image-generation): aspect ratios and
# image sizes per model; 512 and the extreme ratios on 3.1 Flash only, 1K only on Flash Lite. They
# take no function calling (https://ai.google.dev/gemini-api/docs/models). The Imagen models and
# gemini-2.5-flash-image are shut down (https://ai.google.dev/gemini-api/docs/deprecations). A
# closed list: an image model's id has no family prefix of its own. Each size's image output
# tokens, whatever the ratio, are the pricing page's (https://ai.google.dev/gemini-api/docs/pricing)
# and the image generation guide's (G-40).
@dataclass(frozen=True, slots=True, kw_only=True)
class _ImageProfile:
    ratios: frozenset[str]
    sizes: Mapping[str, int]  # each image size it takes, and the tokens of one image at it
    # The model's output token limit, which bounds its thinking: the Gemini 3 image models think,
    # and "thinking cannot be disabled in the API" (the image generation guide); each model's page
    # gives its limit.
    output_limit: int


_RATIOS = frozenset({"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"})
_IMAGE_MODELS: dict[str, _ImageProfile] = {
    "gemini-3.1-flash-image": _ImageProfile(
        ratios=_RATIOS | {"1:4", "4:1", "1:8", "8:1"},
        sizes={"512": 747, "1K": 1120, "2K": 1680, "4K": 2520},
        output_limit=32_768,
    ),
    "gemini-3.1-flash-lite-image": _ImageProfile(
        ratios=_RATIOS, sizes={"1K": 1120}, output_limit=4_096
    ),
    "gemini-3-pro-image": _ImageProfile(
        ratios=_RATIOS, sizes={"1K": 1120, "2K": 1120, "4K": 2000}, output_limit=32_768
    ),
}


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def _is_web_url(source: str) -> bool:
    return source.startswith(("https://", "http://"))


def _content_parts_to_gemini(content: Any) -> list[types.Part]:
    """Convert multimodal content to Gemini Part objects."""
    if isinstance(content, str):
        return [types.Part(text=content)]
    if not isinstance(content, list):
        return [types.Part(text=str(content))]
    return [_part(part) for part in content]


def _part(part: Any) -> types.Part:
    source = part.source if isinstance(part, ImagePart | DocumentPart) else None
    if isinstance(part, ImagePart) and isinstance(source, str) and _is_web_url(source):
        return types.Part(file_data=types.FileData(file_uri=source, mime_type=part.media_type))
    if isinstance(part, ImagePart | DocumentPart) and source is not None:
        if isinstance(source, bytes):
            data = source
        else:  # base64 text, or a data: URL (inline bytes, never a file_uri)
            data = base64.b64decode(
                source.split(",", 1)[1] if source.startswith("data:") else source
            )
        return types.Part(inline_data=types.Blob(data=data, mime_type=part.media_type))
    if isinstance(part, CachePart):
        return types.Part(text=part.content)
    return types.Part(text=part if isinstance(part, str) else str(part))


def _model_turn(msg: dict[str, Any]) -> types.Content:
    """An assistant turn: with tool calls, Gemini's own content when the message carries it (every
    part and thought signature, required for function calling), else rebuilt."""
    raw = msg.get("_raw")
    calls = msg.get("tool_calls") or []
    if calls and isinstance(raw, types.GenerateContentResponse) and raw.candidates:
        content = raw.candidates[0].content
        if content is not None:
            return content
    if not calls:
        return types.Content(role="model", parts=_content_parts_to_gemini(msg.get("content", "")))
    text = msg.get("content")
    parts = [types.Part(text=text)] if text else []
    parts += [
        types.Part(
            function_call=types.FunctionCall(name=call.get("name", ""), args=call.get("input", {}))
        )
        for call in calls
    ]
    return types.Content(role="model", parts=parts)


def _function_response(msg: dict[str, Any], call_ids: set[str]) -> types.Part:
    """A tool result, with its call's id when Gemini gave the call one (Gemini 3 maps results
    to calls by id; an id the toolkit made up was never Gemini's)."""
    content = msg.get("content", "")
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        data = content
    name = msg.get("name", "")
    if not name:
        warnings.warn(
            "Gemini requires 'name' in tool results. Pass name= to tool_result().", stacklevel=4
        )
    call_id = msg["tool_use_id"]
    return types.Part(
        function_response=types.FunctionResponse(
            id=call_id if call_id in call_ids else None,
            name=name,
            response=data if isinstance(data, dict) else {"result": data},
        )
    )


def _messages_to_sdk(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[types.ContentUnion]]:
    """The text of the ``system()`` messages, and the contents.

    The results of a turn's tool calls (marked by ``tool_use_id``) go back together, in one
    ``user`` content (https://ai.google.dev/gemini-api/docs/generate-content/function-calling).
    """
    system: list[str] = []
    contents: list[types.ContentUnion] = []
    results: list[types.Part] = []
    call_ids: set[str] = set()
    for msg in messages:
        if msg.get("tool_use_id"):
            results.append(_function_response(msg, call_ids))
            continue
        if results:
            contents.append(types.Content(role="user", parts=results))
            results = []
        role = msg.get("role", "user")
        if role == "system":
            system.append(system_content_text(msg.get("content", "")))
        elif role == "assistant":
            turn = _model_turn(msg)
            calls = [part.function_call for part in turn.parts or [] if part.function_call]
            call_ids = {call.id for call in calls if call.id}
            contents.append(turn)
        elif role == "user":
            contents.append(
                types.Content(role="user", parts=_content_parts_to_gemini(msg.get("content", "")))
            )
        else:
            raise RequestError(f"Gemini has no {role!r} role")
    if results:
        contents.append(types.Content(role="user", parts=results))
    return ("\n\n".join(system) if system else None), contents


def _tool_to_sdk(tool: dict[str, Any]) -> types.FunctionDeclaration:
    """Map generic tool dict to Gemini FunctionDeclaration.

    ``parameters`` takes Gemini's OpenAPI subset, which the SDK validates client-side and which
    has no ``prefixItems`` (tuples) or ``$defs``/``$ref``. A schema it refuses goes through
    ``parameters_json_schema`` instead, which takes JSON Schema as is; the two are mutually
    exclusive.
    """
    schema = tool.get("input_schema", tool.get("parameters", {}))
    name = tool["name"]
    description = tool.get("description", "")
    try:
        return types.FunctionDeclaration(name=name, description=description, parameters=schema)
    except ValueError:  # pydantic's ValidationError: outside the OpenAPI subset
        return types.FunctionDeclaration(
            name=name, description=description, parameters_json_schema=schema
        )


def _server_tool(tool: dict[str, Any]) -> types.Tool:
    """A server tool the adapter sends as is: its config belongs to C05."""
    kind = tool["type"]
    config = set(tool) - {"_server_tool", "type"}
    if kind not in _SERVER_TOOLS or config:
        detail = f"config {sorted(config)}" if kind in _SERVER_TOOLS else "no such server tool"
        raise RequestError(f"Gemini server tool {kind!r}: {detail} is not supported")
    return _SERVER_TOOLS[kind]


def _tools(tools: list[dict[str, Any]]) -> list[types.Tool]:
    functions = [_tool_to_sdk(tool) for tool in tools if not tool.get("_server_tool")]
    head = [types.Tool(function_declarations=functions)] if functions else []
    return head + [_server_tool(tool) for tool in tools if tool.get("_server_tool")]


def _tool_config(choice: str) -> types.ToolConfig:
    if choice in _MODES:
        calling = types.FunctionCallingConfig(mode=_MODES[choice])
    else:
        calling = types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY, allowed_function_names=[choice]
        )
    return types.ToolConfig(function_calling_config=calling)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def _extract_usage(usage: types.GenerateContentResponseUsageMetadata) -> Usage:
    """The toolkit's usage: tool-use prompt tokens are input, thoughts are output, and the
    candidates' image tokens are image output, billed at the image rate
    (https://ai.google.dev/gemini-api/docs/pricing)."""
    cache_read = usage.cached_content_token_count or 0
    image = sum(
        detail.token_count or 0
        for detail in usage.candidates_tokens_details or []
        if detail.modality == types.MediaModality.IMAGE
    )
    return Usage(
        input_tokens=_uncached_input_tokens(usage.prompt_token_count or 0, cache_read)
        + (usage.tool_use_prompt_token_count or 0),
        output_tokens=(usage.candidates_token_count or 0)
        - image
        + (usage.thoughts_token_count or 0),
        cache_read_tokens=cache_read,
        image_output_tokens=image,
    )


def _citations(candidate: types.Candidate) -> tuple[Citation, ...]:
    grounding = candidate.grounding_metadata
    return tuple(
        Citation(text="", url=chunk.web.uri or "", title=chunk.web.title or "")
        for chunk in (grounding.grounding_chunks or [] if grounding else [])
        if chunk.web
    )


def _parse_sdk_response(
    response: types.GenerateContentResponse,
    model: str,
    *,
    output_schema: OutputSchema | None = None,
) -> Response:
    """The ``Response`` for a ``GenerateContentResponse`` (the base adds usage and cost)."""
    if not response.candidates:
        return Response(raw=response, model=model)
    candidate = response.candidates[0]
    # A candidate cut off by max_tokens while still thinking carries content with no parts.
    parts = (candidate.content.parts if candidate.content else None) or []
    text = "".join(part.text for part in parts if part.text is not None and not part.thought)
    text = text.strip()
    return Response(
        text=text,
        images=tuple(
            # The interim "thought images" of a Gemini 3 image model are not the answer.
            GeneratedImage(
                data=part.inline_data.data,
                media_type=image_media_type(part.inline_data.data, part.inline_data.mime_type),
            )
            for part in parts
            if part.inline_data is not None and part.inline_data.data and not part.thought
        ),
        tool_calls=tuple(
            ToolCall(
                id=part.function_call.id or "",  # the base names a call without one
                name=part.function_call.name or "",
                input=dict(part.function_call.args or {}),
            )
            for part in parts
            if part.function_call
        ),
        thinking=tuple(
            ThinkingBlock(text=part.text) for part in parts if part.thought and part.text
        ),
        parsed=parse_structured(text, output_schema) if output_schema and text else None,
        stop_reason=candidate.finish_reason.value if candidate.finish_reason else "",
        model=response.model_version or model,
        raw=response,
        response_id=response.response_id or "",
        citations=_citations(candidate),
    )


def _chunk_events(chunk: types.GenerateContentResponse, calls: CallPieces) -> list[StreamEvent]:
    """The text and thought fragments of one streamed chunk, its images (an image model's
    interim thought images as previews, ``partial``, its final image whole), and its function
    calls, each as one piece with the whole input.

    The Gemini API sends a function call whole, in one part: streaming its arguments
    (``stream_function_call_arguments``) is Vertex AI's alone, "not supported in Gemini API"
    (``google-genai``'s ``FunctionCallingConfig``).
    """
    candidates = chunk.candidates or []
    content = candidates[0].content if candidates else None
    events: list[StreamEvent] = []
    for part in (content.parts if content else None) or []:
        if part.function_call:
            call = part.function_call
            whole = json.dumps(dict(call.args or {}))
            key = len(calls)  # each part is a call of its own
            events.append(
                calls.start(key, call_id=call.id or "", name=call.name or "", input_json=whole)
            )
        elif part.inline_data is not None and part.inline_data.data:
            data = part.inline_data.data
            image = GeneratedImage(
                data=data, media_type=image_media_type(data, part.inline_data.mime_type)
            )
            events.append(StreamEvent(kind="image", image=image, partial=bool(part.thought)))
        elif part.thought and part.text:
            events.append(
                StreamEvent(kind="thinking", thinking=ThinkingBlock(text=part.text), partial=True)
            )
        elif part.text:
            events.append(StreamEvent(kind="text", text=part.text))
    return events


def _joined(chunks: list[types.GenerateContentResponse]) -> types.GenerateContentResponse:
    """One response from a stream's chunks, which the SDK does not accumulate.

    Every part, in order, with its thought signature; the last finish reason, grounding and
    usage the stream reported.
    """
    candidates = [chunk.candidates[0] for chunk in chunks if chunk.candidates]
    parts = [
        part
        for candidate in candidates
        if candidate.content is not None
        for part in candidate.content.parts or []
    ]
    last = chunks[-1]
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts),
                finish_reason=next(
                    (c.finish_reason for c in reversed(candidates) if c.finish_reason), None
                ),
                grounding_metadata=next(
                    (c.grounding_metadata for c in reversed(candidates) if c.grounding_metadata),
                    None,
                ),
            )
        ],
        usage_metadata=next(
            (chunk.usage_metadata for chunk in reversed(chunks) if chunk.usage_metadata), None
        ),
        model_version=last.model_version,
        response_id=last.response_id,
    )


def _http_options(timeout: float | None) -> types.HttpOptions:
    """One physical attempt (``LLM(RetryConfig(...))`` owns retries, so each is metered), the
    timeout in milliseconds, and an HTTP transport of the adapter's own.

    Given a transport, the SDK sends through ``httpx`` instead of ``aiohttp``, whose path re-sends
    a request after a connection error outside any retry option; the event hook marks a request
    as handed to the transport. The SDK builds and closes the client. The endpoint is always
    given: left out, the SDK reads ``GOOGLE_GEMINI_BASE_URL`` from the environment (D48).
    """
    return types.HttpOptions(
        base_url=OWN_BASE_URLS["gemini"],
        retry_options=types.HttpRetryOptions(attempts=1),
        timeout=None if timeout is None else int(timeout * 1000),
        async_client_args={
            "transport": httpx.AsyncHTTPTransport(),
            "event_hooks": {"request": [on_request]},
        },
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GeminiProvider(
    LoopAwareClientCache, BaseProvider[Prepared[_Generate], types.GenerateContentResponse]
):
    """Google Gemini provider via the official ``google-genai`` SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        timeout: float | None = None,
    ) -> None:
        self._model = model
        self._install_client(
            # The Gemini Developer API, whatever GOOGLE_GENAI_USE_VERTEXAI says (D48).
            lambda: genai.Client(
                api_key=api_key, vertexai=False, http_options=_http_options(timeout)
            )
        )

    async def _close_client(self, client: genai.Client) -> None:
        await client.aio.aclose()
        client.close()

    def _models(self) -> AsyncModels:
        return self._client.aio.models

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
        """Count tokens using Gemini's countTokens API."""
        msg_system, contents = _messages_to_sdk(messages)
        effective_system = merge_system_prompts(system, msg_system)
        config = types.CountTokensConfig(system_instruction=effective_system)
        with self._mapped():
            result = await self._models().count_tokens(
                model=self._model, contents=contents, config=config
            )
        return result.total_tokens or 0

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    def _image_profile(self) -> _ImageProfile | None:
        found = lookup(self._model, _IMAGE_MODELS)
        return found.value if found is not None else None

    def image_token_bound(self, image: ImageRequest) -> int | None:
        """One image's tokens at the size asked, or at the largest the model takes."""
        profile = self._image_profile()
        if profile is None:
            return None
        resolution = image.resolution
        if resolution is not None and resolution in profile.sizes:
            return profile.sizes[resolution]
        return max(profile.sizes.values())

    def image_text_token_bound(self, image: ImageRequest) -> int | None:
        """The model's output token limit: its thinking, billed at the text rate, stays within."""
        profile = self._image_profile()
        return profile.output_limit if profile is not None else None

    def prepare(self, request: Request) -> Prepared[_Generate]:
        if request.tools and self._image_profile() is not None:
            raise RequestError(f"{self._model} draws images and takes no tools")
        options = parse_options(request.kwargs, _FORWARDED, "Gemini")
        msg_system, contents = _messages_to_sdk(request.messages)
        forwarded = dict(options.params)
        if "max_tokens" in forwarded:
            forwarded["max_output_tokens"] = forwarded.pop("max_tokens")
        tools = request.tools or []
        response_json = options.output_schema is not None or options.json_mode
        try:
            config = types.GenerateContentConfig(
                **forwarded,
                system_instruction=merge_system_prompts(request.system, msg_system),
                thinking_config=self._thinking(options),
                tools=_tools(tools) if tools else None,
                tool_config=_tool_config(options.tool_choice) if options.tool_choice else None,
                response_mime_type="application/json" if response_json else None,
                response_json_schema=options.output_schema.schema
                if options.output_schema
                else None,
            )
        except ValueError as refused:  # the SDK's pydantic validation of the caller's values
            if isinstance(refused, RequestError):
                raise
            raise RequestError(f"the Gemini SDK refused the request: {refused}") from refused
        params: _Generate = {"model": self._model, "contents": contents, "config": config}
        return Prepared(params, output_schema=options.output_schema)

    def prepare_image(self, request: Request) -> Prepared[_Generate]:
        """An image generation: ``generate_content`` asking for image output only, the input
        images in the prompt's message."""
        profile = self._image_profile()
        if profile is None:
            raise RequestError(
                f"{self._model} is not an image model; generate_image takes one such as "
                "gemini-3.1-flash-image"
            )
        options = request.image
        assert options is not None  # prepare_image is only called for image generations
        image_prompt(request)  # the same refusals as every adapter's
        if options.n != 1:
            raise RequestError(f"{self._model} returns one image per request, not {options.n}")
        if options.quality is not None:
            raise RequestError(f"{self._model} takes no quality")
        if options.output_format is not None:
            # image_config.output_mime_type is Vertex AI's: google-genai refuses it on the
            # Gemini API (I01 check 4).
            raise RequestError(f"{self._model} on the Gemini API takes no output_format")
        if options.aspect_ratio is not None and options.aspect_ratio not in profile.ratios:
            raise RequestError(
                f"{self._model} takes aspect_ratio in {sorted(profile.ratios)}, "
                f"not {options.aspect_ratio!r}"
            )
        if options.resolution is not None and options.resolution not in profile.sizes:
            raise RequestError(
                f"{self._model} takes resolution in {sorted(profile.sizes)}, "
                f"not {options.resolution!r}"
            )
        _, contents = _messages_to_sdk(request.messages)
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=options.aspect_ratio, image_size=options.resolution
            ),
        )
        return Prepared({"model": self._model, "contents": contents, "config": config})

    def _thinking(self, options: Options) -> types.ThinkingConfig | None:
        """``thinking_effort`` applies on its own (these models think unasked, as in D13/D25);
        ``thinking=True`` asks for thought summaries and, without an effort, thinks hard."""
        profile = self._profile()
        if profile.budget is not None:
            budget = self._budget(options, profile)
            if budget is None:
                return None
            return types.ThinkingConfig(
                include_thoughts=options.thinking or None, thinking_budget=budget
            )
        if options.thinking_budget is not None:
            warnings.warn(
                "thinking_budget is for Gemini 2.5; Gemini 3 takes thinking_effort, ignoring",
                stacklevel=5,
            )
        effort = options.thinking_effort or ("high" if options.thinking else None)
        if effort is None:
            return None
        if effort not in profile.levels:
            raise RequestError(
                f"{self._model} takes thinking_effort in {sorted(profile.levels)}, not {effort!r}"
            )
        return types.ThinkingConfig(
            include_thoughts=options.thinking or None,
            thinking_level=types.ThinkingLevel(effort.upper()),
        )

    def _budget(self, options: Options, profile: _Profile) -> int | None:
        """A Gemini 2.5 thinking budget: the caller's, checked against the model's range, else
        the effort's, else the default when thinking is asked for."""
        low, high = cast("tuple[int, int]", profile.budget)
        budget = options.thinking_budget
        if budget is not None:
            documented = (
                low <= budget <= high or budget == -1 or (budget == 0 and profile.can_disable)
            )
            if not documented:
                off = ", 0 to turn thinking off" if profile.can_disable else ""
                raise RequestError(
                    f"{self._model} takes a thinking_budget from {low} to {high}{off} or -1, "
                    f"not {budget}"
                )
            return budget
        effort = options.thinking_effort
        if effort is not None:
            if effort not in THINKING_EFFORT_BUDGETS:
                raise RequestError(
                    f"{self._model} takes thinking_effort in {sorted(THINKING_EFFORT_BUDGETS)}, "
                    f"not {effort!r}"
                )
            return THINKING_EFFORT_BUDGETS[effort]
        return DEFAULT_THINKING_BUDGET if options.thinking else None

    async def send(self, prepared: Prepared[_Generate]) -> types.GenerateContentResponse:
        return await self._models().generate_content(**prepared.params)

    async def open_stream(
        self, prepared: Prepared[_Generate]
    ) -> AsyncIterator[StreamEvent | Done[types.GenerateContentResponse]]:
        chunks: list[types.GenerateContentResponse] = []
        calls = CallPieces()
        async for chunk in await self._models().generate_content_stream(**prepared.params):
            chunks.append(chunk)
            for event in _chunk_events(chunk, calls):
                yield event
        if chunks:
            yield Done(_joined(chunks))

    def assemble(
        self, final: types.GenerateContentResponse, prepared: Prepared[_Generate]
    ) -> Response:
        return _parse_sdk_response(final, self._model, output_schema=prepared.output_schema)

    def usage(self, final: types.GenerateContentResponse) -> Usage | None:
        return _extract_usage(final.usage_metadata) if final.usage_metadata else None

    def map_error(self, exc: Exception, *, sent: bool) -> ProviderError | None:
        """The one place that knows the SDK's errors.

        Gemini does not bill a request that fails with a 400 or a 500
        (https://ai.google.dev/gemini-api/docs/billing) and never a 429 (D20); other statuses stay
        indeterminate. The SDK raises the same errors for an error object inside a 200 stream,
        whose reply is then not the ``httpx`` response: tokens may already be billed there.
        """
        if isinstance(exc, genai_errors.APIError):
            reply = exc.response if isinstance(exc.response, httpx.Response) else None
            if exc.code == 429:
                retry_after = reply.headers.get("retry-after") if reply is not None else None
                return RateLimitError(429, str(exc), retry_after=_parse_retry_after(retry_after))
            unbilled = reply is not None and exc.code in _UNBILLED_STATUSES
            return APIError(
                exc.code, str(exc), delivery="unbilled" if unbilled else "indeterminate"
            )
        if isinstance(exc, httpx.TransportError):
            return transport_error(exc, not_sent=_NOT_SENT, timeouts=(httpx.TimeoutException,))
        return refused_or_unread(exc, sent=sent)
