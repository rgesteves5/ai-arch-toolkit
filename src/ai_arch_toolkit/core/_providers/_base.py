"""The provider contract: pure preparation, I/O behind one error mapper, one assembly."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import copy
import dataclasses
import json
import logging
import re
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Hashable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from typing import Any, cast

from ai_arch_toolkit.core._content import CachePart, ImagePart
from ai_arch_toolkit.core._exceptions import (
    Delivery,
    ProviderError,
    ProviderTimeout,
    RequestError,
    ResponseError,
    TransportError,
)
from ai_arch_toolkit.core._images import ImageRequest
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._response import (
    OutputSchema,
    Response,
    StreamEvent,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from ai_arch_toolkit.core._stream_lifecycle import close_async

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants — used by providers that map effort → token budget
# ---------------------------------------------------------------------------

# Default budget when thinking_effort maps to token budget (Anthropic, Gemini 2.5)
THINKING_EFFORT_BUDGETS: dict[str, int] = {
    "low": 2048,
    "medium": 5000,
    "high": 10000,
}
DEFAULT_THINKING_BUDGET: int = 10000


def transport_error(
    exc: BaseException,
    *,
    not_sent: tuple[type[BaseException], ...],
    timeouts: tuple[type[BaseException], ...],
) -> TransportError | ProviderTimeout:
    """A transport failure, from the HTTP library's exception (an SDK error's ``__cause__``).

    A failure while connecting or waiting for a pooled connection (``not_sent``) never sent the
    request; any other may have been received and billed.
    """
    delivery: Delivery = "not_sent" if isinstance(exc, not_sent) else "indeterminate"
    if isinstance(exc, timeouts):
        return ProviderTimeout(str(exc) or type(exc).__name__, delivery=delivery)
    return TransportError(str(exc) or type(exc).__name__, delivery=delivery)


def refused_or_unread(exc: Exception, *, sent: bool) -> RequestError | ResponseError:
    """An exception the SDK raised outside its own error types.

    Before the request was handed to the transport, the SDK refused it (its own validation or
    serialization): nothing was sent. After, the response could not be read: it may be billed.
    """
    if sent:
        return ResponseError(f"the provider's response could not be read: {exc!r}")
    return RequestError(f"the SDK refused the request before sending it: {exc!r}")


@dataclass(slots=True)
class Dispatch:
    """Whether one call's request has been handed to the transport."""

    sent: bool = False


_dispatch: ContextVar[Dispatch | None] = ContextVar("provider_dispatch", default=None)


def mark_dispatched() -> None:
    """Record that the current call's request left the SDK for the transport."""
    current = _dispatch.get()
    if current is not None:
        current.sent = True


def dispatched() -> bool:
    """Whether the current call's request has left the SDK for the transport."""
    current = _dispatch.get()
    return current is not None and current.sent


async def on_request(request: object) -> None:
    """An ``httpx``/``httpx2`` request event hook: the SDK hands a built request to the
    transport. Adapters install it on their SDK's HTTP client."""
    mark_dispatched()


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a retry-after header value to seconds.

    Handles numeric seconds (int or float). Returns None if missing or unparseable.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.debug("Could not parse retry-after header: %r", value)
        return None


class LoopAwareClientCache:
    """Build an async SDK client on first use, and rebuild it when its event loop dies.

    Building a provider does no work bound to an event loop (G-17): an app may build its ``LLM``
    in a worker thread or before its loop runs, and a gRPC aio channel (xAI's) cannot be created
    outside a running loop. Providers install a factory with ``_install_client``; the ``_client``
    property builds the client when it is first read, which in a call is inside the loop.

    The sync wrappers drive every call through a fresh ``asyncio.run()`` loop that is closed
    afterwards. An async SDK client binds its connection pool (httpx, gRPC aio) to the loop that
    served its first request, so the next call, on a new loop, would fail with a connection error:
    ``_client`` rebuilds it from the factory once the loop it served is closed. A client assigned
    directly (``provider._client = mock`` in tests) has no factory and is never replaced. Using
    one provider from two concurrently *live* loops remains unsupported.
    """

    _client_value: Any = None
    _client_factory: Callable[[], Any] | None = None
    _client_loop: asyncio.AbstractEventLoop | None = None

    def _install_client(self, factory: Callable[[], Any]) -> None:
        """Install the factory of the SDK client, which ``_client`` builds on first use."""
        self._client_value = None
        self._client_factory = factory
        self._client_loop = None

    @property
    def _client(self) -> Any:
        factory = self._client_factory
        if factory is None:
            return self._client_value
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._client_value is None:
            self._client_value = factory()
            self._client_loop = loop
        elif loop is not None:
            if self._client_loop is None:
                self._client_loop = loop
            elif self._client_loop is not loop and self._client_loop.is_closed():
                # The dead loop's pool cannot be closed without its loop —
                # drop the old client and start fresh on this one.
                self._client_value = factory()
                self._client_loop = loop
        return self._client_value

    @_client.setter
    def _client(self, value: Any) -> None:
        self._client_value = value
        self._client_factory = None
        self._client_loop = None

    async def close(self) -> None:
        """Close the SDK client, if one was built: a provider that never ran built none.

        A client whose event loop has closed (a sync call's) is dropped instead: its connections
        died with that loop, and closing them from this one would touch the dead loop. A later
        call builds a new client.
        """
        client, loop = self._client_value, self._client_loop
        if client is None:
            return
        if loop is not None and loop.is_closed():
            self._client_value = None
            return
        await self._close_client(client)

    async def _close_client(self, client: Any) -> None:
        """Close ``client``; an SDK whose client closes another way overrides this."""
        await client.close()


@dataclass(frozen=True, slots=True)
class Prepared[R]:
    """A request ready to send: the SDK's arguments, and what assembling its answer needs."""

    params: R
    output_schema: OutputSchema | None = None


@dataclass(frozen=True, slots=True)
class Done[F]:
    """The last item of an adapter's stream: the SDK's final object."""

    final: F


@dataclass(frozen=True, slots=True)
class Answer:
    """One attempt's assembled response, and whether the provider reported its usage."""

    response: Response
    usage_reported: bool


class BaseProvider[P, F](ABC):
    """A provider adapter in three phases (costura B).

    An adapter supplies six small pieces: :meth:`prepare` builds the SDK request — synchronous and
    pure, raising ``RequestError`` for what the model or the adapter cannot take (an image
    generation is prepared by :meth:`prepare_image`, which only image-capable adapters override,
    and travels the same way); :meth:`send` and
    :meth:`open_stream` do all the I/O; :meth:`assemble` turns the SDK's final object into a
    ``Response`` and :meth:`usage` reads its usage; :meth:`map_error` is the one place that knows
    the SDK's exceptions. The algorithm lives here, once: :meth:`complete` and :meth:`stream` run
    the I/O inside the mapper and assemble the final object the same way, so a stream and a
    complete give the same ``Response`` by construction.
    """

    _model: str

    @abstractmethod
    def prepare(self, request: Request) -> P:
        """The SDK request for ``request``; synchronous, pure, and raising ``RequestError``."""

    def prepare_image(self, request: Request) -> P:
        """The SDK request for an image generation (``request.image`` is set).

        Pure like :meth:`prepare`. An adapter whose provider generates images overrides it, and
        its :meth:`send` and :meth:`assemble` take the result; this default refuses before any
        charge.
        """
        raise RequestError(
            f"{self._model}: {type(self).__name__} does not generate images; use an image model "
            "such as gpt-image-2.5-flare, gemini-3.1-flash-image or grok-imagine-image-2.0"
        )

    def image_token_bound(self, image: ImageRequest) -> int | None:
        """The most image output tokens one image of ``image`` costs, by the provider's published
        counts for the model, its quality and its size (G-40): ``0`` for a model billed per image,
        ``None`` where no count is published, and the meter holds its allowance."""
        return None

    def image_text_token_bound(self, image: ImageRequest) -> int | None:
        """The most text and thinking output tokens an image generation can bill, for a model
        that thinks or writes beside its images (G-40); ``None`` for one that bills none."""
        return None

    @abstractmethod
    async def send(self, prepared: P) -> F:
        """Send one request and return the SDK's response."""

    @abstractmethod
    def open_stream(self, prepared: P) -> AsyncIterator[StreamEvent | Done[F]]:
        """Stream one request: text and thinking as they arrive, then ``Done`` with the final."""

    @abstractmethod
    def assemble(self, final: F, prepared: P) -> Response:
        """The ``Response`` for the SDK's final object; its usage and cost are set by the base."""

    @abstractmethod
    def usage(self, final: F) -> Usage | None:
        """The final object's usage, or ``None`` when the provider reported none."""

    @abstractmethod
    def map_error(self, exc: Exception, *, sent: bool) -> ProviderError | None:
        """The normalized error for a failure; ``None`` to let it pass unchanged.

        ``sent`` says whether the request had been handed to the transport when ``exc`` was
        raised (see :func:`refused_or_unread`).
        """

    async def complete(self, prepared: P) -> Answer:
        """Send ``prepared`` and assemble the answer."""
        with self._mapped(Dispatch()):
            final = await self.send(prepared)
        return self._answer(final, prepared)

    async def stream(self, prepared: P) -> AsyncIterator[StreamEvent | Answer]:
        """Stream ``prepared``: its events, its tool calls, then the assembled answer."""
        dispatch = Dispatch()
        source = self.open_stream(prepared)
        done: Done[F] | None = None
        try:
            while True:
                with self._mapped(dispatch):
                    item = await anext(source, None)
                if item is None:
                    break
                if isinstance(item, Done):
                    done = item
                else:
                    yield item
        finally:
            await close_async(source)
        if done is None:
            raise ResponseError("the stream ended without a final message")
        answer = self._answer(done.final, prepared)
        for call in answer.response.tool_calls:
            yield StreamEvent(kind="tool_call", tool_call=call)
        yield answer

    @contextlib.contextmanager
    def _mapped(self, dispatch: Dispatch | None = None) -> Iterator[None]:
        """Run one I/O step: its failures leave through :meth:`map_error`."""
        dispatch = dispatch or Dispatch()
        token = _dispatch.set(dispatch)
        try:
            yield
        except ProviderError:
            raise
        except Exception as exc:
            error = self.map_error(exc, sent=dispatch.sent)
            if error is None:
                raise
            raise error from exc
        finally:
            _dispatch.reset(token)

    def _answer(self, final: F, prepared: P, *, batch: bool = False) -> Answer:
        """The ``Response`` for ``final``, with its usage and cost; ``batch`` prices a batch
        result at the batch rates."""
        try:
            response = self.assemble(final, prepared)
            usage = self.usage(final)
        except Exception as exc:
            raise ResponseError(f"could not read the provider's response: {exc!r}") from exc
        cost = response.provider_cost
        if cost is None and usage is not None:
            cost = _estimate_response_cost(self._model, usage, is_batch=batch)
        response = dataclasses.replace(
            response,
            usage=usage or Usage(),
            cost=cost,
            tool_calls=named_calls(response.tool_calls),
        )
        return Answer(response, usage_reported=usage is not None)

    async def batch_submit(
        self,
        requests: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Submit a batch of requests. Returns a batch ID."""
        raise NotImplementedError(f"{type(self).__name__} does not support batch API")

    async def batch_status(self, batch_id: str) -> str:
        """Check batch status. Returns status string."""
        raise NotImplementedError(f"{type(self).__name__} does not support batch API")

    async def batch_results(self, batch_id: str) -> list[Any]:
        """Retrieve batch results."""
        raise NotImplementedError(f"{type(self).__name__} does not support batch API")

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Count tokens for the given messages. Override in providers that support it."""
        raise NotImplementedError(f"{type(self).__name__} does not support token counting")

    # ------------------------------------------------------------------
    # Lifecycle — concrete no-ops, providers override if needed
    # ------------------------------------------------------------------

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in providers that hold clients."""

    async def __aenter__(self) -> BaseProvider[P, F]:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def named_calls(calls: tuple[ToolCall, ...]) -> tuple[ToolCall, ...]:
    """Each call with an id of its own, which its result names: a call that came without one, or
    with the id of an earlier call of the same answer (an OpenAI-compatible server may send
    either), gets a fresh one. Every answer gets it in ``_answer``; a Chat Completions batch line,
    read without an adapter, in ``chat_batch_response``."""
    taken: set[str] = set()
    named: list[ToolCall] = []
    for call in calls:
        fresh = not call.id or call.id in taken
        kept = dataclasses.replace(call, id=f"call_{uuid.uuid4().hex[:24]}") if fresh else call
        taken.add(kept.id)
        named.append(kept)
    return tuple(named)


class CallPieces:
    """The pieces of the tool calls one stream writes, as ``tool_call_delta`` events (G-37, D60).

    A call gets its place among the answer's calls when it starts, and every later piece of it
    carries that place, its id and its name. An adapter keys a call by what its provider's events
    name it with (a content block's index, an output index) and starts only the calls its
    :meth:`BaseProvider.assemble` makes into tool calls, in the same order.
    """

    __slots__ = ("_calls", "_written")

    def __init__(self) -> None:
        self._calls: dict[Hashable, ToolCallDelta] = {}  # each call's identity, no input
        self._written: dict[Hashable, str] = {}  # the input its pieces have written

    def __contains__(self, key: Hashable) -> bool:
        return key in self._calls

    def __len__(self) -> int:
        return len(self._calls)

    def start(
        self, key: Hashable, *, call_id: str, name: str, input_json: str = ""
    ) -> StreamEvent:
        """The first piece of a call: its place, id and name, with any input already written."""
        call = ToolCallDelta(index=len(self._calls), id=call_id, name=name)
        self._calls[key] = call
        self._written[key] = input_json
        return _piece_event(dataclasses.replace(call, input_json=input_json))

    def piece(self, key: Hashable, input_json: str) -> StreamEvent | None:
        """The next piece of a started call's input; none for an empty piece or an unknown key."""
        call = self._calls.get(key)
        if call is None or not input_json:
            return None
        self._written[key] += input_json
        return _piece_event(dataclasses.replace(call, input_json=input_json))

    def finish(self, key: Hashable, whole: str) -> StreamEvent | None:
        """What a call's pieces have not yet written of its whole input, which the provider sends
        at the end; none when they wrote it all (or ``whole`` is not what they began)."""
        written = self._written.get(key)
        if written is None or not whole.startswith(written):
            return None
        return self.piece(key, whole[len(written) :])


def _piece_event(piece: ToolCallDelta) -> StreamEvent:
    return StreamEvent(kind="tool_call_delta", tool_call_delta=piece, partial=True)


# ---------------------------------------------------------------------------
# Shared provider utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Options:
    """The toolkit's call options, read once from a request's keyword arguments."""

    thinking: bool = False
    thinking_effort: str | None = None
    thinking_budget: int | None = None
    output_schema: OutputSchema | None = None
    tool_choice: str | None = None
    json_mode: bool = False
    logprobs: bool = False
    structured_output_mode: str = "native"  # Anthropic's "prompt" fallback; ignored elsewhere
    params: dict[str, Any] = field(default_factory=dict)  # the SDK parameters forwarded as-is


_OPTION_NAMES = frozenset(option.name for option in fields(Options)) - {"params"}


def parse_options(kwargs: Mapping[str, Any], forwarded: frozenset[str], provider: str) -> Options:
    """Split a request's kwargs into the toolkit's options and the SDK parameters ``forwarded``.

    Any other keyword is ignored with a warning that names ``provider``.
    """
    unknown = set(kwargs) - forwarded - _OPTION_NAMES
    if unknown:
        warnings.warn(
            f"Unknown parameter(s) ignored for {provider}: {sorted(unknown)}. "
            f"Valid: {sorted(forwarded)}",
            stacklevel=3,
        )
    options = {name: value for name, value in kwargs.items() if name in _OPTION_NAMES}
    params = {name: value for name, value in kwargs.items() if name in forwarded}
    return Options(**options, params=params)


_JSON_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _json_of(text: str) -> Any:
    """The JSON value of ``text``, or of a Markdown code fence in it; raises ``ValueError``."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fenced = _JSON_FENCE.search(text)
        if fenced is None:
            raise
        return json.loads(fenced.group(1).strip())


def parse_structured(text: str, schema: OutputSchema) -> Any:
    """``text`` as the structured output ``schema`` asks for, validated into its model if any.

    JSON inside a Markdown code fence is accepted. ``None`` when the text holds no JSON; the plain
    data when it does not validate against the schema's Pydantic model.
    """
    try:
        data = _json_of(text.strip())
    except ValueError:
        logger.warning("Failed to parse structured output as JSON")
        return None
    if schema.model_class is None:
        return data
    try:
        return cast("Any", schema.model_class).model_validate(data)
    except Exception:
        logger.warning("Failed to validate structured output against schema")
        return data


def merge_system_prompts(*parts: str | None) -> str | None:
    """Join system prompts in order, separated by a blank line.

    Adapters call this with the ``system=`` argument first, then the text of the
    ``system()`` messages, so neither source replaces the other. ``None`` and empty
    parts are skipped; ``None`` is returned when nothing is left. A part that is not
    a string (provider-native content blocks from untyped callers) cannot be joined
    as text: the first non-empty part is then returned unchanged.
    """
    kept = [part for part in parts if part]
    if not kept:
        return None
    if len(kept) == 1 or not all(isinstance(part, str) for part in kept):
        return kept[0]
    return "\n\n".join(kept)


def system_content_text(content: Any) -> str:
    """The text of a ``system()`` message.

    A system prompt is text: its content is a string, or a list of text parts (strings and
    ``cache()`` parts) joined by a blank line. Only the Anthropic adapter keeps a cache part's
    marker; the others send its text.

    Raises:
        TypeError: The content holds an image, a document, or anything else that is not text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        texts = (_system_part_text(part) for part in content)
        return "\n\n".join(text for text in texts if text)
    return _system_part_text(content)


def _system_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, CachePart):
        return part.content
    msg = (
        f"a system message takes text or cache() parts, not {type(part).__name__}; "
        "send images and documents in a user message"
    )
    raise TypeError(msg)


def image_prompt(request: Request) -> tuple[str, list[ImagePart]]:
    """The prompt and the input images of an image generation (``request.image`` is set).

    ``LLM.generate_image`` sends one user message of text and ``image()`` parts and no options;
    what a middleware adds beyond that is refused rather than dropped.
    """
    if request.kwargs:
        raise RequestError(
            f"an image generation takes no call options, got {sorted(request.kwargs)}"
        )
    if request.system or request.tools:
        raise RequestError("an image generation takes no system prompt and no tools")
    texts: list[str] = []
    images: list[ImagePart] = []
    for message in request.messages:
        if message.get("role") != "user":
            raise RequestError("an image generation takes user messages only")
        content = message.get("content")
        for part in [content] if isinstance(content, str) else content or []:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, ImagePart):
                images.append(part)
            else:
                raise RequestError(
                    "an image generation takes text and image(...) parts, not "
                    f"{type(part).__name__}"
                )
    prompt = "\n\n".join(text for text in texts if text.strip())
    if not prompt:
        raise RequestError("an image generation needs a prompt")
    return prompt, images


def image_bytes(part: ImagePart) -> bytes:
    """The bytes of an input image given as bytes, base64 text or a ``data:`` URL.

    Raises:
        RequestError: The image is a web URL (the provider takes the file itself) or its text is
            not base64.
    """
    source = part.source
    if isinstance(source, bytes):
        return source
    if source.startswith(("https://", "http://")):
        raise RequestError("this provider uploads the image file: pass its bytes, not a URL")
    encoded = source.split(",", 1)[1] if source.startswith("data:") else source
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestError("an image source string must be base64 or a data: URL") from exc


_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
)


def image_media_type(data: bytes, declared: str | None = None) -> str:
    """The MIME type of image bytes by their signature, else the provider's ``declared`` one."""
    for signature, media_type in _SIGNATURES:
        if data.startswith(signature):
            return media_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return declared or "application/octet-stream"


def parse_tool_args(raw_args: str | dict[str, Any]) -> dict[str, Any]:
    """Parse tool call arguments (may be JSON string or dict); empty text is a call without
    arguments, as an OpenAI-compatible server may send for a tool that takes none."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and not raw_args.strip():
        return {}
    try:
        return json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw_args}


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` in OpenAI's strict subset, as the SDK's ``parse()`` helpers do.

    Strict mode rejects an object without ``additionalProperties: false`` or with properties
    missing from ``required``, which is what ``model_json_schema()`` produces for any model with
    a default. Every object is closed and lists all its properties as required (a field with a
    default is then always sent, and validates), ``default: null`` is dropped (the field stays
    nullable), and a ``$ref`` with sibling keys is inlined. Both OpenAI adapters send it: Chat
    Completions in ``response_format``, the Responses API in ``text.format``.
    """
    root = copy.deepcopy(schema)
    return _strict_node(root, root, frozenset())


def _strict_node(
    node: dict[str, Any], root: dict[str, Any], inlining: frozenset[str]
) -> dict[str, Any]:
    for key in ("$defs", "definitions"):
        _strict_values(node.get(key), root, inlining)
    if node.get("type") == "object":
        node.setdefault("additionalProperties", False)
    properties = node.get("properties")
    if isinstance(properties, dict):
        node["required"] = list(properties)
        _strict_values(properties, root, inlining)
    items = node.get("items")
    if isinstance(items, dict):
        node["items"] = _strict_node(items, root, inlining)
    for key in ("anyOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            node[key] = [_strict_variant(v, root, inlining) for v in variants]
    all_of = node.get("allOf")
    if isinstance(all_of, list) and len(all_of) == 1 and isinstance(all_of[0], dict):
        node.update(node.pop("allOf")[0])
    if "default" in node and node["default"] is None:
        node.pop("default")
    return _inline_ref(node, root, inlining)


def _strict_values(mapping: object, root: dict[str, Any], inlining: frozenset[str]) -> None:
    """Make each schema among the values of ``mapping`` (properties, definitions) strict."""
    if isinstance(mapping, dict):
        for name, value in mapping.items():
            mapping[name] = _strict_variant(value, root, inlining)


def _strict_variant(value: object, root: dict[str, Any], inlining: frozenset[str]) -> object:
    return _strict_node(value, root, inlining) if isinstance(value, dict) else value


def _inline_ref(
    node: dict[str, Any], root: dict[str, Any], inlining: frozenset[str]
) -> dict[str, Any]:
    """``node`` with its ``$ref`` inlined when the reference has sibling keys.

    A ref already being inlined is a cycle (a model that holds itself): it stays a reference.
    """
    ref = node.get("$ref")
    if not isinstance(ref, str) or len(node) == 1 or not ref.startswith("#/") or ref in inlining:
        return node
    resolved: Any = root
    for part in ref[2:].split("/"):
        resolved = resolved.get(part) if isinstance(resolved, dict) else None
    if not isinstance(resolved, dict):
        return node
    # A copy, so the definition never ends up containing itself; the node's keys win.
    inlined = {**copy.deepcopy(resolved), **node}
    inlined.pop("$ref")
    return _strict_node(inlined, root, inlining | {ref})
