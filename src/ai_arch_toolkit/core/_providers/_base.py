"""The provider contract: pure preparation, I/O behind one error mapper, one assembly."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from typing import Any, cast

from ai_arch_toolkit.core._content import CachePart
from ai_arch_toolkit.core._exceptions import (
    Delivery,
    ProviderError,
    ProviderTimeout,
    RequestError,
    ResponseError,
    TransportError,
)
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._response import OutputSchema, Response, StreamEvent, Usage
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
    """Rebuild a cached async SDK client whose pool died with its event loop.

    The sync wrappers drive every call through a fresh ``asyncio.run()`` loop
    that is closed afterwards. An async SDK client binds its connection pool
    (httpx, gRPC aio) to the loop that served its first request, so the next
    call — on a new loop — fails with a connection error. Providers install
    their client with ``_install_client``; the ``_client`` property rebuilds it
    from the factory once the loop it served is closed. A client assigned
    directly (``provider._client = mock`` in tests) has no factory and is never
    replaced. Using one provider from two concurrently *live* loops remains
    unsupported.
    """

    _client_value: Any
    _client_factory: Callable[[], Any] | None = None
    _client_loop: asyncio.AbstractEventLoop | None = None

    def _install_client(self, factory: Callable[[], Any]) -> None:
        """Install an SDK client that ``_client`` may rebuild after loop turnover."""
        self._client_value = factory()
        self._client_factory = factory
        self._client_loop = None

    @property
    def _client(self) -> Any:
        if self._client_factory is not None:
            try:
                loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                if self._client_loop is None:
                    self._client_loop = loop
                elif self._client_loop is not loop and self._client_loop.is_closed():
                    # The dead loop's pool cannot be closed without its loop —
                    # drop the old client and start fresh on this one.
                    self._client_value = self._client_factory()
                    self._client_loop = loop
        return self._client_value

    @_client.setter
    def _client(self, value: Any) -> None:
        self._client_value = value
        self._client_factory = None
        self._client_loop = None


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
    pure, raising ``RequestError`` for what the model or the adapter cannot take; :meth:`send` and
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

    def _answer(self, final: F, prepared: P) -> Answer:
        try:
            response = self.assemble(final, prepared)
            usage = self.usage(final)
        except Exception as exc:
            raise ResponseError(f"could not read the provider's response: {exc!r}") from exc
        cost = response.provider_cost
        if cost is None and usage is not None:
            cost = _estimate_response_cost(self._model, usage)
        response = dataclasses.replace(response, usage=usage or Usage(), cost=cost)
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


def parse_tool_args(raw_args: str | dict[str, Any]) -> dict[str, Any]:
    """Parse tool call arguments (may be JSON string or dict)."""
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw_args}
