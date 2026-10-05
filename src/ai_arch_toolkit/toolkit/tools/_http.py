"""The toolkit tools' only door to the network.

Every request a tool in ``toolkit.tools`` sends goes through here: HTTPS to the host its module
declared, redirects kept on that host, a body read bounded in bytes and in time, path segments
quoted one by one, a rate-limit clock shared across threads, and one error type whose text is ready
for the model. ``urllib.request``, ``urllib.error``, ``http.client``, ``socket`` and ``ssl`` are
imported nowhere else in the package (architecture test).
"""

from __future__ import annotations

import email.utils
import http.client
import importlib.metadata
import json
import math
import os
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from email.message import Message
from typing import IO, Any, Protocol, cast

from ai_arch_toolkit.core._tools._billing import bill
from ai_arch_toolkit.core._tools._result import ToolFailure, ToolFailureType


def _version() -> str:
    try:
        return importlib.metadata.version("ai-arch-toolkit")
    except importlib.metadata.PackageNotFoundError:  # run from a source tree, not installed
        return "dev"


# Who is asking, for the services that limit callers who do not say.
USER_AGENT = f"ai-arch-toolkit/{_version()} (+https://github.com/rgesteves5/ai-arch-toolkit)"

type ParamValue = str | int | float | Sequence[str]
type Params = Mapping[str, ParamValue]

_WEB_SCHEMES = frozenset({"http", "https"})
_DOT_SEGMENTS = frozenset({"", ".", ".."})
_CHUNK_BYTES = 64 * 1024
_ERROR_BODY_CHARS = 2000
_SOURCE_TEXT_CHARS = 300  # of a source's own error text, without a reader


class HttpError(ToolFailure):
    """A request that produced no usable body: a tool failure (D42). ``str()`` is the reason shown
    to the model.

    A 429 is ``rate_limited`` and a 5xx ``upstream``, both retryable; any other failure is
    ``upstream`` unless the raise site says otherwise (a timeout or a network error is retryable;
    a URL the module may not reach is a ``validation_error``).

    Attributes:
        status: The HTTP status of an error response; ``None`` when no response arrived, or when
            the API reported the error inside a successful one (``Api.error_reader``).
        body: The start of an error response's body, for APIs that explain errors there.
        retry_after_s: The seconds an error response's ``Retry-After`` asked to wait, if it did.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        retry_after_s: float | None = None,
        kind: ToolFailureType | None = None,
        retryable: bool | None = None,
    ) -> None:
        server = status is not None and status >= 500
        details: dict[str, float] = {}
        if status is not None:
            details["status"] = status
        if retry_after_s is not None:
            details["retry_after_s"] = retry_after_s
        super().__init__(
            kind or ("rate_limited" if status == 429 else "upstream"),
            message,
            retryable=retryable if retryable is not None else status == 429 or server,
            details=details,
        )
        self.status = status
        self.body = body
        self.retry_after_s = retry_after_s


@dataclass(frozen=True, slots=True, kw_only=True)
class Reply:
    """What an API answered, for its ``error_reader``.

    Attributes:
        status: The HTTP status.
        headers: The headers, by lower-case name.
        body: The decoded JSON, or the text when it is not JSON (the start of an error's body).
    """

    status: int
    headers: Mapping[str, str]
    body: object


type ErrorReader = Callable[[Reply], ToolFailure | str | None]
"""Reads the error an answer reports (D38): ``None`` for none, the source's words as text (typed
by the status), or a typed :class:`ToolFailure` when the source says what happened."""


class _Response(Protocol):
    @property
    def status(self) -> int: ...

    @property
    def headers(self) -> Message: ...

    def read1(self, amt: int, /) -> bytes: ...

    def close(self) -> None: ...


class _Redirected(Exception):
    """A redirect the opener refused to follow."""

    def __init__(self, target: str) -> None:
        super().__init__(target)
        self.target = target


class _SameHostRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only on the same host, and never from https down to http."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        old, new = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        downgrade = old.scheme == "https" and new.scheme != "https"
        if new.hostname != old.hostname or downgrade or new.scheme not in _WEB_SCHEMES:
            fp.close()
            raise _Redirected(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _tls_context() -> ssl.SSLContext:
    """The TLS context that verifies every request: the system's certificate store when the
    ``truststore`` extra is installed, and OpenSSL's CA file otherwise (D51).

    On some Pythons that file lacks recent roots: uv's standalone builds on macOS read
    ``/etc/ssl/cert.pem``, which has no GlobalSign Root R46 (Eurostat's, measured 2026-10-04).
    """
    try:
        import truststore
    except ImportError:
        return ssl.create_default_context()
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


_TLS = _tls_context()
_SYSTEM_STORE = type(_TLS) is not ssl.SSLContext
_TRUSTSTORE_HINT = (
    " (to verify with the system's certificate store, install truststore: "
    "pip install truststore, or the ai-arch-toolkit[truststore] extra)"
)


def _build_opener(*handlers: urllib.request.BaseHandler) -> urllib.request.OpenerDirector:
    tls = urllib.request.HTTPSHandler(context=_TLS)
    return urllib.request.build_opener(_SameHostRedirect(), tls, *handlers)


_OPENER = _build_opener()


def _open(request: urllib.request.Request, timeout: float) -> _Response:
    """Send ``request``: the one call in the package that reaches the network."""
    return _OPENER.open(request, timeout=timeout)


class _Throttle:
    """Spaces requests that share a clock: one clock per (host, interval), across threads.

    A request takes a slot before it goes and frees the next ``interval`` after it ends, so a slow
    answer does not let the next request out early. A host that answered 429 rests (``cool``):
    until then, its requests do not go out (D53).
    """

    def __init__(
        self,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._free_at: dict[tuple[str, float], float] = {}
        self._rest_until: dict[str, float] = {}

    def wait(self, host: str, interval_s: float) -> None:
        """Block until this caller's slot: reserved under the lock, slept outside it."""
        if interval_s <= 0:
            return
        key = (host, interval_s)
        with self._lock:
            now = self._clock()
            slot = max(now, self._free_at.get(key, now))
            self._free_at[key] = slot + interval_s
        self._sleep(slot - now)

    def done(self, host: str, interval_s: float) -> None:
        """The request that took a slot has ended: the next goes ``interval_s`` after now."""
        if interval_s <= 0:
            return
        key = (host, interval_s)
        with self._lock:
            end = self._clock() + interval_s
            self._free_at[key] = max(self._free_at.get(key, end), end)

    def cool(self, host: str, seconds: float) -> None:
        """``host`` asked to slow down: it takes no request for ``seconds``."""
        with self._lock:
            until = self._clock() + seconds
            self._rest_until[host] = max(self._rest_until.get(host, until), until)

    def resting(self, host: str) -> float:
        """The seconds left before ``host`` takes requests again; 0 when it does."""
        with self._lock:
            return max(0.0, self._rest_until.get(host, 0.0) - self._clock())


_THROTTLE = _Throttle()


def _split(url: str) -> urllib.parse.SplitResult | None:
    """``url``'s parts, or ``None`` when it does not parse (a bad port or IPv6 literal)."""
    try:
        parts = urllib.parse.urlsplit(url)
        _ = parts.port  # a malformed port raises here
    except ValueError:
        return None
    return parts


def _web_origin(parts: urllib.parse.SplitResult, schemes: frozenset[str]) -> bool:
    """Whether the URL names a host over one of ``schemes``, with no credentials in it."""
    return (
        parts.scheme in schemes
        and bool(parts.hostname)
        and parts.username is None
        and parts.password is None
    )


def _plain_base(url: str) -> bool:
    """Whether ``url`` is ``https://host/path`` with no port, query, fragment or final slash."""
    parts = _split(url)
    return (
        parts is not None
        and _web_origin(parts, frozenset({"https"}))
        and parts.port is None
        and not (parts.query or parts.fragment or parts.path.endswith("/"))
    )


def _segment(segment: str, safe: str) -> str:
    if segment in _DOT_SEGMENTS:
        raise HttpError(f"invalid path segment: {segment!r}", kind="validation_error")
    return urllib.parse.quote(segment, safe=safe)


def _status_text(status: int, reason: str) -> str:
    return f"HTTP error {status}: {reason}"


def _error_body(error: urllib.error.HTTPError) -> str:
    try:
        raw = error.read(_ERROR_BODY_CHARS * 4)
    except (OSError, ValueError, http.client.HTTPException):
        return ""
    finally:
        error.close()
    return raw.decode("utf-8", errors="replace")


def _source_text(body: str) -> str | None:
    """The error text a body gives without a reader of its own: the ``message``, ``error`` or
    ``detail`` of a JSON object (text, or an object's ``message``), or the start of a text body
    that is not an HTML page."""
    value = _decoded(body)
    if isinstance(value, dict):
        for key in ("message", "error", "detail"):
            field_value = value.get(key)
            if isinstance(field_value, dict):
                field_value = field_value.get("message")
            if isinstance(field_value, str) and field_value.strip():
                return " ".join(field_value.split())[:_SOURCE_TEXT_CHARS]
        return None
    text = " ".join(body.split())
    if not text or text.startswith(("<", "{", "[")) or isinstance(value, list | int | float):
        return None  # a page, JSON cut short, or JSON with no message
    return text[:_SOURCE_TEXT_CHARS]


def _body(response: _Response, deadline: float, max_bytes: int) -> tuple[bytes, bool]:
    """Read at most ``max_bytes``; ``False`` when the body went on. Past the deadline, give up.

    ``read1`` makes one read of the socket, so a body that trickles in cannot hold a call past the
    deadline by more than one socket timeout.
    """
    chunks: list[bytes] = []
    size = 0
    while size <= max_bytes:
        if time.monotonic() > deadline:
            raise HttpError("request timed out.", retryable=True)
        chunk = response.read1(min(_CHUNK_BYTES, max_bytes + 1 - size))
        if not chunk:
            return b"".join(chunks), True
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)[:max_bytes], False


def _fetch(
    request: urllib.request.Request,
    *,
    timeout_s: float,
    max_bytes: int,
    failure: Callable[[Reply, str], HttpError],
) -> tuple[int, bytes, str, bool, Mapping[str, str]]:
    """Send ``request`` and read its body: ``(status, body, charset, complete, headers)``.

    ``failure(reply, reason)`` builds the error of a response with an error status, from its
    status, headers and the start of its body (decoded when it is JSON) and the status's reason.
    """
    deadline = time.monotonic() + timeout_s
    try:
        response = _open(request, timeout_s)
        try:
            status = response.status
            body, complete = _body(response, deadline, max_bytes)
            charset = response.headers.get_content_charset() or "utf-8"
            headers = _header_map(response.headers)
        finally:
            response.close()
    except urllib.error.HTTPError as error:
        reply = Reply(
            status=error.code, headers=_header_map(error.headers), body=_error_body(error)
        )
        raise failure(reply, str(error.reason)) from error
    except _Redirected as refused:
        target = refused.target
        raise HttpError(f"refused a redirect to {target} (only same-host HTTPS)") from refused
    except urllib.error.URLError as error:
        unverified = isinstance(error.reason, ssl.SSLCertVerificationError)
        hint = _TRUSTSTORE_HINT if unverified and not _SYSTEM_STORE else ""
        msg = f"URL error: {error.reason}{hint}"
        raise HttpError(msg, retryable=not unverified) from error  # a certificate stays wrong
    except TimeoutError as error:
        raise HttpError("request timed out.", retryable=True) from error
    except http.client.InvalidURL as error:  # a space or a control character in the URL
        raise HttpError(f"invalid URL: {error}", kind="validation_error") from error
    except (OSError, http.client.HTTPException) as error:
        reason = str(error) or type(error).__name__
        raise HttpError(f"network error: {reason}", retryable=True) from error
    return status, body, charset, complete, headers


def _header_map(headers: Message | None) -> Mapping[str, str]:
    """The headers by lower-case name (the last value of a repeated one)."""
    return {name.lower(): value for name, value in (headers or Message()).items()}


def _retry_after_value(value: str | None) -> float | None:
    """The seconds a ``Retry-After`` header asks to wait: a number, or an HTTP date."""
    value = value or ""
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def _text(body: bytes, charset: str) -> str:
    try:
        return body.decode(charset, errors="replace")
    except LookupError as error:
        raise HttpError(f"unknown charset {charset!r} in the response") from error


def _json(text: str) -> object:
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as error:
        raise HttpError(f"could not parse API response: {error}") from error


def _decoded(text: str) -> object:
    """``text`` as JSON, or the text itself when it is not JSON."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return text


def _kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int | float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    return "an array" if isinstance(value, list) else "an object"


# What reading a response of an unexpected shape raises: a missing key, a null where an object
# was, a string where a number was, malformed XML. Everything else is a bug and propagates.
_SHAPE_ERRORS = (
    ArithmeticError,
    AttributeError,
    LookupError,
    RecursionError,
    SyntaxError,
    TypeError,
    ValueError,
)


def _parsed[V, T](parse: Callable[[V], T], value: V) -> T:
    try:
        return parse(value)
    except _SHAPE_ERRORS as error:
        raise HttpError(f"could not parse API response: {error!r}") from error


def _object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise HttpError(f"could not parse API response: expected a JSON object, got {_kind(value)}")


def _array(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    raise HttpError(f"could not parse API response: expected a JSON array, got {_kind(value)}")


@dataclass(frozen=True, slots=True, kw_only=True)
class _Ask:
    """What one request declares about its answers (see :class:`Api`)."""

    missing: str | None = None
    empty: object | None = None
    allow_empty: bool = False
    empty_on_404: bool = False
    body: tuple[bytes, str] | None = None


class _EmptyNotFound(Exception):
    """A 404 the request declared empty: nothing found."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Api:
    """One upstream HTTPS API: requests go to its host and under its base path, nowhere else.

    Every request takes the function that reads its answer (``parse``), and whatever that
    function raises on a shape it did not expect becomes an ``HttpError``: no raw response leaves
    this module unguarded, so a tool cannot crash on a body it did not foresee. An API that
    explains its errors declares how to read them (``error_reader``, D38), so no ``parse``
    mistakes an error for an empty result and no explanation is lost; without one, an error
    status's message carries the source's own error text.

    Each request says what its answers mean. ``missing=`` declares that it asks for one resource:
    a 404 is then ``not_found``, with that message; without it, a 404 is an ``upstream`` "endpoint
    not found". A source that answers "nothing found" with ``204 No Content`` or an empty body
    is declared with ``allow_empty`` (``parse`` reads an empty object or array), one that answers
    it with a 404 with ``empty_on_404``; anywhere else such an answer is a failure.

    Attributes:
        base: ``https://host/path`` without credentials, port, query, fragment or final slash.
        name: The API's name, for the rate-limit message.
        timeout_s: Deadline of one request, checked between reads of the body.
        max_bytes: Largest body read; a longer one is an error.
        min_interval_s: Least time between the end of a request and the start of the next.
            Requests to one host with one interval share a clock, across modules and threads.
        cooldown_s: After a 429 without a ``Retry-After``, how long the host takes no request: a
            request in that time fails at once and says when to try again (D53).
        key_env: The environment variable that holds the API's optional key (D52), read at each
            request and sent in ``key_header``, after ``key_prefix`` (``"Bearer "``); without it,
            a 429 says where to get one (``key_url``). With ``key_required``, a request without
            the key fails before it is sent, saying where to get one.
        billed_as: For a paid API, the tool price its requests are billed at (D56): each request
            it accepts records ``bill_units`` of the answer's text (the units it says it billed),
            or one, for the executor to charge. A refused request records nothing.
        params: Query parameters sent with every request.
        segment_safe: Characters left raw in path segments.
        query_safe: Characters left raw in the query string.
        error_reader: For an API that explains its errors: reads a :class:`Reply` and returns
            the error it reports (an :data:`ErrorReader`). It reads every answer of the JSON
            requests, before ``parse``, and the error statuses of any request. A reader that trips
            on a success's JSON is a parse error (an answer of a shape nobody expected); one that
            trips on any other body, or returns empty text, explains nothing.
        caller_base: ``base`` came from the tool's caller (:meth:`within`): an undeclared 404
            there is the caller's URL, a ``validation_error``, not an endpoint that moved.
    """

    base: str
    name: str
    timeout_s: float = 10.0
    max_bytes: int = 10_000_000
    min_interval_s: float = 0.0
    params: Mapping[str, str] = field(default_factory=dict)
    segment_safe: str = ""
    query_safe: str = ""
    error_reader: ErrorReader | None = None
    cooldown_s: float = 0.0
    key_env: str | None = None
    key_header: str = "x-api-key"
    key_prefix: str = ""
    key_url: str = ""
    key_required: bool = False
    billed_as: str | None = None
    bill_units: Callable[[str], int] | None = None
    caller_base: bool = False

    def __post_init__(self) -> None:
        if not _plain_base(self.base):
            msg = f"Api base must be https://host/path with nothing else, got {self.base!r}"
            raise ValueError(msg)

    @classmethod
    def within(
        cls,
        url: str,
        domains: Iterable[str],
        *,
        name: str,
        timeout_s: float = 10.0,
        error_reader: ErrorReader | None = None,
    ) -> Api:
        """An ``Api`` at ``url``, a URL that came from outside the module.

        Raises:
            HttpError: The host is not one of ``domains`` or a subdomain of one, or the URL is not
                a plain ``https://host/path``.
        """
        parts = _split(url)
        host = (parts.hostname or "") if parts is not None else ""
        if not any(host == domain or host.endswith(f".{domain}") for domain in domains):
            raise HttpError(f"host not allowed: {url!r}", kind="validation_error")
        if not _plain_base(url):
            msg = f"URL not allowed: {url!r} (https://host/path only)"
            raise HttpError(msg, kind="validation_error")
        return cls(
            base=url, name=name, timeout_s=timeout_s, error_reader=error_reader, caller_base=True
        )

    @property
    def host(self) -> str:
        """The one host this API's requests go to."""
        return urllib.parse.urlsplit(self.base).hostname or ""

    def get_json[T](
        self,
        *segments: str,
        parse: Callable[[dict[str, Any]], T],
        params: Params | None = None,
        missing: str | None = None,
        allow_empty: bool = False,
        empty_on_404: bool = False,
    ) -> T:
        """GET a JSON object from ``base/segment/...`` and read it with ``parse``."""
        ask = _Ask(missing=missing, empty={}, allow_empty=allow_empty, empty_on_404=empty_on_404)
        return _parsed(parse, _object(self._json_answer(segments, params, ask)))

    def get_json_list[T](
        self,
        *segments: str,
        parse: Callable[[list[Any]], T],
        params: Params | None = None,
        missing: str | None = None,
        allow_empty: bool = False,
        empty_on_404: bool = False,
    ) -> T:
        """GET a JSON array from ``base/segment/...`` and read it with ``parse``."""
        ask = _Ask(missing=missing, empty=[], allow_empty=allow_empty, empty_on_404=empty_on_404)
        return _parsed(parse, _array(self._json_answer(segments, params, ask)))

    def get_text[T](
        self,
        *segments: str,
        parse: Callable[[str], T],
        params: Params | None = None,
        missing: str | None = None,
    ) -> T:
        """GET a text body (XML, FASTA, a count) and read it with ``parse``."""
        _, text, _ = self._send(segments, params, _Ask(missing=missing))
        return _parsed(parse, text)

    def post_json[T](
        self,
        *segments: str,
        payload: Mapping[str, Any],
        parse: Callable[[dict[str, Any]], T],
        allow_empty: bool = False,
    ) -> T:
        """POST a JSON payload and read the JSON object back with ``parse``."""
        body = (json.dumps(payload).encode(), "application/json")
        ask = _Ask(empty={}, allow_empty=allow_empty, body=body)
        return _parsed(parse, _object(self._json_answer(segments, None, ask)))

    def post_form[T](
        self,
        *segments: str,
        form: Mapping[str, str],
        parse: Callable[[dict[str, Any]], T],
        allow_empty: bool = False,
    ) -> T:
        """POST a form and read the JSON object back with ``parse``."""
        body = (urllib.parse.urlencode(form).encode(), "application/x-www-form-urlencoded")
        ask = _Ask(empty={}, allow_empty=allow_empty, body=body)
        return _parsed(parse, _object(self._json_answer(segments, None, ask)))

    def _json_answer(self, segments: tuple[str, ...], params: Params | None, ask: _Ask) -> object:
        """The decoded JSON answer, unless it reports an error (``error_reader``).

        ``ask.empty`` is what an answer with nothing in it stands for where the request declares
        one: ``204 No Content`` or an empty body (``allow_empty``), a 404 (``empty_on_404``).
        """
        status, text, headers = self._send(segments, params, ask)
        if (ask.empty_on_404 and status == http.HTTPStatus.NOT_FOUND) or (
            ask.allow_empty and (status == http.HTTPStatus.NO_CONTENT or not text.strip())
        ):
            return ask.empty
        try:
            value = _json(text)
        except HttpError as not_json:
            reply = Reply(status=status, headers=headers, body=text[:_ERROR_BODY_CHARS])
            if (error := self._read(reply)) is not None:
                raise self._reported(error, reply) from not_json
            raise
        reply = Reply(status=status, headers=headers, body=value)
        # A reader that trips on a success's JSON is a parse error: nobody expected that shape.
        reported = None if self.error_reader is None else _parsed(self.error_reader, reply)
        if reported is not None and (not isinstance(reported, str) or reported.strip()):
            raise self._reported(reported, reply)
        return value

    def _read(self, reply: Reply) -> ToolFailure | str | None:
        """The error ``reply`` reports, read by ``error_reader``; a reader that trips on the body,
        or says nothing, reports none (the status or the parse error still says what happened)."""
        if self.error_reader is None:
            return None
        try:
            error = self.error_reader(reply)
        except _SHAPE_ERRORS:
            return None
        return None if isinstance(error, str) and not error.strip() else error

    def _reported(self, error: ToolFailure | str, reply: Reply) -> HttpError:
        """The failure a success reports (``error_reader``): its type, and the rest it asks for.

        A source that says it is rate limited in a success rests its host as a 429 does.
        """
        failure = self._failure(error, reply, _retry_after_value(reply.headers.get("retry-after")))
        self._cool(failure)
        return failure

    def _failure(
        self, error: ToolFailure | str, reply: Reply, retry_after_s: float | None = None
    ) -> HttpError:
        """The ``HttpError`` for an error the reader reported: its type, or the status's."""
        status = reply.status if reply.status >= 400 else None
        body = reply.body if isinstance(reply.body, str) else ""
        if isinstance(error, str):
            return HttpError(error, status=status, body=body, retry_after_s=retry_after_s)
        return HttpError(
            error.error.message,
            status=status,
            body=body,
            retry_after_s=retry_after_s,
            kind=cast("ToolFailureType", error.error.type),
            retryable=error.error.retryable,
        )

    def _cool(self, failure: HttpError) -> None:
        """After a 429, or a source's own word that it is rate limited, the host rests for the
        ``Retry-After`` or ``cooldown_s`` (D53)."""
        limited = failure.status == 429 or failure.error.type == "rate_limited"
        if limited and (wait := failure.retry_after_s or self.cooldown_s) > 0:
            _THROTTLE.cool(self.host, wait)

    def _send(
        self, segments: tuple[str, ...], params: Params | None, ask: _Ask
    ) -> tuple[int, str, Mapping[str, str]]:
        """Send one request: its answer's status, text and headers."""
        path = "".join(f"/{_segment(segment, self.segment_safe)}" for segment in segments)
        query = urllib.parse.urlencode(
            {**self.params, **(params or {})}, doseq=True, safe=self.query_safe
        )
        headers = {"User-Agent": USER_AGENT}
        if key := self._key():
            headers[self.key_header] = f"{self.key_prefix}{key}"
        elif self.key_required and self.key_env:
            where = f" (get one: {self.key_url})" if self.key_url else ""
            raise HttpError(f"no key: set {self.key_env}{where}.", kind="validation_error")
        data = None
        if ask.body is not None:
            data, headers["Content-Type"] = ask.body
        url = f"{self.base}{path}?{query}" if query else f"{self.base}{path}"
        request = urllib.request.Request(url, data=data, headers=headers)
        if (rest := _THROTTLE.resting(self.host)) > 0:
            msg = f"{self.name} asked to slow down: try again in {math.ceil(rest)} s."
            raise HttpError(msg, kind="rate_limited", retryable=True, retry_after_s=rest)
        _THROTTLE.wait(self.host, self.min_interval_s)
        try:
            status, raw, charset, complete, answer_headers = _fetch(
                request,
                timeout_s=self.timeout_s,
                max_bytes=self.max_bytes,
                failure=lambda reply, reason: self._status_failure(reply, reason, ask),
            )
        except _EmptyNotFound:
            return http.HTTPStatus.NOT_FOUND, "", {}
        except HttpError as error:
            self._cool(error)
            raise
        finally:
            _THROTTLE.done(self.host, self.min_interval_s)
        if not complete:
            raise HttpError(f"response larger than {self.max_bytes} bytes")
        text = _text(raw, charset)
        if self.billed_as is not None:
            bill(self.billed_as, self._units(text))
        return status, text, answer_headers

    def _units(self, text: str) -> int:
        """The units an accepted request was billed: what the answer says, or one."""
        if self.bill_units is None:
            return 1
        try:
            units = int(self.bill_units(text))
        except _SHAPE_ERRORS:
            return 1
        return max(units, 0)

    def _status_failure(self, reply: Reply, reason: str, ask: _Ask) -> HttpError:
        """The failure of an error status, by what the request declared and the source says.

        A 404 is ``not_found`` for a request that asks for a resource (``missing``), an empty
        answer for one that declares it (``empty_on_404``), and otherwise an endpoint that is not
        there. A typed failure from the reader is the answer; its text, or else the source's own
        error text, is what the source said, in the status's sentence.
        """
        status = reply.status
        full = reply.body if isinstance(reply.body, str) else ""
        body = full[:_ERROR_BODY_CHARS]
        retry_after = _retry_after_value(reply.headers.get("retry-after"))
        if status == http.HTTPStatus.NOT_FOUND and ask.missing is not None:
            return HttpError(ask.missing, status=status, body=body, kind="not_found")
        if status == http.HTTPStatus.NOT_FOUND and ask.empty_on_404:
            raise _EmptyNotFound
        decoded = Reply(status=status, headers=reply.headers, body=_decoded(full))
        error = self._read(decoded)
        if isinstance(error, ToolFailure):
            return self._failure(error, decoded, retry_after)
        source = error or _source_text(full)
        if status == http.HTTPStatus.TOO_MANY_REQUESTS:
            said = f" {self.name} said: {source}" if source else ""
            msg = f"rate limited by {self.name} (HTTP 429). Try again later.{said}"
            return HttpError(
                msg + self._key_hint(), status=status, body=body, retry_after_s=retry_after
            )
        if status == http.HTTPStatus.NOT_FOUND and self.caller_base:
            msg = f"nothing answers at {self.base} (HTTP 404); check the URL"
            return HttpError(
                f"{msg}: {source}" if source else msg,
                status=status,
                body=body,
                kind="validation_error",
            )
        if status == http.HTTPStatus.NOT_FOUND:
            msg = f"{self.name}: endpoint not found (HTTP 404); the API may have changed"
            return HttpError(f"{msg}: {source}" if source else msg, status=status, body=body)
        return HttpError(
            _status_text(status, source or reason),
            status=status,
            body=body,
            retry_after_s=retry_after,
        )

    def _key(self) -> str:
        """The API's optional key, from its environment variable; empty without one."""
        return os.environ.get(self.key_env, "").strip() if self.key_env else ""

    def _key_hint(self) -> str:
        """For an API that takes an optional key, unset: how to set one."""
        if not self.key_env or self._key():
            return ""
        where = f" (a free key: {self.key_url})" if self.key_url else ""
        return f" Set {self.key_env} to send a key of your own{where}."


@dataclass(frozen=True, slots=True)
class Page:
    """A fetched page's text; ``complete`` is ``False`` when it was cut at ``max_bytes``."""

    text: str
    complete: bool


def fetch_page(url: str, *, max_bytes: int, timeout_s: float = 10.0) -> Page:
    """GET any http(s) page a person approved; redirects stay on its host.

    For the ``dangerous`` web tools only: the host is the caller's, not a module's.

    Raises:
        HttpError: The URL is not http(s) with a host and no credentials, or the request failed.
    """
    parts = _split(url)
    if parts is None or not _web_origin(parts, _WEB_SCHEMES):
        msg = f"Invalid URL: {url!r}. Use an http:// or https:// URL without credentials."
        raise HttpError(msg, kind="validation_error")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    _, raw, charset, complete, _ = _fetch(
        request,
        timeout_s=timeout_s,
        max_bytes=max_bytes,
        failure=lambda reply, reason: HttpError(
            _status_text(reply.status, reason),
            status=reply.status,
            body=reply.body[:_ERROR_BODY_CHARS] if isinstance(reply.body, str) else "",
            retry_after_s=_retry_after_value(reply.headers.get("retry-after")),
        ),
    )
    return Page(_text(raw, charset), complete)
