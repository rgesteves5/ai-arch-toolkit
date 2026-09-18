"""The toolkit tools' only door to the network.

Every request a tool in ``toolkit.tools`` sends goes through here: HTTPS to the host its module
declared, redirects kept on that host, a body read bounded in bytes and in time, path segments
quoted one by one, a rate-limit clock shared across threads, and one error type whose text is ready
for the model. ``urllib.request``, ``urllib.error``, ``http.client``, ``socket`` and ``ssl`` are
imported nowhere else in the package (architecture test).
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from email.message import Message
from typing import IO, Any, Protocol

USER_AGENT = "ai-arch-toolkit/1.0 (https://github.com/ai-arch-toolkit)"

type ParamValue = str | int | float | Sequence[str]
type Params = Mapping[str, ParamValue]

_WEB_SCHEMES = frozenset({"http", "https"})
_DOT_SEGMENTS = frozenset({"", ".", ".."})
_CHUNK_BYTES = 64 * 1024
_ERROR_BODY_CHARS = 2000


class HttpError(Exception):
    """A request that produced no usable body. ``str()`` is the reason shown to the model.

    Attributes:
        status: The HTTP status of an error response; ``None`` when no response arrived.
        body: The start of an error response's body, for APIs that explain errors there.
    """

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class _Response(Protocol):
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


def _build_opener(*handlers: urllib.request.BaseHandler) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_SameHostRedirect(), *handlers)


_OPENER = _build_opener()


def _open(request: urllib.request.Request, timeout: float) -> _Response:
    """Send ``request``: the one call in the package that reaches the network."""
    return _OPENER.open(request, timeout=timeout)


class _Throttle:
    """Spaces requests that share a clock: one clock per (host, interval), across threads."""

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
        raise HttpError(f"invalid path segment: {segment!r}")
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
    return raw.decode("utf-8", errors="replace")[:_ERROR_BODY_CHARS]


def _body(response: _Response, deadline: float, max_bytes: int) -> tuple[bytes, bool]:
    """Read at most ``max_bytes``; ``False`` when the body went on. Past the deadline, give up.

    ``read1`` makes one read of the socket, so a body that trickles in cannot hold a call past the
    deadline by more than one socket timeout.
    """
    chunks: list[bytes] = []
    size = 0
    while size <= max_bytes:
        if time.monotonic() > deadline:
            raise HttpError("request timed out.")
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
    describe: Callable[[int, str], str],
) -> tuple[bytes, str, bool]:
    """Send ``request`` and read its body: ``(body, charset, complete)``, or ``HttpError``."""
    deadline = time.monotonic() + timeout_s
    try:
        response = _open(request, timeout_s)
        try:
            body, complete = _body(response, deadline, max_bytes)
            charset = response.headers.get_content_charset() or "utf-8"
        finally:
            response.close()
    except urllib.error.HTTPError as error:
        status, body = error.code, _error_body(error)
        raise HttpError(describe(status, str(error.reason)), status=status, body=body) from error
    except _Redirected as refused:
        target = refused.target
        raise HttpError(f"refused a redirect to {target} (only same-host HTTPS)") from refused
    except urllib.error.URLError as error:
        raise HttpError(f"URL error: {error.reason}") from error
    except TimeoutError as error:
        raise HttpError("request timed out.") from error
    except (OSError, http.client.HTTPException) as error:
        raise HttpError(f"network error: {str(error) or type(error).__name__}") from error
    return body, charset, complete


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
class Api:
    """One upstream HTTPS API: requests go to its host and under its base path, nowhere else.

    Every request takes the function that reads its answer (``parse``), and whatever that
    function raises on a shape it did not expect becomes an ``HttpError``: no raw response leaves
    this module unguarded, so a tool cannot crash on a body it did not foresee.

    Attributes:
        base: ``https://host/path`` without credentials, port, query, fragment or final slash.
        name: The API's name, for the rate-limit message.
        timeout_s: Deadline of one request, checked between reads of the body.
        max_bytes: Largest body read; a longer one is an error.
        min_interval_s: Least time between two requests. Requests to one host with one interval
            share a clock, across modules and threads.
        params: Query parameters sent with every request.
        segment_safe: Characters left raw in path segments.
        query_safe: Characters left raw in the query string.
        status_messages: The text shown for particular HTTP statuses.
    """

    base: str
    name: str
    timeout_s: float = 10.0
    max_bytes: int = 10_000_000
    min_interval_s: float = 0.0
    params: Mapping[str, str] = field(default_factory=dict)
    segment_safe: str = ""
    query_safe: str = ""
    status_messages: Mapping[int, str] = field(default_factory=dict)

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
        status_messages: Mapping[int, str] | None = None,
    ) -> Api:
        """An ``Api`` at ``url``, a URL that came from outside the module.

        Raises:
            HttpError: The host is not one of ``domains`` or a subdomain of one, or the URL is not
                a plain ``https://host/path``.
        """
        parts = _split(url)
        host = (parts.hostname or "") if parts is not None else ""
        if not any(host == domain or host.endswith(f".{domain}") for domain in domains):
            raise HttpError(f"host not allowed: {url!r}")
        if not _plain_base(url):
            raise HttpError(f"URL not allowed: {url!r} (https://host/path only)")
        return cls(base=url, name=name, timeout_s=timeout_s, status_messages=status_messages or {})

    @property
    def host(self) -> str:
        """The one host this API's requests go to."""
        return urllib.parse.urlsplit(self.base).hostname or ""

    def get_json[T](
        self, *segments: str, parse: Callable[[dict[str, Any]], T], params: Params | None = None
    ) -> T:
        """GET a JSON object from ``base/segment/...`` and read it with ``parse``."""
        return _parsed(parse, _object(_json(self._send(segments, params))))

    def get_json_list[T](
        self, *segments: str, parse: Callable[[list[Any]], T], params: Params | None = None
    ) -> T:
        """GET a JSON array from ``base/segment/...`` and read it with ``parse``."""
        return _parsed(parse, _array(_json(self._send(segments, params))))

    def get_text[T](
        self, *segments: str, parse: Callable[[str], T], params: Params | None = None
    ) -> T:
        """GET a text body (XML, FASTA, a count) and read it with ``parse``."""
        return _parsed(parse, self._send(segments, params))

    def post_json[T](
        self, *segments: str, payload: Mapping[str, Any], parse: Callable[[dict[str, Any]], T]
    ) -> T:
        """POST a JSON payload and read the JSON object back with ``parse``."""
        body = (json.dumps(payload).encode(), "application/json")
        return _parsed(parse, _object(_json(self._send(segments, None, body))))

    def post_form[T](
        self, *segments: str, form: Mapping[str, str], parse: Callable[[dict[str, Any]], T]
    ) -> T:
        """POST a form and read the JSON object back with ``parse``."""
        body = (urllib.parse.urlencode(form).encode(), "application/x-www-form-urlencoded")
        return _parsed(parse, _object(_json(self._send(segments, None, body))))

    def _send(
        self,
        segments: tuple[str, ...],
        params: Params | None,
        body: tuple[bytes, str] | None = None,
    ) -> str:
        path = "".join(f"/{_segment(segment, self.segment_safe)}" for segment in segments)
        query = urllib.parse.urlencode(
            {**self.params, **(params or {})}, doseq=True, safe=self.query_safe
        )
        headers = {"User-Agent": USER_AGENT}
        data = None
        if body is not None:
            data, headers["Content-Type"] = body
        url = f"{self.base}{path}?{query}" if query else f"{self.base}{path}"
        request = urllib.request.Request(url, data=data, headers=headers)
        _THROTTLE.wait(self.host, self.min_interval_s)
        raw, charset, complete = _fetch(
            request, timeout_s=self.timeout_s, max_bytes=self.max_bytes, describe=self._describe
        )
        if not complete:
            raise HttpError(f"response larger than {self.max_bytes} bytes")
        return _text(raw, charset)

    def _describe(self, status: int, reason: str) -> str:
        if status in self.status_messages:
            return self.status_messages[status]
        if status == 429:
            return f"rate limited by {self.name} (HTTP 429). Try again later."
        return _status_text(status, reason)


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
        raise HttpError(msg)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    raw, charset, complete = _fetch(
        request, timeout_s=timeout_s, max_bytes=max_bytes, describe=_status_text
    )
    return Page(_text(raw, charset), complete)
