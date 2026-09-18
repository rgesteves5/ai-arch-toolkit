"""Canned answers for the toolkit tools' one network seam, ``_http._open(request, timeout)``.

Patch ``HTTP_OPEN`` and give it ``respond(...)`` as a return value, or an ``http_error(...)``,
``urllib.error.URLError`` or ``TimeoutError`` as a side effect: the same things the real opener
returns and raises. The mock's ``call_args.args[0]`` is the ``urllib.request.Request`` sent.
"""

from __future__ import annotations

import email.message
import io
import json
import urllib.error
from typing import Any

HTTP_OPEN = "ai_arch_toolkit.toolkit.tools._http._open"


class FakeResponse:
    """What ``_http`` reads from a response: its headers and a bounded ``read1``."""

    def __init__(self, body: bytes, content_type: str) -> None:
        self._body = io.BytesIO(body)
        self.headers = email.message.Message()
        self.headers["Content-Type"] = content_type
        self.bytes_read = 0

    def read1(self, amt: int = -1) -> bytes:
        chunk = self._body.read1(amt)
        self.bytes_read += len(chunk)
        return chunk

    def close(self) -> None:
        self._body.close()

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def respond(
    data: dict[str, Any] | list[Any] | str | bytes = b"", *, content_type: str = ""
) -> FakeResponse:
    """A 200 answer: a dict or list as JSON, a str as UTF-8 text, bytes as they are."""
    if isinstance(data, dict | list):
        return FakeResponse(json.dumps(data).encode(), content_type or "application/json")
    if isinstance(data, str):
        return FakeResponse(data.encode(), content_type or "text/plain; charset=utf-8")
    return FakeResponse(data, content_type or "application/octet-stream")


def http_error(status: int, reason: str = "", *, body: bytes = b"") -> urllib.error.HTTPError:
    """The error the opener raises for a 4xx or 5xx answer."""
    return urllib.error.HTTPError(
        "https://api.example/", status, reason, email.message.Message(), io.BytesIO(body)
    )
