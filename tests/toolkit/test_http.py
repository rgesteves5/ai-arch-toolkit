"""The toolkit tools' only door to the network: ``toolkit/tools/_http.py``.

Requests run through the real ``urllib`` opener (redirect handling included); a canned transport
answers them by URL, so no socket is ever opened.
"""

from __future__ import annotations

import email.message
import http.client
import io
import json
import socket
import threading
import time
import urllib.error
import urllib.request
import urllib.response
from collections.abc import Callable
from typing import Any

import pytest

from ai_arch_toolkit.toolkit.tools import _http
from ai_arch_toolkit.toolkit.tools._http import Api, HttpError, fetch_page

type Route = tuple[int, dict[str, str], bytes | io.RawIOBase]  # status, headers, body


class _Transport(urllib.request.BaseHandler):
    """Answers the real opener from canned routes, before the socket handlers are tried."""

    handler_order = 50

    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.seen: list[urllib.request.Request] = []

    def add(
        self,
        url: str,
        body: dict[str, Any] | list[Any] | str | bytes | io.RawIOBase = b"",
        *,
        status: int = 200,
        **headers: str,
    ) -> None:
        if isinstance(body, dict | list):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode()
        headers.setdefault("Content-Type", "application/json; charset=utf-8")
        self.routes[url] = (status, headers, body)

    def _serve(self, request: urllib.request.Request) -> urllib.response.addinfourl:
        self.seen.append(request)
        status, headers, body = self.routes[request.full_url]
        message = email.message.Message()
        for key, value in headers.items():
            message[key] = value
        stream = io.BytesIO(body) if isinstance(body, bytes) else body
        response = urllib.response.addinfourl(stream, message, request.full_url, status)
        response.msg = http.HTTPStatus(status).phrase  # what urllib's error processor reads
        return response

    https_open = _serve
    http_open = _serve


class _SlowBody(io.RawIOBase):
    """A body that trickles one byte per read, forever."""

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        time.sleep(0.02)
        return b"x"

    read1 = read


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> _Transport:
    transport = _Transport()
    monkeypatch.setattr(_http, "_OPENER", _http._build_opener(transport))
    return transport


API = Api(base="https://api.example.org/v1", name="Example")


class TestApiDeclaration:
    @pytest.mark.parametrize(
        "base",
        [
            "http://api.example.org/v1",
            "ftp://api.example.org/v1",
            "https://user:pw@api.example.org/v1",
            "https://api.example.org:8443/v1",
            "https://api.example.org/v1?key=x",
            "https://api.example.org/v1#top",
            "https://api.example.org/v1/",
            "https:///v1",
        ],
    )
    def test_a_base_that_is_not_a_plain_https_origin_and_path_is_refused(self, base: str) -> None:
        with pytest.raises(ValueError, match="base"):
            Api(base=base, name="Example")

    def test_the_host_is_the_bases(self) -> None:
        assert API.host == "api.example.org"


class TestRequests:
    def test_segments_are_quoted_one_by_one_and_params_encoded(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/works/10.1000%2Fxyz%3Fa%23b?q=a+b&page=2", {"ok": 1})

        data = API.get_json("works", "10.1000/xyz?a#b", params={"q": "a b", "page": 2}, parse=dict)

        assert data == {"ok": 1}

    def test_declared_safe_characters_stay_raw(self, web: _Transport) -> None:
        api = Api(
            base="https://api.example.org",
            name="Example",
            segment_safe=";:",
            query_safe="'(",
            params={"format": "json"},
        )
        web.add("https://api.example.org/country/PT;ES/DOI:10.1?format=json&f=a('b", {})

        api.get_json("country", "PT;ES", "DOI:10.1", params={"f": "a('b"}, parse=dict)

        assert web.seen[0].full_url.endswith("/country/PT;ES/DOI:10.1?format=json&f=a('b")

    def test_list_params_repeat_the_key(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/s?fl%5B%5D=a&fl%5B%5D=b", {})

        API.get_json("s", params={"fl[]": ["a", "b"]}, parse=dict)

    @pytest.mark.parametrize("segment", ["", ".", ".."])
    def test_dot_and_empty_segments_are_refused_before_any_request(
        self, web: _Transport, segment: str
    ) -> None:
        with pytest.raises(HttpError, match="invalid path segment"):
            API.get_json("works", segment, parse=dict)
        assert web.seen == []

    def test_every_request_names_the_toolkit(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/a", {})

        API.get_json("a", parse=dict)

        assert web.seen[0].get_header("User-agent") == _http.USER_AGENT

    def test_post_json_sends_the_payload(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/query", {"n": 1})

        assert API.post_json("query", payload={"q": [1]}, parse=dict) == {"n": 1}

        request = web.seen[0]
        assert request.get_method() == "POST"
        assert request.data == b'{"q": [1]}'
        assert request.get_header("Content-type") == "application/json"

    def test_post_form_sends_an_encoded_form(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/interpreter", {"elements": []})

        API.post_form("interpreter", form={"data": "[out:json];"}, parse=dict)

        request = web.seen[0]
        assert request.data == b"data=%5Bout%3Ajson%5D%3B"
        assert request.get_header("Content-type") == "application/x-www-form-urlencoded"


class TestBodies:
    def test_text_is_decoded_with_the_declared_charset(self, web: _Transport) -> None:
        web.add(
            "https://api.example.org/v1/t",
            "café".encode("latin-1"),
            **{"Content-Type": "text/plain; charset=latin-1"},
        )

        assert API.get_text("t", parse=str) == "café"

    def test_undecodable_bytes_are_replaced(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/t", b"\xff\xfe\x00ok")

        assert API.get_text("t", parse=str).endswith("ok")

    def test_an_unknown_charset_is_an_error(self, web: _Transport) -> None:
        web.add(
            "https://api.example.org/v1/t", b"x", **{"Content-Type": "text/plain; charset=nope"}
        )

        with pytest.raises(HttpError, match="charset"):
            API.get_text("t", parse=str)

    @pytest.mark.parametrize(
        ("body", "reason"),
        [
            (b"not json", "could not parse API response"),
            (b"[" * 100_000, "could not parse API response"),
            (b"[1, 2]", "expected a JSON object"),
            (b"null", "expected a JSON object"),
        ],
    )
    def test_json_that_is_not_an_object_is_an_error(
        self, web: _Transport, body: bytes, reason: str
    ) -> None:
        web.add("https://api.example.org/v1/j", body)

        with pytest.raises(HttpError, match=reason):
            API.get_json("j", parse=dict)

    def test_a_json_array_is_read_as_a_list(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/j", [1, 2])

        assert API.get_json_list("j", parse=list) == [1, 2]
        with pytest.raises(HttpError, match="expected a JSON array"):
            web.add("https://api.example.org/v1/o", {"a": 1})
            API.get_json_list("o", parse=list)

    def test_a_body_over_max_bytes_is_an_error(self, web: _Transport) -> None:
        api = Api(base="https://api.example.org/v1", name="Example", max_bytes=1000)
        web.add("https://api.example.org/v1/big", b"x" * 1001)

        with pytest.raises(HttpError, match="larger than 1000 bytes"):
            api.get_text("big", parse=str)

    def test_the_deadline_covers_a_body_that_never_ends(self, web: _Transport) -> None:
        api = Api(base="https://api.example.org/v1", name="Example", timeout_s=0.1)
        web.add("https://api.example.org/v1/slow", _SlowBody())

        started = time.monotonic()
        with pytest.raises(HttpError, match="request timed out"):
            api.get_text("slow", parse=str)
        assert time.monotonic() - started < 1.0


class TestParse:
    @pytest.mark.parametrize(
        "body", [{"message": None}, {"message": {"items": "x"}}, {"message": {"items": [None]}}]
    )
    def test_a_shape_the_reader_did_not_expect_is_an_error(
        self, web: _Transport, body: dict[str, Any]
    ) -> None:
        web.add("https://api.example.org/v1/works", body)

        def titles(data: dict[str, Any]) -> list[str]:
            return [item["title"] for item in data["message"]["items"]]

        with pytest.raises(HttpError, match="could not parse API response"):
            API.get_json("works", parse=titles)

    def test_a_bug_that_is_not_about_the_shape_still_raises(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/works", {})

        def broken(data: dict[str, Any]) -> None:
            raise NotImplementedError

        with pytest.raises(NotImplementedError):
            API.get_json("works", parse=broken)


class TestErrors:
    def test_a_status_error_keeps_the_reason_status_and_body(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/x", b'{"reason": "bad day"}', status=500)

        with pytest.raises(HttpError) as caught:
            API.get_json("x", parse=dict)

        assert str(caught.value) == "HTTP error 500: Internal Server Error"
        assert caught.value.status == 500
        assert json.loads(caught.value.body) == {"reason": "bad day"}

    def test_429_names_the_api(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/x", b"", status=429)

        with pytest.raises(HttpError, match=r"^rate limited by Example \(HTTP 429\)"):
            API.get_json("x", parse=dict)

    def test_declared_status_messages_win(self, web: _Transport) -> None:
        api = Api(
            base="https://api.example.org/v1",
            name="Example",
            status_messages={404: "no matching records found."},
        )
        web.add("https://api.example.org/v1/x", b"", status=404)

        with pytest.raises(HttpError, match=r"^no matching records found\.$"):
            api.get_json("x", parse=dict)

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            (urllib.error.URLError("offline"), "URL error: offline"),
            (TimeoutError(), "request timed out."),
            (ConnectionResetError("reset by peer"), "network error: reset by peer"),
            (http.client.IncompleteRead(b"x"), "network error: IncompleteRead"),
        ],
    )
    def test_network_failures_become_one_error(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception, message: str
    ) -> None:
        def fail(request: urllib.request.Request, timeout: float) -> Any:
            raise error

        monkeypatch.setattr(_http, "_open", fail)

        with pytest.raises(HttpError) as caught:
            API.get_json("x", parse=dict)

        assert str(caught.value).startswith(message)
        assert caught.value.status is None


class TestRedirects:
    def test_a_redirect_on_the_same_host_is_followed(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/old", b"", status=301, Location="/v1/new")
        web.add("https://api.example.org/v1/new", {"moved": True})

        assert API.get_json("old", parse=dict) == {"moved": True}

    @pytest.mark.parametrize(
        "target", ["https://169.254.169.254/latest", "http://api.example.org/v1/new"]
    )
    def test_a_redirect_to_another_host_or_down_to_http_is_refused(
        self, web: _Transport, target: str
    ) -> None:
        web.add("https://api.example.org/v1/old", b"", status=302, Location=target)

        with pytest.raises(HttpError, match="refused a redirect to"):
            API.get_json("old", parse=dict)

        assert [request.full_url for request in web.seen] == ["https://api.example.org/v1/old"]


class TestWithin:
    DOMAINS = frozenset({"wikipedia.org", "wikimedia.org"})

    @pytest.mark.parametrize(
        "url", ["https://en.wikipedia.org/w/api.php", "https://wikipedia.org/w/api.php"]
    )
    def test_a_host_in_the_modules_domains_is_accepted(self, url: str) -> None:
        assert Api.within(url, self.DOMAINS, name="MediaWiki").base == url

    @pytest.mark.parametrize(
        "url",
        [
            "https://169.254.169.254/api.php",
            "https://evilwikipedia.org/api.php",
            "https://wikipedia.org.evil.example/api.php",
            "https://localhost:8443/x/api.php",
            "https://user:pw@en.wikipedia.org/w/api.php",
            "http://en.wikipedia.org/w/api.php",
        ],
    )
    def test_any_other_origin_is_refused(self, url: str) -> None:
        with pytest.raises(HttpError, match="not allowed"):
            Api.within(url, self.DOMAINS, name="MediaWiki")


class TestFetchPage:
    def test_reads_http_and_https_pages(self, web: _Transport) -> None:
        web.add("http://example.com/", "<p>hi</p>", **{"Content-Type": "text/html"})

        page = fetch_page("http://example.com/", max_bytes=1000)

        assert page.text == "<p>hi</p>"
        assert page.complete

    def test_a_page_over_max_bytes_comes_back_cut(self, web: _Transport) -> None:
        web.add("https://example.com/big", b"a" * 5000, **{"Content-Type": "text/plain"})

        page = fetch_page("https://example.com/big", max_bytes=100)

        assert page.text == "a" * 100
        assert not page.complete

    @pytest.mark.parametrize(
        "url", ["file:///etc/passwd", "ftp://example.com/", "https://u:p@example.com/", "example"]
    )
    def test_only_plain_web_urls_are_fetched(self, web: _Transport, url: str) -> None:
        with pytest.raises(HttpError, match="URL"):
            fetch_page(url, max_bytes=100)
        assert web.seen == []

    def test_a_redirect_to_another_host_is_refused_so_approval_covers_one_url(
        self, web: _Transport
    ) -> None:
        web.add("https://example.com/", b"", status=302, Location="http://169.254.169.254/")

        with pytest.raises(HttpError, match=r"refused a redirect to http://169\.254\.169\.254/"):
            fetch_page("https://example.com/", max_bytes=100)

        assert [request.full_url for request in web.seen] == ["https://example.com/"]

    def test_an_upgrade_to_https_on_the_same_host_is_followed(self, web: _Transport) -> None:
        web.add("http://example.com/", b"", status=301, Location="https://example.com/")
        web.add("https://example.com/", "secure", **{"Content-Type": "text/plain"})

        assert fetch_page("http://example.com/", max_bytes=100).text == "secure"


class TestThrottle:
    def _throttle(self) -> tuple[_http._Throttle, list[float], Callable[[float], None]]:
        now = [100.0]
        slept: list[float] = []

        def advance(seconds: float) -> None:
            now[0] += seconds

        throttle = _http._Throttle(sleep=slept.append, clock=lambda: now[0])
        return throttle, slept, advance

    def test_requests_to_one_host_are_spaced_by_the_interval(self) -> None:
        throttle, slept, advance = self._throttle()

        throttle.wait("api.example.org", 1.0)
        advance(0.25)
        throttle.wait("api.example.org", 1.0)
        throttle.wait("other.example.org", 1.0)
        throttle.wait("api.example.org", 0.0)

        assert slept == [0.0, 0.75, 0.0]

    def test_parallel_callers_each_get_their_own_slot(self) -> None:
        slept: list[float] = []
        throttle = _http._Throttle(sleep=slept.append)
        workers = [
            threading.Thread(target=throttle.wait, args=("api.example.org", 10.0))
            for _ in range(5)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        assert [round(seconds, -1) for seconds in sorted(slept)] == [0, 10, 20, 30, 40]


def test_a_toolkit_test_cannot_reach_the_network_whatever_else_runs_with_it() -> None:
    # Guards the root conftest: pytest drops a directory conftest's fixtures when the command line
    # interleaves this directory's files with others' (pytest 9.1, measured in R03).
    with pytest.raises(OSError, match="network access is blocked"):
        socket.getaddrinfo("localhost", 443)  # resolved locally even if the guard were gone
