"""The toolkit tools' only door to the network: ``toolkit/tools/_http.py``.

Requests run through the real ``urllib`` opener (redirect handling included); a canned transport
answers them by URL, so no socket is ever opened.
"""

from __future__ import annotations

import email.message
import http.client
import importlib.metadata
import io
import json
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
import urllib.response
from collections.abc import Callable
from typing import Any

import pytest

from ai_arch_toolkit.core import ToolFailure
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

    def test_the_user_agent_gives_the_packages_version_and_its_repository(self) -> None:
        version = importlib.metadata.version("ai-arch-toolkit")

        assert (
            f"ai-arch-toolkit/{version} (+https://github.com/rgesteves5/ai-arch-toolkit)"
        ) == _http.USER_AGENT

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

    @pytest.mark.parametrize(
        ("status", "kind", "retryable"),
        [(429, "rate_limited", True), (503, "upstream", True), (404, "upstream", False)],
    )
    def test_a_status_error_is_a_typed_tool_failure(
        self, web: _Transport, status: int, kind: str, retryable: bool
    ) -> None:
        web.add("https://api.example.org/v1/x", b"", status=status)

        with pytest.raises(ToolFailure) as caught:
            API.get_json("x", parse=dict)

        assert (caught.value.error.type, caught.value.error.retryable) == (kind, retryable)
        assert caught.value.error.details["status"] == status

    def test_a_timeout_is_a_retryable_upstream_failure(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/x", _SlowBody())
        api = Api(base="https://api.example.org/v1", name="Example", timeout_s=0.05)

        with pytest.raises(HttpError, match="timed out") as caught:
            api.get_json("x", parse=dict)

        assert (caught.value.error.type, caught.value.error.retryable) == ("upstream", True)

    def test_a_url_the_module_may_not_reach_is_a_validation_error(self) -> None:
        with pytest.raises(HttpError) as caught:
            Api.within("https://evil.example.com/x", ("example.org",), name="Example")

        assert caught.value.error.type == "validation_error"

    def test_an_unsendable_url_is_a_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(request: urllib.request.Request, timeout: float) -> Any:
            raise http.client.InvalidURL("URL can't contain control characters")

        monkeypatch.setattr(_http, "_open", refuse)

        with pytest.raises(HttpError) as caught:
            fetch_page("https://example.org/a b", max_bytes=100)

        assert (caught.value.error.type, caught.value.error.retryable) == (
            "validation_error",
            False,
        )

    def test_retry_after_reaches_the_details(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/x", b"", status=503, **{"Retry-After": "7"})

        with pytest.raises(HttpError) as caught:
            API.get_json("x", parse=dict)

        assert caught.value.error.details == {"status": 503, "retry_after_s": 7.0}

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


def _reported(data: object) -> str | None:
    """The example API's own error: ``{"error": text}``, or ``[{"error": text}]``."""
    first = data[0] if isinstance(data, list) and data else data
    return first.get("error") if isinstance(first, dict) else None


REPORTING = Api(base="https://api.example.org/v1", name="Example", body_error=_reported)


def _refuse(data: object) -> None:
    raise AssertionError(f"parse ran on {data!r}")


def _said(answer: object) -> str | None:
    """An API's error sent as text in place of the JSON."""
    return answer.strip() if isinstance(answer, str) else None


def _length(answer: object) -> str | None:
    return f"{len(answer)} characters" if isinstance(answer, str) else None


_HTML = {"Content-Type": "text/html; charset=utf-8"}


class TestBodyErrors:
    """An API that answers errors with a success status declares how to read them."""

    @pytest.mark.parametrize(
        ("method", "kwargs", "body"),
        [
            ("get_json", {}, {"error": "no such thing"}),
            ("get_json_list", {}, [{"error": "no such thing"}]),
            ("post_json", {"payload": {}}, {"error": "no such thing"}),
            ("post_form", {"form": {}}, {"error": "no such thing"}),
        ],
    )
    def test_a_reported_error_is_raised_before_parse_runs(
        self, web: _Transport, method: str, kwargs: dict[str, Any], body: object
    ) -> None:
        web.add("https://api.example.org/v1/x", body)

        with pytest.raises(HttpError) as caught:
            getattr(REPORTING, method)("x", parse=_refuse, **kwargs)

        assert str(caught.value) == "no such thing"
        assert caught.value.status is None

    def test_an_answer_that_reports_no_error_reaches_parse(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/x", {"items": [1]})

        assert REPORTING.get_json("x", parse=dict) == {"items": [1]}

    def test_the_reader_sees_the_answer_before_its_kind_is_checked(self, web: _Transport) -> None:
        # An error object where the API usually sends an array is still the API's error.
        web.add("https://api.example.org/v1/x", {"error": "no such list"})

        with pytest.raises(HttpError, match=r"^no such list$"):
            REPORTING.get_json_list("x", parse=_refuse)

    def test_a_text_answer_is_left_to_its_parse(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/t", '{"error": "just text"}')

        assert REPORTING.get_text("t", parse=str) == '{"error": "just text"}'

    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("get_json", {}),
            ("get_json_list", {}),
            ("post_json", {"payload": {}}),
            ("post_form", {"form": {}}),
        ],
    )
    def test_a_body_that_is_not_json_reaches_the_reader_as_its_text(
        self, web: _Transport, method: str, kwargs: dict[str, Any]
    ) -> None:
        # Some APIs send an error as text in place of the JSON, with a success status.
        api = Api(base="https://api.example.org/v1", name="Example", body_error=_said)
        web.add("https://api.example.org/v1/x", "Query too short.\n", **_HTML)

        with pytest.raises(HttpError) as caught:
            getattr(api, method)("x", parse=_refuse, **kwargs)

        assert str(caught.value) == "Query too short."
        assert caught.value.status is None

    def test_a_body_that_is_not_json_and_reports_no_error_is_a_parse_error(
        self, web: _Transport
    ) -> None:
        web.add("https://api.example.org/v1/x", "not json", **_HTML)

        with pytest.raises(HttpError, match=r"^could not parse API response: Expecting value"):
            REPORTING.get_json("x", parse=_refuse)

    def test_the_reader_gets_only_the_start_of_a_body_that_is_not_json(
        self, web: _Transport
    ) -> None:
        api = Api(base="https://api.example.org/v1", name="Example", body_error=_length)
        web.add("https://api.example.org/v1/x", "x" * 100_000, **_HTML)

        with pytest.raises(HttpError, match=rf"^{_http._ERROR_BODY_CHARS} characters$"):
            api.get_json("x", parse=_refuse)

    def test_a_reader_that_trips_on_the_shape_is_a_parse_error(self, web: _Transport) -> None:
        def strict(data: Any) -> str | None:
            return data["error"]

        api = Api(base="https://api.example.org/v1", name="Example", body_error=strict)
        web.add("https://api.example.org/v1/x", {"items": []})

        with pytest.raises(HttpError, match="could not parse API response: KeyError"):
            api.get_json("x", parse=dict)

    def test_within_keeps_the_reader(self) -> None:
        api = Api.within(
            "https://en.wikipedia.org/w/api.php",
            {"wikipedia.org"},
            name="MediaWiki",
            body_error=_reported,
        )

        assert api.body_error is _reported


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


class TestNoContent:
    """A source that answers "nothing found" with no content is declared per request (RCSB)."""

    @pytest.mark.parametrize(
        ("method", "kwargs", "empty"),
        [
            ("get_json", {}, {}),
            ("get_json_list", {}, []),
            ("post_json", {"payload": {}}, {}),
            ("post_form", {"form": {}}, {}),
        ],
    )
    @pytest.mark.parametrize(("status", "body"), [(204, b""), (200, b""), (200, b" \n")])
    def test_a_request_that_allows_an_empty_answer_reads_it_as_empty(
        self,
        web: _Transport,
        method: str,
        kwargs: dict[str, Any],
        empty: object,
        status: int,
        body: bytes,
    ) -> None:
        web.add("https://api.example.org/v1/x", body, status=status)

        call = getattr(REPORTING, method)
        assert call("x", parse=lambda data: data, allow_empty=True, **kwargs) == empty

    @pytest.mark.parametrize("status", [204, 200])
    def test_elsewhere_an_empty_answer_is_a_parse_error(
        self, web: _Transport, status: int
    ) -> None:
        web.add("https://api.example.org/v1/x", b"", status=status)

        with pytest.raises(HttpError, match=r"^could not parse API response"):
            API.get_json("x", parse=dict)

    def test_an_answer_with_content_is_read_as_usual(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/x", {"error": "no such thing"})

        with pytest.raises(HttpError, match=r"^no such thing$"):
            REPORTING.get_json("x", parse=_refuse, allow_empty=True)


class TestErrorBodies:
    """The ``body_error`` that reads a success also explains an error status in the API's words."""

    @pytest.mark.parametrize("method", ["get_json", "get_json_list", "get_text"])
    def test_an_error_status_is_explained_in_the_apis_words(
        self, web: _Transport, method: str
    ) -> None:
        web.add("https://api.example.org/v1/x", {"error": "no such dataset"}, status=404)

        with pytest.raises(HttpError) as caught:
            getattr(REPORTING, method)("x", parse=_refuse)

        assert str(caught.value) == "HTTP error 404: no such dataset"
        assert caught.value.status == 404
        assert json.loads(caught.value.body) == {"error": "no such dataset"}

    def test_the_apis_words_come_before_a_status_message(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/x", {"error": "no such dataset"}, status=404)

        with pytest.raises(HttpError, match=r"^HTTP error 404: no such dataset$"):
            _STATUS_MESSAGES.get_json("x", parse=_refuse)

    @pytest.mark.parametrize(
        ("status", "message"),
        [
            (404, "no matching records found."),
            (429, "rate limited by Example (HTTP 429). Try again later."),
            (500, "HTTP error 500: Internal Server Error"),
        ],
    )
    def test_a_body_that_explains_nothing_keeps_the_status_text(
        self, web: _Transport, status: int, message: str
    ) -> None:
        web.add("https://api.example.org/v1/x", "<html>Server Error</html>", status=status)

        with pytest.raises(HttpError) as caught:
            _STATUS_MESSAGES.get_json("x", parse=_refuse)

        assert str(caught.value) == message

    def test_a_reader_that_trips_on_an_error_body_keeps_the_status_text(
        self, web: _Transport
    ) -> None:
        def strict(data: Any) -> str | None:
            return data["error"]

        api = Api(base="https://api.example.org/v1", name="Example", body_error=strict)
        web.add("https://api.example.org/v1/x", {"items": []}, status=500)

        with pytest.raises(HttpError, match=r"^HTTP error 500: Internal Server Error$"):
            api.get_json("x", parse=_refuse)


_STATUS_MESSAGES = Api(
    base="https://api.example.org/v1",
    name="Example",
    status_messages={404: "no matching records found."},
    body_error=_reported,
)


# --- D51: TLS with the system's certificate store, when truststore is installed ------------------

_UNVERIFIED = ssl.SSLCertVerificationError(
    1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer"
)


class TestTls:
    def test_with_truststore_installed_the_systems_store_verifies(self) -> None:
        import truststore

        context = _http._tls_context()

        assert isinstance(context, truststore.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_without_truststore_the_standard_context_verifies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "truststore", None)  # the import raises ImportError

        context = _http._tls_context()

        assert type(context) is ssl.SSLContext
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_the_opener_verifies_with_that_context(self) -> None:
        handlers = [
            h for h in _http._OPENER.handlers if isinstance(h, urllib.request.HTTPSHandler)
        ]

        assert [handler._context for handler in handlers] == [_http._TLS]  # type: ignore[attr-defined]

    def _unverified(self, monkeypatch: pytest.MonkeyPatch) -> str:
        def fail(request: urllib.request.Request, timeout: float) -> Any:
            raise urllib.error.URLError(_UNVERIFIED)

        monkeypatch.setattr(_http, "_open", fail)
        with pytest.raises(HttpError) as caught:
            API.get_json("x", parse=dict)
        return str(caught.value)

    def test_a_certificate_the_standard_store_cannot_verify_says_how_to_fix_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_http, "_SYSTEM_STORE", False)

        message = self._unverified(monkeypatch)

        assert message.startswith("URL error: [SSL: CERTIFICATE_VERIFY_FAILED]")
        assert "install truststore: pip install truststore" in message

    def test_with_the_systems_store_the_error_stays_as_it_is(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_http, "_SYSTEM_STORE", True)

        assert "truststore" not in self._unverified(monkeypatch)


# --- D52: an optional key from the environment ---------------------------------------------------

KEYED = Api(
    base="https://api.example.org/v1",
    name="Example",
    key_env="EXAMPLE_API_KEY",
    key_header="x-api-key",
    key_url="https://example.org/key",
)


class TestKeys:
    def test_a_declared_key_goes_in_its_header(
        self, web: _Transport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXAMPLE_API_KEY", "k-123")
        web.add("https://api.example.org/v1/a", {})

        KEYED.get_json("a", parse=dict)

        assert web.seen[0].get_header("X-api-key") == "k-123"

    def test_without_the_key_no_header_goes(
        self, web: _Transport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EXAMPLE_API_KEY", raising=False)
        web.add("https://api.example.org/v1/a", {})

        KEYED.get_json("a", parse=dict)

        assert web.seen[0].get_header("X-api-key") is None

    def test_a_429_without_the_key_says_how_to_get_one(
        self, web: _Transport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EXAMPLE_API_KEY", raising=False)
        web.add("https://api.example.org/v1/x", b"", status=429)

        with pytest.raises(HttpError) as caught:
            KEYED.get_json("x", parse=dict)

        message = str(caught.value)
        assert message.startswith("rate limited by Example (HTTP 429)")
        assert "EXAMPLE_API_KEY" in message
        assert "https://example.org/key" in message

    def test_a_429_with_the_key_asks_for_none(
        self, web: _Transport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXAMPLE_API_KEY", "k-123")
        web.add("https://api.example.org/v1/x", b"", status=429)

        with pytest.raises(HttpError) as caught:
            KEYED.get_json("x", parse=dict)

        assert "EXAMPLE_API_KEY" not in str(caught.value)
        assert "k-123" not in str(caught.value)


# --- D53: a 429 rests the host; the interval counts from the end of a request --------------------

COOLED = Api(base="https://api.example.org/v1", name="Example", cooldown_s=60.0)


class TestRest:
    @pytest.fixture
    def clock(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        now = [1000.0]
        monkeypatch.setattr(
            _http, "_THROTTLE", _http._Throttle(sleep=lambda _s: None, clock=lambda: now[0])
        )
        return now

    def test_after_a_429_the_host_rests_for_the_declared_time(
        self, web: _Transport, clock: list[float]
    ) -> None:
        web.add("https://api.example.org/v1/x", b"", status=429)
        with pytest.raises(HttpError, match="HTTP 429"):
            COOLED.get_json("x", parse=dict)

        clock[0] += 20
        with pytest.raises(HttpError) as resting:
            COOLED.get_json("x", parse=dict)
        assert str(resting.value) == "Example asked to slow down (HTTP 429): try again in 40 s."
        assert resting.value.status == 429
        assert len(web.seen) == 1  # the request did not go out

        clock[0] += 41
        web.add("https://api.example.org/v1/x", {"ok": True})
        assert COOLED.get_json("x", parse=dict) == {"ok": True}
        assert len(web.seen) == 2

    def test_retry_after_wins_over_the_declared_time(
        self, web: _Transport, clock: list[float]
    ) -> None:
        web.add("https://api.example.org/v1/x", b"", status=429, **{"Retry-After": "7"})
        with pytest.raises(HttpError):
            COOLED.get_json("x", parse=dict)

        with pytest.raises(HttpError, match=r"try again in 7 s\.$"):
            COOLED.get_json("x", parse=dict)

    def test_the_rest_covers_every_api_on_the_host(
        self, web: _Transport, clock: list[float]
    ) -> None:
        web.add("https://api.example.org/v1/x", b"", status=429)
        with pytest.raises(HttpError):
            COOLED.get_json("x", parse=dict)

        with pytest.raises(HttpError, match="asked to slow down"):
            API.get_json("y", parse=dict)

    def test_without_a_declared_time_or_retry_after_a_429_rests_nothing(
        self, web: _Transport, clock: list[float]
    ) -> None:
        web.add("https://api.example.org/v1/x", b"", status=429)
        for _ in range(2):
            with pytest.raises(HttpError, match=r"^rate limited by Example"):
                API.get_json("x", parse=dict)

        assert len(web.seen) == 2

    def test_the_interval_counts_from_the_end_of_the_previous_request(self) -> None:
        now = [100.0]
        slept: list[float] = []
        throttle = _http._Throttle(sleep=slept.append, clock=lambda: now[0])

        throttle.wait("api.example.org", 5.0)
        now[0] += 12.0  # a slow answer
        throttle.done("api.example.org", 5.0)
        throttle.wait("api.example.org", 5.0)

        assert slept == [0.0, 5.0]


# --- D56: a request its service accepted is billed ---------------------------------------------

BILLED = Api(base="https://api.example.org/v1", name="Example", billed_as="example_search")


def _credits(text: str) -> int:
    return int(json.loads(text)["usage"]["credits"])


class TestBilling:
    def test_an_accepted_request_is_billed_one_unit(self, web: _Transport) -> None:
        from ai_arch_toolkit.core._tools._billing import billing

        web.add("https://api.example.org/v1/a", {"ok": True})

        with billing() as billed:
            BILLED.get_json("a", parse=dict)

        assert billed == [("example_search", 1)]

    def test_the_units_the_answer_reports_are_billed(self, web: _Transport) -> None:
        from ai_arch_toolkit.core._tools._billing import billing

        api = Api(
            base="https://api.example.org/v1",
            name="Example",
            billed_as="example_search",
            bill_units=_credits,
        )
        web.add("https://api.example.org/v1/a", {"usage": {"credits": 2}})

        with billing() as billed:
            api.get_json("a", parse=dict)

        assert billed == [("example_search", 2)]

    @pytest.mark.parametrize("status", [401, 429, 500])
    def test_a_refused_request_is_not_billed(self, web: _Transport, status: int) -> None:
        from ai_arch_toolkit.core._tools._billing import billing

        web.add("https://api.example.org/v1/a", b"", status=status)

        with billing() as billed, pytest.raises(HttpError):
            BILLED.get_json("a", parse=dict)

        assert billed == []

    def test_a_request_that_never_arrived_is_not_billed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_arch_toolkit.core._tools._billing import billing

        def fail(request: urllib.request.Request, timeout: float) -> Any:
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(_http, "_open", fail)

        with billing() as billed, pytest.raises(HttpError):
            BILLED.get_json("a", parse=dict)

        assert billed == []

    def test_outside_a_tool_call_a_billed_request_records_nothing(self, web: _Transport) -> None:
        web.add("https://api.example.org/v1/a", {})

        assert BILLED.get_json("a", parse=dict) == {}

    def test_a_key_can_go_after_a_prefix(
        self, web: _Transport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api = Api(
            base="https://api.example.org/v1",
            name="Example",
            key_env="EXAMPLE_API_KEY",
            key_header="Authorization",
            key_prefix="Bearer ",
        )
        monkeypatch.setenv("EXAMPLE_API_KEY", "k-123")
        web.add("https://api.example.org/v1/a", {})

        api.get_json("a", parse=dict)

        assert web.seen[0].get_header("Authorization") == "Bearer k-123"
