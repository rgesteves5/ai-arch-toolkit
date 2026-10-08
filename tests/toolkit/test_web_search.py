"""``brave_search`` and ``tavily_search``: web search with one's own key, priced (D55, D56), held
to the tools contract (D37 to D42) and to C08's policy (D65)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import (
    ApprovalDecision,
    MeterScope,
    Money,
    ToolCall,
    ToolGroup,
    ToolResult,
    tool,
)
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools import _http, brave_search, tavily_search
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

BRAVE_ANSWER = {
    "type": "search",
    "query": {"original": "lisbon weather", "more_results_available": False},
    "web": {
        "type": "search",
        "results": [
            {
                "title": "Lisbon <strong>weather</strong> forecast",
                "url": "https://weather.example/lisbon",
                "description": "Sunny, <strong>24°C</strong> today.",
                "age": "2 hours ago",
            },
            {"title": "Portugal climate", "url": "https://climate.example/pt", "description": ""},
        ],
    },
}
TAVILY_ANSWER = {
    "query": "lisbon weather",
    "answer": "Sunny and 24°C.",
    "results": [
        {
            "title": "Lisbon forecast",
            "url": "https://weather.example/lisbon",
            "content": "Sunny, 24°C today, light wind from the north.",
            "score": 0.91,
        }
    ],
    "response_time": 0.8,
    "usage": {"credits": 1},
}
BRAVE_KEY = "brave-secret-0123456789"
TAVILY_KEY = "tvly-secret-0123456789"


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", BRAVE_KEY)
    monkeypatch.setenv("TAVILY_API_KEY", TAVILY_KEY)


def _brave_page(titles: list[str], *, more: bool) -> dict[str, Any]:
    """A Brave answer with one result per title
    (https://api-dashboard.search.brave.com/api-reference/web/search/get)."""
    return {
        "type": "search",
        "query": {"original": "physics", "more_results_available": more},
        "web": {
            "type": "search",
            "results": [
                {"title": title, "url": f"https://{title.lower()}.example/", "description": title}
                for title in titles
            ],
        },
    }


def _tavily_page(count: int) -> dict[str, Any]:
    results = [
        {"title": f"R{n}", "url": f"https://r{n}.example/", "content": f"About R{n}."}
        for n in range(1, count + 1)
    ]
    return {"query": "physics", "results": results, "usage": {"credits": 1}}


def _text(result: ToolResult) -> str:
    assert isinstance(result, ToolResult)
    assert result.ok
    assert isinstance(result.value, str)
    return result.value


def _failure(call: Any) -> ToolFailure:
    with pytest.raises(ToolFailure) as caught:
        call()
    return caught.value


def _sent_query(mock_open: Any) -> dict[str, list[str]]:
    return parse_qs(urlparse(mock_open.call_args.args[0].full_url).query)


def _execute(fn: Any, **arguments: Any) -> ToolResult:
    return ToolGroup(fn).execute(ToolCall(id="t1", name=fn.__name__, input=arguments))


def _now() -> float:
    """A clock that stands still, so a rest the throttle takes is all left when it is read."""
    return 1000.0


class TestBraveSearch:
    @patch(HTTP_OPEN)
    def test_asks_brave_with_the_key_and_lists_the_results(self, mock_open, keys) -> None:
        mock_open.return_value = respond(BRAVE_ANSWER)

        text = _text(brave_search("lisbon weather", max_results=5, country="PT"))

        request = mock_open.call_args.args[0]
        url = urlparse(request.full_url)
        assert url.netloc == "api.search.brave.com"
        assert url.path == "/res/v1/web/search"
        assert parse_qs(url.query) == {"q": ["lisbon weather"], "count": ["5"], "country": ["PT"]}
        assert request.get_header("X-subscription-token") == BRAVE_KEY
        assert text.startswith("Web results for 'lisbon weather' (Brave):")
        assert "1. Lisbon weather forecast\n   https://weather.example/lisbon" in text
        assert "Sunny, 24°C today. (2 hours ago)" in text  # markup stripped
        assert "2. Portugal climate" in text
        assert "[results" not in text  # the only page: nothing to read on

    @patch(HTTP_OPEN)
    def test_the_footer_names_the_next_page_and_the_next_page_follows(
        self, mock_open, keys
    ) -> None:
        mock_open.side_effect = [
            respond(_brave_page(["Alpha", "Beta"], more=True)),
            respond(_brave_page(["Gamma", "Delta"], more=False)),
        ]

        first = brave_search("physics", max_results=2)
        text = _text(first)
        assert text.endswith("[results 1-2 | next: offset=1, max_results=2]")
        assert first.metadata["window"]["next_call"] == {"offset": 1, "max_results": 2}

        second = _text(brave_search("physics", max_results=2, offset=1))
        assert _sent_query(mock_open) == {"q": ["physics"], "count": ["2"], "offset": ["1"]}
        assert "3. Gamma" in second
        assert "4. Delta" in second
        assert second.endswith("[results 3-4 | end]")

    @patch(HTTP_OPEN)
    def test_the_footer_of_an_empty_page_brave_has_more_after_names_the_call(
        self, mock_open, keys
    ) -> None:
        # The text said "[no results from 1 | end]" while metadata named offset=1.
        mock_open.return_value = respond(_brave_page([], more=True))

        result = brave_search("physics")

        assert _text(result).endswith("[no results from 1 | next: offset=1, max_results=10]")
        assert result.metadata["window"]["next_call"] == {"offset": 1, "max_results": 10}

    @patch(HTTP_OPEN)
    def test_the_footer_of_the_tenth_page_says_brave_has_more_it_does_not_serve(
        self, mock_open, keys
    ) -> None:
        mock_open.return_value = respond(_brave_page(["Omega"], more=True))

        assert _text(brave_search("physics", max_results=1, offset=9)).endswith(
            "[results 10-10 | Brave serves no page past offset 9: refine the query for other "
            "results]"
        )

    @patch(HTTP_OPEN)
    def test_an_empty_later_page_says_so(self, mock_open, keys) -> None:
        mock_open.return_value = respond(_brave_page([], more=False))

        text = _text(brave_search("physics", max_results=10, offset=3))

        assert text.endswith("[no results from 31 | end]")

    @pytest.mark.parametrize(
        "arguments",
        [{"max_results": 0}, {"max_results": 21}, {"offset": -1}, {"offset": 10}],
    )
    @patch(HTTP_OPEN)
    def test_limits_outside_braves_are_refused_by_the_schema(
        self, mock_open, keys, arguments
    ) -> None:
        mock_open.return_value = respond(BRAVE_ANSWER)

        result = _execute(brave_search, query="physics", **arguments)

        assert not result.ok
        assert result.error is not None
        assert result.error.type == "validation_error"
        mock_open.assert_not_called()

    @patch(HTTP_OPEN)
    def test_only_web_urls_reach_the_model(self, mock_open, keys) -> None:
        answer = _brave_page(["Alpha", "Beta", "Gamma", "Delta"], more=False)
        results = answer["web"]["results"]
        results[1]["url"] = "javascript:alert(document.cookie)"
        results[2]["url"] = "data:text/html,<script>alert(1)</script>"
        mock_open.return_value = respond(answer)

        text = _text(brave_search("physics"))

        assert "javascript:" not in text
        assert "data:" not in text
        assert "1. Alpha" in text
        assert "4. Delta\n   https://delta.example/" in text
        assert "2 results without an http(s) URL left out" in text

    @patch(HTTP_OPEN)
    def test_a_long_snippet_comes_whole(self, mock_open, keys) -> None:
        long = "word " * 400
        answer = _brave_page(["Alpha"], more=False)
        answer["web"]["results"][0]["description"] = long
        mock_open.return_value = respond(answer)

        text = _text(brave_search("physics"))

        assert long.strip() in text
        assert "…" not in text

    @patch(HTTP_OPEN)
    def test_without_a_key_it_says_where_to_get_one_and_sends_nothing(
        self, mock_open, monkeypatch
    ) -> None:
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

        failure = _failure(lambda: brave_search("lisbon weather"))

        assert failure.error.type == "validation_error"
        assert not failure.error.retryable
        assert "BRAVE_SEARCH_API_KEY" in failure.error.message
        assert "https://brave.com/search/api/" in failure.error.message
        mock_open.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_invalid_key_and_the_rate_limit_are_explained(self, mock_open, keys) -> None:
        mock_open.side_effect = http_error(401, "Unauthorized")
        failure = _failure(lambda: brave_search("x"))
        assert failure.error.type == "upstream"
        assert not failure.error.retryable
        assert failure.error.message == (
            "Brave rejected the key in BRAVE_SEARCH_API_KEY (HTTP 401); set a valid key in "
            "BRAVE_SEARCH_API_KEY (get one: https://brave.com/search/api/)."
        )

        mock_open.side_effect = http_error(429, "Too Many Requests")
        failure = _failure(lambda: brave_search("y"))
        assert failure.error.type == "rate_limited"
        assert "rate limited by Brave Search (HTTP 429)" in failure.error.message

    @patch(HTTP_OPEN)
    def test_a_refused_key_carries_braves_words(self, mock_open, keys) -> None:
        body = {
            "type": "ErrorResponse",
            "error": {"detail": "The token is invalid.", "status": 403},
        }
        mock_open.side_effect = http_error(403, "Forbidden", body=json.dumps(body).encode())

        failure = _failure(lambda: brave_search("x"))

        assert failure.error.type == "upstream"
        assert not failure.error.retryable
        assert failure.error.message.startswith(
            "Brave rejected the key in BRAVE_SEARCH_API_KEY (HTTP 403): The token is invalid;"
        )

    @patch(HTTP_OPEN)
    def test_arguments_brave_cannot_validate_are_a_validation_error_in_its_words(
        self, mock_open, keys
    ) -> None:
        body = {
            "type": "ErrorResponse",
            "error": {"detail": "Unable to validate request parameter(s).", "status": 422},
        }
        mock_open.side_effect = http_error(422, "Unprocessable", body=json.dumps(body).encode())

        failure = _failure(lambda: brave_search("x"))

        assert failure.error.type == "validation_error"
        assert not failure.error.retryable
        assert failure.error.message.startswith(
            "Brave refused the search (HTTP 422): Unable to validate request parameter(s);"
        )
        # It named only the query, whatever Brave refused.
        assert failure.error.message.endswith(
            "; check the arguments: query (at most 600 characters and 75 words), max_results, "
            "offset, country and freshness."
        )

    @patch(HTTP_OPEN)
    def test_the_fields_brave_names_in_its_meta_are_said(self, mock_open, keys) -> None:
        # ``error.meta`` is "non-standard meta-information" in Brave's reference; when it lists
        # the fields it refused, as a validation error's ``loc`` and ``msg``, they are named.
        error = {
            "id": "x",
            "status": 422,
            "code": "VALIDATION",
            "detail": "Unable to validate request parameter(s)",
            "meta": {
                "errors": [
                    {
                        "type": "string_too_long",
                        "loc": ["query", "q"],
                        "msg": "String should have at most 400 characters",
                    }
                ]
            },
        }
        body = json.dumps({"type": "ErrorResponse", "error": error, "time": 1}).encode()
        mock_open.side_effect = http_error(422, "Unprocessable", body=body)

        failure = _failure(lambda: brave_search("x"))

        assert failure.error.type == "validation_error"
        assert failure.error.message.startswith(
            "Brave refused the search (HTTP 422): Unable to validate request parameter(s) "
            "(q: String should have at most 400 characters);"
        )

    @pytest.mark.parametrize(("given", "sent"), [("gb", "GB"), (" pt ", "PT"), ("all", "ALL")])
    @patch(HTTP_OPEN)
    def test_a_country_brave_serves_is_sent_in_its_form(self, mock_open, keys, given, sent):
        mock_open.return_value = respond(BRAVE_ANSWER)

        brave_search("weather", country=given)

        assert _sent_query(mock_open)["country"] == [sent]

    @patch(HTTP_OPEN)
    def test_a_country_brave_does_not_serve_is_refused_before_asking(self, mock_open, keys):
        # "UK" went out, and Brave's 422 sent the agent to shorten the query.
        failure = _failure(lambda: brave_search("weather london", country="uk"))

        assert failure.error.type == "validation_error"
        assert failure.error.message.startswith("invalid country 'UK'; use AR, AU, AT, ")
        assert "GB" in failure.error.message
        mock_open.assert_not_called()

    @patch(HTTP_OPEN)
    def test_after_a_429_brave_rests_for_its_one_second_window(
        self, mock_open, keys, monkeypatch
    ) -> None:
        # Brave limits "using a 1-second sliding window" and sends no Retry-After.
        monkeypatch.setattr(_http, "_THROTTLE", _http._Throttle(sleep=lambda _s: None, clock=_now))
        mock_open.side_effect = http_error(429, "Too Many Requests")
        _failure(lambda: brave_search("x"))

        failure = _failure(lambda: brave_search("y"))

        assert failure.error.type == "rate_limited"
        assert str(failure) == "Brave Search asked to slow down: try again in 1 s."
        assert mock_open.call_count == 1  # the second search did not go out

    @patch(HTTP_OPEN)
    def test_the_rate_limit_carries_braves_words(self, mock_open, keys) -> None:
        body = {
            "type": "ErrorResponse",
            "error": {"detail": "Request rate limit exceeded for plan.", "status": 429},
        }
        mock_open.side_effect = http_error(429, "Too Many", body=json.dumps(body).encode())

        failure = _failure(lambda: brave_search("x"))

        assert failure.error.type == "rate_limited"
        assert failure.error.retryable
        assert "Request rate limit exceeded for plan." in failure.error.message

    @patch(HTTP_OPEN)
    def test_the_key_never_reaches_the_url_or_a_message(self, mock_open, keys) -> None:
        mock_open.return_value = respond(BRAVE_ANSWER)
        brave_search("lisbon weather")
        assert BRAVE_KEY not in mock_open.call_args.args[0].full_url

        for status in (401, 403, 422, 500, 429):  # the 429 last: the host rests after it
            mock_open.side_effect = http_error(status, "Error")
            assert BRAVE_KEY not in _failure(lambda: brave_search("x")).error.message

    @patch(HTTP_OPEN)
    def test_no_results_is_said(self, mock_open, keys) -> None:
        mock_open.return_value = respond({"type": "search", "web": {"results": []}})
        assert _text(brave_search("zzzz")) == "No web results for 'zzzz' (Brave)."

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"query": "  "}, "empty query"),
            ({"query": "x", "freshness": "pz"}, "invalid freshness 'pz'"),
        ],
    )
    def test_invalid_arguments_are_refused(self, keys, kwargs, words) -> None:
        failure = _failure(lambda: brave_search(**kwargs))

        assert failure.error.type == "validation_error"
        assert words in failure.error.message


class TestTavilySearch:
    @patch(HTTP_OPEN)
    def test_asks_tavily_with_the_key_and_lists_the_results(self, mock_open, keys) -> None:
        mock_open.return_value = respond(TAVILY_ANSWER)

        text = _text(
            tavily_search("lisbon weather", max_results=3, topic="news", include_answer=True)
        )

        request = mock_open.call_args.args[0]
        assert request.full_url == "https://api.tavily.com/search"
        assert request.get_header("Authorization") == f"Bearer {TAVILY_KEY}"
        assert json.loads(request.data) == {
            "query": "lisbon weather",
            "search_depth": "basic",
            "max_results": 3,
            "topic": "news",
            "include_answer": True,
            "include_usage": True,
        }
        assert TAVILY_KEY not in request.data.decode()
        assert text.startswith("Web results for 'lisbon weather' (Tavily):")
        assert "Answer: Sunny and 24°C." in text
        assert "1. Lisbon forecast\n   https://weather.example/lisbon" in text
        assert "light wind from the north" in text
        assert "one page" not in text  # fewer than asked for: that is all there is

    @patch(HTTP_OPEN)
    def test_a_full_page_says_tavily_has_no_other(self, mock_open, keys) -> None:
        mock_open.return_value = respond(_tavily_page(5))

        text = _text(tavily_search("physics", max_results=5))
        assert "5. R5" in text
        assert text.endswith(
            "Tavily returns one page: for more results, raise max_results (up to 20) or refine "
            "the query."
        )

        mock_open.return_value = respond(_tavily_page(20))
        text = _text(tavily_search("physics", max_results=20))
        assert text.endswith(
            "Tavily returns one page, of 20 results at most: refine the query for others."
        )

    @patch(HTTP_OPEN)
    def test_no_results_asks_only_for_the_answer_and_otherwise_is_refused(
        self, mock_open, keys
    ) -> None:
        mock_open.return_value = respond(TAVILY_ANSWER)
        failure = _failure(lambda: tavily_search("physics", max_results=0))
        assert failure.error.type == "validation_error"
        assert "include_answer=True" in failure.error.message
        mock_open.assert_not_called()

        mock_open.return_value = respond({**TAVILY_ANSWER, "results": []})
        text = _text(tavily_search("physics", max_results=0, include_answer=True))
        assert json.loads(mock_open.call_args.args[0].data)["max_results"] == 0
        assert text == "Web results for 'physics' (Tavily):\nAnswer: Sunny and 24°C."

    @pytest.mark.parametrize("arguments", [{"max_results": -1}, {"max_results": 21}])
    @patch(HTTP_OPEN)
    def test_limits_outside_tavilys_are_refused_by_the_schema(
        self, mock_open, keys, arguments
    ) -> None:
        mock_open.return_value = respond(TAVILY_ANSWER)

        result = _execute(tavily_search, query="physics", **arguments)

        assert not result.ok
        assert result.error is not None
        assert result.error.type == "validation_error"
        mock_open.assert_not_called()

    @patch(HTTP_OPEN)
    def test_only_web_urls_reach_the_model(self, mock_open, keys) -> None:
        answer = _tavily_page(3)
        answer["results"][0]["url"] = "JavaScript:alert(1)"
        mock_open.return_value = respond(answer)

        text = _text(tavily_search("physics"))

        assert "alert" not in text
        assert "1. R1" not in text
        assert "2. R2\n   https://r2.example/" in text
        assert "1 result without an http(s) URL left out" in text

    @patch(HTTP_OPEN)
    def test_a_long_excerpt_comes_whole(self, mock_open, keys) -> None:
        answer = _tavily_page(1)
        answer["results"][0]["content"] = "word " * 400
        mock_open.return_value = respond(answer)

        text = _text(tavily_search("physics"))

        assert ("word " * 400).strip() in text

    @patch(HTTP_OPEN)
    def test_without_a_key_it_says_where_to_get_one_and_sends_nothing(
        self, mock_open, monkeypatch
    ) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        failure = _failure(lambda: tavily_search("lisbon weather"))

        assert failure.error.type == "validation_error"
        assert not failure.error.retryable
        assert "TAVILY_API_KEY" in failure.error.message
        assert "https://app.tavily.com" in failure.error.message
        mock_open.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_plan_limit_is_explained(self, mock_open, keys) -> None:
        mock_open.side_effect = http_error(432, "Plan limit")

        failure = _failure(lambda: tavily_search("x"))

        # A usage limit stands until the plan changes: retrying does not help.
        assert failure.error.type == "rate_limited"
        assert not failure.error.retryable
        assert failure.error.message == (
            "the search exceeds your Tavily plan's usage limit (HTTP 432); "
            "raise it at https://app.tavily.com."
        )

    @patch(HTTP_OPEN)
    def test_the_pay_as_you_go_limit_is_explained_in_tavilys_words(self, mock_open, keys) -> None:
        body = {"detail": {"error": "This request exceeds the pay-as-you-go limit."}}
        mock_open.side_effect = http_error(433, "Limit", body=json.dumps(body).encode())

        failure = _failure(lambda: tavily_search("x"))

        assert failure.error.type == "rate_limited"
        assert not failure.error.retryable
        assert failure.error.message.startswith(
            "the search exceeds your Tavily pay-as-you-go limit (HTTP 433): "
            "This request exceeds the pay-as-you-go limit;"
        )

    @patch(HTTP_OPEN)
    def test_an_invalid_key_is_not_retryable(self, mock_open, keys) -> None:
        body = {"detail": {"error": "Unauthorized: missing or invalid API key."}}
        mock_open.side_effect = http_error(401, "Unauthorized", body=json.dumps(body).encode())

        failure = _failure(lambda: tavily_search("x"))

        assert failure.error.type == "upstream"
        assert not failure.error.retryable
        assert failure.error.message == (
            "Tavily rejected the key in TAVILY_API_KEY (HTTP 401): Unauthorized: missing or "
            "invalid API key; set a valid key in TAVILY_API_KEY (get one: https://app.tavily.com)."
        )

    @patch(HTTP_OPEN)
    def test_a_bad_request_is_a_validation_error_in_tavilys_words(self, mock_open, keys) -> None:
        body = {"detail": {"error": "Query is too long. Max query length is 400 characters."}}
        mock_open.side_effect = http_error(400, "Bad Request", body=json.dumps(body).encode())

        failure = _failure(lambda: tavily_search("x"))

        assert failure.error.type == "validation_error"
        assert not failure.error.retryable
        assert failure.error.message == (
            "Tavily refused the search (HTTP 400): Query is too long. Max query length is 400 "
            "characters; change the arguments."
        )

    @patch(HTTP_OPEN)
    def test_a_rejected_field_is_named(self, mock_open, keys) -> None:
        body = {
            "detail": [
                {
                    "type": "string_type",
                    "loc": ["body", "query"],
                    "msg": "Input should be a valid string",
                    "input": [],
                }
            ]
        }
        mock_open.side_effect = http_error(422, "Unprocessable", body=json.dumps(body).encode())

        failure = _failure(lambda: tavily_search("x"))

        assert failure.error.type == "validation_error"
        assert "(HTTP 422): query: Input should be a valid string;" in failure.error.message

    @patch(HTTP_OPEN)
    def test_the_rate_limit_carries_tavilys_words(self, mock_open, keys) -> None:
        said = "Your request has been blocked due to excessive requests."
        body = json.dumps({"detail": {"error": said}}).encode()
        mock_open.side_effect = http_error(429, "Too Many", body=body)
        failure = _failure(lambda: tavily_search("x"))
        assert failure.error.type == "rate_limited"
        assert failure.error.retryable
        assert said in failure.error.message

    @patch(HTTP_OPEN)
    def test_after_a_429_without_retry_after_tavily_rests_a_minute(
        self, mock_open, keys, monkeypatch
    ) -> None:
        # Tavily's limits are requests per minute; a 429 carries Retry-After only sometimes.
        monkeypatch.setattr(_http, "_THROTTLE", _http._Throttle(sleep=lambda _s: None, clock=_now))
        mock_open.side_effect = http_error(429, "Too Many Requests")
        _failure(lambda: tavily_search("x"))

        failure = _failure(lambda: tavily_search("y"))

        assert failure.error.type == "rate_limited"
        assert str(failure) == "Tavily asked to slow down: try again in 60 s."
        assert mock_open.call_count == 1

    @patch(HTTP_OPEN)
    def test_a_server_error_carries_tavilys_words(self, mock_open, keys) -> None:
        body = json.dumps({"detail": {"error": "Internal Server Error"}}).encode()
        mock_open.side_effect = http_error(500, "Error", body=body)
        failure = _failure(lambda: tavily_search("y"))
        assert failure.error.type == "upstream"
        assert failure.error.retryable
        assert failure.error.message == "HTTP error 500: Internal Server Error"

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"query": ""}, "empty query"),
            ({"query": "x", "topic": "sports"}, "invalid topic 'sports'; use 'general' or 'news'"),
            ({"query": "x", "time_range": "decade"}, "invalid time_range 'decade'"),
        ],
    )
    def test_invalid_arguments_are_refused(self, keys, kwargs, words) -> None:
        failure = _failure(lambda: tavily_search(**kwargs))

        assert failure.error.type == "validation_error"
        assert words in failure.error.message


class TestThePolicy:
    """C08.3 (D65): a network tool of low risk that runs without approval, as the others do;
    its cost is the budget's (D56)."""

    @pytest.mark.parametrize("fn", [brave_search, tavily_search])
    def test_network_low_risk_no_approval(self, fn) -> None:
        policy = fn.__tool_definition__.policy

        assert policy.capability == "network"
        assert policy.risk_level == "low"
        assert policy.requires_approval is False
        assert policy.approval_reason == ""

    @patch(HTTP_OPEN)
    def test_a_group_without_an_approval_handler_runs_them(self, mock_open, keys) -> None:
        mock_open.return_value = respond(BRAVE_ANSWER)

        assert _execute(brave_search, query="lisbon").ok

    @patch(HTTP_OPEN)
    def test_an_app_that_wants_approval_redecorates_the_tool(self, mock_open, keys) -> None:
        mock_open.return_value = respond(BRAVE_ANSWER)
        approved = tool(
            capability="network",
            risk_level="medium",
            requires_approval=True,
            approval_reason="Sends the query to Brave, billed.",
        )(brave_search)

        denied = _execute(approved, query="lisbon")

        assert approved.__name__ == "brave_search"
        assert approved.__tool_definition__.schema == brave_search.__tool_definition__.schema
        assert denied.error is not None
        assert denied.error.type == "approval_denied"
        mock_open.assert_not_called()

        # Approved, it runs and is billed at brave_search's price.
        group = ToolGroup(approved, approval_handler=lambda _request: ApprovalDecision.approve())
        with MeterScope() as scope:
            result = group.execute(ToolCall(id="t2", name="brave_search", input={"query": "x"}))
        assert result.ok
        assert scope.snapshot().cost == Money.from_usd(0.005)


class TestTheMeterCountsTheirCost:
    @patch(HTTP_OPEN)
    def test_an_accepted_brave_search_costs_its_price(self, mock_open, keys) -> None:
        mock_open.return_value = respond(BRAVE_ANSWER)

        with MeterScope() as scope:
            _execute(brave_search, query="lisbon")

        assert scope.snapshot().cost == Money.from_usd(0.005)

    @patch(HTTP_OPEN)
    def test_tavily_charges_the_credits_it_reports(self, mock_open, keys) -> None:
        mock_open.return_value = respond({**TAVILY_ANSWER, "usage": {"credits": 2}})

        with MeterScope() as scope:
            _execute(tavily_search, query="lisbon")

        assert scope.snapshot().cost == Money.from_usd(0.016)

    @patch(HTTP_OPEN)
    def test_a_search_without_a_key_costs_nothing(self, mock_open, monkeypatch) -> None:
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

        with MeterScope() as scope:
            _execute(brave_search, query="lisbon")

        assert scope.snapshot().cost == Money.zero()
        mock_open.assert_not_called()

    @pytest.mark.parametrize("status", [401, 422, 429, 432, 500])
    @patch(HTTP_OPEN)
    def test_a_refused_search_costs_nothing(self, mock_open, keys, status) -> None:
        mock_open.side_effect = http_error(status, "Refused")

        with MeterScope() as scope:
            brave = _execute(brave_search, query="lisbon")
            tavily = _execute(tavily_search, query="lisbon")

        assert not brave.ok
        assert not tavily.ok
        assert scope.snapshot().cost == Money.zero()

    @patch(HTTP_OPEN)
    def test_a_search_refused_before_it_is_sent_costs_nothing(self, mock_open, keys) -> None:
        mock_open.return_value = respond(TAVILY_ANSWER)

        with MeterScope() as scope:
            result = _execute(tavily_search, query="lisbon", max_results=0)

        assert result.error is not None
        assert result.error.type == "validation_error"
        assert scope.snapshot().cost == Money.zero()
        mock_open.assert_not_called()
