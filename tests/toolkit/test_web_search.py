"""``brave_search`` and ``tavily_search``: web search with one's own key, priced (D55, D56)."""

from __future__ import annotations

import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from ai_arch_toolkit.core import MeterScope, Money, ToolCall, ToolGroup
from ai_arch_toolkit.core._tools._result import ToolFailure
from ai_arch_toolkit.toolkit.tools import brave_search, tavily_search
from tests.toolkit.http_fakes import HTTP_OPEN, http_error, respond

BRAVE_ANSWER = {
    "type": "search",
    "query": {"original": "lisbon weather"},
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


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")


class TestBraveSearch:
    @patch(HTTP_OPEN)
    def test_asks_brave_with_the_key_and_lists_the_results(self, mock_open, keys) -> None:
        mock_open.return_value = respond(BRAVE_ANSWER)

        text = brave_search("lisbon weather", max_results=5, country="PT")

        request = mock_open.call_args.args[0]
        url = urlparse(request.full_url)
        assert url.netloc == "api.search.brave.com"
        assert url.path == "/res/v1/web/search"
        assert parse_qs(url.query) == {"q": ["lisbon weather"], "count": ["5"], "country": ["PT"]}
        assert request.get_header("X-subscription-token") == "brave-key"
        assert text.startswith("Web results for 'lisbon weather' (Brave):")
        assert "1. Lisbon weather forecast\n   https://weather.example/lisbon" in text
        assert "Sunny, 24°C today." in text  # markup stripped
        assert "2. Portugal climate" in text

    @patch(HTTP_OPEN)
    def test_without_a_key_it_says_where_to_get_one_and_sends_nothing(
        self, mock_open, monkeypatch
    ) -> None:
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

        with pytest.raises(ToolFailure) as caught:
            brave_search("lisbon weather")

        assert caught.value.error.type == "upstream"
        assert "BRAVE_SEARCH_API_KEY" in caught.value.error.message
        assert "https://brave.com/search/api/" in caught.value.error.message
        mock_open.assert_not_called()

    @patch(HTTP_OPEN)
    def test_an_invalid_key_and_the_rate_limit_are_explained(self, mock_open, keys) -> None:
        mock_open.side_effect = http_error(401, "Unauthorized")
        with pytest.raises(ToolFailure) as caught:
            brave_search("x")
        assert caught.value.error.type == "upstream"
        assert "Brave rejected the key in BRAVE_SEARCH_API_KEY" in caught.value.error.message

        mock_open.side_effect = http_error(429, "Too Many Requests")
        with pytest.raises(ToolFailure) as caught:
            brave_search("y")
        assert caught.value.error.type == "rate_limited"
        assert "rate limited by Brave Search (HTTP 429)" in caught.value.error.message

    @patch(HTTP_OPEN)
    def test_no_results_is_said(self, mock_open, keys) -> None:
        mock_open.return_value = respond({"type": "search", "web": {"results": []}})
        assert brave_search("zzzz") == "No web results for 'zzzz' (Brave)."

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"query": "  "}, "empty query"),
            ({"query": "x", "freshness": "pz"}, "invalid freshness 'pz'"),
        ],
    )
    def test_invalid_arguments_are_refused(self, keys, kwargs, words) -> None:
        with pytest.raises(ToolFailure) as caught:
            brave_search(**kwargs)

        assert caught.value.error.type == "validation_error"
        assert words in caught.value.error.message


class TestTavilySearch:
    @patch(HTTP_OPEN)
    def test_asks_tavily_with_the_key_and_lists_the_results(self, mock_open, keys) -> None:
        mock_open.return_value = respond(TAVILY_ANSWER)

        text = tavily_search("lisbon weather", max_results=3, topic="news", include_answer=True)

        request = mock_open.call_args.args[0]
        assert request.full_url == "https://api.tavily.com/search"
        assert request.get_header("Authorization") == "Bearer tvly-key"
        assert json.loads(request.data) == {
            "query": "lisbon weather",
            "search_depth": "basic",
            "max_results": 3,
            "topic": "news",
            "include_answer": True,
            "include_usage": True,
        }
        assert text.startswith("Web results for 'lisbon weather' (Tavily):")
        assert "Answer: Sunny and 24°C." in text
        assert "1. Lisbon forecast\n   https://weather.example/lisbon" in text
        assert "light wind from the north" in text

    @patch(HTTP_OPEN)
    def test_without_a_key_it_says_where_to_get_one_and_sends_nothing(
        self, mock_open, monkeypatch
    ) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        with pytest.raises(ToolFailure) as caught:
            tavily_search("lisbon weather")

        assert caught.value.error.type == "upstream"
        assert "TAVILY_API_KEY" in caught.value.error.message
        assert "https://app.tavily.com" in caught.value.error.message
        mock_open.assert_not_called()

    @patch(HTTP_OPEN)
    def test_a_plan_limit_is_explained(self, mock_open, keys) -> None:
        mock_open.side_effect = http_error(432, "Plan limit")

        with pytest.raises(ToolFailure) as caught:
            tavily_search("x")

        assert caught.value.error.type == "upstream"
        assert "your Tavily plan's limit" in caught.value.error.message

    @pytest.mark.parametrize(
        ("kwargs", "words"),
        [
            ({"query": ""}, "empty query"),
            ({"query": "x", "topic": "sports"}, "invalid topic 'sports'; use 'general' or 'news'"),
            ({"query": "x", "time_range": "decade"}, "invalid time_range 'decade'"),
        ],
    )
    def test_invalid_arguments_are_refused(self, keys, kwargs, words) -> None:
        with pytest.raises(ToolFailure) as caught:
            tavily_search(**kwargs)

        assert caught.value.error.type == "validation_error"
        assert words in caught.value.error.message


class TestTheMeterCountsTheirCost:
    @patch(HTTP_OPEN)
    def test_an_accepted_brave_search_costs_its_price(self, mock_open, keys) -> None:
        mock_open.return_value = respond(BRAVE_ANSWER)

        with MeterScope() as scope:
            ToolGroup(brave_search).execute(
                ToolCall(id="t1", name="brave_search", input={"query": "lisbon"})
            )

        assert scope.snapshot().cost == Money.from_usd(0.005)

    @patch(HTTP_OPEN)
    def test_tavily_charges_the_credits_it_reports(self, mock_open, keys) -> None:
        mock_open.return_value = respond({**TAVILY_ANSWER, "usage": {"credits": 2}})

        with MeterScope() as scope:
            ToolGroup(tavily_search).execute(
                ToolCall(id="t1", name="tavily_search", input={"query": "lisbon"})
            )

        assert scope.snapshot().cost == Money.from_usd(0.016)

    @patch(HTTP_OPEN)
    def test_a_search_without_a_key_costs_nothing(self, mock_open, monkeypatch) -> None:
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

        with MeterScope() as scope:
            ToolGroup(brave_search).execute(
                ToolCall(id="t1", name="brave_search", input={"query": "lisbon"})
            )

        assert scope.snapshot().cost == Money.zero()
        mock_open.assert_not_called()
