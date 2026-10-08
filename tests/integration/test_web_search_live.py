"""``brave_search`` and ``tavily_search`` against the real services (C08, D55, D56, D65).

Each test makes one paid search: Brave $0.005, Tavily $0.008 (a basic search, one credit). Run
locally, with the keys in the environment:

    uv run pytest tests/integration/test_web_search_live.py -m live_api

CI deselects ``live_api``; without its key, each test skips.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from ai_arch_toolkit.core import MeterScope, Money, ToolCall, ToolGroup, ToolResult
from ai_arch_toolkit.toolkit.tools import brave_search, tavily_search

pytestmark = [pytest.mark.integration, pytest.mark.live_api]

QUERY = "Python programming language"


def _search(fn: Callable[..., Any], **arguments: Any) -> tuple[ToolResult, Money]:
    """One search through the governed executor, under a meter: its result and its cost."""
    with MeterScope() as scope:
        result = ToolGroup(fn).execute(ToolCall(id="live", name=fn.__name__, input=arguments))
    return result, scope.snapshot().cost


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    not os.environ.get("BRAVE_SEARCH_API_KEY"), reason="BRAVE_SEARCH_API_KEY not set"
)
def test_a_brave_search_lists_web_pages_names_the_next_page_and_costs_its_price() -> None:
    result, cost = _search(brave_search, query=QUERY, max_results=3)

    assert result.ok, result.error
    assert isinstance(result.value, str)
    assert "1. " in result.value
    assert "https://" in result.value
    # A popular query has more than three results: Brave says so, and the footer names page 2.
    assert result.metadata["window"]["next_call"] == {"offset": 1, "max_results": 3}
    assert cost == Money.from_usd(0.005)


@pytest.mark.timeout(60)
@pytest.mark.skipif(not os.environ.get("TAVILY_API_KEY"), reason="TAVILY_API_KEY not set")
def test_a_tavily_search_lists_web_pages_and_costs_the_credit_it_reports() -> None:
    result, cost = _search(tavily_search, query=QUERY, max_results=3)

    assert result.ok, result.error
    assert isinstance(result.value, str)
    assert "1. " in result.value
    assert "https://" in result.value
    assert cost == Money.from_usd(0.008)  # a basic search: one credit
