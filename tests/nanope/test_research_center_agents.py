"""The research manager's web tools run: the agent approves the tools it hands the model."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.core._response import Response, ToolCall, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.nanope.research_center._agents import manager_agent
from ai_arch_toolkit.toolkit.agents.flows._generate_review import generate_review_initial_state
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore
from ai_arch_toolkit.toolkit.tools import _web

_USAGE = Usage(input_tokens=1, output_tokens=1)


async def test_the_manager_fetches_the_page_it_asks_for(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []

    def fake_fetch(url: str, timeout: int = 10) -> str:
        fetched.append(url)
        return "Coverage looks complete."

    monkeypatch.setattr(_web, "_fetch", fake_fetch)
    web_check = ToolCall(id="tc1", name="http_get", input={"url": "https://example.com"})
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            Response(tool_calls=(web_check,), usage=_USAGE),
            Response(text="COVERAGE: 90%\n\nDECISION: DONE", usage=_USAGE),
            Response(text="ACCEPT", usage=_USAGE),
        ]
    )
    flow = manager_agent(llm, GraphStore(NetworkXBackend()))

    await flow.run(State(operational=generate_review_initial_state("A research brief.")))

    assert fetched == ["https://example.com"]
    tool_turn = llm.complete.call_args_list[1].args[0]
    assert tool_turn[-1]["content"] == "Coverage looks complete."
