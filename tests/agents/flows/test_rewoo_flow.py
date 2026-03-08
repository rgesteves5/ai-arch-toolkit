"""Tests for rewoo_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._rewoo import rewoo_flow, rewoo_initial_state


def _make_response(text: str = "", cost: float = 0.001) -> Response:
    return Response(
        text=text,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class TestReWOOFlow:
    async def test_plan_execute_solve(self) -> None:
        plan_text = "#E1 = search[capital of France]\n#E2 = lookup[population of #E1]"

        responses = [
            _make_response(text=plan_text),  # plan
            _make_response(text="Paris, France, population 2.1M"),  # solve
        ]

        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=responses)

        tools = AsyncMock(spec=ToolGroup)
        tools.async_execute = AsyncMock(side_effect=["Paris", "2.1 million"])
        tools.schemas = lambda: {
            "search": {
                "description": "Search",
                "parameters": {"properties": {"query": {"type": "string"}}},
            },
            "lookup": {
                "description": "Lookup",
                "parameters": {"properties": {"query": {"type": "string"}}},
            },
        }

        flow = rewoo_flow(llm, tools)
        task = "What is the population of France's capital?"
        state = State(operational=rewoo_initial_state(task))
        result = await flow.run(state)

        assert state.get("answer") is not None
        assert result.trace.flow_name == "rewoo"
        # Evidence should have been collected
        evidence = state.get("evidence", {})
        assert "#E1" in evidence
        assert "#E2" in evidence

    async def test_unknown_tool_handled(self) -> None:
        plan_text = "#E1 = unknown_tool[test]"

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(text=plan_text),
                _make_response(text="Final answer"),
            ]
        )

        tools = AsyncMock(spec=ToolGroup)
        tools.schemas = lambda: {}

        flow = rewoo_flow(llm, tools)
        state = State(operational=rewoo_initial_state("test"))
        await flow.run(state)

        evidence = state.get("evidence", {})
        assert "#E1" in evidence
        assert "Error" in evidence["#E1"] or "Unknown" in evidence["#E1"]


class TestReWOOInitialState:
    def test_creates_initial_state(self) -> None:
        init = rewoo_initial_state("my task")
        assert init["task"] == "my task"
