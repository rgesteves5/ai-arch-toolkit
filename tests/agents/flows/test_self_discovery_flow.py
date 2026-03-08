"""Tests for self_discovery_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._self_discovery import (
    self_discovery_flow,
    self_discovery_initial_state,
)


def _make_response(text: str = "", cost: float = 0.001) -> Response:
    return Response(
        text=text,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class TestSelfDiscoveryFlow:
    async def test_four_phases(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(text="Selected: Critical Thinking, Analogical"),  # select
                _make_response(text="Adapted: Apply critical thinking to..."),  # adapt
                _make_response(text="1. Analyze\n2. Compare\n3. Conclude"),  # plan
                _make_response(text="The answer is X"),  # solve
            ]
        )
        tools = ToolGroup()

        flow = self_discovery_flow(llm, tools)
        state = State(operational=self_discovery_initial_state("Solve this puzzle"))
        result = await flow.run(state)

        assert state.get("selected_modules") is not None
        assert state.get("adapted_modules") is not None
        assert state.get("reasoning_plan") is not None
        assert state.get("answer") is not None
        assert result.trace.flow_name == "self_discovery"

    async def test_custom_modules(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(text="done"))
        tools = ToolGroup()

        custom = ("Module A: Do thing A.", "Module B: Do thing B.")
        flow = self_discovery_flow(llm, tools, modules=custom)
        state = State(operational=self_discovery_initial_state("task"))
        await flow.run(state)

        # Should work with custom modules — just verify no errors
        assert state.get("answer") is not None


class TestSelfDiscoveryInitialState:
    def test_creates_initial_state(self) -> None:
        init = self_discovery_initial_state("task")
        assert init["task"] == "task"
