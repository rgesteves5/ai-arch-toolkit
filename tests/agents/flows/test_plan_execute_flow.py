"""Tests for plan_execute_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._plan_execute import (
    plan_execute_flow,
    plan_execute_initial_state,
)


def _make_response(text: str = "", cost: float = 0.001) -> Response:
    return Response(
        text=text,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class TestPlanExecuteFlow:
    async def test_plan_and_solve(self) -> None:
        plan_text = "1. Research topic\n2. Write summary"

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(text=plan_text),  # plan
                _make_response(text="Researched"),  # execute step 1
                _make_response(text="Summary done"),  # execute step 2
                _make_response(text="Final answer"),  # solve
            ]
        )
        tools = ToolGroup()

        flow = plan_execute_flow(llm, tools, max_replans=0)
        state = State(operational=plan_execute_initial_state("Write a summary"))
        result = await flow.run(state)

        assert state.get("answer") is not None
        assert result.trace.flow_name == "plan_execute"

    async def test_single_step_plan(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(text="1. Do the thing"),
                _make_response(text="Done"),
                _make_response(text="Final"),
            ]
        )
        tools = ToolGroup()

        flow = plan_execute_flow(llm, tools, max_replans=0)
        state = State(operational=plan_execute_initial_state("task"))
        await flow.run(state)

        assert state.get("answer") == "Final"

    async def test_llm_kwargs_forwarded_to_every_phase(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response("1. Step"),
                _make_response("done"),
                _make_response("final"),
            ]
        )

        flow = plan_execute_flow(llm, ToolGroup(), max_replans=0, llm_kwargs={"temperature": 0.1})
        state = State(operational=plan_execute_initial_state("task"))
        await flow.run(state)

        for call in llm.complete.await_args_list:
            assert call.kwargs.get("temperature") == 0.1

    async def test_planner_sees_executor_tools(self) -> None:
        def lookup(query: str) -> str:
            """Look up a fact."""
            return "fact"

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response("1. Step"),
                _make_response("done"),
                _make_response("final"),
            ]
        )

        flow = plan_execute_flow(llm, ToolGroup(lookup), max_replans=0)
        state = State(operational=plan_execute_initial_state("task"))
        await flow.run(state)

        plan_system = llm.complete.await_args_list[0].kwargs["system"]
        assert "Available tools:" in plan_system
        assert "- lookup:" in plan_system
        assert "{tools}" not in plan_system

    async def test_custom_planner_prompt_without_token_untouched(self) -> None:
        def lookup(query: str) -> str:
            """Look up a fact."""
            return "fact"

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response("1. Step"),
                _make_response("done"),
                _make_response("final"),
            ]
        )

        flow = plan_execute_flow(llm, ToolGroup(lookup), max_replans=0, planner_system="PLAN ONLY")
        state = State(operational=plan_execute_initial_state("task"))
        await flow.run(state)

        assert llm.complete.await_args_list[0].kwargs["system"] == "PLAN ONLY"

    async def test_default_planner_prompt_renders_none_without_tools(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response("1. Step"),
                _make_response("done"),
                _make_response("final"),
            ]
        )

        flow = plan_execute_flow(llm, ToolGroup(), max_replans=0)
        state = State(operational=plan_execute_initial_state("task"))
        await flow.run(state)

        assert "Available tools:\n(none)" in llm.complete.await_args_list[0].kwargs["system"]


class TestPlanExecuteInitialState:
    def test_creates_initial_state(self) -> None:
        init = plan_execute_initial_state("task")
        assert init["task"] == "task"
