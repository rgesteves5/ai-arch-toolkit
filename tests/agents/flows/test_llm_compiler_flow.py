"""Tests for llm_compiler_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._llm_compiler import (
    llm_compiler_flow,
    llm_compiler_initial_state,
)


def _make_response(text: str = "", cost: float = 0.001) -> Response:
    return Response(
        text=text,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class TestLLMCompilerFlow:
    async def test_plan_execute_join(self) -> None:
        dag_plan = "$1. Research topic [deps: none]\n$2. Summarize findings [deps: $1]\n"

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(text=dag_plan),  # plan
                _make_response(text="Research results"),  # execute task 1
                _make_response(text="Summary done"),  # execute task 2
                _make_response(text="Final synthesis"),  # join
            ]
        )
        tools = ToolGroup()

        flow = llm_compiler_flow(llm, tools, max_replans=0)
        state = State(operational=llm_compiler_initial_state("Research and summarize AI"))
        result = await flow.run(state)

        assert state.get("answer") is not None
        assert result.trace.flow_name == "llm_compiler"

    async def test_parallel_tasks(self) -> None:
        dag_plan = "$1. Task A [deps: none]\n$2. Task B [deps: none]\n$3. Combine [deps: $1, $2]\n"

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(text=dag_plan),  # plan
                _make_response(text="Result A"),  # task 1
                _make_response(text="Result B"),  # task 2
                _make_response(text="Combined"),  # task 3
                _make_response(text="Final"),  # join
            ]
        )
        tools = ToolGroup()

        flow = llm_compiler_flow(llm, tools, max_replans=0)
        state = State(operational=llm_compiler_initial_state("parallel task"))
        await flow.run(state)

        assert state.get("answer") is not None

    async def test_replan(self) -> None:
        dag_plan = "$1. Do something [deps: none]"

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(text=dag_plan),  # plan (attempt 1)
                _make_response(text="Partial"),  # execute
                _make_response(text="REPLAN: need more"),  # join → replan
                _make_response(text=dag_plan),  # plan (attempt 2)
                _make_response(text="Better"),  # execute
                _make_response(text="Final answer"),  # join → done
            ]
        )
        tools = ToolGroup()

        flow = llm_compiler_flow(llm, tools, max_replans=1)
        state = State(operational=llm_compiler_initial_state("task"))
        await flow.run(state)

        assert state.get("answer") is not None


class TestLLMCompilerInitialState:
    def test_creates_initial_state(self) -> None:
        init = llm_compiler_initial_state("task")
        assert init["task"] == "task"
