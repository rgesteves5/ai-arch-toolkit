"""Tests for react_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response, ToolCall, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state


def _make_response(
    text: str = "", tool_calls: tuple[ToolCall, ...] = (), cost: float = 0.001
) -> Response:
    return Response(
        text=text,
        tool_calls=tool_calls,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class TestReactFlow:
    async def test_no_tools_completes_immediately(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(text="Hello!"))
        tools = ToolGroup()

        flow = react_flow(llm, tools, max_iterations=5)
        state = State(operational=react_initial_state("Hi"))
        result = await flow.run(state)

        assert state.get("response") is not None
        assert state["response"].text == "Hello!"
        assert result.trace.flow_name == "react"

    async def test_tool_call_loop(self) -> None:
        tc = ToolCall(id="tc1", name="get_weather", input={"city": "NYC"})

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(tool_calls=(tc,)),
                _make_response(text="The weather is sunny."),
            ]
        )

        tools = AsyncMock(spec=ToolGroup)
        tools.async_execute = AsyncMock(return_value="Sunny, 72F")
        tools.schemas = lambda: {}

        flow = react_flow(llm, tools, max_iterations=5)
        state = State(operational=react_initial_state("What's the weather?"))
        await flow.run(state)

        assert state["response"].text == "The weather is sunny."
        tools.async_execute.assert_called_once()

    async def test_max_iterations_stops(self) -> None:
        tc = ToolCall(id="tc1", name="search", input={"q": "test"})

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(tool_calls=(tc,)))

        tools = AsyncMock(spec=ToolGroup)
        tools.async_execute = AsyncMock(return_value="result")
        tools.schemas = lambda: {}

        flow = react_flow(llm, tools, max_iterations=3)
        state = State(operational=react_initial_state("loop forever"))
        result = await flow.run(state)

        # Should stop after max_iterations
        assert result.trace.flow_name == "react"

    async def test_llm_error_captured(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("API down"))
        tools = ToolGroup()

        flow = react_flow(llm, tools, max_iterations=3)
        state = State(operational=react_initial_state("test"))
        result = await flow.run(state)

        # Error should be captured in result, not raised
        error_traces = [t for t in result.trace.steps if t.error is not None]
        assert len(error_traces) > 0

    async def test_sync_run(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(text="sync answer"))
        tools = ToolGroup()

        flow = react_flow(llm, tools)
        state = State(operational=react_initial_state("test"))
        flow.run_sync(state)
        assert state["response"].text == "sync answer"


class TestReactInitialState:
    def test_creates_messages(self) -> None:
        init = react_initial_state("Hello")
        assert "messages" in init
        assert len(init["messages"]) == 1
        assert init["has_tool_calls"] is False
