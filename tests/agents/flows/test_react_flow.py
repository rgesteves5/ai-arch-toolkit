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


class TestReactFinalAnswer:
    """Tests for last-chance final answer behaviour."""

    async def test_final_answer_hint_produces_text(self) -> None:
        """Model always returns tool calls — hint on last turn should still be sent."""
        tc = ToolCall(id="tc1", name="search", input={"q": "x"})

        llm = AsyncMock()
        # All calls return tool calls (simulates Gemini/Haiku behaviour)
        llm.complete = AsyncMock(return_value=_make_response(tool_calls=(tc,)))

        tools = AsyncMock(spec=ToolGroup)
        tools.async_execute = AsyncMock(return_value="result")
        tools.schemas = lambda: {}

        flow = react_flow(llm, tools, max_iterations=3, final_answer_hint=True)
        state = State(operational=react_initial_state("test"))
        await flow.run(state)

        # On the last (3rd) llm_call, the hint message should be appended
        last_call_messages = llm.complete.call_args_list[-1][0][0]
        hint_messages = [
            m
            for m in last_call_messages
            if isinstance(m, dict) and "last turn" in str(m.get("content", "")).lower()
        ]
        assert len(hint_messages) == 1

    async def test_final_answer_hint_not_on_early_turns(self) -> None:
        """Hint message should NOT appear on non-final turns."""
        tc = ToolCall(id="tc1", name="search", input={"q": "x"})

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(tool_calls=(tc,)),
                _make_response(text="done"),
            ]
        )

        tools = AsyncMock(spec=ToolGroup)
        tools.async_execute = AsyncMock(return_value="result")
        tools.schemas = lambda: {}

        flow = react_flow(llm, tools, max_iterations=5, final_answer_hint=True)
        state = State(operational=react_initial_state("test"))
        await flow.run(state)

        # First call should NOT have the hint
        first_call_messages = llm.complete.call_args_list[0][0][0]
        hint_messages = [
            m
            for m in first_call_messages
            if isinstance(m, dict) and "last turn" in str(m.get("content", "")).lower()
        ]
        assert len(hint_messages) == 0

    async def test_strip_tools_on_final(self) -> None:
        """When strip_tools_on_final=True, last call gets empty ToolGroup."""
        tc = ToolCall(id="tc1", name="search", input={"q": "x"})

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(tool_calls=(tc,)))

        tools = AsyncMock(spec=ToolGroup)
        tools.async_execute = AsyncMock(return_value="result")
        tools.schemas = lambda: {}

        flow = react_flow(llm, tools, max_iterations=2, strip_tools_on_final=True)
        state = State(operational=react_initial_state("test"))
        await flow.run(state)

        # Last call should receive an empty ToolGroup (not the original tools)
        last_call = llm.complete.call_args_list[-1]
        call_tools = last_call[1]["tools"]
        assert isinstance(call_tools, ToolGroup)
        assert call_tools is not tools  # Should be a fresh empty ToolGroup

    async def test_show_turn_counter(self) -> None:
        """Turn counter messages appear when show_turn_counter=True."""
        tc = ToolCall(id="tc1", name="search", input={"q": "x"})

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(tool_calls=(tc,)),
                _make_response(text="done"),
            ]
        )

        tools = AsyncMock(spec=ToolGroup)
        tools.async_execute = AsyncMock(return_value="result")
        tools.schemas = lambda: {}

        flow = react_flow(
            llm,
            tools,
            max_iterations=5,
            show_turn_counter=True,
            final_answer_hint=False,
        )
        state = State(operational=react_initial_state("test"))
        await flow.run(state)

        # First call should have [Turn 1/5]
        first_messages = llm.complete.call_args_list[0][0][0]
        turn_msgs = [
            m
            for m in first_messages
            if isinstance(m, dict) and "[Turn 1/5]" in str(m.get("content", ""))
        ]
        assert len(turn_msgs) == 1

    async def test_hint_disabled(self) -> None:
        """When final_answer_hint=False, no hint is injected."""
        tc = ToolCall(id="tc1", name="search", input={"q": "x"})

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(tool_calls=(tc,)))

        tools = AsyncMock(spec=ToolGroup)
        tools.async_execute = AsyncMock(return_value="result")
        tools.schemas = lambda: {}

        flow = react_flow(llm, tools, max_iterations=2, final_answer_hint=False)
        state = State(operational=react_initial_state("test"))
        await flow.run(state)

        # No hint on any call
        for call in llm.complete.call_args_list:
            messages = call[0][0]
            hint_msgs = [
                m
                for m in messages
                if isinstance(m, dict) and "last turn" in str(m.get("content", "")).lower()
            ]
            assert len(hint_msgs) == 0

    async def test_turn_tracked_in_state(self) -> None:
        """Turn counter is stored in state after execution."""
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(text="done"))
        tools = ToolGroup()

        flow = react_flow(llm, tools, max_iterations=5)
        state = State(operational=react_initial_state("test"))
        await flow.run(state)

        assert state.get("turn") == 1


class TestReactInitialState:
    def test_creates_messages(self) -> None:
        init = react_initial_state("Hello")
        assert "messages" in init
        assert len(init["messages"]) == 1
        assert init["has_tool_calls"] is False
