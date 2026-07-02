"""Tests for react_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response, ToolCall, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._tools._result import ToolResult
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.budget import BudgetPolicy


def _make_response(
    text: str = "", tool_calls: tuple[ToolCall, ...] = (), cost: float = 0.001
) -> Response:
    return Response(
        text=text,
        tool_calls=tool_calls,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class _FakeProvider:
    """A real LLM's provider stand-in, so LLM.complete runs its metering charge site."""

    def __init__(self, *responses: Response) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _metered_llm(*responses: Response) -> tuple[LLM, _FakeProvider]:
    llm = LLM("claude-sonnet-4-6", api_key="test")
    provider = _FakeProvider(*responses)
    llm._provider = provider  # type: ignore[assignment]
    return llm, provider


def _budget_dimension(result) -> str:
    info = result.results["budget_exceeded"].artifacts["budget_exceeded"]
    return info.get("dimension") or (info.get("breached") or [None])[0]


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
        tools.async_execute = AsyncMock(return_value=ToolResult.success("Sunny, 72F"))
        tools.schemas = lambda: {}

        flow = react_flow(llm, tools, max_iterations=5)
        state = State(operational=react_initial_state("What's the weather?"))
        await flow.run(state)

        assert state["response"].text == "The weather is sunny."
        tools.async_execute.assert_called_once()

    async def test_tool_error_is_structured_before_model_boundary(self) -> None:
        tc = ToolCall(id="tc1", name="search", input={"q": "test"})

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(tool_calls=(tc,)),
                _make_response(text="Search failed."),
            ]
        )

        tools = AsyncMock(spec=ToolGroup)
        tool_error = ToolResult.failure("runtime_error", "backend down", retryable=True)
        tools.async_execute = AsyncMock(return_value=tool_error)
        tools.schemas = lambda: {}

        flow = react_flow(llm, tools, max_iterations=5)
        state = State(operational=react_initial_state("search"))
        await flow.run(state)

        tool_results = state["tool_results"]
        assert tool_results == [tool_error]

        second_call_messages = llm.complete.call_args_list[1][0][0]
        assert any(
            "Tool error [runtime_error]: backend down" in str(message)
            for message in second_call_messages
        )

    async def test_approval_required_tool_is_blocked_without_handler(self) -> None:
        @tool(capability="shell", risk_level="critical", requires_approval=True)
        def dangerous(command: str) -> str:
            """Run a dangerous command."""
            return command

        tc = ToolCall(id="tc1", name="dangerous", input={"command": "rm -rf /tmp/x"})

        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=[
                _make_response(tool_calls=(tc,)),
                _make_response(text="Blocked."),
            ]
        )

        flow = react_flow(llm, ToolGroup(dangerous), max_iterations=5)
        state = State(operational=react_initial_state("run command"))
        await flow.run(state)

        tool_results = state["tool_results"]
        assert len(tool_results) == 1
        assert tool_results[0].ok is False
        assert tool_results[0].error is not None
        assert tool_results[0].error.type == "approval_denied"

    async def test_llm_call_budget_stops_before_model_call(self) -> None:
        llm, provider = _metered_llm(_make_response(text="should not happen"))

        flow = react_flow(
            llm,
            ToolGroup(),
            max_iterations=5,
            budget_policy=BudgetPolicy(max_llm_calls=0),
        )
        state = State(operational=react_initial_state("test"))
        result = await flow.run(state)

        assert provider.calls == 0  # denied at the charge site, before the provider
        assert "budget_exceeded" in result.results
        assert _budget_dimension(result) == "llm_calls"

    async def test_tool_call_budget_stops_before_tool_execution(self) -> None:
        @tool
        def search(q: str) -> str:
            """Search."""
            return q

        tc = ToolCall(id="tc1", name="search", input={"q": "x"})
        llm, _provider = _metered_llm(_make_response(tool_calls=(tc,)))

        flow = react_flow(
            llm,
            ToolGroup(search),
            max_iterations=5,
            budget_policy=BudgetPolicy(max_tool_calls=0),
        )
        state = State(operational=react_initial_state("test"))
        result = await flow.run(state)

        assert "budget_exceeded" in result.results
        assert _budget_dimension(result) == "tool_calls"

    async def test_max_iterations_stops(self) -> None:
        tc = ToolCall(id="tc1", name="search", input={"q": "test"})

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_response(tool_calls=(tc,)))

        tools = AsyncMock(spec=ToolGroup)
        tools.async_execute = AsyncMock(return_value=ToolResult.success("result"))
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
        tools.async_execute = AsyncMock(return_value=ToolResult.success("result"))
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
        tools.async_execute = AsyncMock(return_value=ToolResult.success("result"))
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
        tools.async_execute = AsyncMock(return_value=ToolResult.success("result"))
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
        tools.async_execute = AsyncMock(return_value=ToolResult.success("result"))
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
        tools.async_execute = AsyncMock(return_value=ToolResult.success("result"))
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
