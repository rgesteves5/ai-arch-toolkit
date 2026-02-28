"""Tests for SelfDiscoveryAgent — Select → Adapt → Operationalize → Solve."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import OutputSchema, Response, Usage
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import AgentConfig, AgentEvent, AgentResult
from ai_arch_toolkit.toolkit.agents._self_discovery import (
    SelfDiscoveryAgent,
    SelfDiscoveryConfig,
)
from tests.agents.conftest import make_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    llm_side_effect: list[Response],
    *,
    config: AgentConfig | None = None,
    self_discovery: SelfDiscoveryConfig | None = None,
    tools: ToolGroup | None = None,
) -> SelfDiscoveryAgent:
    """Build a SelfDiscoveryAgent with mocked LLM."""

    def lookup(query: str = "") -> str:
        """Look up information."""
        return f"info:{query}"

    def calculate(expression: str = "") -> str:
        """Calculate a math expression."""
        return f"calc:{expression}"

    if tools is None:
        tools = ToolGroup(lookup, calculate)
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=llm_side_effect)
    return SelfDiscoveryAgent(
        llm,
        tools,
        config=config or AgentConfig(),
        self_discovery=self_discovery,
    )


# ---------------------------------------------------------------------------
# 1. Full 4-phase flow
# ---------------------------------------------------------------------------


async def test_full_four_phase_flow():
    responses = [
        make_response(text="Critical thinking\nStep-by-step decomposition"),  # Select
        make_response(text="Analyze the problem critically..."),  # Adapt
        make_response(text="1. Break down 2. Analyze 3. Synthesize"),  # Plan
        make_response(text="The answer is 42"),  # Solve (inner ReAct)
    ]
    agent = _make_agent(responses)
    result = await agent.run("What is the meaning of life?")

    assert result.stop_reason == "completed"
    assert result.answer == "The answer is 42"
    assert agent.llm.complete.call_count == 4


# ---------------------------------------------------------------------------
# 2. Custom modules forwarded
# ---------------------------------------------------------------------------


async def test_custom_modules_forwarded():
    custom_modules = ("Module A: Do X.", "Module B: Do Y.")
    responses = [
        make_response(text="Module A"),  # Select
        make_response(text="Adapted A"),  # Adapt
        make_response(text="Plan step"),  # Plan
        make_response(text="Answer"),  # Solve
    ]
    agent = _make_agent(
        responses,
        self_discovery=SelfDiscoveryConfig(modules=custom_modules),
    )
    await agent.run("Test")

    # First call (select) should contain custom modules
    select_call = agent.llm.complete.call_args_list[0]
    select_msg = select_call.args[0][0]["content"]
    assert "Module A: Do X." in select_msg
    assert "Module B: Do Y." in select_msg


# ---------------------------------------------------------------------------
# 3. Custom system prompts forwarded
# ---------------------------------------------------------------------------


async def test_custom_system_prompts():
    responses = [
        make_response(text="Selected"),
        make_response(text="Adapted"),
        make_response(text="Planned"),
        make_response(text="Solved"),
    ]
    custom_select = "Custom select system"
    custom_adapt = "Custom adapt system"
    custom_plan = "Custom plan system"
    custom_solve = "Custom solve system\n\n"
    agent = _make_agent(
        responses,
        self_discovery=SelfDiscoveryConfig(
            select_system=custom_select,
            adapt_system=custom_adapt,
            plan_system=custom_plan,
            solve_system=custom_solve,
        ),
    )
    await agent.run("Test custom prompts")

    calls = agent.llm.complete.call_args_list
    assert calls[0].kwargs["system"] == custom_select
    assert calls[1].kwargs["system"] == custom_adapt
    assert calls[2].kwargs["system"] == custom_plan
    # Solve system is augmented with reasoning plan + task
    assert calls[3].kwargs.get("system", "").startswith(custom_solve)


# ---------------------------------------------------------------------------
# 4. Tool use in solve phase
# ---------------------------------------------------------------------------


async def test_tool_use_in_solve_phase():
    from tests.agents.conftest import make_tool_call

    tc = make_tool_call(name="lookup", input={"query": "test"}, id="tc_1")
    responses = [
        make_response(text="Selected modules"),  # Select
        make_response(text="Adapted modules"),  # Adapt
        make_response(text="Reasoning plan"),  # Plan
        make_response(text="", tool_calls=(tc,)),  # Solve step 1: tool call
        make_response(text="Final answer from tools"),  # Solve step 2: answer
    ]
    agent = _make_agent(responses)
    result = await agent.run("Use tools to answer")

    assert result.stop_reason == "completed"
    assert result.answer == "Final answer from tools"
    assert agent.llm.complete.call_count == 5


# ---------------------------------------------------------------------------
# 5. Event sequence (step_start/step_end for each phase)
# ---------------------------------------------------------------------------


async def test_event_sequence():
    responses = [
        make_response(text="Selected"),
        make_response(text="Adapted"),
        make_response(text="Planned"),
        make_response(text="Solved"),
    ]
    events: list[AgentEvent] = []
    config = AgentConfig(on_event=events.append)
    agent = _make_agent(responses, config=config)
    await agent.run("Test events")

    steps = [e.step for e in events]
    # Steps should be monotonically non-decreasing
    assert steps == sorted(steps)
    # Should have at least 4 phases (select, adapt, plan, solve)
    step_starts = [e for e in events if e.type == "step_start"]
    assert len(step_starts) >= 4


# ---------------------------------------------------------------------------
# 6. run_sync works
# ---------------------------------------------------------------------------


def test_run_sync():
    responses = [
        make_response(text="Selected"),
        make_response(text="Adapted"),
        make_response(text="Planned"),
        make_response(text="Sync answer"),
    ]
    agent = _make_agent(responses)
    result = agent.run_sync("Sync test")

    assert isinstance(result, AgentResult)
    assert result.answer == "Sync answer"
    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 7. Timeout propagation from inner ReAct
# ---------------------------------------------------------------------------


async def test_timeout_propagation(monkeypatch):
    import ai_arch_toolkit.toolkit.agents._react as react_mod

    _counter = {"n": 0}

    def fake_monotonic():
        _counter["n"] += 1
        return _counter["n"] * 100.0

    monkeypatch.setattr(react_mod.time, "monotonic", fake_monotonic)

    responses = [
        make_response(text="Selected"),
        make_response(text="Adapted"),
        make_response(text="Planned"),
        make_response(text="Never reached"),
    ]
    agent = _make_agent(responses, config=AgentConfig(timeout=0.001))
    result = await agent.run("Test timeout")

    assert result.stop_reason == "timeout"


# ---------------------------------------------------------------------------
# 8. Output schema forwarded to solve phase
# ---------------------------------------------------------------------------


async def test_output_schema_forwarded():
    parsed_data = {"answer": "42"}
    solve_response = Response(
        text='{"answer": "42"}',
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=0.001,
        parsed=parsed_data,
    )
    responses = [
        make_response(text="Selected"),
        make_response(text="Adapted"),
        make_response(text="Planned"),
        solve_response,
    ]
    schema = OutputSchema(name="Answer", schema={"type": "object"})
    config = AgentConfig(output_schema=schema)
    agent = _make_agent(responses, config=config)
    result = await agent.run("Get answer")

    assert result.parsed == parsed_data
    # Inner ReAct solve call should have output_schema
    solve_call = agent.llm.complete.call_args_list[-1]
    assert solve_call.kwargs.get("output_schema") is schema


# ---------------------------------------------------------------------------
# 9. Inner ReAct error propagation
# ---------------------------------------------------------------------------


async def test_inner_error_propagation():
    """When inner ReAct returns stop_reason='error', the agent propagates it."""
    call_count = {"n": 0}
    normal_responses = [
        make_response(text="Selected"),  # Select
        make_response(text="Adapted"),  # Adapt
        make_response(text="Planned"),  # Plan
    ]

    async def _side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < 3:
            return normal_responses[idx]
        # Inner ReAct LLM call raises
        raise RuntimeError("LLM call failed")

    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=_side_effect)

    def lookup(query: str = "") -> str:
        """Look up information."""
        return f"info:{query}"

    tools = ToolGroup(lookup)
    agent = SelfDiscoveryAgent(llm, tools, config=AgentConfig())
    result = await agent.run("Test error")

    assert result.stop_reason == "error"
