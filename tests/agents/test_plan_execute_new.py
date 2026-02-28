"""Tests for PlanExecuteAgent — Plan → Execute → Solve."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import OutputSchema, Response, Usage
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import AgentConfig, AgentEvent, AgentResult
from ai_arch_toolkit.toolkit.agents._plan_execute import PlanExecuteAgent, PlanExecuteConfig
from tests.agents.conftest import make_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    llm_side_effect: list[Response],
    *,
    config: AgentConfig | None = None,
    plan_execute: PlanExecuteConfig | None = None,
    tools: ToolGroup | None = None,
) -> PlanExecuteAgent:
    """Build a PlanExecuteAgent with mocked LLM."""

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
    return PlanExecuteAgent(
        llm,
        tools,
        config=config or AgentConfig(),
        plan_execute=plan_execute,
    )


# ---------------------------------------------------------------------------
# 1. Simple plan-execute-solve (2 planned steps)
# ---------------------------------------------------------------------------


async def test_simple_plan_execute_solve():
    plan_text = "1. Look up the population of France\n2. Calculate it times 2"
    responses = [
        make_response(text=plan_text),  # Plan
        make_response(text="67 million"),  # Execute step 1 (inner ReAct answer)
        make_response(text="134 million"),  # Execute step 2 (inner ReAct answer)
        make_response(text="Final answer: 134 million"),  # Solve
    ]
    agent = _make_agent(responses)
    result = await agent.run("Population of France times 2?")

    assert result.stop_reason == "completed"
    assert result.answer == "Final answer: 134 million"


# ---------------------------------------------------------------------------
# 2. Single-step plan
# ---------------------------------------------------------------------------


async def test_single_step_plan():
    plan_text = "1. Look up the capital of France"
    responses = [
        make_response(text=plan_text),  # Plan
        make_response(text="Paris"),  # Execute step 1
        make_response(text="The capital is Paris"),  # Solve
    ]
    agent = _make_agent(responses)
    result = await agent.run("Capital of France?")

    assert result.stop_reason == "completed"
    assert result.answer == "The capital is Paris"


# ---------------------------------------------------------------------------
# 3. Empty plan (no numbered steps → straight to solve)
# ---------------------------------------------------------------------------


async def test_empty_plan():
    responses = [
        make_response(text="I can answer this directly without tools."),  # Plan (no steps)
        make_response(text="Direct answer"),  # Solve
    ]
    agent = _make_agent(responses)
    result = await agent.run("Simple question")

    assert result.stop_reason == "completed"
    assert result.answer == "Direct answer"


# ---------------------------------------------------------------------------
# 4. Replan on failure (max_replans=1)
# ---------------------------------------------------------------------------


async def test_replan_on_inner_error():
    """When inner ReAct hits an error and max_replans>0, replanning occurs."""
    plan_text1 = "1. Do something"
    plan_text2 = "1. Try differently"

    # Inner ReAct will raise an exception on the first execute attempt,
    # causing stop_reason="error". With max_replans=1, the agent replans.
    call_count = {"n": 0}
    original_responses = [
        make_response(text=plan_text1),  # Plan 1
        None,  # Execute step 1 — will raise (see side_effect below)
        make_response(text=plan_text2),  # Replan
        make_response(text="Step result v2"),  # Execute step 1 attempt 2
        make_response(text="Final answer"),  # Solve
    ]

    async def _side_effect(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx == 1:
            raise RuntimeError("LLM call failed")
        return original_responses[idx]

    def lookup(query: str = "") -> str:
        """Look up information."""
        return f"info:{query}"

    tools = ToolGroup(lookup)
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=_side_effect)
    agent = PlanExecuteAgent(
        llm,
        tools,
        config=AgentConfig(),
        plan_execute=PlanExecuteConfig(max_replans=1),
    )
    result = await agent.run("Test replan")

    assert result.stop_reason == "completed"
    assert result.answer == "Final answer"
    # Should have called LLM 5 times: plan1, error, plan2, execute, solve
    assert call_count["n"] == 5


async def test_max_replans_zero_skips_replan():
    """With max_replans=0, no replanning occurs even if steps fail."""
    plan_text = "1. Do something"
    responses = [
        make_response(text=plan_text),  # Plan
        make_response(text="Step result"),  # Execute step 1
        make_response(text="Final answer"),  # Solve
    ]
    agent = _make_agent(
        responses,
        plan_execute=PlanExecuteConfig(max_replans=0),
    )
    result = await agent.run("Test no replan")

    assert result.stop_reason == "completed"
    assert result.answer == "Final answer"


# ---------------------------------------------------------------------------
# 5. Event sequence continuity
# ---------------------------------------------------------------------------


async def test_event_sequence_continuity():
    plan_text = "1. Step one\n2. Step two"
    responses = [
        make_response(text=plan_text),  # Plan
        make_response(text="Result 1"),  # Execute 1
        make_response(text="Result 2"),  # Execute 2
        make_response(text="Done"),  # Solve
    ]
    events: list[AgentEvent] = []
    config = AgentConfig(on_event=events.append)
    agent = _make_agent(responses, config=config)
    await agent.run("Test events")

    steps = [e.step for e in events]
    # Steps should be monotonically non-decreasing
    assert steps == sorted(steps)
    # Should have at least 4 distinct steps: plan, exec1, exec2, solve
    assert len(set(steps)) >= 4


# ---------------------------------------------------------------------------
# 6. run_sync works
# ---------------------------------------------------------------------------


def test_run_sync():
    plan_text = "1. Lookup"
    responses = [
        make_response(text=plan_text),
        make_response(text="Result"),
        make_response(text="Sync answer"),
    ]
    agent = _make_agent(responses)
    result = agent.run_sync("Sync test")

    assert isinstance(result, AgentResult)
    assert result.answer == "Sync answer"
    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 7. Solver receives output_schema
# ---------------------------------------------------------------------------


async def test_solver_receives_output_schema():
    plan_text = "1. Look something up"
    parsed_data = {"result": "42"}
    solve_response = Response(
        text='{"result": "42"}',
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=0.001,
        parsed=parsed_data,
    )
    responses = [
        make_response(text=plan_text),  # Plan
        make_response(text="info"),  # Execute
        solve_response,  # Solve
    ]
    schema = OutputSchema(name="Result", schema={"type": "object"})
    config = AgentConfig(output_schema=schema)
    agent = _make_agent(responses, config=config)
    result = await agent.run("Get result")

    assert result.parsed == parsed_data
    # Solver call should have output_schema
    solver_call = agent.llm.complete.call_args_list[-1]
    assert solver_call.kwargs.get("output_schema") is schema


# ---------------------------------------------------------------------------
# 8. Timeout propagation from inner agent
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8b. Budget exhaustion from inner agent
# ---------------------------------------------------------------------------


async def test_budget_exhaustion_propagation(monkeypatch):
    """Inner ReAct hitting timeout should propagate budget/timeout up."""
    import ai_arch_toolkit.toolkit.agents._react as react_mod

    _counter = {"n": 0}

    def fake_monotonic():
        _counter["n"] += 1
        return _counter["n"] * 100.0

    monkeypatch.setattr(react_mod.time, "monotonic", fake_monotonic)

    plan_text = "1. Do something expensive"
    responses = [
        make_response(text=plan_text),
        make_response(text="Never reached"),
    ]

    def lookup(query: str = "") -> str:
        """Look up information."""
        return f"info:{query}"

    tools = ToolGroup(lookup)
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=responses)
    agent = PlanExecuteAgent(
        llm, tools, config=AgentConfig(timeout=0.001, max_iterations=10)
    )
    result = await agent.run("Test budget")

    assert result.stop_reason == "timeout"


# ---------------------------------------------------------------------------
# 8c. Tool descriptions included in planner prompt
# ---------------------------------------------------------------------------


async def test_tool_descriptions_in_planner():
    """Planner system should include tool names and descriptions."""
    plan_text = "1. Lookup"
    responses = [
        make_response(text=plan_text),
        make_response(text="Result"),
        make_response(text="Answer"),
    ]
    agent = _make_agent(responses)
    await agent.run("Test tools")

    # First LLM call (plan) should have system with tool names
    plan_call = agent.llm.complete.call_args_list[0]
    system = plan_call.kwargs.get("system", "")
    assert "lookup" in system
    assert "calculate" in system


# ---------------------------------------------------------------------------
# 8d. Custom planner/solver system prompts
# ---------------------------------------------------------------------------


async def test_custom_planner_solver_system():
    plan_text = "1. Do it"
    responses = [
        make_response(text=plan_text),
        make_response(text="Step done"),
        make_response(text="Custom answer"),
    ]
    custom_planner = "You are a custom planner."
    custom_solver = "You are a custom solver."
    agent = _make_agent(
        responses,
        plan_execute=PlanExecuteConfig(
            planner_system=custom_planner,
            solver_system=custom_solver,
        ),
    )
    result = await agent.run("Test custom")

    assert result.stop_reason == "completed"
    # Planner call has custom system (augmented with tools)
    plan_call = agent.llm.complete.call_args_list[0]
    assert plan_call.kwargs["system"].startswith(custom_planner)
    # Solver call has custom system
    solver_call = agent.llm.complete.call_args_list[-1]
    assert solver_call.kwargs.get("system") == custom_solver


# ---------------------------------------------------------------------------
# 9. Timeout propagation from inner agent
# ---------------------------------------------------------------------------


async def test_timeout_propagation(monkeypatch):
    import ai_arch_toolkit.toolkit.agents._react as react_mod

    _counter = {"n": 0}

    def fake_monotonic():
        _counter["n"] += 1
        return _counter["n"] * 100.0

    monkeypatch.setattr(react_mod.time, "monotonic", fake_monotonic)

    plan_text = "1. Do something slow"
    responses = [
        make_response(text=plan_text),
        make_response(text="Never reached"),
    ]
    agent = _make_agent(
        responses,
        config=AgentConfig(timeout=0.001),
    )
    result = await agent.run("Test timeout")

    assert result.stop_reason == "timeout"
