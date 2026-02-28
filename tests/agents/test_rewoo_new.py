"""Tests for ReWOOAgent — Plan → Execute → Solve."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import OutputSchema, Response, Usage
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import AgentConfig, AgentEvent, AgentResult
from ai_arch_toolkit.toolkit.agents._rewoo import ReWOOAgent, ReWOOConfig
from tests.agents.conftest import make_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    llm_side_effect: list[Response],
    *,
    config: AgentConfig | None = None,
    rewoo: ReWOOConfig | None = None,
    tools: ToolGroup | None = None,
) -> ReWOOAgent:
    """Build a ReWOOAgent with mocked LLM."""

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
    return ReWOOAgent(
        llm,
        tools,
        config=config or AgentConfig(),
        rewoo=rewoo,
    )


# ---------------------------------------------------------------------------
# 1. Simple plan-execute-solve
# ---------------------------------------------------------------------------


async def test_simple_plan_execute_solve():
    plan_text = (
        "Step 1: Look up the population.\n"
        "#E1 = lookup[population of France]\n"
        "Step 2: Calculate something.\n"
        "#E2 = calculate[#E1 * 2]\n"
    )
    responses = [
        make_response(text=plan_text),  # Plan
        make_response(text="Final answer"),  # Solve
    ]
    agent = _make_agent(responses)
    result = await agent.run("What is the population of France times 2?")

    assert result.stop_reason == "completed"
    assert result.answer == "Final answer"
    # Plan + Solve = 2 steps with responses (tool steps have no response)
    assert len(result.steps) == 2


# ---------------------------------------------------------------------------
# 2. Placeholder substitution (#E1 in #E2's args gets replaced)
# ---------------------------------------------------------------------------


async def test_placeholder_substitution():
    plan_text = "#E1 = lookup[France]\n#E2 = calculate[#E1 times 2]\n"
    responses = [
        make_response(text=plan_text),
        make_response(text="Done"),
    ]
    events: list[AgentEvent] = []
    config = AgentConfig(on_event=events.append)
    agent = _make_agent(responses, config=config)
    await agent.run("Test substitution")

    # Find the tool_call event for calculate
    calc_events = [e for e in events if e.type == "tool_call" and e.tool_name == "calculate"]
    assert len(calc_events) == 1
    # #E1 should have been substituted with the lookup result
    arg_value = calc_events[0].tool_args.get("expression", "")
    assert "info:France" in arg_value
    assert "#E1" not in arg_value


# ---------------------------------------------------------------------------
# 3. Unknown tool → error event + graceful continuation
# ---------------------------------------------------------------------------


async def test_unknown_tool():
    plan_text = "#E1 = nonexistent_tool[hello]\n"
    responses = [
        make_response(text=plan_text),
        make_response(text="Solved anyway"),
    ]
    events: list[AgentEvent] = []
    config = AgentConfig(on_event=events.append)
    agent = _make_agent(responses, config=config)
    result = await agent.run("Test unknown tool")

    assert result.stop_reason == "completed"
    # Error event should be emitted
    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert "nonexistent_tool" in error_events[0].error


# ---------------------------------------------------------------------------
# 4. Single-step plan
# ---------------------------------------------------------------------------


async def test_single_step_plan():
    plan_text = "#E1 = lookup[Paris]\n"
    responses = [
        make_response(text=plan_text),
        make_response(text="Paris is the capital"),
    ]
    agent = _make_agent(responses)
    result = await agent.run("Capital of France?")

    assert result.stop_reason == "completed"
    # Plan + Solve = 2 steps with responses
    assert len(result.steps) == 2


# ---------------------------------------------------------------------------
# 5. Empty plan — no tool steps parsed, goes straight to solve
# ---------------------------------------------------------------------------


async def test_empty_plan():
    """When the planner produces no #E steps, the agent skips to solve."""
    responses = [
        make_response(text="I don't need any tools for this."),
        make_response(text="Direct answer"),
    ]
    agent = _make_agent(responses)
    result = await agent.run("Simple question")

    assert result.stop_reason == "completed"
    assert result.answer == "Direct answer"
    # Plan + Solve = 2 steps, no tool execution steps
    assert len(result.steps) == 2


# ---------------------------------------------------------------------------
# 6. Event sequence (plan → tool calls → solve)
# ---------------------------------------------------------------------------


async def test_event_sequence():
    plan_text = "#E1 = lookup[test]\n"
    responses = [
        make_response(text=plan_text),
        make_response(text="Answer"),
    ]
    events: list[AgentEvent] = []
    config = AgentConfig(on_event=events.append)
    agent = _make_agent(responses, config=config)
    await agent.run("Test events")

    types = [e.type for e in events]
    # Plan: step_start, step_end
    # Execute: step_start, tool_call, tool_result, step_end
    # Solve: step_start, step_end
    assert types == [
        "step_start",
        "step_end",  # Plan
        "step_start",
        "tool_call",
        "tool_result",
        "step_end",  # Execute
        "step_start",
        "step_end",  # Solve
    ]

    # Step numbers should be 1, 1, 2, 2, 2, 2, 3, 3
    steps = [e.step for e in events]
    assert steps == [1, 1, 2, 2, 2, 2, 3, 3]


# ---------------------------------------------------------------------------
# 7. run_sync works
# ---------------------------------------------------------------------------


def test_run_sync():
    plan_text = "#E1 = lookup[sync]\n"
    responses = [
        make_response(text=plan_text),
        make_response(text="Sync answer"),
    ]
    agent = _make_agent(responses)
    result = agent.run_sync("Sync test")

    assert isinstance(result, AgentResult)
    assert result.answer == "Sync answer"
    assert result.stop_reason == "completed"


# ---------------------------------------------------------------------------
# 8. Solver receives all evidence + forwards output_schema
# ---------------------------------------------------------------------------


async def test_solver_receives_evidence_and_schema():
    plan_text = "#E1 = lookup[data]\n"
    parsed_data = {"result": "42"}
    solve_response = Response(
        text='{"result": "42"}',
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=0.001,
        parsed=parsed_data,
    )
    responses = [
        make_response(text=plan_text),
        solve_response,
    ]
    schema = OutputSchema(name="Result", schema={"type": "object"})
    config = AgentConfig(output_schema=schema)
    agent = _make_agent(responses, config=config)
    result = await agent.run("Get result")

    assert result.parsed == parsed_data
    assert result.stop_reason == "completed"

    # Solver call should have output_schema
    solver_call = agent.llm.complete.call_args_list[1]
    assert solver_call.kwargs.get("output_schema") is schema

    # Solver message should contain evidence
    solver_messages = solver_call.args[0]
    solver_text = solver_messages[0]["content"]
    assert "info:data" in solver_text
