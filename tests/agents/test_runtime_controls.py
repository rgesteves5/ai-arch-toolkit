"""Tests for unified runtime controls: stream flag, cancellation, and budgets."""

from __future__ import annotations

from unittest.mock import MagicMock

from ai_arch_toolkit._legacy.agents import (
    AgentConfig,
    AgentResult,
    LATSAgent,
    LLMCompilerAgent,
    PlanExecuteAgent,
    ReActAgent,
    ReflexionAgent,
    ReWOOAgent,
    SelfDiscoveryAgent,
    TreeOfThoughtsAgent,
)
from ai_arch_toolkit._legacy.llm._types import Response, Tool, Usage
from ai_arch_toolkit._legacy.tools._registry import ToolRegistry


def _client_with_responses(responses: list[Response]) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)
    client._provider_name = "openai"
    client._model = "gpt-4o"
    return client


def _tool_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "search",
        lambda input: f"search:{input}",
        Tool(name="search", description="", parameters={}),
    )
    reg.register(
        "summarize",
        lambda input: f"summary:{input}",
        Tool(name="summarize", description="", parameters={}),
    )
    return reg


def test_stream_flag_react_matches_non_stream_steps() -> None:
    responses = [Response(text="done", usage=Usage(total_tokens=5))]
    client = _client_with_responses(responses * 2)
    reg = _tool_registry()
    agent = ReActAgent(client, reg)

    result = agent.run("task")
    streamed = list(agent.run("task", stream=True))

    assert streamed == list(result.steps)


def test_stream_flag_plan_execute_matches_non_stream_steps() -> None:
    responses = [
        Response(text='["step 1"]', usage=Usage(total_tokens=1)),
        Response(text="step one done", usage=Usage(total_tokens=1)),
        Response(text="final", usage=Usage(total_tokens=1)),
    ]
    client = _client_with_responses(responses * 2)
    reg = _tool_registry()
    agent = PlanExecuteAgent(client, reg)

    result = agent.run("task")
    streamed = list(agent.run("task", stream=True))

    assert streamed == list(result.steps)


def test_stream_flag_self_discovery_matches_non_stream_steps() -> None:
    responses = [
        Response(text="Critical Thinking", usage=Usage(total_tokens=1)),
        Response(text="Adapted module", usage=Usage(total_tokens=1)),
        Response(text='{"step":"do"}', usage=Usage(total_tokens=1)),
        Response(text="Solved", usage=Usage(total_tokens=1)),
    ]
    client = _client_with_responses(responses * 2)
    reg = _tool_registry()
    agent = SelfDiscoveryAgent(client, reg)

    result = agent.run("task")
    streamed = list(agent.run("task", stream=True))

    assert streamed == list(result.steps)


def test_stream_flag_reflexion_matches_non_stream_steps() -> None:
    responses = [Response(text="good", usage=Usage(total_tokens=1))]
    client = _client_with_responses(responses * 2)
    reg = _tool_registry()
    agent = ReflexionAgent(client, reg)

    result = agent.run("task", evaluator=lambda _x: 1.0)
    streamed = list(agent.run("task", stream=True, evaluator=lambda _x: 1.0))

    assert streamed == list(result.steps)


def test_stream_flag_rewoo_matches_non_stream_steps() -> None:
    responses = [
        Response(text="#E1 = search[q]", usage=Usage(total_tokens=1)),
        Response(text="final", usage=Usage(total_tokens=1)),
    ]
    client = _client_with_responses(responses * 2)
    reg = _tool_registry()
    agent = ReWOOAgent(client, reg)

    result = agent.run("task")
    streamed = list(agent.run("task", stream=True))

    assert streamed == list(result.steps)


def test_stream_flag_tot_matches_non_stream_steps() -> None:
    responses = [
        Response(text="1. Thought A", usage=Usage(total_tokens=1)),
        Response(text="0.9", usage=Usage(total_tokens=1)),
    ]
    client = _client_with_responses(responses * 2)
    reg = _tool_registry()
    agent = TreeOfThoughtsAgent(client, reg)

    result = agent.run("task", max_depth=1, branching_factor=1, beam_width=1)
    streamed = list(agent.run("task", stream=True, max_depth=1, branching_factor=1, beam_width=1))

    assert streamed == list(result.steps)


def test_stream_flag_lats_matches_non_stream_steps() -> None:
    responses = [
        Response(text="1. Try approach A", usage=Usage(total_tokens=1)),
        Response(text="0.9", usage=Usage(total_tokens=1)),
        Response(text="final answer", usage=Usage(total_tokens=1)),
    ]
    client = _client_with_responses(responses * 2)
    reg = _tool_registry()
    config = AgentConfig(max_iterations=1)
    agent = LATSAgent(client, reg, config=config)

    result = agent.run("task", num_expansions=1)
    streamed = list(agent.run("task", stream=True, num_expansions=1))

    assert streamed == list(result.steps)


def test_stream_flag_compiler_matches_non_stream_steps() -> None:
    responses = [
        Response(
            text='[{"id":"1","tool":"search","args":{"input":"q"},"deps":[]}]',
            usage=Usage(total_tokens=1),
        ),
        Response(text="joined", usage=Usage(total_tokens=1)),
    ]
    client = _client_with_responses(responses * 2)
    reg = _tool_registry()
    agent = LLMCompilerAgent(client, reg)

    result = agent.run("task")
    streamed = list(agent.run("task", stream=True))

    assert streamed == list(result.steps)


def test_token_budget_enforces_limit() -> None:
    client = _client_with_responses([Response(text="too long", usage=Usage(total_tokens=10))])
    reg = _tool_registry()
    config = AgentConfig(max_tokens=5)
    agent = ReActAgent(client, reg, config=config)

    result = agent.run("task")

    assert result.stop_reason == "budget_exhausted"
    assert result.total_usage.total_tokens == 10


def test_nested_budget_propagates_in_plan_execute() -> None:
    responses = [
        Response(text='["step 1"]', usage=Usage(total_tokens=8)),
    ]
    client = _client_with_responses(responses)
    reg = _tool_registry()
    config = AgentConfig(max_tokens=5)
    agent = PlanExecuteAgent(client, reg, config=config)

    result = agent.run("task")

    assert result.stop_reason == "budget_exhausted"
    assert result.answer == "[token budget exceeded]"


def test_plan_execute_repair_usage_counts_toward_budget() -> None:
    responses = [
        Response(text="this is not json", usage=Usage(total_tokens=2)),
        Response(text='["step 1"]', usage=Usage(total_tokens=4)),
    ]
    client = _client_with_responses(responses)
    reg = _tool_registry()
    config = AgentConfig(max_tokens=5, planner_repair_retries=1)
    agent = PlanExecuteAgent(client, reg, config=config)

    result = agent.run("task")

    assert result.stop_reason == "budget_exhausted"
    assert result.total_usage.total_tokens == 6
    assert len(result.steps) == 2
    assert result.steps[1].metadata.get("repair_attempt") is True


def test_rewoo_repair_usage_counts_toward_budget() -> None:
    responses = [
        Response(text="not a plan", usage=Usage(total_tokens=1)),
        Response(
            text='[{"id":"E1","tool":"search","input":"q"}]',
            usage=Usage(total_tokens=2),
        ),
    ]
    client = _client_with_responses(responses)
    reg = _tool_registry()
    config = AgentConfig(max_tokens=2, planner_repair_retries=1)
    agent = ReWOOAgent(client, reg, config=config)

    result = agent.run("task")

    assert result.stop_reason == "budget_exhausted"
    assert result.total_usage.total_tokens == 3
    assert len(result.steps) == 2
    assert result.steps[1].metadata.get("repair_attempt") is True


def test_compiler_repair_usage_counts_toward_budget() -> None:
    responses = [
        Response(text="not json", usage=Usage(total_tokens=1)),
        Response(
            text='[{"id":"1","tool":"search","args":{"input":"q"},"deps":[]}]',
            usage=Usage(total_tokens=3),
        ),
    ]
    client = _client_with_responses(responses)
    reg = _tool_registry()
    config = AgentConfig(max_tokens=3, planner_repair_retries=1)
    agent = LLMCompilerAgent(client, reg, config=config)

    result = agent.run("task")

    assert result.stop_reason == "budget_exhausted"
    assert result.total_usage.total_tokens == 4
    assert len(result.steps) == 2
    assert result.steps[1].metadata.get("repair_attempt") is True


def test_base_result_normalizes_max_iterations_answer() -> None:
    result = AgentResult(answer="[max iterations reached]")
    assert result.stop_reason == "max_iterations"
