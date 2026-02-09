"""Integration tests for all agent architectures."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from ai_arch_toolkit.agents._base import AgentConfig
from ai_arch_toolkit.agents._compiler import LLMCompilerAgent
from ai_arch_toolkit.agents._lats import LATSAgent
from ai_arch_toolkit.agents._plan_execute import PlanExecuteAgent
from ai_arch_toolkit.agents._react import ReActAgent
from ai_arch_toolkit.agents._reflexion import ReflexionAgent
from ai_arch_toolkit.agents._rewoo import ReWOOAgent
from ai_arch_toolkit.agents._self_discovery import SelfDiscoveryAgent
from ai_arch_toolkit.agents._tot import TreeOfThoughtsAgent
from ai_arch_toolkit.llm._types import Response, Tool, ToolCall, Usage
from ai_arch_toolkit.tools._registry import ToolRegistry


def _calc_registry() -> ToolRegistry:
    """Create a registry with real calculator tools."""
    reg = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    def multiply(a: int, b: int) -> int:
        return a * b

    reg.register(
        "add",
        add,
        Tool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        ),
    )
    reg.register(
        "multiply",
        multiply,
        Tool(
            name="multiply",
            description="Multiply two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        ),
    )
    return reg


def test_react_with_registry_e2e():
    """ReAct with real calculator tools executes tool calls."""
    tool_response = Response(
        text="",
        tool_calls=(ToolCall(id="tc_1", name="add", arguments={"a": 5, "b": 3}),),
        usage=Usage(),
    )
    final_response = Response(text="5 + 3 = 8", usage=Usage())
    client = MagicMock()
    client.chat = MagicMock(side_effect=[tool_response, final_response])

    reg = _calc_registry()
    agent = ReActAgent(client, reg)
    result = agent.run("What is 5 + 3?")

    assert result.answer == "5 + 3 = 8"
    assert result.steps[0].tool_results[0].content == "8"


def test_rewoo_with_registry_e2e():
    """ReWOO plans and executes with a real search tool."""
    reg = ToolRegistry()
    reg.register(
        "search",
        lambda input: f"result: {input}",
        Tool(name="search", description="", parameters={}),
    )

    plan_text = "#E1 = search[python]"
    responses = [
        Response(text=plan_text, usage=Usage()),
        Response(text="Python is great.", usage=Usage()),
    ]
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)

    agent = ReWOOAgent(client, reg)
    result = agent.run("Tell me about Python")

    assert result.answer == "Python is great."
    assert len(result.steps) == 3


def test_reflexion_with_react_inner_e2e():
    """Reflexion wraps ReAct and retries on low score."""
    responses = [
        Response(text="first try", usage=Usage()),
        Response(text="try harder", usage=Usage()),
        Response(text="better answer", usage=Usage()),
    ]
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)

    reg = ToolRegistry()
    scores = iter([0.3, 1.0])
    agent = ReflexionAgent(client, reg)
    result = agent.run("solve", evaluator=lambda x: next(scores))

    assert result.answer == "better answer"


def test_plan_execute_with_registry_e2e():
    """PlanExecute with tool registry runs multi-step plan."""
    plan_json = json.dumps(["Calculate 2+3"])
    responses = [
        Response(text=plan_json, usage=Usage()),
        Response(text="2+3 = 5", usage=Usage()),
        Response(text="The answer is 5.", usage=Usage()),
    ]
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)

    reg = _calc_registry()
    agent = PlanExecuteAgent(client, reg)
    result = agent.run("What is 2+3?")

    assert result.answer == "The answer is 5."


def test_compiler_parallel_e2e():
    """LLMCompiler executes parallel DAG tasks."""
    dag_json = json.dumps(
        [
            {"id": "1", "tool": "add", "args": {"a": "1", "b": "2"}, "deps": []},
            {"id": "2", "tool": "multiply", "args": {"a": "3", "b": "4"}, "deps": []},
        ]
    )
    responses = [
        Response(text=dag_json, usage=Usage()),
        Response(text="1+2=3, 3*4=12", usage=Usage()),
    ]
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)

    reg = _calc_registry()
    agent = LLMCompilerAgent(client, reg)
    result = agent.run("Add and multiply")

    assert result.answer == "1+2=3, 3*4=12"


def test_self_discovery_e2e():
    """SelfDiscovery runs four phases with no tools."""
    responses = [
        Response(text="Critical Thinking", usage=Usage()),
        Response(text="Adapted: Analyze critically", usage=Usage()),
        Response(text='{"step1": "Analyze", "step2": "Evaluate"}', usage=Usage()),
        Response(text="Final insight.", usage=Usage()),
    ]
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)

    reg = ToolRegistry()
    agent = SelfDiscoveryAgent(client, reg)
    result = agent.run("Analyze this problem")

    assert result.answer == "Final insight."
    assert len(result.steps) == 4


def test_tot_e2e():
    """TreeOfThoughts explores and scores thoughts."""
    responses = [
        Response(text="1. Think about X\n2. Think about Y", usage=Usage()),
        Response(text="0.9", usage=Usage()),
        Response(text="0.3", usage=Usage()),
    ]
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)

    reg = ToolRegistry()
    agent = TreeOfThoughtsAgent(client, reg)
    result = agent.run("Solve", max_depth=1, branching_factor=2, beam_width=1)

    assert result.answer != "(start)"


def test_lats_e2e():
    """LATS runs MCTS with simulation."""
    responses = [
        Response(text="1. Strategy A", usage=Usage()),
        Response(text="0.8", usage=Usage()),
        Response(text="Solution found!", usage=Usage()),
    ]
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)

    reg = ToolRegistry()
    config = AgentConfig(max_iterations=1)
    agent = LATSAgent(client, reg, config=config)
    result = agent.run("Solve", num_expansions=1)

    assert result.answer == "Solution found!"
