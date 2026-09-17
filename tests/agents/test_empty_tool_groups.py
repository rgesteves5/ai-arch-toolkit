"""An empty ``ToolGroup`` is a value, not an absence.

A group that starts empty keeps its identity (tools can be added to it later), and an empty
per-phase group means "this phase has no tools" — it never falls back to the agent's main tools.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

import ai_arch_toolkit.toolkit.agents as agents_package
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response, ToolCall, Usage
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec


@tool
def ping() -> str:
    """Ping."""
    return "pong"


@tool
def wipe_disk() -> str:
    """A main-agent tool that a tool-less phase must never receive."""
    return "wiped"


def _response(text: str = "", tool_calls: tuple[ToolCall, ...] = ()) -> Response:
    return Response(
        text=text, tool_calls=tool_calls, usage=Usage(input_tokens=10, output_tokens=5), cost=0.001
    )


class _RecordingProvider:
    def __init__(self, *responses: Response) -> None:
        self._responses = list(responses)
        self.tool_names: list[list[str]] = []

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.tool_names.append([t["name"] for t in tools or ()])
        return self._responses[min(len(self.tool_names) - 1, len(self._responses) - 1)]


def _llm(*responses: Response) -> tuple[LLM, _RecordingProvider]:
    llm = LLM("claude-sonnet-4-6", api_key="test")
    provider = _RecordingProvider(*responses)
    llm._provider = provider  # type: ignore[assignment]
    return llm, provider


async def test_a_group_that_starts_empty_is_the_group_the_agent_uses() -> None:
    llm, provider = _llm(
        _response(tool_calls=(ToolCall(id="tc_1", name="ping", input={}),)),
        _response("done"),
    )
    group = ToolGroup()
    agent = Agent(ReasoningSpec(strategy="react"), llm, group)

    group.add(ping)  # e.g. tools that arrive after the agent was built
    result = await agent.run("go")

    assert provider.tool_names[0] == ["ping"]
    assert result.text == "done"
    assert not result.errors


# strategy, the dep that carries the phase's tools, and a reply its planner can parse into work
PHASE_TOOL_DEPS = [
    ("plan_execute", "executor_tools", "1. Do the thing"),
    ("reflexion", "executor_tools", "1. Do the thing"),
    ("llm_compiler", "executor_tools", "$1. Do the thing [deps: ]"),
    ("self_discovery", "solver_tools", "1. Do the thing"),
    ("lats", "rollout_tools", "1. Do the thing"),
]


@pytest.mark.parametrize(("strategy", "dep", "reply"), PHASE_TOOL_DEPS)
async def test_an_empty_phase_group_never_falls_back_to_the_main_tools(
    strategy: str, dep: str, reply: str
) -> None:
    llm, provider = _llm(_response(reply))
    spec = ReasoningSpec(strategy=strategy, max_iterations=2)
    agent = Agent(spec, llm, ToolGroup(wipe_disk), deps={dep: ToolGroup()})

    await agent.run("the task")

    assert provider.tool_names, "the strategy made no LLM call"
    assert all("wipe_disk" not in names for names in provider.tool_names)


def test_no_truthiness_fallback_on_a_tool_group_in_the_agents_package() -> None:
    """``x_tools or default`` treats an empty group as missing; the package uses ``is None``."""
    offenders: list[str] = []
    for path in Path(agents_package.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
                first: Any = node.values[0]
                name = getattr(first, "id", getattr(first, "attr", ""))
                if name == "tools" or name.endswith("_tools"):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
