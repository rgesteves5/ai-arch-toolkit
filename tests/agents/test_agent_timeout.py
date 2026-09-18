"""ReasoningSpec.policy / ReasoningSpec.timeout reach the compiled flow and are enforced."""

from __future__ import annotations

import time
from pathlib import Path
from textwrap import dedent

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents import (
    Agent,
    ReasoningSpec,
    agent_from_manifest,
    build_flow,
    load_agent_manifest,
)
from ai_arch_toolkit.toolkit.agents.flows import react_flow
from tests.fake_provider import Reply, fake_llm

_MODEL = "claude-sonnet-4-6"

_STRATEGIES = (
    "react",
    "completion",
    "plan_execute",
    "rewoo",
    "reflexion",
    "generate_review",
    "self_discovery",
    "llm_compiler",
    "tot",
    "lats",
)


def _llm(text: str = "answer", delay: float = 0.0) -> LLM:
    """A real LLM whose provider waits ``delay`` seconds before every answer."""
    answer = Response(text=text, usage=Usage(input_tokens=10, output_tokens=5), model=_MODEL)
    llm, _ = fake_llm(Reply(response=answer, delay=delay), model=_MODEL)
    return llm


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_policy_and_timeout_both_reach_the_compiled_flow(strategy: str) -> None:
    policy = Policy(retry=RetryConfig(max_retries=2))

    flow = build_flow(
        ReasoningSpec(strategy=strategy, policy=policy, timeout=30.0), _llm(), ToolGroup()
    )

    assert flow.policy is policy
    assert flow.timeout == 30.0


def test_react_factory_keeps_timeout_when_a_policy_is_given() -> None:
    policy = Policy(retry=RetryConfig(max_retries=2))

    flow = react_flow(_llm(), ToolGroup(), policy=policy, timeout=30.0)

    assert flow.policy is policy
    assert flow.timeout == 30.0


async def test_spec_timeout_stops_a_slow_agent() -> None:
    agent = Agent(ReasoningSpec(strategy="react", timeout=0.1), _llm(delay=10.0))

    started = time.monotonic()
    result = await agent.run("hi")

    assert time.monotonic() - started < 3.0
    assert result.text == ""
    assert any("timed out" in error for error in result.errors)


async def test_spec_timeout_cuts_an_inner_flow_of_a_multi_phase_strategy() -> None:
    planner = _llm(text="1. Look it up")
    slow_executor = _llm(delay=10.0)
    agent = Agent(
        ReasoningSpec(strategy="plan_execute", timeout=0.2),
        planner,
        deps={"executor_llm": slow_executor},
    )

    started = time.monotonic()
    result = await agent.run("task")

    assert time.monotonic() - started < 3.0
    assert result.flow_result.trace.steps[-1].name == "flow_timeout"


async def test_spec_policy_timeout_bounds_a_whole_reflexion_attempt() -> None:
    agent = Agent(
        ReasoningSpec(strategy="reflexion", policy=Policy(timeout=0.1)), _llm(delay=10.0)
    )

    started = time.monotonic()
    result = await agent.run("task")

    assert time.monotonic() - started < 3.0
    attempt = next(step for step in result.flow_result.trace.steps if step.name == "attempt")
    assert attempt.error == "Step timed out"


async def test_manifest_timeout_seconds_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "slow.agent.yaml"
    path.write_text(
        dedent(
            """
            version: 1
            id: slow.agent
            strategy:
              name: react
            limits:
              timeout_seconds: 0.1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    manifest = load_agent_manifest(path, allowed_roots=(tmp_path,))
    agent = agent_from_manifest(manifest, _llm(delay=10.0))

    started = time.monotonic()
    result = await agent.run("hi")

    assert time.monotonic() - started < 3.0
    assert any("timed out" in error for error in result.errors)
