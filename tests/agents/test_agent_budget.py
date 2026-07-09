"""Every flow factory and the Agent honour a budget (phase B2).

A ``max_llm_calls=0`` cap must deny the very first model call and surface ``budget_exceeded`` —
proof the ``budget_policy=`` wiring reaches the owning meter scope for each flow.
"""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._agent import Agent
from ai_arch_toolkit.toolkit.agents.flows._generate_review import (
    generate_review_flow,
    generate_review_initial_state,
)
from ai_arch_toolkit.toolkit.agents.flows._plan_execute import (
    plan_execute_flow,
    plan_execute_initial_state,
)
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.agents.flows._reflexion import (
    reflexion_flow,
    reflexion_initial_state,
)
from ai_arch_toolkit.toolkit.agents.flows._rewoo import rewoo_flow, rewoo_initial_state
from ai_arch_toolkit.toolkit.agents.flows._self_discovery import (
    self_discovery_flow,
    self_discovery_initial_state,
)
from ai_arch_toolkit.toolkit.budget import BudgetPolicy


class _FakeProvider:
    """A real LLM's provider stand-in, so LLM.complete runs its metering charge site."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls += 1
        return Response(text="ok", usage=Usage(input_tokens=10, output_tokens=5), cost=0.001)


def _metered_llm() -> tuple[LLM, _FakeProvider]:
    llm = LLM("claude-sonnet-4-6", api_key="test")
    provider = _FakeProvider()
    llm._provider = provider  # type: ignore[assignment]
    return llm, provider


def _budget_dimension(result) -> str:
    info = result.results["budget_exceeded"].artifacts["budget_exceeded"]
    return info.get("dimension") or (info.get("breached") or [None])[0]


ZERO = BudgetPolicy(max_llm_calls=0)


def _build(name: str, budget: BudgetPolicy | None):
    """Return (flow, state, providers) for each newly budget-enabled flow."""
    llm, prov = _metered_llm()
    if name == "plan_execute":
        flow = plan_execute_flow(llm, ToolGroup(), budget_policy=budget)
        return flow, State(operational=plan_execute_initial_state("t")), [prov]
    if name == "reflexion":
        flow = reflexion_flow(llm, ToolGroup(), evaluator=lambda t, r: 1.0, budget_policy=budget)
        return flow, State(operational=reflexion_initial_state("t")), [prov]
    if name == "rewoo":
        flow = rewoo_flow(llm, ToolGroup(), budget_policy=budget)
        return flow, State(operational=rewoo_initial_state("t")), [prov]
    if name == "self_discovery":
        flow = self_discovery_flow(llm, ToolGroup(), budget_policy=budget)
        return flow, State(operational=self_discovery_initial_state("t")), [prov]
    if name == "generate_review":
        llm2, prov2 = _metered_llm()
        flow = generate_review_flow(llm, llm2, budget_policy=budget)
        return flow, State(operational=generate_review_initial_state("t")), [prov, prov2]
    raise AssertionError(name)


FLOW_NAMES = ["plan_execute", "reflexion", "rewoo", "self_discovery", "generate_review"]


@pytest.mark.parametrize("name", FLOW_NAMES)
async def test_flow_honours_budget_policy(name: str) -> None:
    flow, state, providers = _build(name, ZERO)
    result = await flow.run(state)
    assert "budget_exceeded" in result.results
    assert _budget_dimension(result) == "llm_calls"
    assert all(p.calls == 0 for p in providers)  # denied before any provider call


@pytest.mark.parametrize("name", FLOW_NAMES)
async def test_flow_meters_every_call_end_to_end(name: str) -> None:
    # The invariant the rewrite guarantees: with the manual per-strategy accounting gone, the meter
    # counts EXACTLY the calls the provider served — no drift, no double-count — and cost/usage
    # derive from it. Runs to completion under no budget (measure-only).
    flow, state, providers = _build(name, None)
    result = await flow.run(state)
    assert "budget_exceeded" not in result.results  # ran to completion
    calls = sum(p.calls for p in providers)
    assert calls >= 1
    report = result.meter
    assert report is not None
    assert report.llm_calls == calls  # meter == actual provider calls
    assert result.total_cost == report.cost > 0
    assert result.usage.input_tokens == 10 * calls  # meter-summed, not manually threaded


async def test_agent_result_report_and_usage_are_meter_derived() -> None:
    llm, provider = _metered_llm()
    agent = Agent.from_flow(
        react_flow(llm, ToolGroup(), max_iterations=1),
        init_state=lambda task: react_initial_state(task),
    )
    result = await agent.run("t")
    assert provider.calls == 1
    assert result.report is not None and result.report.llm_calls == 1
    assert result.usage.input_tokens == 10  # from the meter, not manual threading
    assert result.cost == result.report.cost > 0


async def test_agent_per_run_budget_denies_the_first_call() -> None:
    llm, provider = _metered_llm()
    agent = Agent.from_flow(
        react_flow(llm, ToolGroup()),
        init_state=lambda task: react_initial_state(task),
    )
    result = await agent.run("t", budget_policy=ZERO)
    assert "budget_exceeded" in result.flow_result.results
    assert provider.calls == 0


async def test_agent_per_run_budget_overrides_flow_construction_budget() -> None:
    # The backing flow is built with a strict cap; a looser per-run policy lifts it for one run.
    llm, provider = _metered_llm()
    agent = Agent.from_flow(
        react_flow(llm, ToolGroup(), max_iterations=1, budget_policy=ZERO),
        init_state=lambda task: react_initial_state(task),
    )
    result = await agent.run("t", budget_policy=BudgetPolicy(max_llm_calls=5))
    assert "budget_exceeded" not in result.flow_result.results
    assert provider.calls == 1
