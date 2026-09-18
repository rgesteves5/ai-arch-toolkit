"""Each factory hands the four Flow options to its Flow; an inner ReAct records as the outer."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import Response
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._trace import StepTrace, TraceCapture
from ai_arch_toolkit.toolkit.agents.flows import (
    generate_review_flow,
    generate_review_initial_state,
    lats_flow,
    lats_initial_state,
    llm_compiler_flow,
    llm_compiler_initial_state,
    plan_execute_flow,
    plan_execute_initial_state,
    react_flow,
    react_initial_state,
    reflexion_flow,
    reflexion_initial_state,
    rewoo_flow,
    rewoo_initial_state,
    self_discovery_flow,
    self_discovery_initial_state,
    tot_flow,
    tot_initial_state,
)
from ai_arch_toolkit.toolkit.agents.flows._common import nested
from ai_arch_toolkit.toolkit.budget import BudgetPolicy
from ai_arch_toolkit.toolkit.flow._flow import Flow

# One answer every phase can read: a one-step plan (plan_execute) that is also a one-task DAG
# (llm_compiler), with no tool call (an inner ReAct ends on it) and no ACCEPT (one review cycle).
_ANSWER = "1. Do the thing\n$1. Do the thing [deps: none]"


def _score(_task: str, _answer: str) -> float:
    return 1.0


type _Factory = Callable[..., Flow]

# Each factory with the arguments it needs to run to its end on ``_ANSWER``, and its initial state.
_FACTORIES: dict[str, tuple[_Factory, Callable[[str], dict[str, Any]]]] = {
    "react": (lambda llm, **o: react_flow(llm, ToolGroup(), **o), react_initial_state),
    "reflexion": (
        lambda llm, **o: reflexion_flow(llm, ToolGroup(), evaluator=_score, **o),
        reflexion_initial_state,
    ),
    "rewoo": (lambda llm, **o: rewoo_flow(llm, ToolGroup(), **o), rewoo_initial_state),
    "plan_execute": (
        lambda llm, **o: plan_execute_flow(llm, ToolGroup(), max_replans=0, **o),
        plan_execute_initial_state,
    ),
    "tot": (
        lambda llm, **o: tot_flow(llm, ToolGroup(), n_candidates=1, max_depth=1, **o),
        tot_initial_state,
    ),
    "lats": (
        lambda llm, **o: lats_flow(llm, ToolGroup(), evaluator_fn=_score, max_rollouts=1, **o),
        lats_initial_state,
    ),
    "self_discovery": (
        lambda llm, **o: self_discovery_flow(llm, ToolGroup(), **o),
        self_discovery_initial_state,
    ),
    "llm_compiler": (
        lambda llm, **o: llm_compiler_flow(llm, ToolGroup(), max_replans=0, **o),
        llm_compiler_initial_state,
    ),
    "generate_review": (
        lambda llm, **o: generate_review_flow(
            llm, llm, gen_tools=ToolGroup(), review_tools=ToolGroup(), max_cycles=1, **o
        ),
        generate_review_initial_state,
    ),
}

# The strategies that run a ReAct loop inside one of their steps.
_WITH_INNER_REACT = (
    "reflexion",
    "plan_execute",
    "lats",
    "self_discovery",
    "llm_compiler",
    "generate_review",
)


def _llm() -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=Response(text=_ANSWER))
    return llm


def _nested_steps(steps: tuple[StepTrace, ...], depth: int = 0) -> Iterator[StepTrace]:
    """The step traces below the outer flow's own steps: the inner flows' steps."""
    for step in steps:
        if depth >= 2:
            yield step
        yield from _nested_steps(step.children, depth + 1)


@pytest.mark.parametrize("name", sorted(_FACTORIES))
def test_the_flow_options_reach_the_flow(name: str) -> None:
    factory, _ = _FACTORIES[name]
    policy = Policy(timeout=5.0)
    budget = BudgetPolicy(max_llm_calls=3)

    flow = factory(_llm(), timeout=30.0, trace_capture="none", policy=policy, budget_policy=budget)

    assert (flow.timeout, flow.trace_capture, flow.policy, flow.budget_policy) == (
        30.0,
        "none",
        policy,
        budget,
    )


@pytest.mark.parametrize("name", sorted(_FACTORIES))
def test_a_factory_without_options_builds_the_flow_defaults(name: str) -> None:
    factory, _ = _FACTORIES[name]

    flow = factory(_llm())

    assert (flow.timeout, flow.trace_capture, flow.policy, flow.budget_policy) == (
        None,
        "keys",
        None,
        None,
    )


@pytest.mark.parametrize("name", sorted(_FACTORIES))
def test_an_unknown_option_is_refused(name: str) -> None:
    factory, _ = _FACTORIES[name]

    with pytest.raises(TypeError, match="deadline"):
        factory(_llm(), deadline=3.0)


@pytest.mark.parametrize("capture", ["full", "none"])
@pytest.mark.parametrize("name", _WITH_INNER_REACT)
async def test_an_inner_react_records_its_trace_as_the_outer_flow_does(
    name: str, capture: TraceCapture
) -> None:
    factory, initial_state = _FACTORIES[name]
    flow = factory(_llm(), trace_capture=capture)

    result = await flow.run(State(operational=initial_state("task")))

    inner = [step for step in _nested_steps(result.trace.steps) if not step.skipped]
    assert inner, f"{name} should record its inner ReAct run"
    if capture == "full":
        assert all(step.input_state for step in inner)
    else:
        assert not any(step.input_state or step.input_keys for step in inner)


def test_an_inner_flow_takes_only_the_trace_capture_of_its_outer_flow() -> None:
    options = {
        "timeout": 3.0,
        "trace_capture": "full",
        "policy": Policy(timeout=1.0),
        "budget_policy": BudgetPolicy(max_llm_calls=1),
    }

    assert nested(options) == {"trace_capture": "full"}
    assert nested({"timeout": 3.0}) == {}
