"""Tests for Flow construction, mode detection, and as_step."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowStep
from ai_arch_toolkit.toolkit.flow._scope import Scope


async def _noop(snap: StateSnapshot) -> Result:
    return Result()


class TestFlowConstruction:
    def test_bare_steps(self) -> None:
        s1 = Step(name="a", fn=_noop)
        s2 = Step(name="b", fn=_noop)
        flow = Flow(s1, s2, name="test")
        assert flow.step_names == ("a", "b")
        assert not flow.is_dag

    def test_flow_steps(self) -> None:
        s1 = Step(name="a", fn=_noop)
        s2 = Step(name="b", fn=_noop)
        flow = Flow(
            FlowStep(step=s1),
            FlowStep(step=s2, after=("a",)),
            name="dag",
        )
        assert flow.is_dag
        assert flow.step_names == ("a", "b")

    def test_duplicate_name_raises(self) -> None:
        s1 = Step(name="a", fn=_noop)
        s2 = Step(name="a", fn=_noop)
        with pytest.raises(ValueError, match="Duplicate step name"):
            Flow(s1, s2)

    def test_unknown_after_raises(self) -> None:
        s = Step(name="a", fn=_noop)
        with pytest.raises(ValueError, match="unknown step"):
            Flow(FlowStep(step=s, after=("nonexistent",)))

    def test_cycle_detection(self) -> None:
        s1 = Step(name="a", fn=_noop)
        s2 = Step(name="b", fn=_noop)
        with pytest.raises(ValueError, match="cycle"):
            Flow(
                FlowStep(step=s1, after=("b",)),
                FlowStep(step=s2, after=("a",)),
            )

    def test_nested_flow(self) -> None:
        inner = Flow(Step(name="inner_step", fn=_noop), name="inner")
        outer = Flow(inner, Step(name="after_inner", fn=_noop), name="outer")
        assert "inner" in outer.step_names
        assert "after_inner" in outer.step_names

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected"):
            Flow("not a step")  # type: ignore[arg-type]

    def test_properties(self) -> None:
        p = Policy(timeout=10.0)
        s = Scope(include=frozenset({"x"}))
        flow = Flow(Step(name="a", fn=_noop), name="f", policy=p, scope=s, max_iterations=5)
        assert flow.name == "f"
        assert flow.policy is p
        assert flow.scope is s
        assert flow.max_iterations == 5


class TestFlowModeDetection:
    def test_sequential(self) -> None:
        flow = Flow(Step(name="a", fn=_noop), Step(name="b", fn=_noop))
        assert not flow.is_dag

    def test_dag(self) -> None:
        flow = Flow(
            FlowStep(step=Step(name="a", fn=_noop)),
            FlowStep(step=Step(name="b", fn=_noop), after=("a",)),
        )
        assert flow.is_dag

    def test_conditional_dag(self) -> None:
        flow = Flow(
            FlowStep(step=Step(name="a", fn=_noop)),
            FlowStep(
                step=Step(name="b", fn=_noop),
                after=("a",),
                when=lambda s: True,
            ),
        )
        # Has deps — DAG mode (when conditions evaluated during execution)
        assert flow.is_dag

    def test_cyclic_requires_max_iterations(self) -> None:
        with pytest.raises(ValueError, match="must set `max_iterations`"):
            Flow(
                FlowStep(step=Step(name="a", fn=_noop), when=lambda s: True),
                name="cyclic_no_limit",
            )

    def test_cyclic_with_max_iterations_ok(self) -> None:
        flow = Flow(
            FlowStep(step=Step(name="a", fn=_noop), when=lambda s: True),
            name="cyclic_limited",
            max_iterations=5,
        )
        assert not flow.is_dag
        assert flow.max_iterations == 5


class TestFlowAsStep:
    async def test_as_step_runs(self) -> None:
        async def add_one(snap: StateSnapshot) -> Result:
            return Result(value=snap.get("x", 0) + 1, artifacts={"x": snap.get("x", 0) + 1})

        inner = Flow(Step(name="add", fn=add_one), name="inner")
        step = inner.as_step()
        assert step.name == "inner"

        state = State(operational={"x": 5})
        result = await step.fn(state.snapshot())
        assert result.value == 6
