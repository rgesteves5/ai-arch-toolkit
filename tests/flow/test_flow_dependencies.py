"""A DAG's weak dependencies: ``after_any`` and ``after_optional``, and why a step was skipped
(``StepTrace.blocked_by``, G-33)."""

from __future__ import annotations

import asyncio

import pytest

from ai_arch_toolkit.core import Result, State, StateSnapshot, Step, StepTrace
from ai_arch_toolkit.toolkit.flow import Flow, FlowEvent, FlowStep


def step(name: str, *, fail: bool = False, delay: float = 0.0) -> Step:
    """A step that leaves its name under its own key, or fails."""

    async def fn(snap: StateSnapshot) -> Result:
        if delay:
            await asyncio.sleep(delay)
        if fail:
            return Result(error=f"{name} broke")
        return Result(value=name, artifacts={name: True})

    return Step(name=name, fn=fn)


def joined(*names: str) -> Step:
    """A step that records which of ``names`` it found in the state."""

    async def fn(snap: StateSnapshot) -> Result:
        return Result(value="join", artifacts={"seen": [name for name in names if name in snap]})

    return Step(name="join", fn=fn)


def branches(
    route: str, *, left_fails: bool = False, right_fails: bool = False
) -> tuple[FlowStep, ...]:
    """start, then left or right by a condition, then a join after either."""
    return (
        FlowStep(step=step("start")),
        FlowStep(
            step=step("left", fail=left_fails), after=("start",), when=lambda _: route == "left"
        ),
        FlowStep(
            step=step("right", fail=right_fails), after=("start",), when=lambda _: route == "right"
        ),
        FlowStep(step=joined("left", "right"), after_any=("left", "right")),
    )


async def events_of(flow: Flow) -> list[FlowEvent]:
    return [event async for event in flow.iter(State())]


def skipped(events: list[FlowEvent]) -> dict[str, StepTrace]:
    found = {}
    for event in events:
        if event.type == "step_skipped":
            assert event.step_trace is not None
            found[event.step_name] = event.step_trace
    return found


async def test_the_paths_join_after_a_condition() -> None:
    flow = Flow(*branches("right"), name="join")

    events = await events_of(flow)

    assert list(skipped(events)) == ["left"]
    ended = [event.step_name for event in events if event.type == "step_end"]
    assert ended == ["start", "right", "join"]
    result = await flow.run(State())
    assert result.state["seen"] == ["right"]


async def test_the_join_is_skipped_when_no_path_succeeded() -> None:
    events = await events_of(Flow(*branches("right", right_fails=True), name="join"))

    trace = skipped(events)["join"]
    assert trace.blocked_by == {"left": "skipped", "right": "failed"}
    assert trace.skip_reason is not None
    assert "'left'" in trace.skip_reason and "'right'" in trace.skip_reason


async def test_after_any_runs_when_one_failed_and_another_succeeded() -> None:
    flow = Flow(
        FlowStep(step=step("a", fail=True)),
        FlowStep(step=step("b")),
        FlowStep(step=joined("a", "b"), after_any=("a", "b")),
        name="either",
    )

    result = await flow.run(State())

    join = result.trace.step("join")
    assert join is not None and not join.skipped
    assert result.state["seen"] == ["b"]


async def test_after_any_waits_for_every_one_of_them() -> None:
    flow = Flow(
        FlowStep(step=step("fast")),
        FlowStep(step=step("slow", delay=0.05)),
        FlowStep(step=joined("fast", "slow"), after_any=("fast", "slow")),
        name="wait",
    )

    result = await flow.run(State())

    assert result.state["seen"] == ["fast", "slow"]


async def test_an_optional_dependency_that_fails_does_not_skip_the_step() -> None:
    flow = Flow(
        FlowStep(step=step("start")),
        FlowStep(step=step("enrich", fail=True), after=("start",)),
        FlowStep(step=joined("start", "enrich"), after=("start",), after_optional=("enrich",)),
        name="optional",
    )

    result = await flow.run(State())

    join = result.trace.step("join")
    assert join is not None and not join.skipped
    assert result.state["seen"] == ["start"]


async def test_an_optional_dependency_that_was_skipped_does_not_skip_the_step() -> None:
    flow = Flow(
        FlowStep(step=step("start")),
        FlowStep(step=step("maybe"), after=("start",), when=lambda _: False),
        FlowStep(step=joined("maybe"), after_optional=("maybe",)),
        name="optional",
    )

    result = await flow.run(State())

    assert result.state["seen"] == []


async def test_a_required_dependency_still_blocks_and_says_which() -> None:
    flow = Flow(
        FlowStep(step=step("a", fail=True)),
        FlowStep(step=step("b")),
        FlowStep(step=joined("a", "b"), after=("a",), after_optional=("b",)),
        FlowStep(step=step("then"), after=("join",)),
        name="strict",
    )

    found = skipped(await events_of(flow))

    assert found["join"].blocked_by == {"a": "failed"}
    assert found["join"].skip_reason == "dependency 'a' failed"
    assert found["then"].blocked_by == {"join": "skipped"}
    assert found["then"].skip_reason == "dependency 'join' was skipped"


async def test_a_step_skipped_by_its_own_condition_has_no_blocker() -> None:
    found = skipped(await events_of(Flow(*branches("right"), name="join")))

    assert found["left"].blocked_by == {}
    assert found["left"].skip_reason == "condition not met"


def test_weak_dependencies_make_a_dag() -> None:
    assert Flow(step("a"), FlowStep(step=step("b"), after_any=("a",)), name="f").is_dag
    assert Flow(step("a"), FlowStep(step=step("b"), after_optional=("a",)), name="f").is_dag


@pytest.mark.parametrize("field", ["after_any", "after_optional"])
def test_a_weak_dependency_must_name_a_step(field: str) -> None:
    with pytest.raises(ValueError, match="depends on unknown step 'nope'"):
        Flow(step("a"), FlowStep(step=step("b"), **{field: ("nope",)}), name="f")


def test_a_dependency_is_declared_once() -> None:
    with pytest.raises(ValueError, match="'a' more than once"):
        Flow(step("a"), FlowStep(step=step("b"), after=("a",), after_optional=("a",)), name="f")
    with pytest.raises(ValueError, match="'a' more than once"):
        Flow(step("a"), FlowStep(step=step("b"), after_any=("a", "a")), name="f")


def test_a_cycle_through_a_weak_dependency_is_refused() -> None:
    with pytest.raises(ValueError, match="cycle"):
        Flow(
            FlowStep(step=step("a"), after_any=("b",)),
            FlowStep(step=step("b"), after_optional=("a",)),
            name="f",
        )


def test_blocked_by_survives_serialization() -> None:
    trace = StepTrace(
        name="join",
        skipped=True,
        skip_reason="dependency 'a' failed",
        blocked_by={"a": "failed"},
    )

    assert StepTrace.from_dict(trace.to_dict()) == trace
    assert StepTrace.from_dict(StepTrace(name="plain").to_dict()).blocked_by == {}
