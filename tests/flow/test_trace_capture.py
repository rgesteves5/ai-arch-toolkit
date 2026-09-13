"""What a flow's trace records from state and step results: ``Flow(trace_capture=...)``."""

from __future__ import annotations

import json
import threading
from collections import deque
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.core._response import Response, ToolCall, Usage
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.core._trace import StepTrace, Trace
from ai_arch_toolkit.toolkit.agents.flows._plan_execute import (
    plan_execute_flow,
    plan_execute_initial_state,
)
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep


async def _answer(snap: StateSnapshot) -> Result:
    return Result(value="done", artifacts={"answer": "done", "notes": [snap["question"]]})


async def _noop(snap: StateSnapshot) -> Result:
    return Result()


def _answer_flow(**kwargs: Any) -> Flow:
    return Flow(Step(name="answer", fn=_answer), name="single", **kwargs)


class TestKeysByDefault:
    async def test_a_step_records_key_names_not_values(self) -> None:
        result = await _answer_flow().run(State(operational={"question": "why?"}))

        step = result.trace.steps[0]
        assert step.input_state == {}
        assert step.input_keys == {"operational": ("question",)}
        assert "artifacts" not in step.output_result
        assert step.output_result["value"] == "done"
        assert step.output_keys == ("answer", "notes")

    async def test_the_initial_state_is_a_copy_later_steps_cannot_rewrite(self) -> None:
        async def append(snap: StateSnapshot) -> Result:
            snap["items"].append(4)
            return Result()

        state = State(operational={"items": [1, 2, 3]})
        result = await Flow(Step(name="append", fn=append)).run(state)

        assert state["items"] == [1, 2, 3, 4]
        assert result.trace.initial_state["operational"]["items"] == [1, 2, 3]

    def test_the_flow_reports_its_capture_mode(self) -> None:
        assert _answer_flow().trace_capture == "keys"
        assert _answer_flow(trace_capture="full").trace_capture == "full"


class TestFull:
    async def test_each_execution_is_recorded_as_it_was(self) -> None:
        # A step that consumes a queue in place used to rewrite every earlier record.
        async def consume(snap: StateSnapshot) -> Result:
            snap["queue"].popleft()
            return Result(artifacts={"consumed": snap.get("consumed", 0) + 1})

        flow = Flow(
            FlowStep(step=Step(name="consume", fn=consume), when=lambda s: len(s["queue"]) > 0),
            name="drain",
            max_iterations=5,
            trace_capture="full",
        )
        result = await flow.run(State(operational={"queue": deque([1, 2, 3])}))

        recorded = [
            list(st.input_state["operational"]["queue"])
            for st in result.trace.steps
            if not st.skipped
        ]
        assert recorded == [[1, 2, 3], [2, 3], [3]]

    async def test_output_artifacts_are_copied(self) -> None:
        shared = [1]

        async def emit(snap: StateSnapshot) -> Result:
            return Result(artifacts={"shared": shared})

        async def mutate(snap: StateSnapshot) -> Result:
            snap["shared"].append(2)
            return Result()

        flow = Flow(
            Step(name="emit", fn=emit), Step(name="mutate", fn=mutate), trace_capture="full"
        )
        result = await flow.run(State())

        emitted = result.trace.step("emit")
        assert emitted is not None
        assert emitted.output_result["artifacts"]["shared"] == [1]
        assert emitted.output_keys == ("shared",)

    async def test_world_is_kept_by_reference_and_uncopyable_values_do_not_fail(self) -> None:
        lock = threading.Lock()
        resources = {"pool": [1]}
        state = State(operational={"lock": lock, "n": 1}, world={"resources": resources})

        result = await Flow(Step(name="noop", fn=_noop), trace_capture="full").run(state)

        step = result.trace.steps[0]
        assert step.input_state["operational"]["lock"] is lock
        assert step.input_state["operational"]["n"] == 1
        assert step.input_state["world"]["resources"] is resources
        assert step.input_keys == {"operational": ("lock", "n"), "world": ("resources",)}


class TestNone:
    async def test_only_metadata_is_recorded(self) -> None:
        flow = _answer_flow(trace_capture="none")
        result = await flow.run(State(operational={"question": "why?"}))

        step = result.trace.steps[0]
        assert step.name == "answer" and step.error is None
        assert (step.input_state, step.input_keys) == ({}, {})
        assert (step.output_result, step.output_keys) == ({}, ())
        assert result.trace.initial_state == {}
        assert result.results["answer"].artifacts["answer"] == "done"


def test_an_unknown_capture_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="trace_capture"):
        _answer_flow(trace_capture="everything")


class TestSerialization:
    def test_key_names_survive_a_json_round_trip(self) -> None:
        step = StepTrace(name="s", input_keys={"operational": ("a", "b")}, output_keys=("c",))

        restored = StepTrace.from_dict(json.loads(json.dumps(step.to_dict())))

        assert restored.input_keys == {"operational": ("a", "b")}
        assert restored.output_keys == ("c",)

    def test_a_trace_saved_before_the_key_fields_still_loads(self) -> None:
        saved = {
            "flow_name": "f",
            "steps": [{"name": "s", "input_state": {"current": {"x": 1}}, "output_result": {}}],
        }

        step = Trace.from_dict(saved).steps[0]

        assert step.input_state == {"current": {"x": 1}}
        assert (step.input_keys, step.output_keys) == ({}, ())


@tool
def lookup(query: str) -> str:
    """Return a long document."""
    return "x" * 2_000


def _conversation(turns: int) -> list[Response]:
    usage = Usage(input_tokens=10, output_tokens=5)
    calls = [
        Response(
            tool_calls=(ToolCall(id=f"tc{i}", name="lookup", input={"query": "q"}),),
            usage=usage,
            raw={"body": "r" * 2_000},
        )
        for i in range(turns)
    ]
    return [*calls, Response(text="final", usage=usage)]


async def _react_trace_size(turns: int, **kwargs: Any) -> int:
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=_conversation(turns))
    flow = react_flow(llm, ToolGroup(lookup), max_iterations=turns + 1, **kwargs)

    result = await flow.run(State(operational=react_initial_state("go")))

    assert result.trace.steps[-1].error is None
    return len(json.dumps(result.trace.to_dict()))


async def test_a_react_trace_grows_linearly_with_turns() -> None:
    ten, twenty = await _react_trace_size(10), await _react_trace_size(20)

    assert twenty / ten < 2.5


async def test_strategy_inner_flows_record_with_the_same_capture_mode() -> None:
    response = Response(text="1. Do the thing", usage=Usage(input_tokens=10, output_tokens=5))
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[response, Response(text="Done"), Response(text="Final")])
    flow = plan_execute_flow(llm, ToolGroup(), max_replans=0, trace_capture="full")

    result = await flow.run(State(operational=plan_execute_initial_state("task")))

    inner = [
        grandchild
        for step in result.trace.steps
        for child in step.children
        for grandchild in child.children
    ]
    assert inner, "plan_execute should record its inner react run"
    assert all(st.input_state for st in inner if not st.skipped)
