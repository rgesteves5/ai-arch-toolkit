"""Tests for Flow executor — sequential, DAG, parallel, cyclic, skip propagation."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._budget import BudgetPolicy
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import Usage
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowEvent, FlowStep


async def _make_step(name: str, value: str = "ok", artifacts: dict | None = None):
    """Helper to create a simple step."""
    arts = artifacts or {}

    async def fn(snap: StateSnapshot) -> Result:
        return Result(value=value, artifacts=arts)

    return Step(name=name, fn=fn)


class TestSequentialExecution:
    async def test_simple_sequential(self) -> None:
        async def step_a(snap: StateSnapshot) -> Result:
            return Result(value="a", artifacts={"from_a": 1})

        async def step_b(snap: StateSnapshot) -> Result:
            return Result(value="b", artifacts={"from_b": snap.get("from_a", 0) + 1})

        flow = Flow(
            Step(name="a", fn=step_a),
            Step(name="b", fn=step_b),
            name="seq",
        )
        state = State()
        result = await flow.run(state)
        assert result.results["a"].value == "a"
        assert result.results["b"].value == "b"
        assert state["from_a"] == 1
        assert state["from_b"] == 2
        assert result.trace.flow_name == "seq"
        assert len(result.trace.steps) == 2

    async def test_halt_on_error(self) -> None:
        async def fail(snap: StateSnapshot) -> Result:
            return Result(error="boom")

        async def never_reached(snap: StateSnapshot) -> Result:
            return Result(value="should not run")

        flow = Flow(
            Step(name="fail", fn=fail),
            Step(name="after", fn=never_reached),
            name="halt",
        )
        state = State()
        result = await flow.run(state)
        assert "fail" in result.results
        assert "after" not in result.results

    async def test_continue_on_error(self) -> None:
        async def fail(snap: StateSnapshot) -> Result:
            return Result(error="boom")

        async def after(snap: StateSnapshot) -> Result:
            return Result(value="reached")

        policy = Policy(on_exhausted="continue")
        flow = Flow(
            Step(name="fail", fn=fail, policy=policy),
            Step(name="after", fn=after),
            name="cont",
        )
        state = State()
        result = await flow.run(state)
        assert "after" in result.results
        assert result.results["after"].value == "reached"

    async def test_cumulative_cost_budget_stops_flow(self) -> None:
        async def expensive(snap: StateSnapshot) -> Result:
            return Result(value="spent", cost=0.75)

        flow = Flow(
            Step(name="a", fn=expensive),
            Step(name="b", fn=expensive),
            name="budgeted",
            budget_policy=BudgetPolicy(max_cost=1.0),
        )
        state = State()

        result = await flow.run(state)

        assert "budget_exceeded" in result.results
        assert result.trace.metadata["budget"]["exceeded"]["limit"] == "total_cost"
        assert result.trace.steps[-1].name == "budget_exceeded"

    async def test_cumulative_token_budget_stops_flow(self) -> None:
        async def use_tokens(snap: StateSnapshot) -> Result:
            return Result(usage=Usage(input_tokens=8, output_tokens=5))

        flow = Flow(
            Step(name="tokens", fn=use_tokens),
            name="token_budgeted",
            budget_policy=BudgetPolicy(max_total_tokens=10),
        )
        state = State()

        result = await flow.run(state)

        assert "budget_exceeded" in result.results
        assert result.trace.metadata["budget"]["exceeded"]["limit"] == "total_tokens"


class TestCyclicExecution:
    async def test_cyclic_with_condition(self) -> None:
        call_count = 0

        async def increment(snap: StateSnapshot) -> Result:
            nonlocal call_count
            call_count += 1
            current = snap.get("counter", 0)
            return Result(value=current + 1, artifacts={"counter": current + 1})

        flow = Flow(
            FlowStep(
                step=Step(name="inc", fn=increment),
                when=lambda s: s.get("counter", 0) < 3,
            ),
            name="cyclic",
            max_iterations=10,
        )
        state = State()
        await flow.run(state)
        assert state["counter"] == 3
        assert call_count == 3

    async def test_cyclic_max_iterations(self) -> None:
        async def always(snap: StateSnapshot) -> Result:
            return Result(artifacts={"tick": snap.get("tick", 0) + 1})

        flow = Flow(
            FlowStep(
                step=Step(name="tick", fn=always),
                when=lambda s: True,  # always true
            ),
            name="bounded",
            max_iterations=5,
        )
        state = State()
        await flow.run(state)
        assert state["tick"] == 5

    async def test_cyclic_stops_when_no_steps_execute(self) -> None:
        async def noop(snap: StateSnapshot) -> Result:
            return Result()

        flow = Flow(
            FlowStep(
                step=Step(name="noop", fn=noop),
                when=lambda s: False,  # never executes
            ),
            name="dead",
            max_iterations=10,
        )
        state = State()
        result = await flow.run(state)
        # Should complete immediately since condition is never met
        assert len(result.results) == 0


class TestDAGExecution:
    async def test_simple_dag(self) -> None:
        async def step_a(snap: StateSnapshot) -> Result:
            return Result(value="a", artifacts={"a_done": True})

        async def step_b(snap: StateSnapshot) -> Result:
            return Result(value="b", artifacts={"b_done": True})

        async def step_c(snap: StateSnapshot) -> Result:
            return Result(value="c", artifacts={"c_done": True})

        flow = Flow(
            FlowStep(step=Step(name="a", fn=step_a)),
            FlowStep(step=Step(name="b", fn=step_b)),
            FlowStep(step=Step(name="c", fn=step_c), after=("a", "b")),
            name="dag",
        )
        state = State()
        result = await flow.run(state)
        assert result.results["a"].value == "a"
        assert result.results["b"].value == "b"
        assert result.results["c"].value == "c"
        assert state["a_done"] is True
        assert state["c_done"] is True

    async def test_dag_parallel_execution(self) -> None:
        """Steps a and b should run in parallel, c depends on both."""
        import asyncio

        execution_order: list[str] = []

        async def step_a(snap: StateSnapshot) -> Result:
            execution_order.append("a_start")
            await asyncio.sleep(0.01)
            execution_order.append("a_end")
            return Result(artifacts={"a": 1})

        async def step_b(snap: StateSnapshot) -> Result:
            execution_order.append("b_start")
            await asyncio.sleep(0.01)
            execution_order.append("b_end")
            return Result(artifacts={"b": 2})

        async def step_c(snap: StateSnapshot) -> Result:
            execution_order.append("c")
            return Result(artifacts={"c": 3})

        flow = Flow(
            FlowStep(step=Step(name="a", fn=step_a)),
            FlowStep(step=Step(name="b", fn=step_b)),
            FlowStep(step=Step(name="c", fn=step_c), after=("a", "b")),
            name="parallel_dag",
        )
        state = State()
        result = await flow.run(state)
        assert result.results["c"].is_ok
        # a and b started before c
        c_idx = execution_order.index("c")
        assert "a_start" in execution_order[:c_idx]
        assert "b_start" in execution_order[:c_idx]

    async def test_dag_skip_on_failure(self) -> None:
        async def fail(snap: StateSnapshot) -> Result:
            return Result(error="boom")

        async def depends_on_fail(snap: StateSnapshot) -> Result:
            return Result(value="should be skipped")

        flow = Flow(
            FlowStep(step=Step(name="fail", fn=fail)),
            FlowStep(step=Step(name="after", fn=depends_on_fail), after=("fail",)),
            name="skip_dag",
        )
        state = State()
        result = await flow.run(state)
        assert result.results["fail"].is_error
        # 'after' should be skipped
        skip_traces = [t for t in result.trace.steps if t.name == "after" and t.skipped]
        assert len(skip_traces) == 1
        assert "failed" in skip_traces[0].skip_reason

    async def test_dag_skip_all_deps_skipped(self) -> None:
        async def cond_skip(snap: StateSnapshot) -> Result:
            return Result(error="fails")

        async def mid(snap: StateSnapshot) -> Result:
            return Result(value="mid")

        async def end(snap: StateSnapshot) -> Result:
            return Result(value="end")

        flow = Flow(
            FlowStep(step=Step(name="root", fn=cond_skip)),
            FlowStep(step=Step(name="mid", fn=mid), after=("root",)),
            FlowStep(step=Step(name="end", fn=end), after=("mid",)),
            name="cascade_skip",
        )
        state = State()
        result = await flow.run(state)
        mid_trace = result.trace.step("mid")
        end_trace = result.trace.step("end")
        assert mid_trace is not None and mid_trace.skipped
        assert end_trace is not None and end_trace.skipped


class TestFlowStreaming:
    async def test_sequential_events(self) -> None:
        async def fn(snap: StateSnapshot) -> Result:
            return Result(value="ok")

        flow = Flow(Step(name="a", fn=fn), name="stream_test")
        state = State()
        events: list[FlowEvent] = []
        async for event in flow.iter(state):
            events.append(event)

        types = [e.type for e in events]
        assert "flow_start" in types
        assert "flow_end" in types
        assert "step_start" in types
        assert "step_end" in types

    async def test_sync_iter(self) -> None:
        async def fn(snap: StateSnapshot) -> Result:
            return Result(value="ok")

        flow = Flow(Step(name="a", fn=fn), name="sync_stream")
        state = State()
        events = list(flow.iter_sync(state))
        types = [e.type for e in events]
        assert "flow_start" in types
        assert "flow_end" in types


class TestFlowResult:
    async def test_final_result(self) -> None:
        async def fn_a(snap: StateSnapshot) -> Result:
            return Result(value="first")

        async def fn_b(snap: StateSnapshot) -> Result:
            return Result(value="last")

        flow = Flow(Step(name="a", fn=fn_a), Step(name="b", fn=fn_b), name="final")
        state = State()
        result = await flow.run(state)
        assert result.final_result is not None
        assert result.final_result.value == "last"

    async def test_total_cost(self) -> None:
        async def fn(snap: StateSnapshot) -> Result:
            return Result(cost=0.5)

        flow = Flow(Step(name="a", fn=fn), Step(name="b", fn=fn), name="cost")
        state = State()
        result = await flow.run(state)
        assert result.total_cost == pytest.approx(1.0)

    async def test_sync_run(self) -> None:
        async def fn(snap: StateSnapshot) -> Result:
            return Result(value="sync")

        flow = Flow(Step(name="a", fn=fn), name="sync")
        state = State()
        result = flow.run_sync(state)
        assert result.results["a"].value == "sync"
