"""RunConfig knobs (review #6/#7): allow_unmetered_batch, sink_error_policy, retain_meter_events,
and per-run Flow.run(config=...)."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._admission import NotMeteredOperationError
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy
from ai_arch_toolkit.toolkit.flow._flow import Flow

MODEL = "claude-sonnet-4-6"


class _FullProvider:
    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        return Response(text="ok", usage=Usage(input_tokens=10, output_tokens=5), model=MODEL)

    async def batch_submit(self, requests) -> str:
        return "batch-1"


def _llm() -> LLM:
    llm = LLM(MODEL, api_key="test")
    llm._provider = _FullProvider()  # type: ignore[assignment]
    return llm


def _enforcing(**cfg_kwargs) -> RunConfig:
    return RunConfig(controller=BudgetController(BudgetPolicy(max_cost=1.0)), **cfg_kwargs)


class _BadSink:
    def emit(self, event) -> None:
        raise RuntimeError("sink boom")


# ── allow_unmetered_batch (#7) ────────────────────────────────────────────────


async def test_batch_is_rejected_under_enforcement_by_default():
    with MeterScope(_enforcing()), pytest.raises(NotMeteredOperationError):
        await _llm().batch_submit([{"x": 1}])


async def test_allow_unmetered_batch_permits_batch_under_enforcement():
    with MeterScope(_enforcing(allow_unmetered_batch=True)):
        assert await _llm().batch_submit([{"x": 1}]) == "batch-1"


# ── sink_error_policy (#6) ────────────────────────────────────────────────────


async def test_sink_error_policy_log_is_the_default():
    with MeterScope(RunConfig(sinks=[_BadSink()])):
        resp = await _llm().complete("hi")  # a raising sink must not break the call
    assert resp.text == "ok"


async def test_sink_error_policy_raise_propagates():
    with (
        MeterScope(RunConfig(sinks=[_BadSink()], sink_error_policy="raise")),
        pytest.raises(RuntimeError),
    ):
        await _llm().complete("hi")


# ── retain_meter_events (#6) ──────────────────────────────────────────────────


async def test_retain_meter_events_keeps_events():
    scope = MeterScope(RunConfig(retain_meter_events=True))
    with scope:
        await _llm().complete("hi")
    events = scope.events()
    assert len(events) >= 1 and events[0].op_id


async def test_events_is_empty_without_retain():
    assert MeterScope(RunConfig()).events() == ()


# ── Flow.run(config=...) (#6) ─────────────────────────────────────────────────


async def _call(snap) -> Result:
    await _llm().complete("hi")
    return Result(value="ok")


async def test_flow_run_config_builds_the_scope_and_enforces():
    flow = Flow(Step(name="a", fn=_call), Step(name="b", fn=_call), name="cfg")
    cfg = RunConfig(controller=BudgetController(BudgetPolicy(max_llm_calls=1)))
    result = await flow.run(State(), config=cfg)
    assert "budget_exceeded" in result.results


async def test_flow_run_config_takes_precedence_over_budget_policy():
    # construction budget denies at 0; a measure-only config lifts it for this run.
    flow = Flow(Step(name="a", fn=_call), name="p", budget_policy=BudgetPolicy(max_llm_calls=0))
    result = await flow.run(State(), config=RunConfig())
    assert "budget_exceeded" not in result.results
