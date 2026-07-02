"""Tool charge site in the executor: metered, failed=free, enforced, gate-blocked=unmetered."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._metering._admission import (
    AdmissionDecision,
    AdmissionDenied,
    MeterSnapshot,
    ResourceLimits,
)
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._response import ToolCall
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._executor import async_execute_tool, execute_tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def boom() -> str:
    """Always raises."""
    raise RuntimeError("kaboom")


@tool(requires_approval=True)
def risky() -> str:
    """Needs approval to run."""
    return "did the risky thing"


class CapController:
    def __init__(self, **limits: int) -> None:
        self._limits = ResourceLimits(**limits)

    def admit(self, snapshot: MeterSnapshot, request: OperationRequest) -> AdmissionDecision:
        return AdmissionDecision.allow(limits=self._limits)


def tc(name: str, **inp: object) -> ToolCall:
    return ToolCall(id="tc_1", name=name, input=inp)


async def test_tool_call_is_metered():
    with MeterScope() as scope:
        result = await async_execute_tool(tc("add", a=1, b=2), [add])
    assert result.ok
    snap = scope.snapshot()
    assert snap.tool_calls == 1 and snap.out_tool_calls == 0
    assert snap.cost == Money.zero() and snap.unknown_cost_count == 0  # a tool is free by default


def test_sync_tool_call_is_metered():
    with MeterScope() as scope:
        execute_tool(tc("add", a=2, b=3), [add])
    assert scope.snapshot().tool_calls == 1


async def test_failed_tool_keeps_the_count_but_stays_free():
    with MeterScope() as scope:
        result = await async_execute_tool(tc("boom"), [boom])
    assert not result.ok  # tools return an error ToolResult, never raise
    snap = scope.snapshot()
    assert snap.tool_calls == 1 and snap.unknown_cost_count == 0  # failed tool is free
    assert snap.out_tool_calls == 0


async def test_enforcing_scope_denies_over_the_tool_cap():
    with (
        MeterScope(RunConfig(controller=CapController(max_tool_calls=0))) as scope,
        pytest.raises(AdmissionDenied),
    ):
        await async_execute_tool(tc("add", a=1, b=2), [add])
    assert scope.snapshot().tool_calls == 0


async def test_unmetered_tool_is_unchanged():
    result = await async_execute_tool(tc("add", a=1, b=2), [add])  # no scope bound
    assert result.ok and result.value == 3


async def test_gate_blocked_tool_is_not_metered():
    # requires_approval + no handler -> ApprovalGate blocks BEFORE the meter opens.
    with MeterScope() as scope:
        result = await async_execute_tool(tc("risky"), [risky])
    assert not result.ok  # blocked by the approval gate
    assert scope.snapshot().tool_calls == 0  # never metered — the tool did not run


async def test_nested_admission_denied_propagates_out_of_the_executor():
    # A tool whose body hits a budget cap (agent-as-tool). The denial must propagate, not become
    # a retryable ToolResult — otherwise the outer loop keeps retrying and the cap is defeated.
    @tool
    def calls_budget() -> str:
        """Simulate a nested metered call that gets denied inside the tool body."""
        raise AdmissionDenied(dimension="cost")

    with MeterScope() as _scope, pytest.raises(AdmissionDenied):
        await async_execute_tool(tc("calls_budget"), [calls_budget])


def test_sync_nested_admission_denied_propagates():
    @tool
    def calls_budget() -> str:
        """Denied inside the tool body (sync path)."""
        raise AdmissionDenied(dimension="cost")

    with MeterScope() as _scope, pytest.raises(AdmissionDenied):
        execute_tool(tc("calls_budget"), [calls_budget])


async def test_tool_settles_free_when_a_pricer_returns_an_estimate():
    # A misbehaving custom pricer returning an estimate for a tool must not make settle() raise
    # (which the tool's except would turn into an error result).
    from ai_arch_toolkit.core._metering._cost import Cost
    from ai_arch_toolkit.core._metering._scope import RunConfig

    class EstimatePricer:
        def price(self, request, usage):
            return Cost.estimated(Money.from_usd(0.01))

    with MeterScope(RunConfig(pricer=EstimatePricer())) as scope:
        result = await async_execute_tool(tc("add", a=1, b=2), [add])
    assert result.ok and result.value == 3  # not flipped to an error
    assert scope.snapshot().tool_calls == 1
