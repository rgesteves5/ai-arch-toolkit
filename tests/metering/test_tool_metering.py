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
