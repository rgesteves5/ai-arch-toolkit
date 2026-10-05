"""A paid tool is charged what its service billed, at the table's price (D56)."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core import (
    AdmissionDenied,
    MeterScope,
    Money,
    RunConfig,
    ToolCall,
    ToolGroup,
    ToolPricing,
    pricing,
    tool,
)
from ai_arch_toolkit.core._tools._billing import bill
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy

RAN: list[int] = []


@pytest.fixture(autouse=True)
def priced() -> object:
    RAN.clear()
    pricing.register_tool("paid", ToolPricing(per_unit=0.01))
    yield
    pricing.unregister_tool("paid")


@tool
def paid(units: int) -> str:
    """Make a request its service bills ``units`` for (none when 0)."""
    RAN.append(units)
    if units:
        bill("paid", units)
    return "done"


@tool
async def paid_async(units: int) -> str:
    """The same, as a coroutine."""
    if units:
        bill("paid", units)
    return "done"


@tool
def paid_then_broken(units: int) -> str:
    """Bill, then fail."""
    bill("paid", units)
    raise RuntimeError("after the request")


def _call(name: str, units: int) -> ToolCall:
    return ToolCall(id="t1", name=name, input={"units": units})


def test_a_tool_is_charged_the_units_its_service_billed() -> None:
    with MeterScope() as scope:
        result = ToolGroup(paid).execute(_call("paid", 2))

    assert result.ok
    assert scope.snapshot().cost == Money.from_usd(0.02)


def test_a_call_its_service_did_not_bill_costs_nothing() -> None:
    with MeterScope() as scope:
        ToolGroup(paid).execute(_call("paid", 0))

    snapshot = scope.snapshot()
    assert snapshot.cost == Money.zero()
    assert snapshot.tool_calls == 1


async def test_the_async_path_charges_the_same() -> None:
    with MeterScope() as scope:
        await ToolGroup(paid_async).async_execute(_call("paid_async", 3))

    assert scope.snapshot().cost == Money.from_usd(0.03)


async def test_a_sync_tool_bills_from_its_own_thread() -> None:
    with MeterScope() as scope:
        await ToolGroup(paid).async_execute(_call("paid", 1))

    assert scope.snapshot().cost == Money.from_usd(0.01)


def test_a_tool_that_fails_after_its_request_still_pays_for_it() -> None:
    with MeterScope() as scope:
        result = ToolGroup(paid_then_broken).execute(_call("paid_then_broken", 1))

    assert not result.ok
    assert scope.snapshot().cost == Money.from_usd(0.01)


def test_a_strict_budget_below_one_unit_refuses_before_the_tool_runs() -> None:
    policy = BudgetPolicy(max_cost=0.005, reserve="strict")

    with (
        MeterScope(RunConfig(controller=BudgetController(policy))),
        pytest.raises(AdmissionDenied),
    ):
        ToolGroup(paid).execute(_call("paid", 1))

    assert RAN == []


def test_outside_a_meter_nothing_is_charged_and_nothing_breaks() -> None:
    assert ToolGroup(paid).execute(_call("paid", 1)).ok
