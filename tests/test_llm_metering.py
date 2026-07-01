"""LLM.complete charge site: metered per attempt, enforced, failed-attempt kept, sync + pricer."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._admission import (
    AdmissionDecision,
    AdmissionDenied,
    MeterSnapshot,
    ResourceLimits,
)
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._response import Response, Usage

MODEL = "claude-sonnet-4-6"  # priced in _default_pricing.toml


class FakeProvider:
    """Stands in for a real provider — returns a canned Response or raises."""

    def __init__(
        self, *, response: Response | None = None, error: Exception | None = None
    ) -> None:
        self._response = response
        self._error = error
        self.calls = 0

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class CapController:
    def __init__(self, **limits: int) -> None:
        self._limits = ResourceLimits(**limits)

    def admit(self, snapshot: MeterSnapshot, request: OperationRequest) -> AdmissionDecision:
        return AdmissionDecision.allow(limits=self._limits)


def make_llm(provider: FakeProvider) -> LLM:
    llm = LLM(MODEL, api_key="test")
    llm._provider = provider  # type: ignore[assignment]  # inject the fake
    return llm


def resp(**usage: int) -> Response:
    return Response(text="ok", usage=Usage(**usage), model=MODEL)


async def test_complete_without_a_scope_is_unchanged():
    llm = make_llm(FakeProvider(response=resp(input_tokens=10)))
    out = await llm.complete("hi")  # no MeterScope bound -> charge site is inert
    assert out.text == "ok"


async def test_complete_meters_one_llm_call_with_cost():
    llm = make_llm(FakeProvider(response=resp(input_tokens=1000, output_tokens=500)))
    with MeterScope() as scope:
        await llm.complete("hi")
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.out_llm_calls == 0
    assert snap.input_tokens == 1000 and snap.output_tokens == 500
    assert snap.cost.pico > 0 and snap.unknown_cost_count == 0


async def test_enforcing_scope_denies_over_the_call_cap():
    prov = FakeProvider(response=resp(input_tokens=10))
    llm = make_llm(prov)
    with (
        MeterScope(RunConfig(controller=CapController(max_llm_calls=0))) as scope,
        pytest.raises(AdmissionDenied),
    ):
        await llm.complete("hi")
    assert scope.snapshot().llm_calls == 0
    assert prov.calls == 0  # denied before the provider was ever touched


async def test_failed_attempt_keeps_the_count_as_unknown_cost():
    prov = FakeProvider(error=ValueError("boom"))  # non-retryable, not a PROVIDER_ERROR
    llm = make_llm(prov)
    with MeterScope() as scope, pytest.raises(ValueError, match="boom"):
        await llm.complete("hi")
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1
    assert snap.out_llm_calls == 0 and snap.out_cost == Money.zero()


async def test_runconfig_pricer_overrides_the_default():
    class FixedPricer:
        def price(self, request: OperationRequest, usage: Usage) -> Cost:
            return Cost.known(Money.from_usd(0.42))

    llm = make_llm(FakeProvider(response=resp(input_tokens=100)))
    with MeterScope(RunConfig(pricer=FixedPricer())) as scope:
        await llm.complete("hi")
    assert scope.snapshot().cost == Money.from_usd(0.42)


def test_complete_sync_is_metered_too():
    # Plain sync test: complete_sync -> _run_sync (no running loop -> same-thread asyncio.run),
    # so the scope bound in this thread is visible to the coroutine.
    llm = make_llm(FakeProvider(response=resp(input_tokens=20, output_tokens=5)))
    with MeterScope() as scope:
        llm.complete_sync("hi")
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.input_tokens == 20 and snap.output_tokens == 5
