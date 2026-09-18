"""A foreign (user-supplied) pricer or redactor must never break an already-paid call.

Review findings N6 (raising/estimate-returning pricer flipped a success into a failed op) and
N3 (an unguarded redactor in the store's dispatch broke the settled call). The pricer is also
asked, with zero usage, whether it prices the model before anything is sent (D16); the pricers
below answer that probe and misbehave only when a paid call settles.
"""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._exceptions import UnpricedModelError
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._response import Response, Usage
from tests.fake_provider import fake_llm

MODEL = "claude-sonnet-4-6"
# A paid call reports non-zero usage, so the pricers below misbehave when it settles.
_OK = Response(text="ok", usage=Usage(input_tokens=10, output_tokens=5), model=MODEL)


def _llm() -> LLM:
    llm, _ = fake_llm(_OK, model=MODEL)
    return llm


def _probe(usage: Usage) -> bool:
    return usage == Usage()


class _RaisingPricer:
    def price(self, request, usage) -> Cost:
        if _probe(usage):
            return Cost.known(Money.zero())
        raise RuntimeError("pricer boom")


class _EstimatingPricer:
    def price(self, request, usage) -> Cost:
        if _probe(usage):
            return Cost.known(Money.zero())
        return Cost.estimated(Money.from_usd(0.01))  # settle rejects estimates


class _RaisingRedactor:
    def redact(self, value):
        raise RuntimeError("redactor boom")


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)


async def test_raising_pricer_settles_unknown_not_fails():
    llm = _llm()
    with MeterScope(RunConfig(pricer=_RaisingPricer())) as scope:
        resp = await llm.complete("hi")  # must NOT raise
    assert resp.text == "ok"
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1  # settled as unknown, not failed


async def test_estimate_returning_pricer_settles_unknown_not_fails():
    # settle() records ACTUALS and rejects an estimated cost; a pricer returning one must not raise
    # out of the success path — it degrades to unknown.
    llm = _llm()
    with MeterScope(RunConfig(pricer=_EstimatingPricer())) as scope:
        resp = await llm.complete("hi")
    assert resp.text == "ok"
    snap = scope.snapshot()
    assert snap.llm_calls == 1 and snap.unknown_cost_count == 1


async def test_a_pricer_that_raises_before_the_call_blocks_it_unsent():
    class AlwaysRaising:
        def price(self, request, usage) -> Cost:
            raise RuntimeError("pricer boom")

    llm, provider = fake_llm(_OK, model=MODEL)
    with (
        MeterScope(RunConfig(pricer=AlwaysRaising())) as scope,
        pytest.raises(UnpricedModelError) as raised,
    ):
        await llm.complete("hi")
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert provider.calls == 0 and scope.snapshot().llm_calls == 0


async def test_raising_redactor_does_not_break_a_paid_call():
    llm = _llm()
    sink = _RecordingSink()
    with MeterScope(RunConfig(redactor=_RaisingRedactor(), sinks=[sink])) as scope:
        resp = await llm.complete("hi")  # must NOT raise despite the redactor blowing up
    assert resp.text == "ok"
    assert scope.snapshot().llm_calls == 1
    assert len(sink.events) >= 1  # the event was still emitted (with metadata dropped)
