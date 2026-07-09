"""A foreign (user-supplied) pricer or redactor must never break an already-paid call.

Review findings N6 (raising/estimate-returning pricer flipped a success into a failed op) and
N3 (an unguarded redactor in the store's dispatch broke the settled call).
"""

from __future__ import annotations

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._response import Response, Usage

MODEL = "claude-sonnet-4-6"


class _OkProvider:
    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        return Response(text="ok", usage=Usage(input_tokens=10, output_tokens=5), model=MODEL)


def _llm() -> LLM:
    llm = LLM(MODEL, api_key="test")
    llm._provider = _OkProvider()  # type: ignore[assignment]
    return llm


class _RaisingPricer:
    def price(self, request, usage) -> Cost:
        raise RuntimeError("pricer boom")


class _EstimatingPricer:
    def price(self, request, usage) -> Cost:
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


async def test_raising_redactor_does_not_break_a_paid_call():
    llm = _llm()
    sink = _RecordingSink()
    with MeterScope(RunConfig(redactor=_RaisingRedactor(), sinks=[sink])) as scope:
        resp = await llm.complete("hi")  # must NOT raise despite the redactor blowing up
    assert resp.text == "ok"
    assert scope.snapshot().llm_calls == 1
    assert len(sink.events) >= 1  # the event was still emitted (with metadata dropped)
