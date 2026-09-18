"""AdmissionDenied must stay terminal — never swallowed, retried, or masked by a fallback.

Covers review findings: N1 (fallback loops mask a denial), #1 (completion builder swallows it),
N7 (LLMModerator swallows it). Contract: the denial escapes even under fallback_on=(Exception,).
"""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._admission import (
    AdmissionDecision,
    AdmissionDenied,
    MeterSnapshot,
)
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.toolkit.agents._agent import Agent
from ai_arch_toolkit.toolkit.agents._spec import ReasoningSpec
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetExceeded, BudgetPolicy
from ai_arch_toolkit.toolkit.moderation._llm import LLMModerator
from tests.fake_provider import FakeProvider

MODEL = "claude-sonnet-4-6"  # every LLM here, primary or fallback, unless noted
OTHER = "claude-haiku-4-5"


def _denial() -> BudgetExceeded:
    return BudgetExceeded(dimension="cost", limit=1.0, current=2.0, attempted=0.0)


def _ok_provider() -> FakeProvider:
    verdict = Response(
        text='{"flagged": false, "categories": []}',
        usage=Usage(input_tokens=5, output_tokens=2),
    )
    return FakeProvider(verdict, model=MODEL)


def _healthy() -> FakeProvider:
    """A fallback that would answer (and so mask a denial) if it were ever reached."""
    return FakeProvider(Response(text="MASKED", usage=Usage()), model=MODEL)


def _real_llm(provider: FakeProvider) -> LLM:
    """An actual LLM, so terminality crosses the shared pipeline."""
    llm = LLM(MODEL, api_key="test")
    llm._provider = provider
    return llm


async def test_complete_fallback_does_not_mask_a_denial():
    # Primary provider-errors -> enters fallbacks; the FIRST fallback is budget-denied. Under a
    # broad fallback_on the denial must escape, NOT be swallowed by a healthy later fallback.
    primary = _real_llm(FakeProvider(RuntimeError("primary down"), model=MODEL))
    denied = FakeProvider(_denial(), model=MODEL)
    healthy = _healthy()
    primary._fallbacks = [_real_llm(denied), _real_llm(healthy)]
    primary._fallback_on = (Exception,)  # type: ignore[assignment]

    with pytest.raises(AdmissionDenied):
        await primary.complete("hi")
    assert denied.calls == 1  # the denial came from the first fallback's call
    assert healthy.calls == 0  # short-circuited on the denial; no later fallback tried


class _DenyModel:
    """Denies only one model's operations, so a fallback on another model is admissible."""

    def __init__(self, model: str) -> None:
        self.model = model

    def admit(self, snapshot: MeterSnapshot, request: OperationRequest) -> AdmissionDecision:
        if request.model == self.model:
            return AdmissionDecision.deny(_denial())
        return AdmissionDecision.allow()


async def test_complete_primary_denial_does_not_enter_fallbacks():
    # A real denial on the PRIMARY (from scope.open) must not trigger the fallback chain, even
    # though the fallback's own admission would pass and it would answer.
    unreachable = FakeProvider(RuntimeError("provider must not be reached"), model=MODEL)
    primary = _real_llm(unreachable)
    healthy = FakeProvider(Response(text="MASKED", usage=Usage()), model=OTHER)
    fallback = LLM(OTHER, api_key="test")
    fallback._provider = healthy
    primary._fallbacks = [fallback]
    primary._fallback_on = (Exception,)  # type: ignore[assignment]

    with MeterScope(RunConfig(controller=_DenyModel(MODEL))), pytest.raises(AdmissionDenied):
        await primary.complete("hi")
    assert unreachable.calls == 0
    assert healthy.calls == 0
    # The fallback alone is admitted: had the denial been masked, it would have answered.
    with MeterScope(RunConfig(controller=_DenyModel(MODEL))):
        assert (await fallback.complete("hi")).text == "MASKED"


async def test_completion_builder_surfaces_a_budget_denial():
    # The completion strategy must surface a denial as budget_exceeded, not a swallowed error.
    agent = Agent(ReasoningSpec(strategy="completion"), _real_llm(_ok_provider()))
    result = await agent.run("hi", budget_policy=BudgetPolicy(max_llm_calls=0))
    assert "budget_exceeded" in result.flow_result.results


async def test_llm_moderator_reraises_a_budget_denial():
    # A budget denial from the classifier LLM must escape, not become a moderation fail-result.
    mod = LLMModerator(_real_llm(_ok_provider()), ["hate"])
    scope = MeterScope(RunConfig(controller=BudgetController(BudgetPolicy(max_llm_calls=0))))
    with scope, pytest.raises(AdmissionDenied):
        await mod.moderate("some text")


async def test_stream_fallback_does_not_mask_a_denial():
    # A denial raised by the primary's stream — after admission, *inside* the try block — must
    # escape the `except self._fallback_on` guard under a broad fallback_on, never masked by a
    # healthy later fallback.
    primary = _real_llm(FakeProvider(_denial(), model=MODEL))
    healthy = _healthy()
    primary._fallbacks = [_real_llm(healthy)]
    primary._fallback_on = (Exception,)  # type: ignore[assignment]

    stream = primary.stream("hi")
    with pytest.raises(AdmissionDenied):
        async for _ in stream:
            pass
    assert healthy.calls == 0  # short-circuited on the denial; no fallback stream tried


async def test_stream_events_fallback_does_not_mask_a_denial():
    # Same terminality contract for the stream_events path.
    primary = _real_llm(FakeProvider(_denial(), model=MODEL))
    healthy = _healthy()
    primary._fallbacks = [_real_llm(healthy)]
    primary._fallback_on = (Exception,)  # type: ignore[assignment]

    stream = primary.stream_events("hi")
    with pytest.raises(AdmissionDenied):
        async for _ in stream:
            pass
    assert healthy.calls == 0
