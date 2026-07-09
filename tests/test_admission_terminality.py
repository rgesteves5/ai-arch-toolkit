"""AdmissionDenied must stay terminal — never swallowed, retried, or masked by a fallback.

Covers review findings: N1 (fallback loops mask a denial), #1 (completion builder swallows it),
N7 (LLMModerator swallows it). Contract: the denial escapes even under fallback_on=(Exception,).
"""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering._admission import AdmissionDenied
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.toolkit.agents._agent import Agent
from ai_arch_toolkit.toolkit.agents._spec import ReasoningSpec
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetExceeded, BudgetPolicy
from ai_arch_toolkit.toolkit.moderation._llm import LLMModerator


def _denial() -> BudgetExceeded:
    return BudgetExceeded(dimension="cost", limit=1.0, current=2.0, attempted=0.0)


class _RaisingProvider:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def complete(self, *a, **k) -> Response:
        raise self._exc


class _RaisingStreamProvider:
    """Provider whose stream openers raise — to drive a denial *inside* the try block,
    so the `except self._fallback_on` guard (not the pre-try `_open_stream_op`) is exercised."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def stream(self, *a, **k):
        raise self._exc

    def stream_events(self, *a, **k):
        raise self._exc


class _OkProvider:
    async def complete(self, *a, **k) -> Response:
        return Response(
            text='{"flagged": false, "categories": []}',
            usage=Usage(input_tokens=5, output_tokens=2),
        )


class _FakeFallback:
    """Minimal fallback-LLM stand-in for the fallback loop (needs .complete / ._model)."""

    def __init__(
        self, model: str, *, response: Response | None = None, raises: Exception | None = None
    ):
        self._model = model
        self._response = response
        self._raises = raises
        self.called = False

    async def complete(self, messages, **kwargs) -> Response:
        self.called = True
        if self._raises is not None:
            raise self._raises
        assert self._response is not None
        return self._response

    def stream(self, messages, **kwargs):
        self.called = True
        raise AssertionError("fallback stream must not be reached after a denial")

    def stream_events(self, messages, **kwargs):
        self.called = True
        raise AssertionError("fallback stream_events must not be reached after a denial")


def _real_llm(provider) -> LLM:
    llm = LLM("claude-sonnet-4-6", api_key="test")
    llm._provider = provider  # type: ignore[assignment]
    return llm


async def test_complete_fallback_does_not_mask_a_denial():
    # Primary provider-errors -> enters fallbacks; the FIRST fallback is budget-denied. Under a
    # broad fallback_on the denial must escape, NOT be swallowed by a healthy later fallback.
    primary = _real_llm(_RaisingProvider(RuntimeError("primary down")))
    denied = _FakeFallback("fb-denied", raises=_denial())
    healthy = _FakeFallback("fb-healthy", response=Response(text="MASKED", usage=Usage()))
    primary._fallbacks = [denied, healthy]  # type: ignore[assignment]
    primary._fallback_on = (Exception,)  # type: ignore[assignment]

    with pytest.raises(AdmissionDenied):
        await primary.complete("hi")
    assert healthy.called is False  # short-circuited on the denial; no later fallback tried


async def test_complete_primary_denial_does_not_enter_fallbacks():
    # A real budget denial on the PRIMARY (from scope.open) must not trigger the fallback chain.
    primary = _real_llm(_RaisingProvider(RuntimeError("provider must not be reached")))
    healthy = _FakeFallback("fb", response=Response(text="MASKED", usage=Usage()))
    primary._fallbacks = [healthy]  # type: ignore[assignment]
    primary._fallback_on = (Exception,)  # type: ignore[assignment]

    scope = MeterScope(RunConfig(controller=BudgetController(BudgetPolicy(max_llm_calls=0))))
    with scope, pytest.raises(AdmissionDenied):
        await primary.complete("hi")
    assert healthy.called is False


async def test_completion_builder_surfaces_a_budget_denial():
    # The completion strategy must surface a denial as budget_exceeded, not a swallowed error.
    agent = Agent(ReasoningSpec(strategy="completion"), _real_llm(_OkProvider()))
    result = await agent.run("hi", budget_policy=BudgetPolicy(max_llm_calls=0))
    assert "budget_exceeded" in result.flow_result.results


async def test_llm_moderator_reraises_a_budget_denial():
    # A budget denial from the classifier LLM must escape, not become a moderation fail-result.
    mod = LLMModerator(_real_llm(_OkProvider()), ["hate"])
    scope = MeterScope(RunConfig(controller=BudgetController(BudgetPolicy(max_llm_calls=0))))
    with scope, pytest.raises(AdmissionDenied):
        await mod.moderate("some text")


async def test_stream_fallback_does_not_mask_a_denial():
    # A denial surfacing from the primary stream opener must escape the `except self._fallback_on`
    # guard under a broad fallback_on — never masked by a healthy later fallback.
    primary = _real_llm(_RaisingStreamProvider(_denial()))
    healthy = _FakeFallback("fb", response=Response(text="MASKED", usage=Usage()))
    primary._fallbacks = [healthy]  # type: ignore[assignment]
    primary._fallback_on = (Exception,)  # type: ignore[assignment]

    with pytest.raises(AdmissionDenied):
        primary.stream("hi")
    assert healthy.called is False  # short-circuited on the denial; no fallback stream tried


async def test_stream_events_fallback_does_not_mask_a_denial():
    # Same terminality contract for the stream_events path.
    primary = _real_llm(_RaisingStreamProvider(_denial()))
    healthy = _FakeFallback("fb", response=Response(text="MASKED", usage=Usage()))
    primary._fallbacks = [healthy]  # type: ignore[assignment]
    primary._fallback_on = (Exception,)  # type: ignore[assignment]

    with pytest.raises(AdmissionDenied):
        primary.stream_events("hi")
    assert healthy.called is False
