"""D16/D20: under a meter, a model without a price fails before any request."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.fake_provider import FakeProvider
from tests.test_failure_matrix import invoke

from ai_arch_toolkit.core import (
    LLM,
    Cost,
    MeterScope,
    ModelPricing,
    Money,
    RequestError,
    Response,
    RunConfig,
    Usage,
    pricing,
)
from ai_arch_toolkit.core._exceptions import APIError, UnpricedModelError
from ai_arch_toolkit.core._metering._operation import OperationRequest

MODEL = "house-model-7"


def configured(provider: FakeProvider) -> LLM:
    llm = LLM(provider._model, provider="openai", api_key="test", max_tokens=32)
    llm._provider = provider
    return llm


def house(*script: Response | BaseException) -> FakeProvider:
    return FakeProvider(*script, model=MODEL)


@pytest.fixture
def registered() -> Iterator[None]:
    pricing.register(MODEL, ModelPricing())
    try:
        yield
    finally:
        pricing.unregister(MODEL)


@pytest.mark.parametrize("path", ("complete", "stream", "stream_events"))
async def test_an_unpriced_model_fails_before_any_request(path: str) -> None:
    provider = house()
    with MeterScope(RunConfig(retain_meter_events=True)) as scope:
        with pytest.raises(UnpricedModelError) as raised:
            await invoke(configured(provider), path)
        snapshot = scope.snapshot()
    assert isinstance(raised.value, RequestError)
    assert f"pricing.register('{MODEL}'" in str(raised.value)
    assert "ModelPricing()" in str(raised.value)  # how a local model registers zero
    assert provider.calls == 0
    assert (snapshot.llm_calls, snapshot.out_llm_calls) == (0, 0)
    assert scope.events() == ()


async def test_without_a_scope_the_call_runs_and_its_cost_is_unknown() -> None:
    response = await configured(house()).complete("hi")
    assert response.text == "ok"
    assert response.cost is None


@pytest.mark.usefixtures("registered")
async def test_an_explicit_zero_price_lets_a_local_model_run() -> None:
    with MeterScope() as scope:
        response = await configured(house()).complete("hi")
    assert response.text == "ok"
    assert scope.snapshot().cost == Money.zero()
    assert scope.snapshot().unknown_cost_count == 0


async def test_the_scope_pricer_decides_what_is_priced() -> None:
    class HousePricer:
        def price(self, request: OperationRequest, usage: Usage) -> Cost:
            if request.model == MODEL:
                return Cost.known(Money.from_usd(usage.output_tokens / 1_000_000))
            return Cost.unknown("not a house model")

    with MeterScope(RunConfig(pricer=HousePricer())) as scope:
        response = await configured(house()).complete("hi")
    assert response.text == "ok"
    assert scope.snapshot().cost == Money.from_usd(5 / 1_000_000)


async def test_an_unpriced_fallback_is_terminal() -> None:
    backup = house()
    primary = configured(FakeProvider(APIError(503, "busy"), model="gpt-4o"))
    primary._fallbacks = [configured(backup)]
    with MeterScope() as scope, pytest.raises(UnpricedModelError):
        await primary.complete("hi")
    assert scope.snapshot().llm_calls == 1  # only the primary's attempt started
    assert backup.calls == 0
