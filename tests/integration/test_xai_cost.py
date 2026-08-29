"""Live verification that local xAI token pricing matches the provider's exact charge."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core import LLM, MeterScope
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from tests.integration.conftest import skip_no_xai

MODEL = "grok-4.6"
pytestmark = pytest.mark.live_api


@skip_no_xai
@pytest.mark.timeout(120)
@pytest.mark.integration
async def test_grok_46_local_cost_matches_reported_ticks() -> None:
    """Exercise reasoning, cache, exact-cost propagation, and the local pricing fallback."""
    async with LLM(MODEL, max_tokens=128) as llm:
        with MeterScope() as scope:
            response = await llm.complete(
                "Calculate 17 * 19, check the result internally, and answer with only the number."
            )

    reported_ticks = response.raw.usage.cost_in_usd_ticks
    reported_cost = reported_ticks / 10_000_000_000
    local_cost = _estimate_response_cost(MODEL, response.usage)

    assert response.raw.usage.reasoning_tokens > 0
    assert local_cost is not None
    assert local_cost == pytest.approx(reported_cost, abs=1e-10)
    assert response.provider_cost == pytest.approx(reported_cost, abs=1e-12)
    assert response.cost == pytest.approx(reported_cost, abs=1e-12)
    assert scope.snapshot().cost.to_float() == pytest.approx(reported_cost, abs=1e-12)
