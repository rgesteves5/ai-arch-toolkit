"""A request through each adapter and its real SDK never follows an endpoint from the environment
(D48, the ai-network's G-16).

The environment points at one loopback server; the provider's own API is replaced by another.
Every request reaches the second, and the first, which would get the key, receives nothing.
"""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._providers import OWN_BASE_URLS, create_provider
from tests.integration import fakeserver
from tests.integration.test_provider_transport import CLAUDE_OK, GEMINI_OK, META_OK, RESPONSES_OK
from tests.provider_calls import complete

pytestmark = pytest.mark.integration

HI = [{"role": "user", "content": "hi"}]

CASES = {
    "openai": ("gpt-6-luna", RESPONSES_OK, "/v1"),
    "anthropic": ("claude-haiku-4-5", CLAUDE_OK, ""),
    "gemini": ("gemini-3.8-flash", GEMINI_OK, ""),
    "meta": ("muse-spark-1.3", META_OK, "/v1"),
}
ENVIRONMENT = (
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "GOOGLE_GEMINI_BASE_URL",
    "GOOGLE_VERTEX_BASE_URL",
)


@pytest.mark.parametrize("provider", sorted(CASES))
async def test_the_key_never_reaches_an_endpoint_from_the_environment(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, answer, path = CASES[provider]
    elsewhere, elsewhere_port, elsewhere_stats = await fakeserver.start("status", body=answer)
    own, own_port, own_stats = await fakeserver.start("status", body=answer)
    async with elsewhere, own:
        for name in ENVIRONMENT:
            monkeypatch.setenv(name, f"http://127.0.0.1:{elsewhere_port}{path}")
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
        monkeypatch.setitem(OWN_BASE_URLS, provider, f"http://127.0.0.1:{own_port}{path}")
        adapter = create_provider(model, api_key="the-key")
        response = await complete(adapter, HI, max_tokens=64)
        await adapter.close()
    assert response.text
    assert own_stats.requests == 1
    assert elsewhere_stats.requests == 0
