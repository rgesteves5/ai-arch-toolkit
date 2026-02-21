"""16 — Fallback Client (OpenAI -> Anthropic).

Try one provider first, then automatically fall back to another
provider on retryable API errors.
"""

import asyncio

from ai_arch_toolkit import AsyncClient, Client, FallbackClient

sync_primary = Client("openai", model="gpt-5-nano")
sync_fallback = Client("anthropic", model="claude-haiku-4-5-20251001")

sync_client = FallbackClient([sync_primary, sync_fallback])
sync_resp = sync_client.chat("Give me a one-sentence definition of entropy.")
print("Sync:", sync_resp.text)


async def run_async():
    async_primary = AsyncClient("openai", model="gpt-5-nano")
    async_fallback = AsyncClient("anthropic", model="claude-haiku-4-5-20251001")
    async_client = FallbackClient([async_primary, async_fallback])
    resp = await async_client.achat("Give one practical use of Bayesian inference.")
    print("Async:", resp.text)


asyncio.run(run_async())

