#!/usr/bin/env python3
"""Compare the providers' model lists with the toolkit's pricing table.

Owner-only: it calls each provider's model-list API (free, no prompt is sent) with the keys in
the environment, and prints which listed ids have a price of their own, a price through a dated
snapshot, or none (a metered call to those fails before it is sent), plus table ids no provider
listed. Never run in CI.

    set -a; source .env; set +a
    uv run python scripts/audit_models.py              # every provider with a key
    uv run python scripts/audit_models.py openai xai   # a subset
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from ai_arch_toolkit.core._model_id import snapshot_base
from ai_arch_toolkit.core._pricing import PricingRegistry, pricing

type Price = Literal["exact", "snapshot", "prefix", "none"]


@dataclass(frozen=True, slots=True)
class Row:
    """One listed model id and how it finds a price."""

    provider: str
    model: str
    price: Price


def price_of(model: str, registry: PricingRegistry) -> Price:
    """How ``model`` finds its price in ``registry``."""
    ids = set(registry.list_models())
    if model in ids:
        return "exact"
    base = snapshot_base(model)
    if base is not None and base in ids:
        return "snapshot"
    return "prefix" if registry.has(model) else "none"


def audit(listed: Mapping[str, Iterable[str]], registry: PricingRegistry) -> list[Row]:
    """A row per listed id, grouped by provider and sorted."""
    return [
        Row(provider, model, price_of(model, registry))
        for provider, models in sorted(listed.items())
        for model in sorted(set(models))
    ]


def unlisted(listed: Mapping[str, Iterable[str]], registry: PricingRegistry) -> list[str]:
    """Table ids that no audited provider listed (retired, or an alias the API hides)."""
    seen = {model for models in listed.values() for model in models}
    return [model for model in registry.list_models() if model not in seen]


def render(rows: list[Row], stale: list[str]) -> str:
    lines = ["| Provider | Model | Price |", "|---|---|---|"]
    lines += [f"| {row.provider} | `{row.model}` | {row.price} |" for row in rows]
    missing = [row for row in rows if row.price == "none"]
    lines += ["", f"{len(missing)} listed ids without a price."]
    lines += ["", "Table ids no audited provider listed:", *(f"- `{model}`" for model in stale)]
    return "\n".join(lines)


async def _openai() -> list[str]:
    import openai

    async with openai.AsyncOpenAI() as client:
        return [model.id async for model in client.models.list()]


async def _anthropic() -> list[str]:
    import anthropic

    async with anthropic.AsyncAnthropic() as client:
        return [model.id async for model in client.models.list()]


async def _gemini() -> list[str]:
    from google import genai

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    pager = await client.aio.models.list()
    return [
        (model.name or "").removeprefix("models/")
        async for model in pager
        if "generateContent" in (model.supported_actions or [])
    ]


async def _xai() -> list[str]:
    import xai_sdk

    client = xai_sdk.AsyncClient(api_key=os.environ["XAI_API_KEY"])
    try:
        models = await client.models.list_language_models()
    finally:
        await client.close()
    return [name for model in models for name in (model.name, *model.aliases)]


async def _meta() -> list[str]:
    import openai

    client = openai.AsyncOpenAI(
        api_key=os.environ["MODEL_API_KEY"], base_url="https://api.meta.ai/v1"
    )
    # OPENAI_ORG_ID / OPENAI_PROJECT_ID from the environment identify an OpenAI account (D14).
    client.organization = None
    client.project = None
    async with client:
        return [model.id async for model in client.models.list()]


_FETCH: dict[str, tuple[tuple[str, ...], Callable[[], Awaitable[list[str]]]]] = {
    "anthropic": (("ANTHROPIC_API_KEY",), _anthropic),
    "openai": (("OPENAI_API_KEY",), _openai),
    "gemini": (("GOOGLE_API_KEY", "GEMINI_API_KEY"), _gemini),
    "xai": (("XAI_API_KEY",), _xai),
    "meta": (("MODEL_API_KEY",), _meta),
}


async def _listed(providers: list[str]) -> dict[str, list[str]]:
    listed: dict[str, list[str]] = {}
    for provider in providers:
        keys, fetch = _FETCH[provider]
        if not any(os.environ.get(key) for key in keys):
            print(f"skip {provider}: none of {', '.join(keys)} is set")
            continue
        listed[provider] = await fetch()
    return listed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model lists with the pricing table.")
    parser.add_argument("providers", nargs="*", help=f"any of {', '.join(_FETCH)}; default all")
    args = parser.parse_args()
    if unknown := set(args.providers) - set(_FETCH):
        parser.error(f"unknown providers: {sorted(unknown)}")
    listed = asyncio.run(_listed(args.providers or list(_FETCH)))
    print(render(audit(listed, pricing), unlisted(listed, pricing)))


if __name__ == "__main__":
    main()
