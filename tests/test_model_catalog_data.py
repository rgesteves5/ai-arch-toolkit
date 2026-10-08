"""The shipped seed of the model catalog (C06b, D63): what it holds, and that it agrees with the
provider routing and with the adapters' own tables."""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from ai_arch_toolkit.core import ImageRequest, model_catalog, resolve_provider_name
from ai_arch_toolkit.core._model_id import snapshot_base
from ai_arch_toolkit.core._providers import create_provider

SEED = Path(__file__).resolve().parents[1] / "src/ai_arch_toolkit/core/_default_catalog.toml"

# The current line with an official page (C06.6): the 30 models of the probe inventory that are
# still served, and the image models (C06.7).
_CHAT = {
    "anthropic": {
        "claude-opus-5-5",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-opus-4-6",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
    },
    "openai": {
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-4.1",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5",
        "o3",
    },
    "gemini": {
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    },
    # xAI's own model names; the inventory's ids are their aliases.
    "xai": {
        "grok-4.7",
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-multi-agent-0309",
    },
    "meta": {"muse-spark-1.3"},
}
_IMAGE = {
    "openai": {
        "gpt-image-2.5-sunburst",
        "gpt-image-2.5-flare",
        "gpt-image-2",
        "gpt-image-1.5",
        "gpt-image-1",
        "gpt-image-1-mini",
    },
    "gemini": {"gemini-3.1-flash-image", "gemini-3.1-flash-lite-image", "gemini-3-pro-image"},
    "xai": {"grok-imagine-image-2.0", "grok-imagine-image", "grok-imagine-image-quality"},
    "meta": {"muse-image-1.0"},
}
_INVENTORY_ALIASES = {
    "grok-4.20-reasoning": "grok-4.20-0309-reasoning",
    "grok-4.20-non-reasoning": "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent": "grok-4.20-multi-agent-0309",
}
# Each provider's documentation, the only place a seed fact comes from.
_DOCS_HOSTS = {
    "anthropic": "platform.claude.com",
    "openai": "developers.openai.com",
    "gemini": "ai.google.dev",
    "xai": "docs.x.ai",
    "meta": "dev.meta.ai",
}
# What the seed may say (C06.4): the published limits and modalities, nothing the adapter knows.
_PUBLISHED = {
    "context_window",
    "input_token_limit",
    "output_token_limit",
    "input_modalities",
    "output_modalities",
}


def _pairs(groups: dict[str, set[str]]) -> set[tuple[str, str]]:
    return {(provider, model) for provider, models in groups.items() for model in models}


def _raw() -> dict[str, Any]:
    with SEED.open("rb") as file:
        return tomllib.load(file)


def _raw_entries() -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (provider, model, values)
        for provider, models in _raw().items()
        if isinstance(models, dict)
        for model, values in models.items()
    ]


def test_the_seed_is_the_current_line_and_the_image_models() -> None:
    seeded = {(entry.provider, entry.model) for entry in model_catalog.entries()}

    assert seeded == _pairs(_CHAT) | _pairs(_IMAGE)
    assert len(_pairs(_CHAT)) == 30


def test_the_inventory_ids_find_their_entries() -> None:
    for alias, model in _INVENTORY_ALIASES.items():
        found = model_catalog.get(alias)
        assert found is not None and found.model == model


def test_the_routing_agrees_with_every_id_and_alias() -> None:
    for entry in model_catalog.entries():
        for name in (entry.model, *entry.aliases):
            assert resolve_provider_name(name) == entry.provider, name


def test_no_alias_moves_between_models() -> None:
    """A ``-latest`` pointer changes model, so it is never an alias here (C06.2)."""
    for entry in model_catalog.entries():
        for alias in entry.aliases:
            assert not alias.endswith("-latest"), alias
            assert snapshot_base(alias) != entry.model, alias  # a snapshot needs no alias


def test_the_seed_holds_only_published_limits_and_modalities() -> None:
    for provider, model, values in _raw_entries():
        assert set(values) - {"aliases", "source", "sources"} <= _PUBLISHED, (provider, model)


def test_every_seed_fact_comes_from_its_providers_docs_on_a_past_day() -> None:
    today = datetime.now(UTC).date()
    for entry in model_catalog.entries():
        for name in sorted(_PUBLISHED):
            source = entry.provenance(name)
            if source is None or source.kind == "adapter":
                continue
            assert source.kind == "docs", (entry.model, name)
            assert source.verified_at is not None and source.verified_at <= today
            for url in source.ref.split(", "):
                parts = urlsplit(url)
                assert parts.scheme == "https", (entry.model, name, url)
                assert parts.hostname == _DOCS_HOSTS[entry.provider], (entry.model, name, url)


def test_every_entry_says_what_it_outputs() -> None:
    for entry in model_catalog.entries():
        assert entry.output_modalities is not None, entry.model
        is_image = entry.model in _IMAGE.get(entry.provider, set())
        assert ("image" in entry.output_modalities) is is_image, entry.model


@pytest.mark.parametrize(("provider", "model"), sorted(_pairs(_IMAGE)))
def test_an_image_model_takes_no_tools(provider: str, model: str) -> None:
    found = model_catalog.get(model, provider=provider)

    assert found is not None and found.tools is False
    assert found.server_tools == frozenset()


def test_the_limits_fit_inside_each_other() -> None:
    for entry in model_catalog.entries():
        window = entry.context_window
        for limit in (entry.input_token_limit, entry.output_token_limit):
            assert limit is None or limit > 0
            assert window is None or limit is None or limit <= window, entry.model


@pytest.mark.parametrize("model", sorted(_IMAGE["gemini"]))
def test_a_gemini_image_models_output_limit_is_the_one_its_adapter_reserves(model: str) -> None:
    """The adapter keeps the published output limit to bound an image's thinking (D61); the docs
    give the same number today."""
    found = model_catalog.get(model)
    adapter = create_provider(model, api_key="test-key")

    assert found is not None
    assert found.output_token_limit == adapter.image_text_token_bound(ImageRequest())
