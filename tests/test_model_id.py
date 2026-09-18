"""The model-id grammar: an id's own entry, a dated snapshot of one, a family, or unknown."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._model_id import ModelMatch, family, lookup, snapshot_base


@pytest.mark.parametrize(
    ("model", "base"),
    [
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),  # Anthropic
        ("gpt-4.1-2025-04-14", "gpt-4.1"),  # OpenAI
        ("claude-opus-4-5@20251101", "claude-opus-4-5"),  # Vertex AI
        ("grok-4-latest", "grok-4"),
        ("gemini-2.0-flash-001", "gemini-2.0-flash"),
        ("gemini-2.5-flash-preview-05-20", "gemini-2.5-flash"),
        ("gemini-2.5-flash-preview-09-2025", "gemini-2.5-flash"),
        ("grok-4-0709", "grok-4"),  # xAI
        ("gpt-3.5-turbo-0125", "gpt-3.5-turbo"),  # older OpenAI
    ],
)
def test_snapshot_suffixes_of_every_provider(model: str, base: str) -> None:
    assert snapshot_base(model) == base


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-6",
        "claude-fable-5-1",
        "gpt-4o-audio-preview",
        "gpt-5.4",
        "grok-4.6-mini",
        "grok-build-0.1",
        "muse-spark-1.3",
        "gemini-3.1-pro-preview",
    ],
)
def test_versions_and_variants_are_not_snapshots(model: str) -> None:
    assert snapshot_base(model) is None


TABLE = {
    "o3": "o3",
    "gpt-4o": "gpt-4o",
    "gpt-5": "gpt-5",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "grok-4.6": "grok-4.6",
    "claude-fable-5": "claude-fable-5",
}


@pytest.mark.parametrize(
    "model",
    [
        "o3-pro",
        "o3-deep-research",
        "gpt-4o-audio-preview",
        "gpt-5-codex",
        "gemini-2.5-flash-image",
        "grok-4.6-mini",
        "claude-fable-5-1",
    ],
)
def test_a_variant_of_a_known_id_is_unknown(model: str) -> None:
    assert lookup(model, TABLE) is None


def test_resolution_order_is_exact_then_snapshot_then_family() -> None:
    exact = {"gpt-4o": "base", "gpt-4o-2024-05-13": "own tariff"}
    families = {"gpt-": "any gpt", "gpt-4": "gpt-4 family"}
    assert lookup("gpt-4o", exact, families) == ModelMatch("exact", "gpt-4o", "base")
    assert lookup("gpt-4o-2024-05-13", exact, families) == ModelMatch(
        "exact", "gpt-4o-2024-05-13", "own tariff"
    )
    assert lookup("gpt-4o-2024-08-06", exact, families) == ModelMatch("snapshot", "gpt-4o", "base")
    assert lookup("gpt-4-turbo", exact, families) == ModelMatch("family", "gpt-4", "gpt-4 family")
    assert lookup("gpt-4-turbo", exact) is None  # families only when the caller passes them
    assert lookup("claude-opus-5", exact, families) is None


def test_family_is_the_longest_prefix() -> None:
    table = {"claude-": 1.15, "claude-3": 1.12}
    assert family("claude-3-7-sonnet", table) == 1.12
    assert family("claude-opus-5", table) == 1.15
    assert family("gpt-4o", table) is None
