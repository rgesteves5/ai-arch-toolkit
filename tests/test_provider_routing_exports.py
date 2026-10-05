"""The model routing and the loopback rule are public (G-14): an app reads them instead of
copying them."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core import MODEL_IDS, MODEL_PREFIXES, is_local_url, resolve_provider_name


def test_every_prefix_routes_to_its_provider() -> None:
    for prefix, provider in MODEL_PREFIXES.items():
        assert resolve_provider_name(f"{prefix}x") == provider
    for model, provider in MODEL_IDS.items():
        assert resolve_provider_name(model) == provider


def test_the_table_has_every_family_the_registry_routes() -> None:
    assert MODEL_PREFIXES["muse-spark-"] == "meta"
    assert MODEL_PREFIXES["claude-"] == "anthropic"
    assert set(MODEL_PREFIXES.values()) == {"anthropic", "openai", "xai", "gemini", "meta"}


def test_the_tables_cannot_be_changed_from_outside() -> None:
    with pytest.raises(TypeError):
        MODEL_PREFIXES["mistral-"] = "openai"  # type: ignore[index]
    with pytest.raises(TypeError):
        MODEL_IDS["o5"] = "openai"  # type: ignore[index]


def test_an_unknown_model_with_a_base_url_goes_to_the_openai_compatible_adapter() -> None:
    assert resolve_provider_name("llama3.2", base_url="http://localhost:11434/v1") == "openai"
    with pytest.raises(ValueError, match="Cannot detect provider"):
        resolve_provider_name("llama3.2")


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:8000", True),
        ("http://127.5.6.7", True),
        ("http://[::1]:8080/v1", True),
        ("http://0.0.0.0:1234", True),
        ("http://lmstudio.localhost/v1", True),
        ("https://api.openai.com/v1", False),
        ("http://192.168.1.10:11434", False),
        ("", False),
        (None, False),
    ],
)
def test_the_loopback_rule(url: str | None, local: bool) -> None:
    assert is_local_url(url) is local
