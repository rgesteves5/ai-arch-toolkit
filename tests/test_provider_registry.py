"""Tests for _providers/__init__.py — provider detection and registry."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from ai_arch_toolkit.core._providers import (
    _detect_provider,
    _is_local_url,
    _match_provider,
    _resolve_key,
    create_provider,
    resolve_provider_name,
)
from ai_arch_toolkit.core._providers._imports import require_sdk


class TestDetectProvider:
    def test_claude(self):
        assert _detect_provider("claude-sonnet-4-20250514") == "anthropic"

    def test_gpt(self):
        assert _detect_provider("gpt-4o") == "openai"

    def test_chat_latest(self):
        assert _detect_provider("chat-latest") == "openai"

    def test_o1(self):
        assert _detect_provider("o1-mini") == "openai"

    def test_o3(self):
        assert _detect_provider("o3-mini") == "openai"

    def test_exact_o3(self):
        assert _detect_provider("o3") == "openai"

    def test_o4(self):
        assert _detect_provider("o4-mini") == "openai"

    def test_grok(self):
        assert _detect_provider("grok-3-latest") == "xai"

    def test_gemini(self):
        assert _detect_provider("gemini-2.5-flash") == "gemini"

    def test_muse_spark(self):
        assert _detect_provider("muse-spark-1.3-contributor") == "meta"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot detect provider"):
            _detect_provider("unknown-model-v1")

    def test_unknown_error_mentions_provider_hint(self):
        with pytest.raises(ValueError, match="provider="):
            _detect_provider("gemma4:e4b")


class TestMatchProvider:
    def test_known_prefix(self):
        assert _match_provider("gpt-4o") == "openai"

    def test_known_model_id(self):
        assert _match_provider("o3") == "openai"

    def test_unknown_returns_none(self):
        assert _match_provider("gemma4:e4b") is None


class TestResolveProviderName:
    """The routing decision create_provider acts on, without building a provider."""

    @pytest.mark.parametrize(
        ("model", "kwargs", "expected"),
        [
            ("claude-sonnet-4-6", {}, "anthropic"),
            ("o3", {}, "openai"),
            ("grok-4", {}, "xai"),
            ("gemini-2.5-flash", {}, "gemini"),
            ("muse-spark-1.3", {}, "meta"),
            # Self-hosted Muse Glimmer is served by local OpenAI-compatible runtimes.
            ("muse-glimmer-1.0", {"base_url": "http://localhost:8000/v1"}, "openai"),
            ("gemma4:e4b", {"base_url": "http://localhost:11434/v1"}, "openai"),
            ("claude-sonnet-4-6", {"provider": "openai"}, "openai"),
        ],
    )
    def test_routes_like_create_provider(self, model, kwargs, expected):
        assert resolve_provider_name(model, **kwargs) == expected

    @pytest.mark.parametrize("kwargs", [{}, {"base_url": ""}, {"provider": "nope"}])
    def test_raises_the_same_error_as_create_provider(self, kwargs):
        with pytest.raises(ValueError) as resolved:
            resolve_provider_name("gemma4:e4b", **kwargs)
        with pytest.raises(ValueError) as created:
            create_provider("gemma4:e4b", api_key="k", **kwargs)
        assert str(resolved.value) == str(created.value)


class TestRequireSdk:
    def test_installed_package_passes(self):
        with require_sdk("test"):
            __import__("json")  # json is always available

    def test_missing_package_raises(self):
        with pytest.raises(ImportError, match="pip install ai-arch-toolkit"), require_sdk("x"):
            __import__("nonexistent_package_xyz")

    def test_error_message_includes_extra_name(self):
        with pytest.raises(ImportError, match=r"\[myextra\]") as missing, require_sdk("myextra"):
            __import__("nonexistent_package_xyz")
        assert isinstance(missing.value.__cause__, ModuleNotFoundError)  # which import failed


class TestResolveKey:
    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("FOO_KEY", "env-value")
        assert _resolve_key("FOO_KEY", "explicit") == "explicit"

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("FOO_KEY", "env-value")
        assert _resolve_key("FOO_KEY", None) == "env-value"

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("FOO_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key provided"):
            _resolve_key("FOO_KEY", None)

    def test_local_returns_placeholder(self, monkeypatch):
        monkeypatch.delenv("FOO_KEY", raising=False)
        assert _resolve_key("FOO_KEY", None, local=True) == "not-needed"

    def test_local_ignores_env_key(self, monkeypatch):
        # A real cloud key is never sent to a local server.
        monkeypatch.setenv("FOO_KEY", "env-value")
        assert _resolve_key("FOO_KEY", None, local=True) == "not-needed"

    def test_local_explicit_still_wins(self, monkeypatch):
        monkeypatch.delenv("FOO_KEY", raising=False)
        assert _resolve_key("FOO_KEY", "explicit", local=True) == "explicit"

    def test_multiple_env_vars_use_first_set(self, monkeypatch):
        monkeypatch.setenv("FIRST_KEY", "first")
        monkeypatch.setenv("SECOND_KEY", "second")
        assert _resolve_key(("FIRST_KEY", "SECOND_KEY"), None) == "first"

    def test_multiple_env_vars_fall_back_in_order(self, monkeypatch):
        monkeypatch.delenv("FIRST_KEY", raising=False)
        monkeypatch.setenv("SECOND_KEY", "second")
        assert _resolve_key(("FIRST_KEY", "SECOND_KEY"), None) == "second"

    def test_multiple_env_vars_error_mentions_all_names(self, monkeypatch):
        monkeypatch.delenv("FIRST_KEY", raising=False)
        monkeypatch.delenv("SECOND_KEY", raising=False)
        with pytest.raises(ValueError, match="FIRST_KEY, or SECOND_KEY"):
            _resolve_key(("FIRST_KEY", "SECOND_KEY"), None)


class TestCreateProvider:
    """Verify the factory routes to the right adapter and honors base_url / timeout."""

    def test_anthropic_route(self):
        with patch("ai_arch_toolkit.core._providers._anthropic.AnthropicProvider") as cls:
            create_provider(
                "claude-haiku-4-5", api_key="test-key", base_url="https://x", timeout=10.0
            )
            cls.assert_called_once_with(
                "claude-haiku-4-5", "test-key", base_url="https://x", timeout=10.0
            )

    def test_openai_route(self):
        with patch("ai_arch_toolkit.core._providers._openai.OpenAIProvider") as cls:
            create_provider("gpt-4.1-nano", api_key="test-key", base_url="https://x", timeout=20.0)
            cls.assert_called_once_with(
                "gpt-4.1-nano", "test-key", base_url="https://x", timeout=20.0
            )

    def test_xai_route(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        with patch("ai_arch_toolkit.core._providers._xai.XAIProvider") as cls:
            create_provider("grok-4", timeout=15.0)
            cls.assert_called_once_with("grok-4", "test-key", timeout=15.0)

    def test_xai_warns_on_base_url(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        with (
            patch("ai_arch_toolkit.core._providers._xai.XAIProvider"),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            create_provider("grok-4", base_url="https://override")
            assert any("base_url is not supported" in str(w.message) for w in caught)

    def test_gemini_route(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch("ai_arch_toolkit.core._providers._gemini.GeminiProvider") as cls:
            create_provider("gemini-2.5-flash", timeout=12.0)
            cls.assert_called_once_with("gemini-2.5-flash", "test-key", timeout=12.0)

    def test_gemini_route_accepts_gemini_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        with patch("ai_arch_toolkit.core._providers._gemini.GeminiProvider") as cls:
            create_provider("gemini-2.5-flash", timeout=12.0)
            cls.assert_called_once_with("gemini-2.5-flash", "gemini-key", timeout=12.0)

    def test_gemini_route_prefers_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        with patch("ai_arch_toolkit.core._providers._gemini.GeminiProvider") as cls:
            create_provider("gemini-2.5-flash", timeout=12.0)
            cls.assert_called_once_with("gemini-2.5-flash", "google-key", timeout=12.0)

    def test_gemini_route_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        with patch("ai_arch_toolkit.core._providers._gemini.GeminiProvider") as cls:
            create_provider("gemini-2.5-flash", api_key="explicit-key", timeout=12.0)
            cls.assert_called_once_with("gemini-2.5-flash", "explicit-key", timeout=12.0)

    def test_gemini_missing_key_mentions_supported_env_vars(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY, or GEMINI_API_KEY"):
            create_provider("gemini-2.5-flash")

    def test_gemini_warns_on_base_url(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        with (
            patch("ai_arch_toolkit.core._providers._gemini.GeminiProvider"),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            create_provider("gemini-2.5-flash", base_url="https://override")
            assert any("base_url is not supported" in str(w.message) for w in caught)

    def test_meta_route(self):
        with patch("ai_arch_toolkit.core._providers._meta.MetaProvider") as cls:
            create_provider(
                "muse-spark-1.3",
                api_key="meta-key",
                base_url="https://gateway.example/v1",
                timeout=30.0,
            )
            cls.assert_called_once_with(
                "muse-spark-1.3", "meta-key", base_url="https://gateway.example/v1", timeout=30.0
            )

    def test_meta_never_falls_back_to_the_openai_key(self, monkeypatch):
        monkeypatch.delenv("MODEL_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        with pytest.raises(ValueError, match="MODEL_API_KEY"):
            create_provider("muse-spark-1.3")

    def test_meta_builds_a_real_provider(self, monkeypatch):
        from ai_arch_toolkit.core._providers._meta import MetaProvider

        monkeypatch.setenv("MODEL_API_KEY", "meta-key")
        assert isinstance(create_provider("muse-spark-1.3"), MetaProvider)
        assert isinstance(create_provider("future-muse", provider="meta"), MetaProvider)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Cannot detect provider"):
            create_provider("unknown-model-id")

    def test_explicit_provider_overrides_detection(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("ai_arch_toolkit.core._providers._openai.OpenAIProvider") as cls:
            create_provider(
                "gemma4:e4b",
                provider="openai",
                api_key="k",
                base_url="http://localhost:11434/v1",
            )
            cls.assert_called_once_with(
                "gemma4:e4b", "k", base_url="http://localhost:11434/v1", timeout=None
            )

    def test_explicit_provider_invalid_raises(self):
        with pytest.raises(ValueError, match="Valid providers"):
            create_provider("gemma4:e4b", provider="nope")

    def test_unknown_model_with_localhost_infers_openai(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("ai_arch_toolkit.core._providers._openai.OpenAIProvider") as cls:
            create_provider("gemma4:e4b", base_url="http://localhost:11434/v1")
            cls.assert_called_once_with(
                "gemma4:e4b", "not-needed", base_url="http://localhost:11434/v1", timeout=None
            )

    def test_localhost_known_prefix_uses_placeholder(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("ai_arch_toolkit.core._providers._openai.OpenAIProvider") as cls:
            create_provider("gpt-4o", base_url="http://127.0.0.1:8000/v1")
            cls.assert_called_once_with(
                "gpt-4o", "not-needed", base_url="http://127.0.0.1:8000/v1", timeout=None
            )

    def test_localhost_ignores_env_key(self, monkeypatch):
        # A real cloud key in the environment is never sent to a local server.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        with patch("ai_arch_toolkit.core._providers._openai.OpenAIProvider") as cls:
            create_provider("gemma4:e4b", base_url="http://localhost:11434/v1")
            cls.assert_called_once_with(
                "gemma4:e4b", "not-needed", base_url="http://localhost:11434/v1", timeout=None
            )

    @pytest.mark.parametrize(
        ("model", "env_var", "own_base_url"),
        [
            ("claude-haiku-4-5", "ANTHROPIC_API_KEY", "https://api.anthropic.com"),
            ("gpt-4o", "OPENAI_API_KEY", "https://api.openai.com/v1"),
            ("muse-spark-1.3", "MODEL_API_KEY", "https://api.meta.ai/v1"),
        ],
    )
    def test_environment_keys_only_reach_the_providers_own_host(
        self, monkeypatch, model, env_var, own_base_url
    ):
        monkeypatch.setenv(env_var, "env-key")

        with pytest.raises(ValueError, match=f"api_key=.*{env_var}"):
            create_provider(model, base_url="https://gateway.example/v1")

        provider = create_provider(model, base_url=own_base_url)
        assert provider._client.api_key == "env-key"  # type: ignore[attr-defined]

    def test_another_vendors_openai_compatible_server_never_gets_the_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        base_url = "https://api.together.xyz/v1"

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_provider("llama-3.3-70b", base_url=base_url)

        with patch("ai_arch_toolkit.core._providers._openai.OpenAIProvider") as cls:
            create_provider("llama-3.3-70b", base_url=base_url, api_key="together-key")
            cls.assert_called_once_with(
                "llama-3.3-70b", "together-key", base_url=base_url, timeout=None
            )

    def test_remote_base_url_missing_key_raises(self, monkeypatch):
        # A remote endpoint (gateway/proxy) still needs a key — fail fast.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key"):
            create_provider("gpt-4o", base_url="https://openrouter.ai/api/v1")

    def test_empty_base_url_normalized_requires_key(self, monkeypatch):
        # base_url="" must not disable the key check for a real cloud model.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key"):
            create_provider("gpt-4o", base_url="")

    def test_explicit_provider_without_base_url_still_requires_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key"):
            create_provider("gemma4:e4b", provider="openai")

    def test_unknown_model_without_base_url_still_raises(self):
        with pytest.raises(ValueError, match="Cannot detect provider"):
            create_provider("gemma4:e4b")


class TestIsLocalUrl:
    def test_loopback_hosts(self):
        assert _is_local_url("http://localhost:11434/v1")
        assert _is_local_url("http://127.0.0.1:8000/v1")
        assert _is_local_url("http://127.5.5.5/v1")
        assert _is_local_url("http://[::1]:8000/v1")
        assert _is_local_url("http://0.0.0.0:1234")

    def test_remote_hosts(self):
        assert not _is_local_url("https://openrouter.ai/api/v1")
        assert not _is_local_url("https://api.openai.com/v1")
        assert not _is_local_url("http://192.168.1.50:11434/v1")

    def test_none_and_empty(self):
        assert not _is_local_url(None)
        assert not _is_local_url("")
