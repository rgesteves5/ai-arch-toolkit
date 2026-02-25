"""Tests for _providers/__init__.py — provider detection and registry."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._providers import _detect_provider
from ai_arch_toolkit.core._providers._imports import require_sdk


class TestDetectProvider:
    def test_claude(self):
        assert _detect_provider("claude-sonnet-4-20250514") == "anthropic"

    def test_gpt(self):
        assert _detect_provider("gpt-4o") == "openai"

    def test_o1(self):
        assert _detect_provider("o1-mini") == "openai"

    def test_o3(self):
        assert _detect_provider("o3-mini") == "openai"

    def test_o4(self):
        assert _detect_provider("o4-mini") == "openai"

    def test_grok(self):
        assert _detect_provider("grok-3-latest") == "xai"

    def test_gemini(self):
        assert _detect_provider("gemini-2.5-flash") == "gemini"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot detect provider"):
            _detect_provider("unknown-model-v1")


class TestRequireSdk:
    def test_installed_package_passes(self):
        require_sdk("json", "test")  # json is always available

    def test_missing_package_raises(self):
        with pytest.raises(ImportError, match="pip install ai-arch-toolkit"):
            require_sdk("nonexistent_package_xyz", "nonexistent")

    def test_error_message_includes_extra_name(self):
        with pytest.raises(ImportError, match=r"\[myextra\]"):
            require_sdk("nonexistent_package_xyz", "myextra")
