"""Unit tests for HTTP timeout support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._sync import (
    _read_positive_float_env,
    configure_sync_timeouts,
)


@pytest.fixture
def mock_provider():
    """Patch create_provider for all tests that need it."""
    with patch("ai_arch_toolkit.core._llm.create_provider") as mock_cp:
        mock_cp.return_value = MagicMock()
        yield mock_cp


def test_llm_passes_timeout_to_create_provider(mock_provider):
    """LLM passes timeout to create_provider."""
    LLM("gpt-4", timeout=30.0, api_key="k")
    mock_provider.assert_called_once_with(
        "gpt-4", provider=None, api_key="k", base_url=None, timeout=30.0
    )


def test_llm_without_timeout(mock_provider):
    """LLM without timeout passes None."""
    LLM("gpt-4", api_key="k")
    mock_provider.assert_called_once_with(
        "gpt-4", provider=None, api_key="k", base_url=None, timeout=None
    )


def test_llm_repr_includes_timeout(mock_provider):
    """LLM repr includes timeout when set."""
    llm = LLM("gpt-4", timeout=60.0, api_key="k")
    assert "timeout=60.0" in repr(llm)


def test_llm_repr_without_timeout(mock_provider):
    """LLM repr omits timeout when not set."""
    llm = LLM("gpt-4", api_key="k")
    assert "timeout" not in repr(llm)


def test_configure_sync_timeouts_works():
    """configure_sync_timeouts updates module-level vars."""
    import ai_arch_toolkit.core._sync as sync_mod

    original_sync = sync_mod._sync_timeout
    original_stream = sync_mod._stream_join_timeout
    try:
        configure_sync_timeouts(sync_timeout=10.0, stream_join_timeout=2.0)
        assert sync_mod._sync_timeout == 10.0
        assert sync_mod._stream_join_timeout == 2.0
    finally:
        configure_sync_timeouts(sync_timeout=original_sync, stream_join_timeout=original_stream)


def test_configure_sync_timeouts_rejects_negative():
    """configure_sync_timeouts rejects negative sync_timeout."""
    with pytest.raises(ValueError):
        configure_sync_timeouts(sync_timeout=-1)


def test_configure_sync_timeouts_rejects_zero_stream():
    """configure_sync_timeouts rejects zero stream_join_timeout."""
    with pytest.raises(ValueError):
        configure_sync_timeouts(stream_join_timeout=0)


def test_sync_timeout_env_default(monkeypatch):
    """Sync timeout env helper uses default when unset."""
    monkeypatch.delenv("AI_ARCH_SYNC_TIMEOUT", raising=False)
    assert _read_positive_float_env("AI_ARCH_SYNC_TIMEOUT", 300.0) == 300.0


def test_sync_timeout_env_accepts_positive_number(monkeypatch):
    """Sync timeout env helper parses positive numeric values."""
    monkeypatch.setenv("AI_ARCH_SYNC_TIMEOUT", "12.5")
    assert _read_positive_float_env("AI_ARCH_SYNC_TIMEOUT", 300.0) == 12.5


def test_sync_timeout_env_rejects_invalid_number(monkeypatch):
    """Sync timeout env helper rejects invalid values with a clear error."""
    monkeypatch.setenv("AI_ARCH_SYNC_TIMEOUT", "soon")
    with pytest.raises(
        ValueError,
        match="AI_ARCH_SYNC_TIMEOUT must be a positive number of seconds, got 'soon'",
    ):
        _read_positive_float_env("AI_ARCH_SYNC_TIMEOUT", 300.0)


def test_sync_timeout_env_rejects_non_positive_number(monkeypatch):
    """Sync timeout env helper rejects non-positive values with a clear error."""
    monkeypatch.setenv("AI_ARCH_STREAM_JOIN_TIMEOUT", "0")
    with pytest.raises(
        ValueError,
        match="AI_ARCH_STREAM_JOIN_TIMEOUT must be a positive number of seconds, got '0'",
    ):
        _read_positive_float_env("AI_ARCH_STREAM_JOIN_TIMEOUT", 5.0)


def test_llm_rejects_negative_timeout(mock_provider):
    """LLM rejects negative timeout."""
    with pytest.raises(ValueError):
        LLM("gpt-4", timeout=-1.0, api_key="k")
