"""Unit tests for HTTP timeout support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._sync import (
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
    mock_provider.assert_called_once_with("gpt-4", api_key="k", base_url=None, timeout=30.0)


def test_llm_without_timeout(mock_provider):
    """LLM without timeout passes None."""
    LLM("gpt-4", api_key="k")
    mock_provider.assert_called_once_with("gpt-4", api_key="k", base_url=None, timeout=None)


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


def test_llm_rejects_negative_timeout(mock_provider):
    """LLM rejects negative timeout."""
    with pytest.raises(ValueError):
        LLM("gpt-4", timeout=-1.0, api_key="k")
