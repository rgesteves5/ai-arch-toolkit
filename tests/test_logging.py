"""Tests for package logging setup and warning events."""

from __future__ import annotations

import logging

import pytest

import ai_arch_toolkit
from ai_arch_toolkit.llm._async_client import AsyncClient
from ai_arch_toolkit.llm._client import Client
from ai_arch_toolkit.llm._providers import _resolve_key, create_provider


def test_package_logger_has_null_handler() -> None:
    _ = ai_arch_toolkit.__version__
    logger = logging.getLogger("ai_arch_toolkit")
    assert any(isinstance(handler, logging.NullHandler) for handler in logger.handlers)


def test_sync_client_logs_warning_for_empty_messages(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="ai_arch_toolkit")
    client = Client.__new__(Client)

    _ = client._normalize_input([])

    assert "Received empty prompt_or_messages sequence" in caplog.text


def test_async_client_logs_warning_for_empty_messages(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="ai_arch_toolkit")
    client = AsyncClient.__new__(AsyncClient)

    _ = client._normalize_input([])

    assert "Received empty prompt_or_messages sequence" in caplog.text


def test_provider_logs_warning_for_missing_api_key(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="ai_arch_toolkit")

    with pytest.raises(ValueError):
        _resolve_key("MISSING_ENV_KEY", api_key=None)

    assert "Missing API key for env var MISSING_ENV_KEY" in caplog.text


def test_provider_logs_warning_for_unknown_provider(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="ai_arch_toolkit")

    with pytest.raises(ValueError):
        create_provider("not-a-provider", "gpt-4o")

    assert "Unknown provider requested: not-a-provider" in caplog.text
