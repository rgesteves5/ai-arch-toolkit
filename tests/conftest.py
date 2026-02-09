"""Shared test fixtures."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_arch_toolkit.llm._types import Tool


@pytest.fixture
def weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )


class MockResponse:
    """Mimics ``requests.Response`` for testing post_json / stream_sse / stream_ndjson."""

    def __init__(
        self,
        json_data: dict[str, Any] | None = None,
        status_code: int = 200,
        text: str = "",
        lines: list[str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or ""
        self.ok = 200 <= status_code < 300
        self._lines = lines or []
        self.headers: dict[str, str] = headers or {}

    def json(self) -> dict[str, Any]:
        if self._json_data is None:
            raise ValueError("No JSON")
        return self._json_data

    def raise_for_status(self) -> None:
        if not self.ok:
            raise Exception(f"HTTP {self.status_code}")

    def iter_lines(self, **_kwargs: object) -> list[str]:
        return self._lines

    def __enter__(self) -> MockResponse:
        return self

    def __exit__(self, *args: object) -> None:
        pass


@pytest.fixture
def mock_post(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Monkeypatch ``requests.post`` and return the mock."""
    mock = MagicMock()
    monkeypatch.setattr("requests.post", mock)
    return mock
