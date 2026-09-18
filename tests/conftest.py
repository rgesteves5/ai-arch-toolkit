"""Shared test fixtures."""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._providers._base import Prepared
from ai_arch_toolkit.toolkit.tools import _http
from tests.wire_contract import ADAPTERS, WireLog


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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "wire_contract(tolerate): regexes of the SDK-contract violations a test sends on purpose",
    )


def _checked(adapter: type[Any], log: WireLog) -> Callable[[Any, Request], Prepared[Any]]:
    prepare = adapter.prepare
    name = adapter.__name__

    def checked(self: Any, request: Request) -> Prepared[Any]:
        prepared = prepare(self, request)
        log.record(name, prepared.params)
        return prepared

    return checked


@pytest.fixture(autouse=True)
def wire_log(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[WireLog]:
    """Every request an adapter prepares in a test, checked against its SDK's own contract
    (``tests/wire_contract.py``, R02 step 4). A violation fails the test, unless the test sends it
    on purpose and says so with ``@pytest.mark.wire_contract(tolerate=[regex, ...])``."""
    log = WireLog()
    for adapter in ADAPTERS:
        monkeypatch.setattr(adapter, "prepare", _checked(adapter, log))
    yield log
    marker = request.node.get_closest_marker("wire_contract")
    if found := log.unexpected(marker.kwargs["tolerate"] if marker else ()):
        pytest.fail("requests outside the SDK's contract:\n  " + "\n  ".join(found), pytrace=False)


# --- Toolkit tool tests run offline ---------------------------------------------------------
#
# They simulate the network at the tools' one seam, ``_http._open`` (see ``tests/toolkit/
# http_fakes.py``); a test that forgets to meets a blocked socket, and the tool returns its error
# string instead of making a real call. These fixtures live here and not in a
# ``tests/toolkit/conftest.py``: pytest 9.1 drops a directory conftest's fixtures when a command
# line interleaves that directory's files with other directories' (measured in R03).

_TOOLKIT_TESTS = Path(__file__).resolve().parent / "toolkit"


def _refuse(*_args: object, **_kwargs: object) -> None:
    msg = "network access is blocked in toolkit tests"
    raise OSError(msg)


@pytest.fixture(autouse=True)
def _toolkit_sockets_blocked(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.path.is_relative_to(_TOOLKIT_TESTS):
        monkeypatch.setattr(socket.socket, "connect", _refuse)
        monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
        monkeypatch.setattr(socket, "create_connection", _refuse)
        monkeypatch.setattr(socket, "getaddrinfo", _refuse)


@pytest.fixture(autouse=True)
def throttle_waits(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The tools' throttle waits, in seconds, recorded instead of slept (in toolkit tests)."""
    waits: list[float] = []
    if request.path.is_relative_to(_TOOLKIT_TESTS):
        monkeypatch.setattr(_http, "_THROTTLE", _http._Throttle(sleep=waits.append))
    return waits
