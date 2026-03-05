"""Shared fixtures for core graph tests."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core.graph._networkx import NetworkXBackend
from ai_arch_toolkit.core.graph._store import Graph


@pytest.fixture
def backend() -> NetworkXBackend:
    return NetworkXBackend()


@pytest.fixture
def graph(backend: NetworkXBackend) -> Graph:
    return Graph(backend)
